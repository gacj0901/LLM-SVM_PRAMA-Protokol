#!/usr/bin/env python
"""Evaluate final-state discrimination against logprob-derived baselines."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from aptadynamic_llm.evaluation.state_discrimination_metrics import (
    BASELINE_SCORE_FIELDS,
    PRIMARY_PRAMA_SCORE,
    PRAMA_SCORE_FIELDS,
    REQUIRED_LABEL_FIELDS,
    auroc,
    baseline_score_from_tokens,
    confusion_at_threshold,
    prama_score_from_row,
    permutation_pvalue,
    precision_recall_at_threshold,
    safe_float,
    threshold_at_fpr,
)

def load_labels(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"labels file not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SystemExit("labels.csv is empty or missing a header")
        missing = REQUIRED_LABEL_FIELDS.difference(reader.fieldnames)
        if missing:
            raise SystemExit(f"labels.csv missing required fields: {', '.join(sorted(missing))}")
        rows = {row["session_id"]: row for row in reader if row.get("session_id")}
    if not rows:
        raise SystemExit("labels.csv contains no labeled sessions")
    return rows


def _session_id_from_result(raw: dict[str, Any], path: Path) -> str:
    return str(raw.get("session_id") or path.stem)


def _flatten_turns(raw: dict[str, Any]) -> list[dict[str, Any]]:
    turns = raw.get("turns") or []
    out: list[dict[str, Any]] = []
    cumulative_tokens: list[dict[str, Any]] = []
    for turn in turns:
        tokens = list(turn.get("tokens") or [])
        cumulative_tokens.extend(tokens)
        token_count = int(turn.get("token_count") or len(tokens) or 0)
        cumulative_position = len(cumulative_tokens) if cumulative_tokens else sum(int(row.get("token_count", 0)) for row in out) + token_count
        summary = turn.get("metrics_summary") or turn.get("summary") or turn
        row = dict(summary)
        row["turn_index"] = int(turn.get("turn_index") or len(out))
        row["token_position"] = cumulative_position
        row["token_count"] = token_count
        row["tokens_causal_prefix"] = list(cumulative_tokens)
        for field in BASELINE_SCORE_FIELDS:
            row.setdefault(field, baseline_score_from_tokens(field, cumulative_tokens))
        out.append(row)
    return out


def _derive_score(row: dict[str, Any], field: str) -> float:
    if field in PRAMA_SCORE_FIELDS:
        return prama_score_from_row(field, row)
    return safe_float(row.get(field))


def load_examples(paths: list[Path], labels: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        session_id = _session_id_from_result(raw, path)
        label = labels.get(session_id)
        if not label:
            raise SystemExit(f"missing label for session_id {session_id} from {path}")
        turns = _flatten_turns(raw)
        if not turns:
            raise SystemExit(f"input has no turns: {path}")
        event_token = int(float(label["event_token"]))
        final_token = int(turns[-1].get("token_position", event_token))
        final_outcome_proxy = event_token >= final_token
        # Causal choice: latest available row at or before event_token.
        chosen = turns[0]
        for turn in turns:
            if int(turn.get("token_position", 0)) <= event_token:
                chosen = turn
        example = {
            "session_id": session_id,
            "label": label["label"],
            "event_token": event_token,
            "event_turn": int(float(label["event_turn"])),
            "event_type": label["event_type"],
            "token_position": int(chosen.get("token_position", event_token)),
            "turn_index": int(chosen.get("turn_index", 0)),
            "split": (label.get("split") or "test").strip().lower(),
            "final_outcome_proxy": final_outcome_proxy,
            "scores": {},
        }
        for field in PRAMA_SCORE_FIELDS + BASELINE_SCORE_FIELDS:
            example["scores"][field] = _derive_score(chosen, field)
        examples.append(example)
    return examples


def _safe_auroc(scores: list[float], labels: list[Any]) -> float | None:
    try:
        return auroc(scores, labels)
    except ValueError:
        return None


def _metric_block(train: list[dict[str, Any]], test: list[dict[str, Any]], field: str, target_fpr: float) -> dict[str, Any]:
    labels_train = [row["label"] for row in train]
    labels_test = [row["label"] for row in test]
    train_scores = [row["scores"][field] for row in train]
    test_scores = [row["scores"][field] for row in test]
    try:
        threshold = threshold_at_fpr(train_scores, labels_train, target_fpr)
        confusion = confusion_at_threshold(test_scores, labels_test, threshold)
        precision_recall = precision_recall_at_threshold(test_scores, labels_test, threshold)
        error = None
    except ValueError as exc:
        threshold = None
        confusion = None
        precision_recall = None
        error = str(exc)
    return {
        "threshold": threshold,
        "auroc": _safe_auroc(test_scores, labels_test),
        "confusion": confusion,
        "precision_recall": precision_recall,
        "error": error,
    }


def _rate(confusion: dict[str, int] | None, numerator: str, denominator: tuple[str, str]) -> float | None:
    if confusion is None:
        return None
    total = confusion[denominator[0]] + confusion[denominator[1]]
    return confusion[numerator] / total if total else None


def _matched_fpr_block(prama: dict[str, Any], baseline: dict[str, Any], target_fpr: float) -> dict[str, Any]:
    """Compare test performance using thresholds calibrated on the train split."""

    p_conf = prama.get("confusion")
    b_conf = baseline.get("confusion")
    p_tpr = _rate(p_conf, "tp", ("tp", "fn"))
    b_tpr = _rate(b_conf, "tp", ("tp", "fn"))
    p_fpr = _rate(p_conf, "fp", ("fp", "tn"))
    b_fpr = _rate(b_conf, "fp", ("fp", "tn"))
    p_auc, b_auc = prama.get("auroc"), baseline.get("auroc")
    return {
        "target_fpr": target_fpr,
        "prama_threshold_train": prama.get("threshold"),
        "baseline_threshold_train": baseline.get("threshold"),
        "prama_test_fpr": p_fpr,
        "baseline_test_fpr": b_fpr,
        "prama_test_tpr": p_tpr,
        "baseline_test_tpr": b_tpr,
        "tpr_delta": None if p_tpr is None or b_tpr is None else p_tpr - b_tpr,
        "prama_auroc": p_auc,
        "baseline_auroc": b_auc,
        "auroc_delta": None if p_auc is None or b_auc is None else p_auc - b_auc,
        "prama_confusion": p_conf,
        "baseline_confusion": b_conf,
    }


def _confirmatory_verdict(
    primary: dict[str, Any],
    baselines: dict[str, dict[str, Any]],
    permutation_p: float | None,
) -> dict[str, Any]:
    primary_auc = primary.get("auroc")
    primary_tpr = _rate(primary.get("confusion"), "tp", ("tp", "fn"))
    baseline_aucs = [row.get("auroc") for row in baselines.values() if row.get("auroc") is not None]
    baseline_tprs = [
        value
        for row in baselines.values()
        if (value := _rate(row.get("confusion"), "tp", ("tp", "fn"))) is not None
    ]
    if primary_auc is None or primary_tpr is None or not baseline_aucs or not baseline_tprs:
        verdict = "inconclusive"
        best_auc = best_tpr = None
    else:
        best_auc = max(baseline_aucs)
        best_tpr = max(baseline_tprs)
        verdict = (
            "positive"
            if permutation_p is not None
            and permutation_p < 0.01
            and primary_auc > best_auc
            and primary_tpr > best_tpr
            else "honest_null"
        )
    return {
        "verdict": verdict,
        "primary_auroc": primary_auc,
        "best_baseline_auroc": best_auc,
        "auroc_delta": None if primary_auc is None or best_auc is None else primary_auc - best_auc,
        "primary_test_tpr": primary_tpr,
        "best_baseline_test_tpr": best_tpr,
        "tpr_delta": None if primary_tpr is None or best_tpr is None else primary_tpr - best_tpr,
        "permutation_p": permutation_p,
        "rule": "permutation p < 0.01 and primary exceeds every baseline by AUROC and test TPR at train-calibrated target FPR",
    }


def evaluate_split(
    train: list[dict[str, Any]],
    test: list[dict[str, Any]],
    target_fpr: float,
    primary_score: str = PRIMARY_PRAMA_SCORE,
    permutations: int = 1000,
    permutation_seed: int = 0,
) -> dict[str, Any]:
    if primary_score not in PRAMA_SCORE_FIELDS:
        raise ValueError(f"unknown primary PRAMA score: {primary_score}")
    if not train:
        train = test
    output: dict[str, Any] = {"target_fpr": target_fpr, "primary_score": primary_score, "prama": {}, "baselines": {}, "matched_fpr": {}}

    for field in PRAMA_SCORE_FIELDS:
        output["prama"][field] = _metric_block(train, test, field, target_fpr)

    for field in BASELINE_SCORE_FIELDS:
        output["baselines"][field] = _metric_block(train, test, field, target_fpr)

    for field in BASELINE_SCORE_FIELDS:
        output["matched_fpr"][field] = _matched_fpr_block(
            output["prama"][primary_score], output["baselines"][field], target_fpr
        )

    comparable_prama = [(name, data["auroc"]) for name, data in output["prama"].items() if data["auroc"] is not None]
    comparable_base = [(name, data["auroc"]) for name, data in output["baselines"].items() if data["auroc"] is not None]
    try:
        primary_permutation_p = permutation_pvalue(
            [row["scores"][primary_score] for row in test],
            [row["label"] for row in test],
            permutations=permutations,
            seed=permutation_seed,
        )
    except ValueError:
        primary_permutation_p = None
    output["primary_confirmatory"] = _confirmatory_verdict(
        output["prama"][primary_score], output["baselines"], primary_permutation_p
    )
    output["best_prama_by_auroc_exploratory"] = max(comparable_prama, key=lambda item: item[1], default=None)
    output["best_baseline_by_auroc"] = max(comparable_base, key=lambda item: item[1], default=None)
    output["status"] = output["primary_confirmatory"]["verdict"]
    return output


def write_report(path: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# Final-State Discrimination Evaluation",
        "",
        "The preregistered latent-occupancy channel is evaluated as final-state discrimination, not as token-localized early warning.",
        "",
        "## Null Hypothesis",
        "",
        "The preregistered latent-occupancy channel adds no discriminative signal beyond surprisal-derived baselines.",
        "",
        "## Summary",
        "",
        f"- target_fpr: `{metrics['target_fpr']}`",
        f"- primary_score: `{metrics['primary_score']}`",
        f"- permutations: `{metrics['permutations']}` (seed `{metrics['permutation_seed']}`)",
        f"- n_examples: `{metrics['n_examples']}`",
        f"- split_aware: `{metrics['split_aware']}`",
        f"- final_outcome_proxy_count: `{metrics['final_outcome_proxy_count']}`",
    ]
    for split_name, split_metrics in metrics["splits"].items():
        lines.extend([
            "",
            f"## Split: {split_name}",
            "",
            f"- status: `{split_metrics.get('status')}`",
            f"- primary_confirmatory: `{split_metrics.get('primary_confirmatory')}`",
            f"- best_prama_by_auroc_exploratory: `{split_metrics.get('best_prama_by_auroc_exploratory')}`",
            f"- best_baseline_by_auroc: `{split_metrics.get('best_baseline_by_auroc')}`",
            "",
            "### Certified-kernel channels",
            "",
        ])
        for name, data in split_metrics["prama"].items():
            lines.append(f"- {name}: AUROC={data['auroc']} threshold={data['threshold']} confusion={data['confusion']}")
        lines.extend(["", "### Baselines", ""])
        for name, data in split_metrics["baselines"].items():
            lines.append(f"- {name}: AUROC={data['auroc']} threshold={data['threshold']} confusion={data['confusion']}")
    lines.extend([
        "",
        "## Methodological Note",
        "",
        "Ground truth must come from finish_reason or external automatic verification, never from the Protokol channel itself.",
        "The confirmatory verdict uses only latent_occupancy. Best-of-panel kernel channels are exploratory.",
        "If event_token equals the final token, any reported lead time is a final-outcome proxy and not anticipation.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--target-fpr", type=float, default=0.10)
    parser.add_argument("--primary-score", default=PRIMARY_PRAMA_SCORE)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--permutation-seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("results/state_discrimination_eval"))
    args = parser.parse_args()

    labels = load_labels(args.labels)
    examples = load_examples(args.inputs, labels)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split_values = {row["split"] for row in examples}
    split_aware = bool(split_values.difference({"", "test"}))
    metrics: dict[str, Any] = {
        "target_fpr": args.target_fpr,
        "primary_score": args.primary_score,
        "permutations": args.permutations,
        "permutation_seed": args.permutation_seed,
        "n_examples": len(examples),
        "split_aware": split_aware,
        "final_outcome_proxy_count": sum(1 for row in examples if row["final_outcome_proxy"]),
        "label_counts": dict(Counter(str(row["label"]) for row in examples)),
        "splits": {},
    }

    if split_aware:
        train = [row for row in examples if row["split"] in {"train", "calibration", "calib"}]
        test = [row for row in examples if row["split"] == "test"]
        if not test:
            raise SystemExit("split column exists but no test examples were found")
        metrics["splits"]["test"] = evaluate_split(
            train, test, args.target_fpr, args.primary_score, args.permutations, args.permutation_seed
        )
    else:
        metrics["splits"]["all"] = evaluate_split(
            examples, examples, args.target_fpr, args.primary_score, args.permutations, args.permutation_seed
        )

    (args.output_dir / "state_discrimination_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_report(args.output_dir / "state_discrimination_report.md", metrics)
    print(f"wrote {args.output_dir / 'state_discrimination_metrics.json'}")
    print(f"wrote {args.output_dir / 'state_discrimination_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
