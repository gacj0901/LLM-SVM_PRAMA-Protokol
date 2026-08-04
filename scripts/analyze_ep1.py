#!/usr/bin/env python
"""Run the complete preregistered E-P1 analysis and emit its sole verdict."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for candidate in (REPO_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from aptadynamic_llm.evaluation.state_discrimination_metrics import (
    BASELINE_SCORE_FIELDS,
    PRIMARY_PRAMA_SCORE,
)
from aptadynamic_llm.ep1_config import (
    BASE_SEED,
    EXTENSION_BLOCK,
    FROZEN_NUM_PREDICT,
    FROZEN_PROMPT_SUITE_SHA256,
    MAX_TOTAL_N,
    K,
    MIN_SESSIONS,
    MIN_VALID_TOKENS,
    MIN_OLLAMA_VERSION,
    MIN_TEST_POSITIVES,
    MODEL,
    MODEL_ID_PREFIX,
    PERMUTATIONS,
    PERMUTATION_SEED,
    SEED_PER_INDEX,
    TARGET_FPR,
    TEST_FRACTION,
    TEMPERATURE,
    TOP_LOGPROBS,
    TOP_P,
)
from aptadynamic_llm.ingest import load_sessions
from aptadynamic_llm.omega import omega_sessions
from scripts.evaluate_state_discrimination import (
    evaluate_split,
    load_examples,
    load_labels,
    write_report,
)
from scripts.score_sessions_prama import (
    StudyInvalid,
    parse_args as parse_score_args,
    run as run_scoring,
)
from scripts.validate_state_discrimination_inputs import validate_inputs, write_markdown


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()

def _version_tuple(value: str) -> tuple[int, int, int]:
    parts = value.split(".")
    return tuple(int(part.split("-")[0]) for part in (parts + ["0", "0"])[:3])


def _validate_collection_manifest(sessions_dir: Path) -> dict[str, Any]:
    path = sessions_dir / "manifest.json"
    if not path.exists():
        raise ValueError("confirmatory collection manifest.json is missing")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("mode") != "confirmatory":
        raise ValueError("authoritative E-P1 analysis accepts confirmatory collections only")
    if manifest.get("model") != MODEL:
        raise ValueError(f"model must be {MODEL!r}")
    digest = str(manifest.get("model_digest") or "").removeprefix("sha256:")
    if not digest.startswith(MODEL_ID_PREFIX):
        raise ValueError("model identifier does not match frozen D7 identity")
    if _version_tuple(str(manifest.get("ollama_version") or "0.0.0")) < MIN_OLLAMA_VERSION:
        raise ValueError("Ollama version is below the frozen D7 minimum")
    expected = {
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_logprobs": TOP_LOGPROBS,
        "seed": BASE_SEED,
        "seed_per_index": SEED_PER_INDEX,
    }
    mismatches = {
        name: {"expected": value, "observed": manifest.get(name)}
        for name, value in expected.items()
        if manifest.get(name) != value
    }
    if mismatches:
        raise ValueError(f"collection manifest violates frozen D7 sampling: {mismatches}")
    if FROZEN_NUM_PREDICT is None or FROZEN_PROMPT_SUITE_SHA256 is None:
        raise ValueError("post-pilot freeze is incomplete: num_predict and prompt-suite SHA-256 are pending")
    if manifest.get("num_predict") != FROZEN_NUM_PREDICT:
        raise ValueError("num_predict differs from the post-pilot freeze")
    if manifest.get("prompt_suite_sha256") != FROZEN_PROMPT_SUITE_SHA256:
        raise ValueError("prompt-suite SHA-256 differs from the post-pilot freeze")
    return manifest


def _validate_nvidia_replication_manifest(
    sessions_dir: Path, freeze_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = sessions_dir / "manifest.json"
    if not manifest_path.exists():
        raise ValueError("confirmatory collection manifest.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze_hash = _file_sha256(freeze_path)
    if freeze.get("schema") != "LLM-SVM-E-P1-NVIDIA-model-freeze/1":
        raise ValueError("unsupported NVIDIA model freeze schema")
    if freeze.get("status") != "CONFIRMATORY_FROZEN":
        raise ValueError("NVIDIA model freeze is not confirmatory")
    design_path = Path(str(freeze.get("design") or ""))
    if not design_path.exists():
        design_path = REPO_ROOT / "config" / "ep1_nvidia_replication_v1.json"
    if _file_sha256(design_path) != freeze.get("design_sha256"):
        raise ValueError("NVIDIA replication design differs from the model freeze")
    design = json.loads(design_path.read_text(encoding="utf-8"))
    confirmatory = design["confirmatory"]
    collection_n = int(manifest.get("n") or 0)
    initial_n = int(confirmatory["initial_n"])
    block = int(confirmatory["extension_block"])
    maximum_n = int(confirmatory["maximum_total_n"])
    if (
        not initial_n <= collection_n <= maximum_n
        or (collection_n - initial_n) % block
    ):
        raise ValueError("collection N violates the frozen extension schedule")
    sampling = freeze["sampling"]
    expected = {
        "schema": "LLM-SVM-E-P1-NVIDIA-collection/1",
        "study_id": freeze["study_id"],
        "mode": "confirmatory",
        "provider": freeze["provider"],
        "provider_endpoint": freeze["provider_endpoint"],
        "model": freeze["model"],
        "design_sha256": freeze["design_sha256"],
        "model_freeze_sha256": freeze_hash,
        "prompt_suite_sha256": freeze["prompt_suite_sha256"],
        "temperature": sampling["temperature"],
        "top_p": sampling["top_p"],
        "top_logprobs": sampling["top_logprobs"],
        "seed": sampling["base_seed"],
        "seed_per_index": sampling["seed_per_index"],
        "max_tokens": freeze["selected_max_tokens"],
        "enable_thinking": sampling.get("enable_thinking"),
        "reasoning_effort": sampling.get("reasoning_effort"),
        "stream": sampling["stream"],
        "completed_n": collection_n,
        "complete": True,
    }
    mismatches = {
        name: {"expected": value, "observed": manifest.get(name)}
        for name, value in expected.items()
        if manifest.get(name) != value
    }
    if mismatches:
        raise ValueError(f"NVIDIA collection violates its model freeze: {mismatches}")
    if not manifest.get("collection_content_sha256"):
        raise ValueError("collection lacks its canonical content binding")
    if {str(value) for value in manifest.get("resolved_models") or []} != {
        str(freeze["model"])
    }:
        raise ValueError("provider resolved model differs from the frozen model")
    return manifest, {
        "program": "E-P1-NVIDIA-R1",
        "relationship_to_ep1": design["relationship_to_ep1"],
        "claim_boundary": design["claim_boundary"],
        "model_freeze_sha256": freeze_hash,
        "design_sha256": freeze["design_sha256"],
        "expected_model": freeze["model"],
    }


def _preflight_power(
    sessions_dir: Path, expected_model: str = MODEL
) -> dict[str, Any]:
    """Count eligible outcomes without calling project() or materializing scores."""

    df = load_sessions(str(sessions_dir))
    if df.empty:
        raise ValueError("no verifiable raw sessions found")
    models = {str(value) for value in df["model"].dropna().unique()}
    if models != {expected_model}:
        raise ValueError(f"raw-session model identity differs from D7: {sorted(models)!r}")
    labels: list[int] = []
    for gid, finish_reason, omega, expected in omega_sessions(df, min_sessions=MIN_SESSIONS):
        observed = np.asarray(omega[:K], dtype=float)
        causal_expected = np.asarray(expected[:K], dtype=float)
        if len(observed) < MIN_VALID_TOKENS:
            continue
        if int((~np.isnan(causal_expected)).sum()) < MIN_VALID_TOKENS:
            continue
        reason = str(finish_reason or "").strip().lower()
        if reason not in {"length", "stop"}:
            raise ValueError(f"{gid}: unsupported finish_reason {finish_reason!r}")
        labels.append(int(reason == "length"))
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives < 2 or negatives < 2:
        raise ValueError("each outcome class needs at least two eligible sessions")

    def test_count(count: int) -> int:
        return min(count - 1, max(1, int(round(count * TEST_FRACTION))))

    test_positives = test_count(positives)
    total_collected = int(df["session_id"].nunique())
    passed = test_positives >= MIN_TEST_POSITIVES
    next_n = min(MAX_TOTAL_N, ((total_collected // EXTENSION_BLOCK) + 1) * EXTENSION_BLOCK)
    can_extend = total_collected < MAX_TOTAL_N
    return {
        "passed": passed,
        "eligible_sessions": len(labels),
        "eligible_positives": positives,
        "eligible_negatives": negatives,
        "test_positives": test_positives,
        "minimum": MIN_TEST_POSITIVES,
        "total_collected_n": total_collected,
        "extension_block": EXTENSION_BLOCK,
        "maximum_total_n": MAX_TOTAL_N,
        "next_total_n": next_n if not passed and can_extend else None,
        "extension_allowed": not passed and can_extend,
        "scores_computed_at_gate": False,
    }


def _write_verdict(out: Path, payload: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "ep1_verdict.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    gates = payload.get("gates") or {}
    lines = [
        "# E-P1 Authoritative Verdict",
        "",
        f"- verdict: `{payload['verdict']}`",
        f"- terminal: `{payload.get('terminal', True)}`",
        f"- detail: {payload.get('detail', '')}",
        "",
        "## Preregistered gates",
        "",
    ]
    lines.extend(f"- {name}: `{value}`" for name, value in gates.items())
    (out / "ep1_verdict.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"VERDICT: {payload['verdict']}")


def _interface_failure(
    out: Path,
    detail: str,
    c3: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> int:
    _write_verdict(
        out,
        {
            "schema": "LLM-SVM-E-P1-verdict/1",
            "verdict": "interface_failure",
            "terminal": True,
            "detail": detail,
            "gates": {"C3": c3 or {"passed": False}},
            "analysis_executed": False,
            **(context or {}),
        },
    )
    return 2


def run(args: argparse.Namespace) -> int:
    if args.out.exists():
        raise FileExistsError(f"analysis output already exists: {args.out}")
    replication_freeze = getattr(args, "replication_freeze", None)
    context: dict[str, Any] = {}
    expected_model = MODEL
    try:
        if replication_freeze is None:
            collection_manifest = _validate_collection_manifest(args.sessions_dir)
        else:
            collection_manifest, context = _validate_nvidia_replication_manifest(
                args.sessions_dir, replication_freeze
            )
            expected_model = context["expected_model"]
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        return _interface_failure(
            args.out, f"D7 collection identity failure: {exc}", context=context
        )
    try:
        power_gate = _preflight_power(args.sessions_dir, expected_model)
    except (ValueError, OSError) as exc:
        return _interface_failure(
            args.out, f"preflight outcome/power failure: {exc}", context=context
        )
    if not power_gate["passed"]:
        can_extend = power_gate["extension_allowed"]
        next_n = power_gate["next_total_n"]
        _write_verdict(
            args.out,
            {
                "schema": "LLM-SVM-E-P1-verdict/1",
                "verdict": "underpowered",
                "terminal": not can_extend,
                "detail": (
                    f"test contains {power_gate['test_positives']} positives; at least {MIN_TEST_POSITIVES} are required. "
                    + (f"Collect to N={next_n} without inspecting scores." if can_extend else "Maximum N reached.")
                ),
                "gates": {"C3": {"passed": None, "status": "not_run"}, "power": power_gate},
                "analysis_executed": False,
                "collection_manifest": collection_manifest,
                **context,
            },
        )
        return 3

    scored = args.out / "scored"
    score_argv = ["--sessions-dir", str(args.sessions_dir), "--out", str(scored)]
    try:
        scoring = run_scoring(parse_score_args(score_argv))
    except StudyInvalid as exc:
        return _interface_failure(args.out, str(exc), context=context)
    except (ValueError, OSError) as exc:
        return _interface_failure(
            args.out,
            f"Observation Interface/scoring failure: {exc}",
            context=context,
        )

    c3 = scoring["statistics"]["C3"]
    scored_manifest = json.loads((scored / "manifest.json").read_text(encoding="utf-8"))
    if scored_manifest.get("models") != [expected_model]:
        return _interface_failure(
            args.out,
            f"raw-session model identity differs from D7: {scored_manifest.get('models')!r}",
            c3,
            context,
        )
    labels_path = scored / "labels.csv"
    raw_paths = sorted((scored / "sessions").rglob("raw.json"))
    labels = load_labels(labels_path)

    validation = validate_inputs(labels_path, raw_paths, PRIMARY_PRAMA_SCORE)
    validation_dir = args.out / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    (validation_dir / "state_discrimination_input_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_markdown(validation_dir / "state_discrimination_input_validation.md", validation)
    if not validation["valid"]:
        return _interface_failure(
            args.out,
            "Input validation failed: " + "; ".join(validation["errors"]),
            c3,
            context,
        )

    examples = load_examples(raw_paths, labels)
    train = [row for row in examples if row["split"] in {"train", "calibration", "calib"}]
    test = [row for row in examples if row["split"] == "test"]
    split_metrics = evaluate_split(
        train,
        test,
        TARGET_FPR,
        PRIMARY_PRAMA_SCORE,
        PERMUTATIONS,
        PERMUTATION_SEED,
    )
    metrics = {
        "target_fpr": TARGET_FPR,
        "primary_score": PRIMARY_PRAMA_SCORE,
        "permutations": PERMUTATIONS,
        "permutation_seed": PERMUTATION_SEED,
        "n_examples": len(examples),
        "split_aware": True,
        "final_outcome_proxy_count": sum(1 for row in examples if row["final_outcome_proxy"]),
        "label_counts": dict(Counter(str(row["label"]) for row in examples)),
        "splits": {"test": split_metrics},
    }
    evaluation_dir = args.out / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    (evaluation_dir / "state_discrimination_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_report(evaluation_dir / "state_discrimination_report.md", metrics)

    confirmatory = split_metrics["primary_confirmatory"]
    comparisons = split_metrics["matched_fpr"]
    permutation_gate = {
        "passed": confirmatory["permutation_p"] is not None and confirmatory["permutation_p"] < 0.01,
        "p": confirmatory["permutation_p"],
        "threshold": 0.01,
        "permutations": PERMUTATIONS,
    }
    auroc_gate = {
        "passed": all(comparisons[name]["auroc_delta"] is not None and comparisons[name]["auroc_delta"] > 0 for name in BASELINE_SCORE_FIELDS),
        "by_baseline": {name: comparisons[name]["auroc_delta"] for name in BASELINE_SCORE_FIELDS},
    }
    matched_fpr_gate = {
        "passed": all(comparisons[name]["tpr_delta"] is not None and comparisons[name]["tpr_delta"] > 0 for name in BASELINE_SCORE_FIELDS),
        "by_baseline": {name: comparisons[name]["tpr_delta"] for name in BASELINE_SCORE_FIELDS},
        "target_fpr": TARGET_FPR,
        "threshold_source": "train",
    }
    gates = {
        "C3": c3,
        "power": power_gate,
        "permutation": permutation_gate,
        "auroc_vs_all_four_baselines": auroc_gate,
        "tpr_at_fpr_vs_all_four_baselines": matched_fpr_gate,
    }
    positive = all(
        gate.get("passed", False)
        for gate in (c3, power_gate, permutation_gate, auroc_gate, matched_fpr_gate)
    )
    verdict = "positive" if positive else "honest_null"
    _write_verdict(
        args.out,
        {
            "schema": "LLM-SVM-E-P1-verdict/1",
            "verdict": verdict,
            "terminal": True,
            "detail": "All D6 gates passed." if positive else "At least one confirmatory D6 signal gate failed.",
            "gates": gates,
            "analysis_executed": True,
            "baseline_fields": list(BASELINE_SCORE_FIELDS),
            "confirmatory": confirmatory,
            "collection_manifest": collection_manifest,
            **context,
        },
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--replication-freeze",
        type=Path,
        help="Model-specific E-P1 NVIDIA freeze; omitted for Hermes/Ollama E-P1.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except (FileExistsError, OSError, ValueError) as exc:
        print(f"analysis failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
