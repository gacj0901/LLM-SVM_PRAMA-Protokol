#!/usr/bin/env python
"""Post-hoc audit of termination and trajectory-length confounding in CoCC.

This audit is descriptive and cannot alter a frozen confirmatory verdict.
"""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics
from typing import Any

import numpy as np


SCORES = ("max_delta", "max_xi", "max_negative_balance")


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def auroc(scores: list[float], labels: list[int]) -> float:
    positives = [score for score, label in zip(scores, labels) if label == 1]
    negatives = [score for score, label in zip(scores, labels) if label == 0]
    if not positives or not negatives:
        raise ValueError("AUROC requires both outcome classes")
    favorable = 0.0
    for positive in positives:
        for negative in negatives:
            favorable += positive > negative
            favorable += 0.5 * (positive == negative)
    return favorable / (len(positives) * len(negatives))


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile requires values")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def summary(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "minimum": min(values),
        "q25": quantile(values, 0.25),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "q75": quantile(values, 0.75),
        "maximum": max(values),
    }


def trajectory_scores(path: Path) -> tuple[int, dict[str, float]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"empty trajectory: {path}")
    deltas = [float(row["delta"]) for row in rows if row.get("delta") is not None]
    xis = [float(row["xi"]) for row in rows if row.get("xi") is not None]
    balances = [
        float(row["balance"]) for row in rows if row.get("balance") is not None
    ]
    if not deltas or not xis or not balances:
        raise ValueError(f"incomplete trajectory: {path}")
    return len(rows), {
        "max_delta": max(deltas),
        "max_xi": max(xis),
        "max_negative_balance": max(-value for value in balances),
    }


def load_rows(blind_join: Path, sessions_dir: Path) -> list[dict[str, Any]]:
    raw_by_session: dict[str, dict[str, Any]] = {}
    for path in sessions_dir.rglob("raw.json"):
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw_by_session[str(raw["session_id"])] = raw
    rows = []
    with blind_join.open(newline="", encoding="utf-8") as handle:
        for joined in csv.DictReader(handle):
            if joined["split"].strip().lower() != "test":
                continue
            session_id = joined["session_id"]
            raw = raw_by_session[session_id]
            turn = raw["turns"][0]
            n_windows, scores = trajectory_scores(Path(joined["trajectory_path"]))
            rows.append(
                {
                    "session_id": session_id,
                    "problem_id": str(raw["problem_id"]),
                    "perturbation_type": str(raw["perturbation_type"]),
                    "label": int(joined["label"]),
                    "finish_reason": str(turn["finish_reason"]),
                    "token_count": int(turn["token_count"]),
                    "n_windows": n_windows,
                    "scores": scores,
                }
            )
    if not rows:
        raise ValueError("blind join contains no test rows")
    return rows


def crosstab(
    rows: list[dict[str, Any]], left: str, right: str
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for row in rows:
        left_value = str(row[left])
        right_value = str(row[right])
        result.setdefault(left_value, {}).setdefault(right_value, 0)
        result[left_value][right_value] += 1
    return result


def stratified_auc(
    rows: list[dict[str, Any]], score: str, stratum: str
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row[stratum]), []).append(row)
    favorable = 0.0
    pairs = 0
    informative_sessions: set[str] = set()
    informative_strata = []
    for key, members in sorted(groups.items()):
        positives = [m for m in members if m["label"] == 1]
        negatives = [m for m in members if m["label"] == 0]
        if not positives or not negatives:
            continue
        stratum_pairs = len(positives) * len(negatives)
        pairs += stratum_pairs
        informative_sessions.update(m["session_id"] for m in members)
        stratum_favorable = 0.0
        for positive in positives:
            for negative in negatives:
                left = positive["scores"][score]
                right = negative["scores"][score]
                stratum_favorable += left > right
                stratum_favorable += 0.5 * (left == right)
        favorable += stratum_favorable
        informative_strata.append(
            {
                "stratum": key,
                "fail_n": len(positives),
                "pass_n": len(negatives),
                "pair_count": stratum_pairs,
                "auroc": stratum_favorable / stratum_pairs,
            }
        )
    return {
        "method": "pooled_within_stratum_pair_concordance",
        "stratum": stratum,
        "auroc": favorable / pairs if pairs else None,
        "informative_pair_count": pairs,
        "informative_session_count": len(informative_sessions),
        "informative_stratum_count": len(informative_strata),
        "strata": informative_strata,
    }


def residualized_auc(rows: list[dict[str, Any]], score: str) -> dict[str, Any]:
    lengths = np.asarray([row["n_windows"] for row in rows], dtype=float)
    values = np.asarray([row["scores"][score] for row in rows], dtype=float)
    labels = [row["label"] for row in rows]
    design = np.column_stack(
        [np.ones(len(rows)), lengths, lengths**2]
    )
    coefficients, _, rank, _ = np.linalg.lstsq(design, values, rcond=None)
    residuals = values - design @ coefficients
    return {
        "method": "outcome_blind_ols_score_on_n_windows_and_n_windows_squared",
        "design_rank": int(rank),
        "coefficients": [float(value) for value in coefficients],
        "residual_auroc": auroc(residuals.tolist(), labels),
    }


def score_audit(rows: list[dict[str, Any]], score: str) -> dict[str, Any]:
    labels = [row["label"] for row in rows]
    values = [row["scores"][score] for row in rows]
    by_finish_reason = {}
    for reason in sorted({row["finish_reason"] for row in rows}):
        members = [row for row in rows if row["finish_reason"] == reason]
        member_labels = [row["label"] for row in members]
        by_finish_reason[reason] = {
            "n": len(members),
            "fail_n": sum(member_labels),
            "pass_n": len(member_labels) - sum(member_labels),
            "auroc": (
                auroc([row["scores"][score] for row in members], member_labels)
                if len(set(member_labels)) == 2
                else None
            ),
        }
    return {
        "full_trajectory_auroc": auroc(values, labels),
        "exact_n_windows_stratified": stratified_auc(rows, score, "n_windows"),
        "finish_reason_stratified": stratified_auc(
            rows, score, "finish_reason"
        ),
        "outcome_blind_length_residualization": residualized_auc(rows, score),
        "within_finish_reason": by_finish_reason,
    }


def build_report(
    rows: list[dict[str, Any]],
    blind_join: Path,
    run_manifest: Path,
    confirmatory_report: Path,
) -> dict[str, Any]:
    labels = [row["label"] for row in rows]
    by_outcome = {
        name: {
            "token_count": summary(
                [row["token_count"] for row in rows if row["label"] == label]
            ),
            "n_windows": summary(
                [row["n_windows"] for row in rows if row["label"] == label]
            ),
        }
        for name, label in (("PASS", 0), ("FAIL", 1))
    }
    length_indicator = [
        1.0 if row["finish_reason"] == "length" else 0.0 for row in rows
    ]
    token_counts = [float(row["token_count"]) for row in rows]
    n_windows = [float(row["n_windows"]) for row in rows]
    return {
        "schema": "LLM-SVM-CoCC-posthoc-length-audit/1",
        "analysis_status": "POST_HOC_DESCRIPTIVE_DOES_NOT_MODIFY_CONFIRMATORY_VERDICT",
        "confirmatory_verdict": "honest_null",
        "scope": "frozen test partition only",
        "test_n": len(rows),
        "fail_n": sum(labels),
        "pass_n": len(labels) - sum(labels),
        "source_binding": {
            "blind_join_sha256": file_sha256(blind_join),
            "run_manifest_sha256": file_sha256(run_manifest),
            "confirmatory_report_sha256": file_sha256(confirmatory_report),
            "audit_script_sha256": file_sha256(Path(__file__)),
        },
        "finish_reason_by_outcome": crosstab(rows, "label", "finish_reason"),
        "finish_reason_by_perturbation_type": crosstab(
            rows, "perturbation_type", "finish_reason"
        ),
        "length_summary_by_outcome": by_outcome,
        "length_only_discrimination": {
            "finish_reason_length_indicator_auroc": auroc(length_indicator, labels),
            "token_count_auroc": auroc(token_counts, labels),
            "n_windows_auroc": auroc(n_windows, labels),
        },
        "score_audits": {score: score_audit(rows, score) for score in SCORES},
        "interpretation_limits": [
            "All analyses were selected after viewing the confirmatory result.",
            "No post-hoc p-value changes the frozen honest_null verdict.",
            "Exact-length stratification may cover only a subset of sessions and pairs.",
            "Outcome-blind polynomial residualization removes only the specified functional relationship with n_windows.",
            "finish_reason=length is a generation-cap event, not an independent semantic outcome.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blind-join", required=True, type=Path)
    parser.add_argument("--sessions-dir", required=True, type=Path)
    parser.add_argument("--run-manifest", required=True, type=Path)
    parser.add_argument("--confirmatory-report", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        rows = load_rows(args.blind_join, args.sessions_dir)
        report = build_report(
            rows, args.blind_join, args.run_manifest, args.confirmatory_report
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"CoCC length-confound audit failed: {exc}")
        return 1
    print(json.dumps({"output": str(args.out), "sha256": file_sha256(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
