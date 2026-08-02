#!/usr/bin/env python
"""Create a descriptive report for an incomplete CoCC acquisition."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean, median
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT, REPO_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts import evaluate_break_the_chain_prama as ev  # noqa: E402


FIELDS = (
    "max_delta",
    "max_xi",
    "max_negative_balance",
    "final_accumulated_excess",
    "capacity_loss",
    "mean_surprisal",
    "mean_entropy",
    "negative_mean_gap",
)
HORIZONS = (1, 2, 4, 8, 16)


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "minimum": min(values) if values else None,
        "median": median(values) if values else None,
        "mean": mean(values) if values else None,
        "maximum": max(values) if values else None,
    }


def metric(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    labels = [row["label"] for row in rows]
    if not rows or len(set(labels)) < 2:
        return {"n": len(rows), "auroc": None, "average_precision": None}
    scores = [row["features"][field] for row in rows]
    return {
        "n": len(rows),
        "fail_n": sum(labels),
        "pass_n": len(labels) - sum(labels),
        "auroc": ev.auroc(scores, labels),
        "average_precision": ev._average_precision(scores, labels),
    }


def scalar_auc(values: list[float], labels: list[int]) -> float | None:
    return ev.auroc(values, labels) if len(set(labels)) == 2 else None


def crosstab(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    table: dict[str, dict[str, int]] = {}
    for row in rows:
        a = str(row[left])
        b = str(row[right])
        table.setdefault(a, {}).setdefault(b, 0)
        table[a][b] += 1
    return table


def load_rows(base: Path) -> list[dict[str, Any]]:
    raw_by_session = {}
    for path in (base / "sessions").rglob("raw.json"):
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw_by_session[str(raw["session_id"])] = raw
    rows = []
    with (base / "evaluation" / "blind_join.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        for joined in csv.DictReader(handle):
            session_id = joined["session_id"]
            raw = raw_by_session[session_id]
            trajectory = ev._trajectory(Path(joined["trajectory_path"]))
            tokens = ev._numeric_tokens(Path(joined["projection_request_path"]))
            rows.append(
                {
                    "session_id": session_id,
                    "problem_id": str(raw["problem_id"]),
                    "perturbation_type": str(raw["perturbation_type"]),
                    "label": int(joined["label"]),
                    "outcome": "FAIL" if int(joined["label"]) else "PASS",
                    "finish_reason": str(raw["turns"][0]["finish_reason"]),
                    "token_count": int(raw["turns"][0]["token_count"]),
                    "response_time_seconds": float(raw["response_time_seconds"]),
                    "n_windows": len(trajectory),
                    "trajectory": trajectory,
                    "tokens": tokens,
                    "features": ev._features(trajectory, tokens),
                }
            )
    return rows


def build_report(base: Path) -> dict[str, Any]:
    rows = load_rows(base)
    labels = [row["label"] for row in rows]
    full = {field: metric(rows, field) for field in FIELDS}
    horizons = {}
    for horizon in HORIZONS:
        prefix = [
            {
                **row,
                "features": ev._features(
                    row["trajectory"][:horizon], row["tokens"][: horizon * 16]
                ),
            }
            for row in rows
            if len(row["trajectory"]) >= horizon
        ]
        horizon_metrics = {field: metric(prefix, field) for field in FIELDS}
        horizons[str(horizon)] = {
            "window_horizon": horizon,
            "nominal_token_horizon": horizon * 16,
            "at_risk_n": len(prefix),
            "metrics": horizon_metrics,
            "prama_minus_delta": (
                horizon_metrics["max_negative_balance"]["auroc"]
                - horizon_metrics["max_delta"]["auroc"]
                if horizon_metrics["max_negative_balance"]["auroc"] is not None
                else None
            ),
        }
    by_outcome = {
        outcome: {
            "n": len(members),
            "token_count": summary([row["token_count"] for row in members]),
            "n_windows": summary([row["n_windows"] for row in members]),
            "response_time_seconds": summary(
                [row["response_time_seconds"] for row in members]
            ),
        }
        for outcome in ("PASS", "FAIL")
        for members in [[row for row in rows if row["outcome"] == outcome]]
    }
    stop_rows = [row for row in rows if row["finish_reason"] == "stop"]
    length_rows = [row for row in rows if row["finish_reason"] == "length"]
    length_rate = len(length_rows) / len(rows)
    return {
        "schema": "LLM-SVM-CoCC-PRAMA-partial-descriptive/1",
        "analysis_status": "PARTIAL_POST_HOC_DESCRIPTIVE_NOT_CONFIRMATORY",
        "generated_from": str(base),
        "source_binding": {
            "manifest_sha256": file_sha256(base / "manifest.json"),
            "blind_join_sha256": file_sha256(
                base / "evaluation" / "blind_join.csv"
            ),
        },
        "scope": {
            "session_count": len(rows),
            "dataset_order": "first completed holdout sessions",
            "perturbation_types": sorted(
                {row["perturbation_type"] for row in rows}
            ),
            "complete_paired_problem_clusters": 0,
            "limitations": [
                "The acquisition stopped after the first 53 holdout sessions.",
                "All observed sessions are clean controls; negation sessions are absent.",
                "No cluster-aware or confirmatory inference is valid.",
                "The generation-cap criterion was examined during acquisition.",
            ],
        },
        "outcomes": {
            "pass_n": len(rows) - sum(labels),
            "fail_n": sum(labels),
            "pass_rate": (len(rows) - sum(labels)) / len(rows),
            "by_outcome": by_outcome,
        },
        "generation_cap_audit": {
            "finish_reason_by_outcome": crosstab(
                rows, "outcome", "finish_reason"
            ),
            "finish_reason_by_perturbation": crosstab(
                rows, "perturbation_type", "finish_reason"
            ),
            "length_n": len(length_rows),
            "length_rate": length_rate,
            "frozen_maximum_rate": 0.05,
            "passes_frozen_limit": length_rate <= 0.05,
            "token_count_auroc": scalar_auc(
                [float(row["token_count"]) for row in rows], labels
            ),
            "n_windows_auroc": scalar_auc(
                [float(row["n_windows"]) for row in rows], labels
            ),
        },
        "full_trajectory_metrics": full,
        "full_prama_minus_delta": (
            full["max_negative_balance"]["auroc"]
            - full["max_delta"]["auroc"]
            if full["max_negative_balance"]["auroc"] is not None
            else None
        ),
        "absolute_horizon_metrics": horizons,
        "finish_reason_sensitivity": {
            "stop_only_n": len(stop_rows),
            "stop_only_metrics": {
                field: metric(stop_rows, field) for field in FIELDS
            },
            "length_only_n": len(length_rows),
            "length_only_metrics": {
                field: metric(length_rows, field) for field in FIELDS
            },
        },
        "rank_equivalence_audit": ev._rank_equivalence_audit(rows),
        "activation_audit": ev._activation_audit(rows, 1e-12),
        "trajectory_length_audit": ev._trajectory_length_audit(rows),
        "interpretation": (
            "This report characterizes the observed high-reasoning failure mode; "
            "it cannot validate or refute the frozen confirmatory hypothesis."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    report = build_report(args.run_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.out), "sha256": file_sha256(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
