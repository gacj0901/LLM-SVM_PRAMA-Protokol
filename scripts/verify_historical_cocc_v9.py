#!/usr/bin/env python3
"""Verify historical CoCC answers and blind-join outcomes to saved v9 projections."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for candidate in (REPO_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from aptadynamic_llm.structural_coherence_v9 import StructuralCoherenceV9Config  # noqa: E402
from scripts.analyze_frontend_test_battery import atomic_json, utc_now  # noqa: E402
from scripts.analyze_frontend_test_battery_v9 import session_endpoints  # noqa: E402
from scripts.cocc_external_verifier import load_dataset, verify  # noqa: E402
from scripts.project_cocc_operator_geometry import file_sha256, load_json  # noqa: E402


def auc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    positives = [score for score, label in zip(scores, labels) if label == 1]
    negatives = [score for score, label in zip(scores, labels) if label == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def score_values(endpoints: Mapping[str, Any], token_count: int, prama: Mapping[str, Any]) -> dict[str, Any]:
    alert = endpoints["alert_endpoints_post_stabilization"]
    transport = endpoints["transport_status_occupancy_among_geometry_ready"]
    mobility = endpoints["mobility_regime_occupancy_among_assigned_windows"]
    channels = endpoints["channels"]
    terminal = channels["terminal_transport_coherence"]
    min_balance = prama.get("min_balance")
    return {
        "token_count": token_count,
        "post_stabilization_disrupted_fraction": alert["transport_disrupted_fraction"]["fraction"],
        "post_stabilization_mean_transport_deficit": alert["mean_transport_deficit"],
        "maximum_disrupted_dwell": alert["maximum_transport_disrupted_dwell"],
        "full_transport_disrupted_fraction": transport["disrupted"]["fraction"],
        "mean_recurrence_persistence": channels["mean_recurrence_persistence"],
        "crystallizing_fraction": mobility["crystallizing"]["fraction"],
        "negative_terminal_transport_coherence": -float(terminal) if terminal is not None else None,
        "max_delta": prama.get("max_delta"),
        "max_xi": prama.get("max_xi"),
        "negative_min_balance": -float(min_balance) if min_balance is not None else None,
    }


def discrimination(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = [1 if row["outcome"] == "FAIL" else 0 for row in rows]
    names = sorted({name for row in rows for name in row["scores"]})
    metrics: dict[str, Any] = {}
    for name in names:
        pairs = [
            (float(row["scores"][name]), 1 if row["outcome"] == "FAIL" else 0)
            for row in rows if row["scores"].get(name) is not None
        ]
        metrics[name] = {
            "evaluable_n": len(pairs),
            "pass_n": sum(label == 0 for _, label in pairs),
            "fail_n": sum(label == 1 for _, label in pairs),
            "auroc_fail_positive": auc([score for score, _ in pairs], [label for _, label in pairs]),
        }
    return {
        "n": len(rows), "pass_n": sum(label == 0 for label in labels),
        "fail_n": sum(label == 1 for label in labels), "metrics": metrics,
    }


_REQUIRED_DIAGNOSTIC_BUCKETS = (
    "PASS",
    "missing_callable",
    "candidate_AttributeError",
    "SyntaxError",
    "NameError",
    "wrong_answer",
    "timeout",
    "worker_error",
)


def verifier_diagnostic_bucket(row: Mapping[str, Any]) -> str:
    """Return a stable audit bucket without erasing more-specific failures."""
    if row.get("outcome") == "PASS":
        return "PASS"
    failure_kind = str(row.get("failure_kind") or "")
    exception_type = str(row.get("exception_type") or "")
    if failure_kind == "missing_callable":
        return "missing_callable"
    if exception_type == "AttributeError":
        return "candidate_AttributeError"
    if failure_kind:
        return failure_kind
    if exception_type:
        return exception_type
    return "unclassified_failure"


def verifier_diagnostic_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    observed = sorted({verifier_diagnostic_bucket(row) for row in rows})
    buckets = list(dict.fromkeys((*_REQUIRED_DIAGNOSTIC_BUCKETS, *observed)))

    def counts(members: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        return {
            bucket: sum(verifier_diagnostic_bucket(row) == bucket for row in members)
            for bucket in buckets
        }

    return {
        "classification_rule": {
            "missing_callable": (
                "exception_type == 'AttributeError' and exception_message exactly matches "
                "\"callable '<identifier>' not found\""
            ),
            "candidate_AttributeError": (
                "exception_type == 'AttributeError' and failure_kind != 'missing_callable'"
            ),
            "extraction_failures_are_not_missing_callable": True,
        },
        "total": counts(rows),
        "by_model": [
            {
                "model": model,
                "counts": counts([row for row in rows if row["model"] == model]),
            }
            for model in sorted({str(row["model"]) for row in rows})
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backfill-report", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, action="append", required=True)
    parser.add_argument("--v9-contract", type=Path, default=Path("config/sequor_structural_coherence_observer_v9.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=20)
    args = parser.parse_args()

    report = load_json(args.backfill_report)
    dataset = load_dataset(args.dataset)
    dataset_sha = file_sha256(args.dataset)
    verifier_path = Path(__file__).with_name("cocc_external_verifier.py")
    verifier_sha = file_sha256(verifier_path)
    v9_config = StructuralCoherenceV9Config.from_contract(load_json(args.v9_contract))
    worker = Path(__file__).with_name("_historical_cocc_verify_worker.py").resolve()
    worker_sha = file_sha256(worker)
    relevant = {
        str(row["problem_id"]): row for row in dataset.values()
        if str(row.get("perturbation_type")) == "negation_objective"
    }
    raw_map: dict[tuple[str, str], Path] = {}
    for source_run in args.source_run:
        for path in source_run.glob("sessions/**/raw.json"):
            raw = load_json(path)
            raw_map[(source_run.name, str(raw.get("session_id")))] = path

    verification_dir = args.output_dir / "verification"
    joined: list[dict[str, Any]] = []
    candidates = [row for row in report["items"] if str(row["problem_id"]) in relevant]
    for index, row in enumerate(candidates, 1):
        key = (str(row["source_run"]), str(row["session_id"]))
        raw_path = raw_map[key]
        raw_sha = file_sha256(raw_path)
        dataset_row = relevant[str(row["problem_id"])]
        answer = str((load_json(raw_path).get("turns") or [{}])[0].get("assistant_message") or "")
        answer_sha = hashlib.sha256(answer.encode("utf-8")).hexdigest()
        cache_path = verification_dir / f"{row['session_id']}.json"
        cached = load_json(cache_path) if cache_path.exists() else None
        if (
            cached
            and cached.get("source_raw_sha256") == raw_sha
            and cached.get("dataset_sha256") == dataset_sha
            and cached.get("verifier_sha256") == verifier_sha
            and cached.get("worker_sha256") == worker_sha
        ):
            result = cached["verifier_result"]
            print(f"[{index}/{len(candidates)}] reusing {row['session_id']}", flush=True)
        else:
            print(f"[{index}/{len(candidates)}] verifying {row['session_id']}", flush=True)
            result = verify(
                {"item_id": dataset_row["item_id"], "observed_answer": answer},
                dataset, worker, args.timeout,
            )
            atomic_json(cache_path, {
                "schema": "LLM-SVM-historical-CoCC-verification/1",
                "session_id": row["session_id"], "problem_id": row["problem_id"],
                "source_raw_sha256": raw_sha, "answer_sha256": answer_sha,
                "dataset_sha256": dataset_sha,
                "verifier_sha256": verifier_sha,
                "worker_sha256": worker_sha,
                "contains_answer": False, "verifier_result": result,
            })
        outcome = "PASS" if result.get("passed") else "FAIL"
        joined.append({
            "model": row["model"], "problem_id": row["problem_id"],
            "session_id": row["session_id"], "finish_reason": row["finish_reason"],
            "token_count": row["token_count"], "outcome": outcome,
            "failure_kind": result.get("failure_kind"),
            "exception_type": result.get("exception_type"),
            "exception_message": result.get("exception_message"),
            "source_extraction": result.get("source_extraction"),
            "tests_passed": result.get("tests_passed"), "tests_total": result.get("tests_total"),
            "projection_sha256": row["projection_sha256"],
            "verification_sha256": file_sha256(cache_path),
            "scores": score_values(row["endpoints"], int(row["token_count"]), row["prama"]),
            "projection_path": row["projection_path"],
        })

    model_results = []
    for model in sorted({row["model"] for row in joined}):
        members = [row for row in joined if row["model"] == model]
        model_results.append({
            "model": model,
            "finish_reason_by_outcome": {
                finish: {
                    outcome.lower(): sum(
                        row["finish_reason"] == finish and row["outcome"] == outcome for row in members
                    ) for outcome in ("PASS", "FAIL")
                } for finish in sorted({row["finish_reason"] for row in members})
            },
            "full_trajectory": discrimination(members),
        })

    horizons = []
    for horizon in load_json(args.v9_contract)["fixed_causal_horizons_windows"]:
        horizon_rows = []
        for row in joined:
            projection = load_json(Path(row["projection_path"]))
            windows = projection["structural_windows"]
            if len(windows) < int(horizon):
                continue
            endpoints = session_endpoints(windows[: int(horizon)], v9_config)
            if endpoints["session_evaluation_status"] != "EVALUABLE":
                continue
            horizon_rows.append({
                **row,
                "scores": score_values(
                    endpoints,
                    min(int(row["token_count"]), int(horizon) * 16),
                    prama={},
                ),
            })
        by_model = []
        for model in sorted({row["model"] for row in horizon_rows}):
            members = [row for row in horizon_rows if row["model"] == model]
            by_model.append({"model": model, **discrimination(members)})
        horizons.append({
            "horizon_windows": int(horizon),
            "analysis_status": "DESCRIPTIVE_WARMUP_ONLY" if int(horizon) < 32 else "EXPLORATORY_FIXED_HORIZON",
            "models": by_model,
        })

    output = {
        "schema": "LLM-SVM-historical-CoCC-v9-outcome-join/1",
        "generated_at": utc_now(), "status": "EXPLORATORY_RETROSPECTIVE",
        "backfill_report_sha256": file_sha256(args.backfill_report),
        "historical_generation_context": report.get("historical_generation_context"),
        "reprojection_context": report.get("reprojection_context"),
        "interpretation_boundary": report.get("interpretation_boundary"),
        "dataset_sha256": dataset_sha,
        "v9_contract_sha256": file_sha256(args.v9_contract),
        "projection_was_blind_to_outcome": True,
        "external_outcomes_joined_after_projection": True,
        "contains_prompt_or_answer": False,
        "matched_problem_count": len({row["problem_id"] for row in joined}),
        "verified_response_count": len(joined),
        "verifier_diagnostic_audit": verifier_diagnostic_audit(joined),
        "model_results": model_results, "fixed_horizons": horizons,
        "items": [{key: value for key, value in row.items() if key != "projection_path"} for row in joined],
    }
    output_path = args.output_dir / "report.json"
    atomic_json(output_path, output)
    print(json.dumps({
        "output": str(output_path), "verified_responses": len(joined),
        "matched_problems": output["matched_problem_count"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
