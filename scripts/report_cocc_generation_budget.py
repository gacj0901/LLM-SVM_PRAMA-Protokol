"""Build a calibration-only CoCC generation-budget report.

The report consumes only a budget-selection artifact, run manifests and raw
generation observations. It never reads verifier, projector or holdout outcome
artifacts.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean, median
from typing import Any


RETRY_CONTROLLED_RUNNER_SHA256 = (
    "945687ed62b528f40847564a013be469bd353c1f9bf7bae4358bbcf13c35b2d1"
)


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "minimum": round(min(values), 6),
        "median": round(median(values), 6),
        "mean": round(mean(values), 6),
        "p90": round(_percentile(values, 0.90), 6),
        "maximum": round(max(values), 6),
        "sum": round(sum(values), 6),
    }


def build_report(selection_path: Path) -> dict[str, Any]:
    selection = _load_json(selection_path)
    if selection.get("interpretation") != "CALIBRATION_ONLY_GENERATION_CAP_SELECTION":
        raise ValueError("selection is not a calibration-only artifact")
    root = selection_path.resolve().parents[1]
    candidate_reports: list[dict[str, Any]] = []
    for audit in selection.get("candidate_audits", []):
        budget = int(audit["max_tokens"])
        run_dir = root / Path(audit["run_directory"])
        manifest_path = run_dir / "manifest.json"
        manifest = _load_json(manifest_path)
        if _sha256(manifest_path) != audit["run_manifest_sha256"]:
            raise ValueError(f"manifest hash mismatch for budget {budget}")
        if not manifest.get("queue_only"):
            raise ValueError(f"candidate {budget} is not queue-only")
        if any(
            int(manifest.get(field, -1)) != 0
            for field in ("verified_count", "projected_count", "blind_join_count")
        ):
            raise ValueError(f"candidate {budget} contains forbidden downstream results")
        raw_paths = sorted((run_dir / "sessions").rglob("raw.json"))
        rows = [_load_json(path) for path in raw_paths]
        if len(rows) != int(manifest["session_count"]):
            raise ValueError(f"raw/session count mismatch for budget {budget}")
        finish_reasons = Counter(
            str(row["turns"][0]["finish_reason"]) for row in rows
        )
        token_counts = [int(row["turns"][0]["token_count"]) for row in rows]
        response_times = [float(row["response_time_seconds"]) for row in rows]
        attempts = [int(row["attempts"]) for row in rows]
        runner_sha = str(manifest["observation_interface"]["runner_sha256"])
        retry_controlled = runner_sha == RETRY_CONTROLLED_RUNNER_SHA256
        candidate_reports.append(
            {
                "max_tokens": budget,
                "run_directory": str(audit["run_directory"]),
                "run_manifest_sha256": audit["run_manifest_sha256"],
                "runner_sha256": runner_sha,
                "session_count": len(rows),
                "finish_reason_counts": dict(sorted(finish_reasons.items())),
                "observed_length_rate": finish_reasons.get("length", 0) / len(rows),
                "token_count": _summary([float(value) for value in token_counts]),
                "successful_request_response_time_seconds": {
                    **_summary(response_times),
                    "definition": (
                        "Elapsed time from immediately before the successful explicit "
                        "provider request until its response returned. Failed explicit "
                        "attempt durations are not stored in raw.json."
                        if retry_controlled
                        else "Elapsed time around the successful outer SDK call. The SDK "
                        "could perform up to two hidden internal retries, so this is not "
                        "guaranteed to represent one provider request."
                    ),
                    "single_provider_request_identifiable": retry_controlled,
                },
                "explicit_attempts": {
                    "total": sum(attempts),
                    "mean_per_session": round(mean(attempts), 6),
                    "maximum_per_session": max(attempts),
                    "sessions_with_retry": sum(value > 1 for value in attempts),
                    "failed_attempt_count": sum(attempts) - len(attempts),
                    "count_distribution": {
                        str(key): value
                        for key, value in sorted(Counter(attempts).items())
                    },
                    "sdk_internal_retries_disabled": retry_controlled,
                },
                "model": manifest["model"],
                "generation_parameter_set": manifest["generation_parameter_set"],
                "provider_response_identity": manifest["provider_response_identity"],
            }
        )

    return {
        "schema": "LLM-SVM-CoCC-generation-budget-calibration-report/1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "report_type": "CALIBRATION_ONLY_NOT_CONFIRMATORY",
        "selection_artifact": str(selection_path),
        "selection_artifact_sha256": _sha256(selection_path),
        "protocol_id": selection["protocol_id"],
        "dataset": selection["dataset"],
        "dataset_sha256": selection["dataset_sha256"],
        "calibration_item_count": selection["calibration_item_count"],
        "outcome_labels_consumed": False,
        "holdout_rows_consumed": False,
        "verifier_executed": False,
        "projector_executed": False,
        "prama_coordinates_available": False,
        "auroc_available": False,
        "selection": {
            "status": selection["status"],
            "selection_rule": selection["selection_rule"],
            "maximum_observed_length_rate": selection[
                "maximum_observed_length_rate"
            ],
            "selected_max_tokens": selection["selected_max_tokens"],
            "next_candidate_max_tokens": selection["next_candidate_max_tokens"],
            "design_freeze_allowed": selection["design_freeze_allowed"],
        },
        "candidate_reports": candidate_reports,
        "conclusion": (
            f"No evaluated budget met the frozen length-rate threshold; continue with "
            f"the preregistered {selection['next_candidate_max_tokens']}-token candidate."
            if selection["status"] == "MORE_CALIBRATION_REQUIRED"
            else "The smallest passing preregistered budget has been selected."
        ),
        "interpretation_boundary": (
            "This report supports generation-budget selection only. It contains no "
            "evidence about PASS/FAIL, PRAMA discrimination, structural state or early warning."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = build_report(args.selection)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": _sha256(args.output),
                "status": report["selection"]["status"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
