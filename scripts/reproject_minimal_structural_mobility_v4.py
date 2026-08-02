#!/usr/bin/env python
"""Reproject the completed minimal-perturbation study with mobility v4.

This script is strictly offline: it reads saved numeric projection requests and
PRAMA trajectories, writes v4 observer artifacts, and joins arm labels only for
the final report.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for candidate in (REPO_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts.project_cocc_operator_geometry import file_sha256, load_json  # noqa: E402
from scripts.project_cocc_structural_mobility_v4 import build_artifact  # noqa: E402


ARMS = ("control_neutral", "concrete_content", "abstract_content", "minimal_structural")


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def endpoints(windows: Sequence[Mapping[str, Any]], horizon: int | None = None) -> dict[str, Any]:
    ready = [row for row in windows if row["geometry_ready"]]
    if horizon is not None:
        ready = ready[:horizon]
    if not ready:
        return {
            "geometry_ready_windows": 0,
            "structural_evaluation_status": "NOT_EVALUABLE_SHORT_TRAJECTORY",
            "active_window_fraction": None,
            "stagnant_fraction": None,
            "stagnation_onset": None,
            "minimum_transport_efficiency_active": None,
            "raw_recurrence_persistence_area": None,
            "mobile_recurrence_persistence_area": None,
            "maximum_mobile_recurrence_persistence": None,
            "recurrent_fraction": None,
            "crystallizing_fraction": None,
            "crystallized_fraction": None,
            "crystallization_onset": None,
        }
    active = [row for row in ready if row["transport_efficiency_valid"]]
    states = [row["structural_mobility_state"] for row in ready]
    crystallized = [row for row in ready if row["structural_mobility_state"] == "CRYSTALLIZED"]
    stagnant = [row for row in ready if row["structural_mobility_state"] == "STAGNANT"]
    return {
        "geometry_ready_windows": len(ready),
        "structural_evaluation_status": "EVALUABLE",
        "active_window_fraction": len(active) / len(ready) if ready else 0.0,
        "stagnant_fraction": len(stagnant) / len(ready) if ready else 0.0,
        "stagnation_onset": int(stagnant[0]["window_index"]) if stagnant else None,
        "minimum_transport_efficiency_active": min(
            (float(row["transport_efficiency_active"]) for row in active), default=None
        ),
        "raw_recurrence_persistence_area": statistics.fmean(
            float(row["recurrence_persistence_raw"]) for row in ready
        ) if ready else 0.0,
        "mobile_recurrence_persistence_area": statistics.fmean(
            float(row["mobile_recurrence_persistence"]) for row in ready
        ) if ready else 0.0,
        "maximum_mobile_recurrence_persistence": max(
            (float(row["mobile_recurrence_persistence"]) for row in ready), default=0.0
        ),
        "recurrent_fraction": states.count("RECURRENT") / len(ready) if ready else 0.0,
        "crystallizing_fraction": states.count("CRYSTALLIZING") / len(ready) if ready else 0.0,
        "crystallized_fraction": len(crystallized) / len(ready) if ready else 0.0,
        "crystallization_onset": int(crystallized[0]["window_index"]) if crystallized else None,
    }


def summarize(rows: Sequence[Mapping[str, Any]], suffix: str = "") -> dict[str, Any]:
    metrics = (
        "active_window_fraction",
        "stagnant_fraction",
        "raw_recurrence_persistence_area",
        "mobile_recurrence_persistence_area",
        "maximum_mobile_recurrence_persistence",
        "recurrent_fraction",
        "crystallizing_fraction",
        "crystallized_fraction",
    )
    if suffix:
        metrics = tuple(f"{metric}{suffix}" for metric in metrics)
    result: dict[str, Any] = {}
    for arm in ARMS:
        members = [row for row in rows if row["arm"] == arm]
        metric_summaries = {}
        for metric in metrics:
            values = [float(row[metric]) for row in members if row[metric] is not None]
            metric_summaries[metric] = {
                "evaluable_n": len(values),
                "mean": statistics.fmean(values) if values else None,
                "median": statistics.median(values) if values else None,
            }
        result[arm] = {
            "n": len(members),
            "structurally_evaluable_n": sum(
                row[f"structural_evaluation_status{suffix}"] == "EVALUABLE"
                for row in members
            ),
            **metric_summaries,
        }
    return result


def contrasts(rows: Sequence[Mapping[str, Any]], comparator_arm: str, suffix: str = "") -> list[dict[str, Any]]:
    metrics = (
        "stagnant_fraction",
        "mobile_recurrence_persistence_area",
        "maximum_mobile_recurrence_persistence",
        "crystallized_fraction",
    )
    output = []
    for item_id in sorted({str(row["item_id"]) for row in rows}):
        minimal = next(row for row in rows if row["item_id"] == item_id and row["arm"] == "minimal_structural")
        comparator = next(row for row in rows if row["item_id"] == item_id and row["arm"] == comparator_arm)
        differences = {}
        for metric in metrics:
            minimal_value = minimal[f"{metric}{suffix}"]
            comparator_value = comparator[f"{metric}{suffix}"]
            evaluable = minimal_value is not None and comparator_value is not None
            differences[f"{metric}{suffix}_contrast_evaluable"] = evaluable
            differences[f"{metric}{suffix}_minimal_minus_comparator"] = (
                float(minimal_value) - float(comparator_value) if evaluable else None
            )
        output.append({
            "item_id": item_id,
            "comparator_arm": comparator_arm,
            **differences,
        })
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--geometry-contract", type=Path, default=Path("config/cocc_operator_geometry_observer_v1.json"))
    parser.add_argument("--mobility-contract", type=Path, default=Path("config/cocc_structural_mobility_observer_v4.json"))
    parser.add_argument("--mobility-freeze", type=Path, default=Path("config/cocc_structural_mobility_observer_v4.freeze.json"))
    parser.add_argument("--fixed-ready-horizon", type=int, default=16)
    args = parser.parse_args()

    freeze = load_json(args.mobility_freeze)
    contract_hash = file_sha256(args.mobility_contract)
    if freeze["observer_contract_sha256"] != contract_hash:
        raise SystemExit("v4 contract differs from its freeze")

    rows: list[dict[str, Any]] = []
    for session_dir in sorted((args.input_dir / "sessions").iterdir()):
        if not session_dir.is_dir():
            continue
        request_path = session_dir / "projection_request.json"
        trajectory_path = session_dir / "trajectory.jsonl"
        raw_path = session_dir / "raw.json"
        if not (request_path.exists() and trajectory_path.exists() and raw_path.exists()):
            raise SystemExit(f"incomplete saved session: {session_dir}")
        artifact = build_artifact(request_path, trajectory_path, args.geometry_contract, args.mobility_contract)
        atomic_json(session_dir / "structural_mobility_v4.json", artifact)
        raw = load_json(raw_path)
        full = endpoints(artifact["windows"])
        fixed = endpoints(artifact["windows"], args.fixed_ready_horizon)
        rows.append({
            "session_id": str(raw["session_id"]),
            "item_id": str(raw["item_id"]),
            "arm": str(raw["arm"]),
            "finish_reason": str(raw["turns"][-1]["finish_reason"]),
            "token_count": int(raw["turns"][-1]["token_count"]),
            **full,
            **{f"{key}_h{args.fixed_ready_horizon}": value for key, value in fixed.items()},
        })

    csv_path = args.input_dir / "session_endpoints_v4.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    suffix = f"_h{args.fixed_ready_horizon}"
    report = {
        "schema": "LLM-SVM-minimal-structural-mobility-v4-report/1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_report": str(args.input_dir / "report.json"),
        "source_report_sha256": file_sha256(args.input_dir / "report.json"),
        "mobility_contract_sha256": contract_hash,
        "offline_reprojection_only": True,
        "llm_calls": 0,
        "observer_receives_arm_or_item_labels": False,
        "fixed_geometry_ready_horizon": args.fixed_ready_horizon,
        "summary_full": summarize(rows),
        "summary_fixed_horizon": summarize(rows, suffix),
        "minimal_vs_abstract_full": contrasts(rows, "abstract_content"),
        "minimal_vs_abstract_fixed_horizon": contrasts(rows, "abstract_content", suffix),
        "minimal_vs_concrete_full": contrasts(rows, "concrete_content"),
        "minimal_vs_concrete_fixed_horizon": contrasts(rows, "concrete_content", suffix),
        "session_endpoints": rows,
    }
    output = args.input_dir / "report_structural_mobility_v4.json"
    atomic_json(output, report)
    print(json.dumps({"output": str(output), "sha256": file_sha256(output), "sessions": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
