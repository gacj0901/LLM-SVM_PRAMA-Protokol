#!/usr/bin/env python
"""Offline v5 coherence reprojection of the three completed minimal-perturbation runs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.project_cocc_operator_geometry import file_sha256, load_json  # noqa: E402
from scripts.project_cocc_structural_coherence_v5 import build_artifact  # noqa: E402
from scripts.reproject_minimal_structural_mobility_v4 import atomic_json  # noqa: E402


RUNS = (
    Path("run_outputs/minimal_structural_nemotron_super_dynamic_v4"),
    Path("run_outputs/minimal_structural_mistral_medium_3_5_dynamic_v4"),
    Path("run_outputs/minimal_structural_nemotron_ultra_dynamic_v4"),
)
ARMS = ("control_neutral", "concrete_content", "abstract_content", "minimal_structural")
STATES = ("VIABLE", "STAGNANT", "IMITATIVE_ECHO", "RECURRENT", "CRYSTALLIZING", "CRYSTALLIZED", "COHERENCE_LOSS", "UNRESOLVED")


def endpoints(windows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ready = [row for row in windows if row["geometry_ready"]]
    if not ready:
        return {
            "structural_evaluation_status": "NOT_EVALUABLE_SHORT_TRAJECTORY",
            "geometry_ready_windows": 0,
            "inertial_echo_recurrence": None,
            "recurrence_persistence": None,
            "mean_transport_coherence": None,
            "terminal_transport_coherence": None,
            "mean_variation_capacity": None,
            "maximum_variation_contraction": None,
            **{f"fraction_{state.lower()}": None for state in STATES},
        }
    coherence = [float(row["transport_coherence"]) for row in ready if row["transport_coherence"] is not None]
    capacity = [float(row["variation_capacity"]) for row in ready if row["variation_capacity"] is not None]
    contractions = [float(row["variation_contraction"]) for row in ready if row["variation_contraction"] is not None]
    states = [row["structural_coherence_state"] for row in ready]
    return {
        "structural_evaluation_status": "EVALUABLE",
        "geometry_ready_windows": len(ready),
        "inertial_echo_recurrence": statistics.fmean(float(row["inertial_echo_recurrence"]) for row in ready),
        "recurrence_persistence": statistics.fmean(float(row["recurrence_persistence"]) for row in ready),
        "mean_transport_coherence": statistics.fmean(coherence) if coherence else None,
        "terminal_transport_coherence": coherence[-1] if coherence else None,
        "mean_variation_capacity": statistics.fmean(capacity) if capacity else None,
        "maximum_variation_contraction": max(contractions) if contractions else None,
        **{f"fraction_{state.lower()}": states.count(state) / len(states) for state in STATES},
    }


def metric_summary(rows: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    members = [row for row in rows if row["arm"] == arm]
    metrics = (
        "inertial_echo_recurrence",
        "recurrence_persistence",
        "mean_transport_coherence",
        "terminal_transport_coherence",
        "mean_variation_capacity",
        "maximum_variation_contraction",
        *[f"fraction_{state.lower()}" for state in STATES],
    )
    output = {"n": len(members), "structurally_evaluable_n": sum(row["structural_evaluation_status"] == "EVALUABLE" for row in members)}
    for metric in metrics:
        values = [float(row[metric]) for row in members if row[metric] is not None]
        output[metric] = {
            "evaluable_n": len(values),
            "mean": statistics.fmean(values) if values else None,
            "median": statistics.median(values) if values else None,
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, default=list(RUNS))
    parser.add_argument("--geometry-contract", type=Path, default=Path("config/cocc_operator_geometry_observer_v1.json"))
    parser.add_argument("--coherence-contract", type=Path, default=Path("config/cocc_structural_coherence_observer_v5.json"))
    parser.add_argument("--coherence-freeze", type=Path, default=Path("config/cocc_structural_coherence_observer_v5.freeze.json"))
    parser.add_argument("--out", type=Path, default=Path("run_outputs/minimal_structural_three_model_coherence_v5_report.json"))
    args = parser.parse_args()
    contract_hash = file_sha256(args.coherence_contract)
    if load_json(args.coherence_freeze)["observer_contract_sha256"] != contract_hash:
        raise SystemExit("v5 coherence contract differs from freeze")
    model_reports = []
    all_rows = []
    for run_dir in args.runs:
        source_report = load_json(run_dir / "report.json")
        rows = []
        for session_dir in sorted((run_dir / "sessions").iterdir()):
            if not session_dir.is_dir():
                continue
            raw = load_json(session_dir / "raw.json")
            artifact = build_artifact(
                session_dir / "projection_request.json",
                session_dir / "trajectory.jsonl",
                args.geometry_contract,
                args.coherence_contract,
            )
            atomic_json(session_dir / "structural_coherence_v5.json", artifact)
            row = {
                "model": source_report["model"],
                "session_id": str(raw["session_id"]),
                "item_id": str(raw["item_id"]),
                "arm": str(raw["arm"]),
                "token_count": int(raw["turns"][-1]["token_count"]),
                "finish_reason": str(raw["turns"][-1]["finish_reason"]),
                **endpoints(artifact["windows"]),
            }
            rows.append(row)
            all_rows.append(row)
        model_reports.append({
            "model": source_report["model"],
            "source_report": str(run_dir / "report.json"),
            "source_report_sha256": file_sha256(run_dir / "report.json"),
            "by_arm": {arm: metric_summary(rows, arm) for arm in ARMS},
            "sessions": rows,
        })
    report = {
        "schema": "LLM-SVM-minimal-structural-three-model-coherence-v5-report/1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "coherence_contract_sha256": contract_hash,
        "offline_reprojection_only": True,
        "llm_calls": 0,
        "observer_receives_labels": False,
        "model_reports": model_reports,
        "session_count": len(all_rows),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out, report)
    print(json.dumps({"output": str(args.out), "sha256": file_sha256(args.out), "sessions": len(all_rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
