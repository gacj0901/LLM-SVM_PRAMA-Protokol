#!/usr/bin/env python
"""Exploratory evaluation of structural mobility v3 on completed CoCC runs."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_cocc_operator_geometry import (  # noqa: E402
    file_sha256,
    paired_result,
    residualize_by_length,
    resolve_path,
    spearman,
)
from scripts.evaluate_break_the_chain_prama import auroc  # noqa: E402
from scripts.project_cocc_structural_mobility_v3 import build_artifact  # noqa: E402


ENDPOINTS = {
    "minimum_transport_efficiency": "negative_minimum_transport_efficiency",
    "recurrence_persistence_area": "recurrence_persistence_area",
    "maximum_recurrence_persistence": "maximum_recurrence_persistence",
    "crystallized_fraction": "crystallized_fraction",
    "crystallization_onset": "early_crystallization_score",
}


def endpoints(artifact: dict[str, Any], max_xi: float) -> dict[str, Any]:
    windows = artifact["windows"]
    ready = [window for window in windows if window["geometry_ready"]]
    active = [window for window in ready if float(window["path_length"]) > 0.5]
    crystallized = [window for window in ready if window["structural_mobility_state"] == "CRYSTALLIZED"]
    onset = int(crystallized[0]["window_index"]) if crystallized else None
    state_fractions = {
        state.lower() + "_fraction": (
            sum(window["structural_mobility_state"] == state for window in ready) / len(ready)
            if ready else 0.0
        )
        for state in ("MOBILE", "RECURRENT", "CRYSTALLIZING", "CRYSTALLIZED")
    }
    minimum_efficiency = min((float(window["transport_efficiency"]) for window in active), default=1.0)
    return {
        "n_windows": len(windows),
        "geometry_ready_windows": len(ready),
        "minimum_transport_efficiency": minimum_efficiency,
        "negative_minimum_transport_efficiency": -minimum_efficiency,
        "recurrence_persistence_area": statistics.fmean(
            [float(window["recurrence_persistence"]) for window in ready]
        ) if ready else 0.0,
        "maximum_recurrence_persistence": max(
            (float(window["recurrence_persistence"]) for window in ready), default=0.0
        ),
        "crystallized_fraction": len(crystallized) / len(ready) if ready else 0.0,
        "crystallization_onset": onset,
        "early_crystallization_score": 1.0 / (1.0 + onset) if onset is not None else 0.0,
        "max_xi": max_xi,
        **state_fractions,
    }


def summarize_run(run_dir: Path, geometry_contract: Path, mobility_contract: Path, artifact_root: Path) -> dict[str, Any]:
    source_report_path = run_dir / "evaluation" / "report.json"
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    with (run_dir / "evaluation" / "blind_join.csv").open(encoding="utf-8", newline="") as handle:
        joined = list(csv.DictReader(handle))
    rows = []
    for joined_row in joined:
        session_id = str(joined_row["session_id"])
        request_path = run_dir / "projection" / "requests" / f"{session_id}.json"
        trajectory_path = resolve_path(joined_row["trajectory_path"])
        artifact = build_artifact(request_path, trajectory_path, geometry_contract, mobility_contract)
        artifact_path = artifact_root / run_dir.name / f"{session_id}.json"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raw = json.loads((run_dir / "sessions" / session_id / "raw.json").read_text(encoding="utf-8"))
        trajectory = [json.loads(line) for line in trajectory_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        row_endpoints = endpoints(artifact, max(float(window["xi"]) for window in trajectory))
        rows.append({
            "session_id": session_id,
            "problem_id": str(raw["problem_id"]),
            "condition": str(raw["perturbation_type"]),
            "label": int(joined_row["label"]),
            **row_endpoints,
        })
    labels = [row["label"] for row in rows]
    metrics = {}
    for endpoint_name, score_field in ENDPOINTS.items():
        scores = [float(row[score_field]) for row in rows]
        residuals = residualize_by_length(scores, [row["n_windows"] for row in rows])
        metrics[endpoint_name] = {
            "score_field": score_field,
            "auroc": auroc(scores, labels),
            "length_residualized_auroc": auroc(residuals, labels),
            "spearman_with_n_windows": spearman(scores, [row["n_windows"] for row in rows]),
            "spearman_with_max_xi": spearman(scores, [row["max_xi"] for row in rows]),
            "paired_inference": paired_result(rows, score_field),
        }
    by_outcome = {}
    for name, label in (("PASS", 0), ("FAIL", 1)):
        members = [row for row in rows if row["label"] == label]
        by_outcome[name] = {
            "n": len(members),
            **{
                "median_" + endpoint: statistics.median(
                    [float(row[endpoint]) for row in members if row[endpoint] is not None]
                ) if any(row[endpoint] is not None for row in members) else None
                for endpoint in ENDPOINTS
            },
            "median_n_windows": statistics.median(row["n_windows"] for row in members),
            "median_max_xi": statistics.median(row["max_xi"] for row in members),
        }
    return {
        "model": source_report["model"],
        "source_report_sha256": file_sha256(source_report_path),
        "source_design_sha256": source_report["design_sha256"],
        "session_count": len(rows),
        "outcomes": source_report["outcomes"],
        "mobility_metrics": metrics,
        "endpoint_medians_by_outcome": by_outcome,
        "session_endpoints": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-contract", type=Path, default=Path("config/cocc_operator_geometry_observer_v1.json"))
    parser.add_argument("--mobility-contract", type=Path, default=Path("config/cocc_structural_mobility_observer_v3.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("run_outputs/cocc_10pair_structural_mobility_v3"))
    args = parser.parse_args()
    run_dirs = [
        Path("run_outputs/cocc_10pair_nemotron3_super_v1"),
        Path("run_outputs/cocc_10pair_mistral_medium_3_5_v1"),
        Path("run_outputs/cocc_10pair_nemotron3_ultra_v1"),
    ]
    try:
        mobility_hash = file_sha256(args.mobility_contract)
        geometry_hash = file_sha256(args.geometry_contract)
        freeze = json.loads(Path("config/cocc_structural_mobility_observer_v3.freeze.json").read_text(encoding="utf-8"))
        if freeze["observer_contract_sha256"] != mobility_hash:
            raise ValueError("mobility v3 contract differs from freeze record")
        if freeze["base_geometry_contract_sha256"] != geometry_hash:
            raise ValueError("base geometry contract differs from freeze record")
        models = [
            summarize_run(path, args.geometry_contract, args.mobility_contract, args.output_dir / "artifacts")
            for path in run_dirs
        ]
        report = {
            "schema": "LLM-SVM-CoCC-structural-mobility-exploratory-report/3",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "inference_status": "EXPLORATORY_ONLY",
            "mobility_contract_sha256": mobility_hash,
            "base_geometry_contract_sha256": geometry_hash,
            "kernel_modified": False,
            "llm_calls_executed": 0,
            "models": models,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output = args.output_dir / "report.json"
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"structural-mobility v3 analysis failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"output": str(output), "models": len(models)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
