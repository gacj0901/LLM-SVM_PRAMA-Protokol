#!/usr/bin/env python
"""Reproject an existing run to audit the universal dynamic observer offline."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT, REPO_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts import evaluate_break_the_chain_prama as ev  # noqa: E402
from scripts.project_cocc_prama import file_sha256, load_json, validate_identity  # noqa: E402
from scripts.project_cocc_prama_dynamic import project  # noqa: E402


HORIZONS = (1, 2, 4, 8, 16)


def _resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def _auc(rows: list[dict[str, Any]], field: str) -> float | None:
    labels = [int(row["label"]) for row in rows]
    if len(set(labels)) < 2:
        return None
    return ev.auroc([float(row[field]) for row in rows], labels)


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "pass_n": sum(int(row["label"]) == 0 for row in rows),
        "fail_n": sum(int(row["label"]) == 1 for row in rows),
        "auroc": {
            field: _auc(rows, field)
            for field in (
                "max_delta",
                "max_xi",
                "max_negative_balance",
                "resource_occupancy",
                "terminal_exhaustion",
            )
        },
    }


def build_report(
    run_dir: Path,
    contract_path: Path,
    declaration_path: Path,
    recertification_path: Path,
) -> dict[str, Any]:
    contract = load_json(contract_path)
    contract_hash = file_sha256(contract_path)
    config, columns, identity = validate_identity(
        declaration_path, recertification_path
    )
    manifest = load_json(run_dir / "manifest.json")
    max_tokens = int(manifest["generation_parameter_set"]["max_tokens"])
    old_report_path = run_dir / "evaluation" / "partial_report.json"
    old_report = load_json(old_report_path) if old_report_path.exists() else None
    raw_by_session = {
        path.parent.name: load_json(path)
        for path in (run_dir / "sessions").glob("*/raw.json")
    }
    rows = []
    with (run_dir / "evaluation" / "blind_join.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        for joined in csv.DictReader(handle):
            session_id = str(joined["session_id"])
            request = load_json(_resolve(joined["projection_request_path"]))
            trajectory = project(
                request,
                contract,
                contract_hash,
                config,
                columns,
                identity,
            )
            raw = raw_by_session[session_id]
            token_count = int(raw["turns"][0]["token_count"])
            finish_reason = str(raw["turns"][0]["finish_reason"])
            rows.append(
                {
                    "session_id": session_id,
                    "label": int(joined["label"]),
                    "trajectory": trajectory,
                    "max_delta": max(float(point["delta"]) for point in trajectory),
                    "max_xi": max(float(point["xi"]) for point in trajectory),
                    "max_negative_balance": max(
                        -float(point["balance"]) for point in trajectory
                    ),
                    "resource_occupancy": min(1.0, token_count / max_tokens),
                    "terminal_exhaustion": float(finish_reason == "length"),
                }
            )
    full = _metrics(rows)
    horizons = {}
    for horizon in HORIZONS:
        prefix_rows = []
        for row in rows:
            trajectory = row["trajectory"]
            if len(trajectory) < horizon:
                continue
            prefix = trajectory[:horizon]
            prefix_rows.append(
                {
                    **row,
                    "max_delta": max(float(point["delta"]) for point in prefix),
                    "max_xi": max(float(point["xi"]) for point in prefix),
                    "max_negative_balance": max(
                        -float(point["balance"]) for point in prefix
                    ),
                }
            )
        horizons[str(horizon)] = _metrics(prefix_rows)
    threshold_crossing = sum(
        any(float(point["xi"]) > float(point["theta"]) for point in row["trajectory"])
        for row in rows
    )
    excess_activation = sum(
        any(float(point["accumulated_excess"]) > 0.0 for point in row["trajectory"])
        for row in rows
    )
    capacity_degradation = sum(
        any(float(point["capacity"]) < 1.0 for point in row["trajectory"])
        for row in rows
    )
    max_xi_minus_theta = max(
        float(point["xi"]) - float(point["theta"])
        for row in rows
        for point in row["trajectory"]
    )
    return {
        "schema": "LLM-SVM-CoCC-dynamic-observer-development-audit/1",
        "analysis_status": "POST_HOC_ARTIFACT_DEVELOPMENT_NOT_CONFIRMATORY",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_run": str(run_dir),
        "historical_verdict_modified": False,
        "observer": {
            "id": contract["observer_id"],
            "contract_sha256": contract_hash,
            "model_specific_parameters": False,
            "requires_external_calibration": False,
            "causal": True,
        },
        "diagnosis": {
            "replaced_observer_unit_mismatch": (
                "The historical observer normalized window-mean surprisal with a "
                "scale estimated from individual-token surprisal. The dynamic observer "
                "estimates and applies location/scale on the same window-level statistic."
            ),
            "labels_or_text_used_by_observer": False,
        },
        "full_trajectory": full,
        "absolute_window_horizons": horizons,
        "activation_audit": {
            "threshold_crossing_session_count": threshold_crossing,
            "accumulated_excess_activation_session_count": excess_activation,
            "capacity_degradation_session_count": capacity_degradation,
            "max_xi_minus_theta": max_xi_minus_theta,
        },
        "operational_channel": {
            "kept_separate_from_structural_state": True,
            "resource_occupancy_auroc": full["auroc"]["resource_occupancy"],
            "terminal_exhaustion_auroc": full["auroc"]["terminal_exhaustion"],
            "interpretation": (
                "Descriptive only: the partial run is generation-cap confounded. "
                "Neither value modifies Xi, accumulated_excess, or capacity."
            ),
        },
        "historical_projector_metrics": (
            {
                "report_sha256": file_sha256(old_report_path),
                "full_trajectory_metrics": old_report.get("full_trajectory_metrics"),
                "activation_audit": old_report.get("activation_audit"),
            }
            if old_report is not None
            else None
        ),
        "claim_boundary": (
            "This audit may select an observer for a future freeze. It cannot revise "
            "the source run's frozen design, projections, or verdict."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--observer-contract", required=True, type=Path)
    parser.add_argument("--declaration", required=True, type=Path)
    parser.add_argument("--recertification", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    report = build_report(
        args.run_dir,
        args.observer_contract,
        args.declaration,
        args.recertification,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"output": str(args.out), "sha256": sha256(args.out.read_bytes()).hexdigest()}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
