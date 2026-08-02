#!/usr/bin/env python3
"""V8 causal-hysteresis reprojection of the completed frontend battery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for candidate in (REPO_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from aptadynamic_llm.structural_coherence_v8 import (  # noqa: E402
    DIAGNOSTICS_V8,
    PRIMARY_STATES_V8,
    StructuralCoherenceV8Config,
    classify_structural_coherence_v8,
    episode_summary,
)
from scripts.analyze_frontend_test_battery_v7 import (  # noqa: E402
    atomic_json,
    endpoint_value,
    fraction_record,
    mean_or_none,
    spearman,
    utc_now,
)
from scripts.project_cocc_operator_geometry import file_sha256, load_json  # noqa: E402


def session_endpoints(
    windows: Sequence[Mapping[str, Any]], config: StructuralCoherenceV8Config
) -> dict[str, Any]:
    total = len(windows)
    ready = [row for row in windows if row.get("geometry_ready")]
    transport = [row for row in ready if row.get("transport_coherence") is not None]
    ready_n, transport_n = len(ready), len(transport)
    state_counts = {
        state: sum(row.get("primary_state_v8") == state for row in ready) for state in PRIMARY_STATES_V8
    }
    if sum(state_counts.values()) != ready_n:
        raise RuntimeError("v8 state exhaustiveness invariant failed")
    diagnostic_counts = {
        diagnostic: sum(diagnostic in (row.get("diagnostics_v8") or []) for row in ready)
        for diagnostic in DIAGNOSTICS_V8 if diagnostic != "INSUFFICIENT_GEOMETRY"
    }
    confirmed = sum(row.get("viability_status_v8") == "CONFIRMED" for row in ready)
    provisional = sum(row.get("viability_status_v8") == "PROVISIONAL" for row in ready)
    status = (
        "EVALUABLE" if transport_n >= config.minimum_session_transport_windows
        else "SESSION_NOT_EVALUABLE_SHORT_TRAJECTORY"
    )
    return {
        "session_evaluation_status": status,
        "coverage": {
            "total_windows": {"count": total, "denominator": total, "fraction": 1.0 if total else None},
            "geometry_ready": fraction_record(ready_n, total),
            "transport_evaluable": fraction_record(transport_n, total),
            "insufficient_geometry": fraction_record(total - ready_n, total),
        },
        "state_occupancy_among_geometry_ready": {
            state.lower(): fraction_record(count, ready_n) for state, count in state_counts.items()
        },
        "viability_occupancy_among_geometry_ready": {
            "confirmed": fraction_record(confirmed, ready_n),
            "provisional": fraction_record(provisional, ready_n),
            "total_viable": fraction_record(state_counts["VIABLE"], ready_n),
        },
        "diagnostic_prevalence_among_geometry_ready": {
            diagnostic.lower(): fraction_record(count, ready_n)
            for diagnostic, count in diagnostic_counts.items()
        },
        "channels": {
            "terminal_transport_coherence": endpoint_value(transport, "transport_coherence", "terminal"),
            "mean_transport_coherence": endpoint_value(transport, "transport_coherence"),
            "mean_recurrence_persistence": endpoint_value(ready, "recurrence_persistence"),
            "maximum_variation_contraction": endpoint_value(ready, "variation_contraction", "max"),
            "mean_inertial_echo_recurrence": endpoint_value(ready, "inertial_echo_recurrence"),
            "mean_variation_capacity": endpoint_value(ready, "variation_capacity"),
        },
        "episodes": {
            "crystallizing": episode_summary(ready, {"CRYSTALLIZING", "CRYSTALLIZED"}),
            "transport_disrupted": episode_summary(ready, {"TRANSPORT_DISRUPTED"}),
        },
        "onsets": {
            "first_recurrent_window": next((int(row["absolute_window_index"]) for row in windows if row.get("primary_state_v8") == "RECURRENT"), None),
            "first_crystallizing_window": next((int(row["absolute_window_index"]) for row in windows if row.get("primary_state_v8") == "CRYSTALLIZING"), None),
            "first_crystallized_window": next((int(row["absolute_window_index"]) for row in windows if row.get("primary_state_v8") == "CRYSTALLIZED"), None),
            "first_transport_disrupted_window": next((int(row["absolute_window_index"]) for row in windows if row.get("primary_state_v8") == "TRANSPORT_DISRUPTED"), None),
        },
    }


def aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = {
        "terminal_transport_coherence": lambda row: row["endpoints"]["channels"]["terminal_transport_coherence"],
        "mean_recurrence_persistence": lambda row: row["endpoints"]["channels"]["mean_recurrence_persistence"],
        "fraction_viable_confirmed": lambda row: row["endpoints"]["viability_occupancy_among_geometry_ready"]["confirmed"]["fraction"],
        "fraction_viable_provisional": lambda row: row["endpoints"]["viability_occupancy_among_geometry_ready"]["provisional"]["fraction"],
        "fraction_recurrent": lambda row: row["endpoints"]["state_occupancy_among_geometry_ready"]["recurrent"]["fraction"],
        "fraction_crystallizing": lambda row: row["endpoints"]["state_occupancy_among_geometry_ready"]["crystallizing"]["fraction"],
        "fraction_transport_disrupted": lambda row: row["endpoints"]["state_occupancy_among_geometry_ready"]["transport_disrupted"]["fraction"],
        "maximum_crystallizing_dwell": lambda row: row["endpoints"]["episodes"]["crystallizing"]["maximum_consecutive_windows"],
        "maximum_transport_disrupted_dwell": lambda row: row["endpoints"]["episodes"]["transport_disrupted"]["maximum_consecutive_windows"],
    }
    output: dict[str, Any] = {"n": len(rows)}
    for name, getter in fields.items():
        values = [float(value) for row in rows if (value := getter(row)) is not None]
        output[name] = {"evaluable_n": len(values), "mean": mean_or_none(values)}
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--battery-dir", type=Path, default=Path("FrontEnd/results/blind_interaction_battery_v1"))
    parser.add_argument("--contract", type=Path, default=Path("config/sequor_structural_coherence_observer_v8.json"))
    args = parser.parse_args()
    contract = load_json(args.contract)
    config = StructuralCoherenceV8Config.from_contract(contract)
    v6_report_path = args.battery_dir / "structural_analysis_v6_report.json"
    v6_report = load_json(v6_report_path)
    evaluations = {
        (row["model"], row["test_id"]): row["external_evaluation"] for row in v6_report["items"]
    }

    projection_dir = args.battery_dir / "structural_projection_v8"
    rows = []
    for path in sorted((args.battery_dir / "structural_projection_v6").glob("*.json")):
        source = load_json(path)
        windows = classify_structural_coherence_v8(source["structural_windows"], config)
        projection = {
            "schema": "LLM-SVM-frontend-structural-projection/8", "generated_at": utc_now(),
            "session_id": source["session_id"], "model": source["model"],
            "source_v6_projection_sha256": file_sha256(path),
            "v8_contract_sha256": file_sha256(args.contract),
            "contains_prompt_or_answer": False, "contains_external_evaluation": False,
            "numeric_channels_recomputed": False, "kernel_modified": False,
            "windows": windows,
        }
        target = projection_dir / path.name
        atomic_json(target, projection)
        test_id = path.stem.rsplit("--", 1)[-1]
        evaluation = evaluations[(source["model"], test_id)]
        original = next(
            row for row in v6_report["items"]
            if row["model"] == source["model"] and row["test_id"] == test_id
        )
        rows.append({
            "model": source["model"], "test_id": test_id,
            "token_count": original["token_count"], "finish_reason": original["finish_reason"],
            "protocol_completion": evaluation["protocol_completion"], "failure_mode": evaluation["failure_mode"],
            "projection_path": str(target.resolve()), "projection_sha256": file_sha256(target),
            "endpoints": session_endpoints(windows, config), "windows": windows,
        })

    horizons = []
    for horizon in contract["fixed_causal_horizons_windows"]:
        reached = []
        for row in rows:
            if len(row["windows"]) < int(horizon):
                continue
            endpoint = session_endpoints(row["windows"][: int(horizon)], config)
            reached.append({
                "model": row["model"], "test_id": row["test_id"],
                "protocol_completion": row["protocol_completion"], "failure_mode": row["failure_mode"],
                "endpoints": endpoint,
            })
        structurally_evaluable = [
            row for row in reached if row["endpoints"]["session_evaluation_status"] == "EVALUABLE"
        ]
        completed = [row for row in structurally_evaluable if row["protocol_completion"]]
        failed = [row for row in structurally_evaluable if not row["protocol_completion"]]
        warmup_only = int(horizon) < int(contract["horizon_rules"]["minimum_inferential_horizon"])
        horizons.append({
            "horizon_windows": int(horizon), "approximate_token_boundary": int(horizon) * 16,
            "analysis_status": "DESCRIPTIVE_WARMUP_ONLY" if warmup_only else "STRUCTURALLY_EVALUABLE_FIXED_HORIZON",
            "inferential_use_allowed": False if warmup_only else True,
            "sessions_reaching_horizon": len(reached),
            "structurally_evaluable_session_count": len(structurally_evaluable),
            "completed_evaluable_n": len(completed), "failed_evaluable_n": len(failed),
            "completed_evaluable": aggregate(completed), "failed_evaluable": aggregate(failed),
            "sessions": reached,
        })

    correlation_fields = {
        "mean_recurrence_persistence": lambda row: row["endpoints"]["channels"]["mean_recurrence_persistence"],
        "fraction_crystallizing": lambda row: row["endpoints"]["state_occupancy_among_geometry_ready"]["crystallizing"]["fraction"],
        "fraction_transport_disrupted": lambda row: row["endpoints"]["state_occupancy_among_geometry_ready"]["transport_disrupted"]["fraction"],
        "fraction_viable_confirmed": lambda row: row["endpoints"]["viability_occupancy_among_geometry_ready"]["confirmed"]["fraction"],
    }
    correlations = {}
    for name, getter in correlation_fields.items():
        pairs = [(float(row["token_count"]), getter(row)) for row in rows if getter(row) is not None]
        correlations[name] = {
            "n": len(pairs),
            "spearman_rho_with_token_count": spearman([a for a, _ in pairs], [float(b) for _, b in pairs]),
        }

    report_rows = [{key: value for key, value in row.items() if key != "windows"} for row in rows]
    report = {
        "schema": "LLM-SVM-frontend-structural-analysis/8", "generated_at": utc_now(),
        "status": "EXPLORATORY_OFFLINE_REPROJECTION",
        "source_v6_report_sha256": file_sha256(v6_report_path),
        "v8_contract_sha256": file_sha256(args.contract),
        "projection_boundary": "saved_numeric_v6_channels_only",
        "external_evaluation_joined_after_projection": True,
        "numeric_channels_recomputed": False, "kernel_modified": False,
        "items": report_rows, "fixed_horizons": horizons, "length_correlations": correlations,
    }
    report_path = args.battery_dir / "structural_analysis_v8_report.json"
    atomic_json(report_path, report)
    print(json.dumps({"output": str(report_path), "items": len(rows), "horizons": len(horizons)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
