#!/usr/bin/env python3
"""V9 offline reprojection with layered transport status and mobility regimes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for candidate in (REPO_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from aptadynamic_llm.structural_coherence_v9 import (  # noqa: E402
    DIAGNOSTICS_V9,
    MOBILITY_REGIMES_V9,
    SUMMARY_CLASSES_V9,
    TRANSPORT_STATUSES_V9,
    StructuralCoherenceV9Config,
    classify_structural_coherence_v9,
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


def _count_record(count: int, denominator: int) -> dict[str, Any]:
    return fraction_record(int(count), int(denominator))


def _runs(
    windows: Sequence[Mapping[str, Any]], predicate: Callable[[Mapping[str, Any]], bool]
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    start: int | None = None
    last: int | None = None
    for offset, row in enumerate(windows):
        index = int(row.get("absolute_window_index", offset))
        if predicate(row):
            if start is None:
                start = index
            last = index
        elif start is not None:
            runs.append({"start_window": start, "end_window": last, "length_windows": last - start + 1})
            start = last = None
    if start is not None:
        runs.append({"start_window": start, "end_window": last, "length_windows": last - start + 1})
    return runs


def _episode_summary(
    windows: Sequence[Mapping[str, Any]], predicate: Callable[[Mapping[str, Any]], bool]
) -> dict[str, Any]:
    runs = _runs(windows, predicate)
    lengths = [int(run["length_windows"]) for run in runs]
    terminal = lengths[-1] if runs and int(runs[-1]["end_window"]) == int(
        windows[-1].get("absolute_window_index", len(windows) - 1)
    ) else 0
    return {
        "episode_count": len(runs),
        "maximum_consecutive_windows": max(lengths) if lengths else 0,
        "median_episode_length_windows": statistics.median(lengths) if lengths else None,
        "terminal_dwell_windows": terminal,
        "episodes": runs,
    }


def _transport_recovery(
    windows: Sequence[Mapping[str, Any]], *, eligible_only: bool = True
) -> dict[str, Any]:
    selected = [row for row in windows if not eligible_only or row.get("alert_eligible_v9")]
    runs = _runs(selected, lambda row: row.get("transport_status_v9") == "DISRUPTED")
    if not selected:
        return {
            "episode_count": 0, "recovered_episode_count": 0,
            "interrupted_without_coherent_recovery_count": 0,
            "right_censored_episode_count": 0,
            "observed_recovery_rate": None,
        }
    by_index = {
        int(row.get("absolute_window_index", offset)): row for offset, row in enumerate(selected)
    }
    final_index = max(by_index)
    recovered = interrupted = right_censored = 0
    for run in runs:
        next_row = by_index.get(int(run["end_window"]) + 1)
        if next_row is None:
            if int(run["end_window"]) == final_index:
                right_censored += 1
            else:
                interrupted += 1
        elif next_row.get("transport_status_v9") == "COHERENT":
            recovered += 1
        else:
            interrupted += 1
    evaluable = recovered + interrupted
    return {
        "episode_count": len(runs),
        "recovered_episode_count": recovered,
        "interrupted_without_coherent_recovery_count": interrupted,
        "right_censored_episode_count": right_censored,
        "observed_recovery_rate": recovered / evaluable if evaluable else None,
    }


def session_endpoints(
    windows: Sequence[Mapping[str, Any]], config: StructuralCoherenceV9Config
) -> dict[str, Any]:
    total = len(windows)
    ready = [row for row in windows if row.get("geometry_ready")]
    active = [
        row for row in ready
        if float(row.get("movement") or 0.0) > config.activity_path_length_threshold
    ]
    transport_evaluable = [row for row in active if row.get("transport_coherence") is not None]
    alert = [row for row in windows if row.get("alert_eligible_v9")]
    alert_evaluable = [row for row in alert if row.get("transport_deficit_v9") is not None]
    mobility = [row for row in ready if row.get("mobility_regime_v9") is not None]
    ready_n = len(ready)
    active_n = len(active)
    transport_n = len(transport_evaluable)
    mobility_n = len(mobility)

    transport_counts = {
        status: sum(row.get("transport_status_v9") == status for row in ready)
        for status in TRANSPORT_STATUSES_V9
    }
    summary_counts = {
        state: sum(row.get("summary_class_v9") == state for row in ready)
        for state in SUMMARY_CLASSES_V9
    }
    mobility_counts = {
        state: sum(row.get("mobility_regime_v9") == state for row in mobility)
        for state in MOBILITY_REGIMES_V9
    }
    if sum(transport_counts.values()) != ready_n:
        raise RuntimeError("v9 transport-status exhaustiveness invariant failed")
    if sum(summary_counts.values()) != ready_n:
        raise RuntimeError("v9 summary-class exhaustiveness invariant failed")
    if sum(mobility_counts.values()) != mobility_n:
        raise RuntimeError("v9 mobility-regime exhaustiveness invariant failed")

    diagnostic_counts = {
        diagnostic: sum(diagnostic in (row.get("diagnostics_v9") or []) for row in ready)
        for diagnostic in DIAGNOSTICS_V9 if diagnostic != "INSUFFICIENT_GEOMETRY"
    }
    local = diagnostic_counts["LOCAL_TRANSPORT_DISCONTINUITY"]
    inherited = diagnostic_counts["HYSTERESIS_INHERITANCE"]
    disrupted_diag = diagnostic_counts["TRANSPORT_DISRUPTION"]
    partition_difference = local - inherited - disrupted_diag
    if partition_difference != 0:
        raise RuntimeError("v9 local-discontinuity diagnostic partition failed")

    alert_disrupted = sum(row.get("transport_status_v9") == "DISRUPTED" for row in alert)
    deficits = [float(row["transport_deficit_v9"]) for row in alert_evaluable]
    disrupted_episodes = _episode_summary(
        alert, lambda row: row.get("transport_status_v9") == "DISRUPTED"
    )
    crystallizing_episodes = _episode_summary(
        mobility,
        lambda row: row.get("mobility_regime_v9") in {"CRYSTALLIZING", "CRYSTALLIZED"},
    )
    status = (
        "EVALUABLE" if transport_n >= config.minimum_session_transport_windows
        else "SESSION_NOT_EVALUABLE_SHORT_TRAJECTORY"
    )
    return {
        "session_evaluation_status": status,
        "coverage": {
            "total_windows": {"count": total, "denominator": total, "fraction": 1.0 if total else None},
            "geometry_ready": _count_record(ready_n, total),
            "active_geometry": _count_record(active_n, ready_n),
            "transport_evaluable": _count_record(transport_n, total),
            "mobility_regime_assigned": _count_record(mobility_n, ready_n),
            "alert_eligible": _count_record(len(alert), total),
            "alert_transport_evaluable": _count_record(len(alert_evaluable), len(alert)),
            "insufficient_geometry": _count_record(total - ready_n, total),
        },
        "transport_status_occupancy_among_geometry_ready": {
            status_name.lower(): _count_record(count, ready_n)
            for status_name, count in transport_counts.items()
        },
        "mobility_regime_occupancy_among_assigned_windows": {
            state.lower(): _count_record(count, mobility_n)
            for state, count in mobility_counts.items()
        },
        "summary_class_occupancy_among_geometry_ready": {
            state.lower(): _count_record(count, ready_n)
            for state, count in summary_counts.items()
        },
        "diagnostic_prevalence_among_geometry_ready": {
            diagnostic.lower(): _count_record(count, ready_n)
            for diagnostic, count in diagnostic_counts.items()
        },
        "diagnostic_partition_audit": {
            "local_transport_discontinuity_count": local,
            "hysteresis_inheritance_count": inherited,
            "transport_disruption_count": disrupted_diag,
            "difference": partition_difference,
            "exact": partition_difference == 0,
        },
        "channels": {
            "terminal_transport_coherence": endpoint_value(transport_evaluable, "transport_coherence", "terminal"),
            "mean_transport_coherence": endpoint_value(transport_evaluable, "transport_coherence"),
            "mean_recurrence_persistence": endpoint_value(ready, "recurrence_persistence"),
            "maximum_variation_contraction": endpoint_value(ready, "variation_contraction", "max"),
            "mean_inertial_echo_recurrence": endpoint_value(ready, "inertial_echo_recurrence"),
            "mean_variation_capacity": endpoint_value(ready, "variation_capacity"),
        },
        "alert_endpoints_post_stabilization": {
            "eligible_window_count": len(alert),
            "transport_evaluable_window_count": len(alert_evaluable),
            "transport_disrupted_fraction": _count_record(alert_disrupted, len(alert)),
            "transport_deficit_area": sum(deficits) if deficits else None,
            "mean_transport_deficit": mean_or_none(deficits),
            "maximum_transport_disrupted_dwell": disrupted_episodes["maximum_consecutive_windows"],
            "number_of_transport_disrupted_episodes": disrupted_episodes["episode_count"],
            "median_transport_disrupted_episode_length": disrupted_episodes["median_episode_length_windows"],
            "terminal_transport_disrupted_dwell": disrupted_episodes["terminal_dwell_windows"],
            "transport_recovery": _transport_recovery(windows),
        },
        "episodes": {
            "crystallizing": crystallizing_episodes,
            "transport_disrupted_post_stabilization": disrupted_episodes,
        },
        "onsets_diagnostic_only": {
            "first_transport_disrupted_window": next((
                int(row["absolute_window_index"]) for row in windows
                if row.get("transport_status_v9") == "DISRUPTED"
            ), None),
            "first_alert_eligible_transport_disrupted_window": next((
                int(row["absolute_window_index"]) for row in windows
                if row.get("alert_eligible_v9") and row.get("transport_status_v9") == "DISRUPTED"
            ), None),
            "first_recurrent_window": next((
                int(row["absolute_window_index"]) for row in windows
                if row.get("mobility_regime_v9") == "RECURRENT"
            ), None),
            "first_crystallizing_window": next((
                int(row["absolute_window_index"]) for row in windows
                if row.get("mobility_regime_v9") == "CRYSTALLIZING"
            ), None),
        },
    }


def aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = {
        "terminal_transport_coherence": lambda row: row["endpoints"]["channels"]["terminal_transport_coherence"],
        "mean_recurrence_persistence": lambda row: row["endpoints"]["channels"]["mean_recurrence_persistence"],
        "fraction_transport_coherent": lambda row: row["endpoints"]["transport_status_occupancy_among_geometry_ready"]["coherent"]["fraction"],
        "fraction_transport_provisional": lambda row: row["endpoints"]["transport_status_occupancy_among_geometry_ready"]["provisional"]["fraction"],
        "fraction_transport_disrupted": lambda row: row["endpoints"]["transport_status_occupancy_among_geometry_ready"]["disrupted"]["fraction"],
        "fraction_recurrent_among_assigned": lambda row: row["endpoints"]["mobility_regime_occupancy_among_assigned_windows"]["recurrent"]["fraction"],
        "fraction_crystallizing_among_assigned": lambda row: row["endpoints"]["mobility_regime_occupancy_among_assigned_windows"]["crystallizing"]["fraction"],
        "post_stabilization_disrupted_fraction": lambda row: row["endpoints"]["alert_endpoints_post_stabilization"]["transport_disrupted_fraction"]["fraction"],
        "post_stabilization_transport_deficit_area": lambda row: row["endpoints"]["alert_endpoints_post_stabilization"]["transport_deficit_area"],
        "post_stabilization_mean_transport_deficit": lambda row: row["endpoints"]["alert_endpoints_post_stabilization"]["mean_transport_deficit"],
        "maximum_transport_disrupted_dwell": lambda row: row["endpoints"]["alert_endpoints_post_stabilization"]["maximum_transport_disrupted_dwell"],
        "observed_transport_recovery_rate": lambda row: row["endpoints"]["alert_endpoints_post_stabilization"]["transport_recovery"]["observed_recovery_rate"],
    }
    output: dict[str, Any] = {"n": len(rows)}
    for name, getter in fields.items():
        values = [float(value) for row in rows if (value := getter(row)) is not None]
        output[name] = {"evaluable_n": len(values), "mean": mean_or_none(values)}
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--battery-dir", type=Path, default=Path("FrontEnd/results/blind_interaction_battery_v1"))
    parser.add_argument("--contract", type=Path, default=Path("config/sequor_structural_coherence_observer_v9.json"))
    args = parser.parse_args()
    contract = load_json(args.contract)
    config = StructuralCoherenceV9Config.from_contract(contract)
    v6_report_path = args.battery_dir / "structural_analysis_v6_report.json"
    v6_report = load_json(v6_report_path)
    originals = {(row["model"], row["test_id"]): row for row in v6_report["items"]}

    projection_dir = args.battery_dir / "structural_projection_v9"
    rows: list[dict[str, Any]] = []
    for path in sorted((args.battery_dir / "structural_projection_v6").glob("*.json")):
        source = load_json(path)
        windows = classify_structural_coherence_v9(source["structural_windows"], config)
        projection = {
            "schema": "LLM-SVM-frontend-structural-projection/9",
            "generated_at": utc_now(), "session_id": source["session_id"], "model": source["model"],
            "source_v6_projection_sha256": file_sha256(path),
            "v9_contract_sha256": file_sha256(args.contract),
            "contains_prompt_or_answer": False, "contains_external_evaluation": False,
            "numeric_channels_recomputed": False, "kernel_modified": False,
            "windows": windows,
        }
        target = projection_dir / path.name
        atomic_json(target, projection)
        test_id = path.stem.rsplit("--", 1)[-1]
        original = originals[(source["model"], test_id)]
        evaluation = original["external_evaluation"]
        rows.append({
            "model": source["model"], "test_id": test_id,
            "token_count": original["token_count"], "finish_reason": original["finish_reason"],
            "protocol_completion": evaluation["protocol_completion"],
            "failure_mode": evaluation["failure_mode"],
            "projection_path": str(target.resolve()), "projection_sha256": file_sha256(target),
            "endpoints": session_endpoints(windows, config), "windows": windows,
        })

    horizons: list[dict[str, Any]] = []
    for horizon in contract["fixed_causal_horizons_windows"]:
        reached = []
        for row in rows:
            if len(row["windows"]) < int(horizon):
                continue
            endpoints = session_endpoints(row["windows"][: int(horizon)], config)
            reached.append({
                "model": row["model"], "test_id": row["test_id"],
                "protocol_completion": row["protocol_completion"], "failure_mode": row["failure_mode"],
                "endpoints": endpoints,
            })
        evaluable = [row for row in reached if row["endpoints"]["session_evaluation_status"] == "EVALUABLE"]
        completed = [row for row in evaluable if row["protocol_completion"]]
        failed = [row for row in evaluable if not row["protocol_completion"]]
        warmup_only = int(horizon) < int(contract["horizon_rules"]["minimum_inferential_horizon"])
        horizons.append({
            "horizon_windows": int(horizon), "approximate_token_boundary": int(horizon) * 16,
            "analysis_status": "DESCRIPTIVE_WARMUP_ONLY" if warmup_only else "STRUCTURALLY_EVALUABLE_FIXED_HORIZON",
            "inferential_use_allowed": not warmup_only,
            "sessions_reaching_horizon": len(reached),
            "structurally_evaluable_session_count": len(evaluable),
            "completed_evaluable_n": len(completed), "failed_evaluable_n": len(failed),
            "completed_evaluable": aggregate(completed), "failed_evaluable": aggregate(failed),
            "sessions": reached,
        })

    correlation_fields = {
        "mean_recurrence_persistence": lambda row: row["endpoints"]["channels"]["mean_recurrence_persistence"],
        "post_stabilization_transport_disrupted_fraction": lambda row: row["endpoints"]["alert_endpoints_post_stabilization"]["transport_disrupted_fraction"]["fraction"],
        "post_stabilization_transport_deficit_area": lambda row: row["endpoints"]["alert_endpoints_post_stabilization"]["transport_deficit_area"],
        "post_stabilization_mean_transport_deficit": lambda row: row["endpoints"]["alert_endpoints_post_stabilization"]["mean_transport_deficit"],
        "fraction_crystallizing_among_assigned": lambda row: row["endpoints"]["mobility_regime_occupancy_among_assigned_windows"]["crystallizing"]["fraction"],
    }
    correlations: dict[str, Any] = {}
    for name, getter in correlation_fields.items():
        pairs = [(float(row["token_count"]), value) for row in rows if (value := getter(row)) is not None]
        correlations[name] = {
            "n": len(pairs),
            "spearman_rho_with_token_count": spearman(
                [a for a, _ in pairs], [float(b) for _, b in pairs]
            ),
        }

    report_rows = [{key: value for key, value in row.items() if key != "windows"} for row in rows]
    partition_difference = sum(
        row["endpoints"]["diagnostic_partition_audit"]["difference"] for row in rows
    )
    report = {
        "schema": "LLM-SVM-frontend-structural-analysis/9", "generated_at": utc_now(),
        "status": "EXPLORATORY_OFFLINE_REPROJECTION",
        "source_v6_report_sha256": file_sha256(v6_report_path),
        "v9_contract_sha256": file_sha256(args.contract),
        "projection_boundary": "saved_numeric_v6_channels_only",
        "external_evaluation_joined_after_projection": True,
        "numeric_channels_recomputed": False, "kernel_modified": False,
        "architecture": "layered_transport_status_then_coherent_mobility_regime",
        "global_invariants": {
            "diagnostic_partition_difference": partition_difference,
            "diagnostic_partition_exact": partition_difference == 0,
            "viability_by_residual_fallback": False,
            "first_disruption_onset_used_as_alert_endpoint": False,
        },
        "items": report_rows, "fixed_horizons": horizons,
        "length_correlations": correlations,
    }
    report_path = args.battery_dir / "structural_analysis_v9_report.json"
    atomic_json(report_path, report)
    print(json.dumps({
        "output": str(report_path), "items": len(rows), "horizons": len(horizons),
        "diagnostic_partition_exact": partition_difference == 0,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
