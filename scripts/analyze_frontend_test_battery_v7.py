#!/usr/bin/env python3
"""V7 offline state, horizon, denominator, and episode analysis for the frontend battery."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for candidate in (REPO_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from aptadynamic_llm.structural_coherence_v7 import (  # noqa: E402
    DIAGNOSTICS_V7,
    PRIMARY_STATES_V7,
    StructuralCoherenceV7Config,
    classify_structural_coherence_v7,
    crystallizing_episode_summary,
)
from scripts.project_cocc_operator_geometry import file_sha256, load_json  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def fraction_record(count: int, denominator: int) -> dict[str, Any]:
    return {"count": count, "denominator": denominator, "fraction": count / denominator if denominator else None}


def mean_or_none(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def endpoint_value(windows: Sequence[Mapping[str, Any]], name: str, operation: str = "mean") -> float | None:
    values = [float(row[name]) for row in windows if row.get(name) is not None]
    if not values:
        return None
    return values[-1] if operation == "terminal" else max(values) if operation == "max" else statistics.fmean(values)


def session_endpoints(
    windows: Sequence[Mapping[str, Any]], config: StructuralCoherenceV7Config
) -> dict[str, Any]:
    total = len(windows)
    ready = [row for row in windows if row.get("geometry_ready")]
    transport = [row for row in ready if row.get("transport_coherence") is not None]
    ready_n = len(ready)
    transport_n = len(transport)
    state_counts = {
        state: sum(row.get("primary_state_v7") == state for row in ready) for state in PRIMARY_STATES_V7
    }
    if sum(state_counts.values()) != ready_n:
        raise RuntimeError("v7 state exhaustiveness invariant failed")
    diagnostic_counts = {
        diagnostic: sum(diagnostic in (row.get("diagnostics_v7") or []) for row in ready)
        for diagnostic in DIAGNOSTICS_V7 if diagnostic != "INSUFFICIENT_GEOMETRY"
    }
    confirmed_viable = sum(bool(row.get("viability_confirmed")) for row in ready)
    viable_with_discontinuity = sum(
        row.get("primary_state_v7") == "VIABLE"
        and "LOCAL_TRANSPORT_DISCONTINUITY" in (row.get("diagnostics_v7") or [])
        for row in ready
    )
    session_status = (
        "EVALUABLE" if transport_n >= config.minimum_session_transport_windows
        else "SESSION_NOT_EVALUABLE_SHORT_TRAJECTORY"
    )
    episodes = crystallizing_episode_summary(ready)
    return {
        "session_evaluation_status": session_status,
        "coverage": {
            "total_windows": {"count": total, "denominator": total, "fraction": 1.0 if total else None},
            "geometry_ready": fraction_record(ready_n, total),
            "transport_evaluable": fraction_record(transport_n, total),
            "insufficient_geometry": fraction_record(total - ready_n, total),
        },
        "state_occupancy_among_geometry_ready": {
            state.lower(): fraction_record(count, ready_n) for state, count in state_counts.items()
        },
        "state_qualification": {
            "confirmed_viable_among_geometry_ready": fraction_record(confirmed_viable, ready_n),
            "viable_with_local_discontinuity_among_viable": fraction_record(
                viable_with_discontinuity, state_counts["VIABLE"]
            ),
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
        "episodes": episodes,
        "onsets": {
            "first_recurrent_window": next((int(row["absolute_window_index"]) for row in windows if row.get("primary_state_v7") == "RECURRENT"), None),
            "first_crystallizing_window": next((int(row["absolute_window_index"]) for row in windows if row.get("primary_state_v7") == "CRYSTALLIZING"), None),
            "first_crystallized_window": next((int(row["absolute_window_index"]) for row in windows if row.get("primary_state_v7") == "CRYSTALLIZED"), None),
        },
    }


def ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    output = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + 1 + end) / 2.0
        for index in order[cursor:end]:
            output[index] = rank
        cursor = end
    return output


def pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    lm, rm = statistics.fmean(left), statistics.fmean(right)
    numerator = sum((a - lm) * (b - rm) for a, b in zip(left, right, strict=True))
    ld = math.sqrt(sum((a - lm) ** 2 for a in left))
    rd = math.sqrt(sum((b - rm) ** 2 for b in right))
    return numerator / (ld * rd) if ld and rd else None


def spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    return pearson(ranks(left), ranks(right))


def aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = {
        "terminal_transport_coherence": lambda row: row["endpoints"]["channels"]["terminal_transport_coherence"],
        "mean_recurrence_persistence": lambda row: row["endpoints"]["channels"]["mean_recurrence_persistence"],
        "maximum_variation_contraction": lambda row: row["endpoints"]["channels"]["maximum_variation_contraction"],
        "fraction_viable": lambda row: row["endpoints"]["state_occupancy_among_geometry_ready"]["viable"]["fraction"],
        "fraction_confirmed_viable": lambda row: row["endpoints"]["state_qualification"]["confirmed_viable_among_geometry_ready"]["fraction"],
        "fraction_crystallizing": lambda row: row["endpoints"]["state_occupancy_among_geometry_ready"]["crystallizing"]["fraction"],
        "fraction_local_transport_discontinuity": lambda row: row["endpoints"]["diagnostic_prevalence_among_geometry_ready"]["local_transport_discontinuity"]["fraction"],
        "maximum_consecutive_crystallizing_windows": lambda row: row["endpoints"]["episodes"]["maximum_consecutive_crystallizing_windows"],
    }
    result: dict[str, Any] = {"n": len(rows)}
    for name, getter in fields.items():
        values = [float(value) for row in rows if (value := getter(row)) is not None]
        result[name] = {"evaluable_n": len(values), "mean": mean_or_none(values)}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--battery-dir", type=Path, default=Path("FrontEnd/results/blind_interaction_battery_v1"))
    parser.add_argument("--contract", type=Path, default=Path("config/sequor_structural_coherence_observer_v7.json"))
    args = parser.parse_args()
    contract = load_json(args.contract)
    config = StructuralCoherenceV7Config.from_contract(contract)
    v6_report_path = args.battery_dir / "structural_analysis_v6_report.json"
    v6_report = load_json(v6_report_path)
    evaluations = {
        (row["model"], row["test_id"]): row["external_evaluation"] for row in v6_report["items"]
    }

    projection_dir = args.battery_dir / "structural_projection_v7"
    rows = []
    for path in sorted((args.battery_dir / "structural_projection_v6").glob("*.json")):
        source = load_json(path)
        windows = classify_structural_coherence_v7(source["structural_windows"], config)
        projection = {
            "schema": "LLM-SVM-frontend-structural-projection/7",
            "generated_at": utc_now(),
            "session_id": source["session_id"], "model": source["model"],
            "source_v6_projection_sha256": file_sha256(path),
            "v7_contract_sha256": file_sha256(args.contract),
            "contains_prompt_or_answer": False, "contains_external_evaluation": False,
            "numeric_channels_recomputed": False, "kernel_modified": False,
            "windows": windows,
        }
        target = projection_dir / path.name
        atomic_json(target, projection)
        test_id = path.stem.rsplit("--", 1)[-1]
        evaluation = evaluations[(source["model"], test_id)]
        item_v6 = next(row for row in v6_report["items"] if row["model"] == source["model"] and row["test_id"] == test_id)
        rows.append({
            "model": source["model"], "test_id": test_id,
            "token_count": item_v6["token_count"], "finish_reason": item_v6["finish_reason"],
            "protocol_completion": evaluation["protocol_completion"], "failure_mode": evaluation["failure_mode"],
            "projection_path": str(target.resolve()), "projection_sha256": file_sha256(target),
            "endpoints": session_endpoints(windows, config),
            "windows": windows,
        })

    horizons = []
    for horizon in contract["fixed_causal_horizons_windows"]:
        horizon_rows = []
        for row in rows:
            if len(row["windows"]) < int(horizon):
                continue
            horizon_rows.append({
                "model": row["model"], "test_id": row["test_id"],
                "protocol_completion": row["protocol_completion"], "failure_mode": row["failure_mode"],
                "endpoints": session_endpoints(row["windows"][: int(horizon)], config),
            })
        completed = [row for row in horizon_rows if row["protocol_completion"]]
        failed = [row for row in horizon_rows if not row["protocol_completion"]]
        horizons.append({
            "horizon_windows": int(horizon), "approximate_token_boundary": int(horizon) * 16,
            "eligible_session_count": len(horizon_rows),
            "completed_n": len(completed), "failed_n": len(failed),
            "completed": aggregate(completed), "failed": aggregate(failed),
            "sessions": horizon_rows,
        })

    correlation_fields = {
        "mean_recurrence_persistence": lambda row: row["endpoints"]["channels"]["mean_recurrence_persistence"],
        "fraction_recurrent": lambda row: row["endpoints"]["state_occupancy_among_geometry_ready"]["recurrent"]["fraction"],
        "fraction_crystallizing": lambda row: row["endpoints"]["state_occupancy_among_geometry_ready"]["crystallizing"]["fraction"],
        "local_transport_discontinuity": lambda row: row["endpoints"]["diagnostic_prevalence_among_geometry_ready"]["local_transport_discontinuity"]["fraction"],
        "imitative_echo": lambda row: row["endpoints"]["diagnostic_prevalence_among_geometry_ready"]["imitative_echo"]["fraction"],
    }
    correlations = {}
    for name, getter in correlation_fields.items():
        pairs = [(float(row["token_count"]), getter(row)) for row in rows if getter(row) is not None]
        correlations[name] = {
            "n": len(pairs),
            "spearman_rho_with_token_count": spearman([a for a, _ in pairs], [float(b) for _, b in pairs]),
        }

    # Keep full window series in blind projection artifacts, not in the joined report.
    report_rows = [{key: value for key, value in row.items() if key != "windows"} for row in rows]
    report = {
        "schema": "LLM-SVM-frontend-structural-analysis/7", "generated_at": utc_now(),
        "status": "EXPLORATORY_OFFLINE_REPROJECTION",
        "source_v6_report_sha256": file_sha256(v6_report_path),
        "v7_contract_sha256": file_sha256(args.contract),
        "projection_boundary": "saved_numeric_v6_channels_only",
        "external_evaluation_joined_after_projection": True,
        "numeric_channels_recomputed": False, "kernel_modified": False,
        "items": report_rows, "fixed_horizons": horizons,
        "length_correlations": correlations,
    }
    report_path = args.battery_dir / "structural_analysis_v7_report.json"
    atomic_json(report_path, report)
    print(json.dumps({"output": str(report_path), "items": len(rows), "horizons": len(horizons)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
