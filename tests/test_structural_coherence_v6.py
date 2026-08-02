import json
from pathlib import Path

import pytest

from aptadynamic_llm.operator_geometry import OperatorGeometryConfig
from aptadynamic_llm.structural_coherence_v6 import (
    PRIMARY_STATES,
    StructuralCoherenceV6Config,
    _ridge_prediction,
    observe_structural_coherence_v6,
)


CONTRACT = Path("config/sequor_structural_coherence_observer_v6.json")


def token(entropy=0.2, gap=1.0):
    return {"entropy": entropy, "gap": gap, "top_logprobs": [-0.1, -1.1], "top1_logprob": -0.1}


def point(index, delta, xi):
    return {"turn_index": index // 2, "window_index": index % 2, "delta": delta, "xi": xi, "theta": 0.2}


def geometry(activity=0.05):
    return OperatorGeometryConfig(
        window_size_tokens=1, geometry_window=8, minimum_geometry_points=4,
        recurrence_lag_exclusion=1, recurrence_radius=0.08,
        activity_path_length_threshold=activity, recirculation_dwell_threshold=2,
    )


def config():
    return StructuralCoherenceV6Config(
        operator_window_tau=2, minimum_operator_transitions=4, ridge_alpha=1e-5,
        residual_scale_floor=0.05, recurrence_window_tau=1,
        recurrence_relative_radius=0.5, recurrence_threshold=0.1,
        coherence_threshold=0.3, variation_reference_tau=2,
        variation_contraction_threshold=0.2, tau_windows=4,
        crystallizing_duration_tau=1,
    )


def test_contract_loads_and_distinguishes_openness_from_coherence():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    parsed = StructuralCoherenceV6Config.from_contract(contract)
    assert parsed.minimum_operator_transitions == 8
    assert contract["channels"]["trajectory_openness"]["interpretation"].endswith("never coherence")


def test_ridge_operator_is_strictly_causal_and_recognizes_closed_orbit():
    orbit = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]] * 4
    coherent, residual, support = _ridge_prediction(orbit, 12, config())
    broken = orbit[:12] + [[0.7, 0.7]]
    incoherent, broken_residual, _ = _ridge_prediction(broken, 12, config())
    assert support >= 4
    assert coherent > 0.95
    assert residual < broken_residual
    assert incoherent < coherent


def test_static_echo_is_stagnant_and_does_not_invent_coherence():
    windows = [[token()]] * 16
    trajectory = [point(i, 0.1, 0.02) for i in range(16)]
    rows = observe_structural_coherence_v6(windows, trajectory, geometry(), config())
    ready = [row for row in rows if row["geometry_ready"]]
    assert ready
    assert all(row["primary_state"] == "STAGNANT" for row in ready)
    assert all(row["transport_coherence"] is None for row in ready)
    assert any(row["inertial_echo_recurrence"] > 0 for row in ready)


def test_prefix_is_invariant_and_classification_closes_explicitly():
    windows = [[token(0.1 + 0.02 * (i % 3), 0.5 + 0.1 * (i % 2))] for i in range(24)]
    trajectory = [point(i, 0.1 + 0.1 * (i % 3), 0.02 + 0.01 * (i % 4)) for i in range(24)]
    prefix = observe_structural_coherence_v6(windows[:18], trajectory[:18], geometry(), config())
    full = observe_structural_coherence_v6(windows, trajectory, geometry(), config())
    assert prefix == full[:18]
    for row in full:
        if row["primary_state"] is not None:
            assert row["primary_state"] in PRIMARY_STATES
            assert row["classification_status"] == "PRIMARY"
        else:
            assert row["classification_status"] in {"DIAGNOSTIC_ONLY", "INSUFFICIENT_GEOMETRY"}
