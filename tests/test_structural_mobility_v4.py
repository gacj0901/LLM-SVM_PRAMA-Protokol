import json
from hashlib import sha256
from pathlib import Path

import pytest

from aptadynamic_llm.operator_geometry import OperatorGeometryConfig
from aptadynamic_llm.structural_mobility_v4 import (
    StructuralMobilityV4Config,
    _causal_recurrence_intensity,
    observe_structural_mobility_v4,
)


CONTRACT = Path("config/cocc_structural_mobility_observer_v4.json")
FREEZE = Path("config/cocc_structural_mobility_observer_v4.freeze.json")


def token():
    return {"entropy": 0.2, "gap": 1.0, "top_logprobs": [-0.1, -1.1]}


def point(index, delta, xi):
    return {"window_index": index, "delta": delta, "xi": xi, "theta": 0.2}


def geometry_config(activity=0.1):
    return OperatorGeometryConfig(
        window_size_tokens=1,
        geometry_window=8,
        minimum_geometry_points=4,
        recurrence_lag_exclusion=1,
        recurrence_radius=0.05,
        activity_path_length_threshold=activity,
        recirculation_dwell_threshold=4,
    )


def mobility_config(duration_tau=0.25):
    return StructuralMobilityV4Config(
        recurrence_persistence_window_tau=0.25,
        recurrence_relative_radius=0.5,
        recurrence_persistence_threshold=0.1,
        transport_efficiency_threshold=0.3,
        tau_windows=4,
        crystallizing_duration_tau=duration_tau,
    )


def test_contract_is_frozen_and_conditions_recurrence_on_movement():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    assert freeze["observer_contract_sha256"] == sha256(CONTRACT.read_bytes()).hexdigest()
    assert contract["channels"]["mobile_recurrence_persistence"]["definition"].startswith("activity_indicator")
    assert "STAGNANT" in contract["state_machine"]["states"]
    assert contract["kernel_boundary"]["diagnostic_only"] is True


def test_uniform_static_trajectory_is_stagnant_not_recurrent():
    tokens = [token()] * 10
    static = [point(i, 0.1, 0.02) for i in range(10)]
    result = observe_structural_mobility_v4(tokens, static, geometry_config(), mobility_config())
    ready = [row for row in result if row["geometry_ready"]]
    assert ready
    assert all(row["structural_mobility_state"] == "STAGNANT" for row in ready)
    assert all(row["mobile_recurrence_persistence"] == 0.0 for row in ready)
    assert all(row["transport_efficiency_active"] is None for row in ready)
    assert any(row["recurrence_persistence_raw"] > 0.0 for row in ready)


def test_active_closed_orbit_can_crystallize():
    tokens = [token()] * 12
    orbit = [point(i, 0.1 if i % 2 else 0.5, 0.02 if i % 2 else 0.08) for i in range(12)]
    result = observe_structural_mobility_v4(tokens, orbit, geometry_config(), mobility_config(duration_tau=0.5))
    crystallized = [row for row in result if row["structural_mobility_state"] == "CRYSTALLIZED"]
    assert crystallized
    assert all(row["transport_efficiency_valid"] for row in crystallized)
    assert all(row["mobile_recurrence_persistence"] > 0.0 for row in crystallized)


def test_channels_are_causal_and_pre_ready_state_is_null():
    tokens = [token()] * 8
    points = [point(i, 0.1 if i % 2 else 0.5, 0.02) for i in range(8)]
    prefix = observe_structural_mobility_v4(tokens[:6], points[:6], geometry_config(), mobility_config())
    extended = observe_structural_mobility_v4(tokens, points, geometry_config(), mobility_config())
    assert prefix == extended[:6]
    assert all(row["structural_mobility_state"] is None for row in extended[:3])
    assert all("accumulated_excess" not in row and "capacity" not in row for row in extended)


def test_relative_recurrence_remains_scale_invariant():
    states = [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]
    scaled = [[value * 9.0 for value in state] for state in states]
    intensity, _ = _causal_recurrence_intensity(states, 1, 0.5, 1e-12)
    scaled_intensity, _ = _causal_recurrence_intensity(scaled, 1, 0.5, 1e-12)
    assert scaled_intensity == pytest.approx(intensity)
