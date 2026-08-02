import json
from hashlib import sha256
from pathlib import Path

import pytest

from aptadynamic_llm.operator_geometry import OperatorGeometryConfig
from aptadynamic_llm.structural_mobility_v3 import (
    StructuralMobilityV3Config,
    _causal_recurrence_intensity,
    observe_structural_mobility_v3,
)


CONTRACT = Path("config/cocc_structural_mobility_observer_v3.json")
FREEZE = Path("config/cocc_structural_mobility_observer_v3.freeze.json")


def token():
    return {"entropy": 0.2, "gap": 1.0, "top_logprobs": [-0.1, -1.1]}


def point(index, delta, xi):
    return {"window_index": index, "delta": delta, "xi": xi, "theta": 0.2}


def geometry_config(dwell=4):
    return OperatorGeometryConfig(
        window_size_tokens=1,
        geometry_window=8,
        minimum_geometry_points=4,
        recurrence_lag_exclusion=1,
        recurrence_radius=0.05,
        activity_path_length_threshold=0.1,
        recirculation_dwell_threshold=dwell,
    )


def mobility_config(duration_tau=0.25):
    return StructuralMobilityV3Config(
        recurrence_persistence_window_tau=0.25,
        recurrence_relative_radius=0.5,
        recurrence_persistence_threshold=0.1,
        transport_efficiency_threshold=0.3,
        tau_windows=4,
        crystallizing_duration_tau=duration_tau,
    )


def test_contract_is_frozen_and_dwell_is_excluded_from_state_machine():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    assert freeze["observer_contract_sha256"] == sha256(CONTRACT.read_bytes()).hexdigest()
    assert contract["state_machine"]["uses_recurrent_dwell"] is False
    assert "recurrent_dwell_fraction" not in contract["channels"]
    assert contract["kernel_boundary"]["diagnostic_only"] is True


def test_relative_recurrence_is_invariant_to_uniform_scale():
    states = [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]
    scaled = [[value * 7.0 for value in state] for state in states]
    intensity, _ = _causal_recurrence_intensity(states, 1, 0.5, 1e-12)
    scaled_intensity, _ = _causal_recurrence_intensity(scaled, 1, 0.5, 1e-12)
    assert scaled_intensity == pytest.approx(intensity)


def test_crystallization_requires_consecutive_crystallizing_duration():
    tokens = [token()] * 10
    orbit = [point(i, 0.1 if i % 2 else 0.5, 0.02 if i % 2 else 0.08) for i in range(10)]
    result = observe_structural_mobility_v3(
        tokens, orbit, geometry_config(), mobility_config(duration_tau=0.5)
    )
    crystallized = [row for row in result if row["structural_mobility_state"] == "CRYSTALLIZED"]
    assert crystallized
    first = crystallized[0]
    assert first["crystallizing_duration_windows"] >= 2
    assert first["crystallization_onset"] == first["window_index"]


def test_old_dwell_threshold_cannot_change_v3_states():
    tokens = [token()] * 10
    orbit = [point(i, 0.1 if i % 2 else 0.5, 0.02 if i % 2 else 0.08) for i in range(10)]
    low = observe_structural_mobility_v3(tokens, orbit, geometry_config(1), mobility_config())
    high = observe_structural_mobility_v3(tokens, orbit, geometry_config(99), mobility_config())
    assert [row["structural_mobility_state"] for row in low] == [
        row["structural_mobility_state"] for row in high
    ]


def test_channels_are_causal_and_kernel_state_is_not_created():
    tokens = [token()] * 7
    points = [point(i, 0.1 if i % 2 else 0.5, 0.02) for i in range(7)]
    prefix = observe_structural_mobility_v3(tokens[:5], points[:5], geometry_config(), mobility_config())
    extended = observe_structural_mobility_v3(tokens, points, geometry_config(), mobility_config())
    assert prefix == extended[:5]
    assert all("accumulated_excess" not in row and "capacity" not in row for row in extended)
    assert all(0.0 <= row["crystallized_fraction"] <= 1.0 for row in extended)
