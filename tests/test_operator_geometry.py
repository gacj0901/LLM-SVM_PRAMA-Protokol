import json
from hashlib import sha256
from pathlib import Path

from aptadynamic_llm.operator_geometry import (
    OperatorGeometryConfig,
    observe_operator_geometry,
)


CONTRACT = Path("config/cocc_operator_geometry_observer_v1.json")
FREEZE = Path("config/cocc_operator_geometry_observer_v1.freeze.json")


def token(entropy=0.2, gap=1.0):
    return {"entropy": entropy, "gap": gap, "top_logprobs": [-0.1, -1.1]}


def point(index, delta, xi):
    return {"window_index": index, "delta": delta, "xi": xi, "theta": 0.2}


def config():
    return OperatorGeometryConfig(
        window_size_tokens=1,
        geometry_window=8,
        minimum_geometry_points=4,
        recurrence_lag_exclusion=1,
        recurrence_radius=0.05,
        activity_path_length_threshold=0.1,
        recirculation_efficiency_threshold=0.3,
        recirculation_rate_threshold=0.2,
        recirculation_dwell_threshold=2,
    )


def test_contract_is_frozen_and_model_independent():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    assert contract["model_specific_parameters"] is False
    assert contract["requires_external_calibration"] is False
    assert contract["kernel_boundary"]["diagnostic_only"] is True
    assert freeze["observer_contract_sha256"] == sha256(CONTRACT.read_bytes()).hexdigest()


def test_closed_orbit_has_lower_efficiency_and_more_recurrence_than_transport():
    tokens = [token()] * 8
    straight = [point(i, i / 10, i / 100) for i in range(8)]
    orbit = [point(i, 0.1 if i % 2 else 0.5, 0.02 if i % 2 else 0.08) for i in range(8)]
    straight_result = observe_operator_geometry(tokens, straight, config())[-1]
    orbit_result = observe_operator_geometry(tokens, orbit, config())[-1]
    assert orbit_result["transport_efficiency"] < straight_result["transport_efficiency"]
    assert orbit_result["recurrence_rate"] > straight_result["recurrence_rate"]
    assert orbit_result["structural_condition"] == "CRYSTALLIZED_RECIRCULATION"


def test_observer_is_causal():
    prefix_tokens = [token(entropy=0.1 + i / 100) for i in range(6)]
    prefix_points = [point(i, (i % 3) / 10, (i % 2) / 100) for i in range(6)]
    prefix = observe_operator_geometry(prefix_tokens, prefix_points, config())
    extended = observe_operator_geometry(
        prefix_tokens + [token(entropy=0.9), token(entropy=0.01)],
        prefix_points + [point(6, 0.9, 0.19), point(7, 0.0, 0.0)],
        config(),
    )
    assert prefix == extended[: len(prefix)]


def test_kernel_coordinates_are_not_returned_as_modified_values():
    result = observe_operator_geometry(
        [token()] * 4,
        [point(i, 0.1, 0.02) for i in range(4)],
        config(),
    )
    assert all("accumulated_excess" not in row and "capacity" not in row for row in result)
