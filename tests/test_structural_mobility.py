import json
from hashlib import sha256
from pathlib import Path

from aptadynamic_llm.operator_geometry import OperatorGeometryConfig
from aptadynamic_llm.structural_mobility import (
    StructuralMobilityConfig,
    observe_structural_mobility,
)


CONTRACT = Path("config/cocc_structural_mobility_observer_v2.json")
FREEZE = Path("config/cocc_structural_mobility_observer_v2.freeze.json")


def token():
    return {"entropy": 0.2, "gap": 1.0, "top_logprobs": [-0.1, -1.1]}


def point(index, delta, xi):
    return {"window_index": index, "delta": delta, "xi": xi, "theta": 0.2}


def geometry_config():
    return OperatorGeometryConfig(
        window_size_tokens=1,
        geometry_window=8,
        minimum_geometry_points=4,
        recurrence_lag_exclusion=1,
        recurrence_radius=0.05,
        activity_path_length_threshold=0.1,
    )


def mobility_config():
    return StructuralMobilityConfig(
        persistence_window=4,
        activity_path_length_threshold=0.1,
        transport_efficiency_threshold=0.3,
        recurrence_persistence_threshold=0.1,
        recurrent_dwell_fraction_threshold=0.2,
        early_crystallization_window=8,
    )


def test_contract_is_frozen_and_keeps_kernel_separate():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    assert contract["kernel_boundary"]["diagnostic_only"] is True
    assert contract["model_specific_parameters"] is False
    assert freeze["observer_contract_sha256"] == sha256(CONTRACT.read_bytes()).hexdigest()


def test_closed_orbit_has_more_persistent_recurrence_than_open_transport():
    tokens = [token()] * 8
    straight = [point(i, i / 10, i / 100) for i in range(8)]
    orbit = [point(i, 0.1 if i % 2 else 0.5, 0.02 if i % 2 else 0.08) for i in range(8)]
    open_result = observe_structural_mobility(
        tokens, straight, geometry_config(), mobility_config()
    )[-1]
    orbit_result = observe_structural_mobility(
        tokens, orbit, geometry_config(), mobility_config()
    )[-1]
    assert orbit_result["recurrence_persistence"] > open_result["recurrence_persistence"]
    assert orbit_result["transport_efficiency"] < open_result["transport_efficiency"]
    assert orbit_result["recirculating"] is True


def test_channels_are_causal_and_dwell_fraction_is_bounded():
    prefix_tokens = [token()] * 6
    prefix_points = [point(i, 0.1 if i % 2 else 0.5, 0.02) for i in range(6)]
    prefix = observe_structural_mobility(
        prefix_tokens, prefix_points, geometry_config(), mobility_config()
    )
    extended = observe_structural_mobility(
        prefix_tokens + [token(), token()],
        prefix_points + [point(6, 0.9, 0.19), point(7, 0.0, 0.0)],
        geometry_config(),
        mobility_config(),
    )
    assert prefix == extended[: len(prefix)]
    assert all(0.0 <= row["recurrent_dwell_fraction"] <= 1.0 for row in extended)


def test_no_kernel_debt_or_capacity_output_is_created():
    result = observe_structural_mobility(
        [token()] * 4,
        [point(i, 0.1, 0.02) for i in range(4)],
        geometry_config(),
        mobility_config(),
    )
    assert all("accumulated_excess" not in row and "capacity" not in row for row in result)
