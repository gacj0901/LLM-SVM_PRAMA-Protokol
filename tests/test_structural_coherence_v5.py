import json
from hashlib import sha256
from pathlib import Path

import pytest

from aptadynamic_llm.operator_geometry import OperatorGeometryConfig
from aptadynamic_llm.structural_coherence_v5 import (
    StructuralCoherenceV5Config,
    _operator_similarity,
    observe_structural_coherence_v5,
)


CONTRACT = Path("config/cocc_structural_coherence_observer_v5.json")
FREEZE = Path("config/cocc_structural_coherence_observer_v5.freeze.json")


def token():
    return {"entropy": 0.2, "gap": 1.0, "top_logprobs": [-0.1, -1.1]}


def point(index, delta, xi):
    return {"window_index": index, "delta": delta, "xi": xi, "theta": 0.2}


def geometry(activity=0.1):
    return OperatorGeometryConfig(
        window_size_tokens=1,
        geometry_window=8,
        minimum_geometry_points=4,
        recurrence_lag_exclusion=1,
        recurrence_radius=0.05,
        activity_path_length_threshold=activity,
        recirculation_dwell_threshold=4,
    )


def coherence(duration_tau=0.5):
    return StructuralCoherenceV5Config(
        persistence_window_tau=0.25,
        recurrence_relative_radius=0.5,
        recurrence_threshold=0.1,
        coherence_threshold=0.3,
        coherence_memory_tau=1.0,
        variation_reference_tau=1.0,
        variation_contraction_threshold=0.2,
        tau_windows=4,
        crystallizing_duration_tau=duration_tau,
    )


def test_contract_is_frozen_and_coherence_is_not_net_transport():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    assert freeze["observer_contract_sha256"] == sha256(CONTRACT.read_bytes()).hexdigest()
    assert contract["channels"]["net_transport_ratio"]["interpretation"] == "path directness, not coherence"
    assert contract["channels"]["transport_coherence"]["recursion"].startswith("C_t=")


def test_static_echo_is_stagnant_and_does_not_create_coherence():
    tokens = [token()] * 12
    points = [point(i, 0.1, 0.02) for i in range(12)]
    result = observe_structural_coherence_v5(tokens, points, geometry(), coherence())
    ready = [row for row in result if row["geometry_ready"]]
    assert ready
    assert all(row["structural_coherence_state"] == "STAGNANT" for row in ready)
    assert all(row["recurrence_persistence"] == 0.0 for row in ready)
    assert all(row["transport_coherence"] is None for row in ready)
    assert any(row["inertial_echo_recurrence"] > 0.0 for row in ready)


def test_equivalent_transitions_have_maximum_operator_similarity():
    assert _operator_similarity([1.0, -0.5], [1.0, -0.5], 1.0, 1e-12) == pytest.approx(1.0)
    assert _operator_similarity([1.0, 0.0], [-1.0, 0.0], 1.0, 1e-12) < 0.3


def test_coherent_closed_orbit_can_be_recurrent_despite_low_net_transport():
    tokens = [token()] * 20
    points = [point(i, 0.1 if i % 2 else 0.6, 0.02 if i % 2 else 0.12) for i in range(20)]
    result = observe_structural_coherence_v5(tokens, points, geometry(), coherence())
    recurrent = [row for row in result if row["structural_coherence_state"] == "RECURRENT"]
    assert recurrent
    assert max(float(row["transport_coherence"] or 0.0) for row in recurrent) > 0.3
    assert min(float(row["net_transport_ratio"]) for row in recurrent if row["net_transport_ratio"] is not None) < 0.3


def test_coherence_is_causal_and_kernel_is_untouched():
    tokens = [token()] * 12
    points = [point(i, 0.1 if i % 2 else 0.6, 0.02 if i % 2 else 0.12) for i in range(12)]
    prefix = observe_structural_coherence_v5(tokens[:9], points[:9], geometry(), coherence())
    extended = observe_structural_coherence_v5(tokens, points, geometry(), coherence())
    assert prefix == extended[:9]
    assert all("accumulated_excess" not in row and "capacity" not in row for row in extended)
