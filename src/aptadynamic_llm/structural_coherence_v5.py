"""Causal structural coherence, recurrence, and variation-capacity observer."""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from typing import Any, Mapping, Sequence

from .operator_geometry import OperatorGeometryConfig, _distance, observe_operator_geometry
from .structural_mobility_v4 import _causal_recurrence_intensity


STATES = (
    "VIABLE",
    "STAGNANT",
    "IMITATIVE_ECHO",
    "RECURRENT",
    "CRYSTALLIZING",
    "CRYSTALLIZED",
    "COHERENCE_LOSS",
    "UNRESOLVED",
)


@dataclass(frozen=True)
class StructuralCoherenceV5Config:
    persistence_window_tau: float = 1.0
    recurrence_relative_radius: float = 0.5
    recurrence_threshold: float = 0.3
    coherence_threshold: float = 0.5
    coherence_memory_tau: float = 1.0
    variation_reference_tau: float = 2.0
    variation_contraction_threshold: float = 0.25
    tau_windows: int = 16
    crystallizing_duration_tau: float = 1.0
    scale_epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if self.tau_windows <= 0:
            raise ValueError("tau_windows must be positive")
        for name in (
            "persistence_window_tau",
            "recurrence_relative_radius",
            "recurrence_threshold",
            "coherence_threshold",
            "coherence_memory_tau",
            "variation_reference_tau",
            "variation_contraction_threshold",
            "crystallizing_duration_tau",
            "scale_epsilon",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("recurrence_threshold", "coherence_threshold", "variation_contraction_threshold"):
            if float(getattr(self, name)) >= 1.0:
                raise ValueError(f"{name} must be below one")

    @property
    def persistence_window(self) -> int:
        return max(1, math.ceil(self.persistence_window_tau * self.tau_windows))

    @property
    def variation_reference_window(self) -> int:
        return max(1, math.ceil(self.variation_reference_tau * self.tau_windows))

    @property
    def crystallizing_duration_windows(self) -> int:
        return max(1, math.ceil(self.crystallizing_duration_tau * self.tau_windows))

    @property
    def coherence_alpha(self) -> float:
        return math.exp(-1.0 / (self.coherence_memory_tau * self.tau_windows))

    @classmethod
    def from_contract(cls, contract: Mapping[str, Any]) -> "StructuralCoherenceV5Config":
        if contract.get("schema") != "LLM-SVM-CoCC-structural-coherence-contract/5":
            raise ValueError("unexpected structural-coherence v5 contract schema")
        if contract.get("model_specific_parameters") is not False:
            raise ValueError("structural coherence must forbid model-specific parameters")
        if contract.get("requires_external_calibration") is not False:
            raise ValueError("structural coherence must not require calibration")
        channels = contract["channels"]
        state = contract["state_machine"]
        return cls(
            persistence_window_tau=float(channels["inertial_echo_recurrence"]["trailing_window_tau"]),
            recurrence_relative_radius=float(channels["inertial_echo_recurrence"]["relative_radius"]),
            recurrence_threshold=float(state["minimum_recurrence_persistence"]),
            coherence_threshold=float(state["minimum_transport_coherence"]),
            coherence_memory_tau=float(channels["transport_coherence"]["memory_tau"]),
            variation_reference_tau=float(channels["variation_capacity"]["reference_window_tau"]),
            variation_contraction_threshold=float(state["minimum_variation_contraction"]),
            tau_windows=int(state["tau_windows"]),
            crystallizing_duration_tau=float(state["crystallizing_duration_tau"]),
            scale_epsilon=float(channels["inertial_echo_recurrence"]["scale_epsilon"]),
        )


def _vector(left: Sequence[float], right: Sequence[float]) -> list[float]:
    return [float(b) - float(a) for a, b in zip(left, right, strict=True)]


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in vector) / len(vector)) if vector else 0.0


def _operator_similarity(left: Sequence[float], right: Sequence[float], scale: float, epsilon: float) -> float:
    left_norm = _norm(left)
    right_norm = _norm(right)
    if left_norm <= epsilon or right_norm <= epsilon:
        return 0.0
    distance = _norm([a - b for a, b in zip(left, right, strict=True)])
    directional = math.exp(-distance / max(scale, epsilon))
    magnitude = min(left_norm, right_norm) / max(left_norm, right_norm)
    return min(1.0, max(0.0, directional * magnitude))


def _transported_inherence_evidence(
    states: Sequence[Sequence[float]],
    index: int,
    start: int,
    lag_exclusion: int,
    relative_radius: float,
    state_scale: float,
    epsilon: float,
) -> tuple[float | None, str, int]:
    """Estimate whether equivalent source states transport equivalent transitions."""

    if index < 2:
        return None, "INSUFFICIENT_TRANSITIONS", 0
    current = _vector(states[index - 1], states[index])
    if _norm(current) <= epsilon:
        return None, "NO_CURRENT_MOVEMENT", 0
    steps = [_vector(states[i - 1], states[i]) for i in range(max(start + 1, 1), index + 1)]
    positive_norms = [_norm(step) for step in steps if _norm(step) > epsilon]
    velocity_scale = statistics.median(positive_norms) if positive_norms else epsilon
    radius = max(epsilon, relative_radius * max(state_scale, epsilon))
    weighted_scores: list[tuple[float, float]] = []
    current_source = states[index - 1]
    latest_prior_source = index - lag_exclusion - 2
    for prior in range(start, latest_prior_source + 1):
        if prior + 1 >= index:
            continue
        source_distance = _distance(current_source, states[prior])
        if source_distance >= radius:
            continue
        historical = _vector(states[prior], states[prior + 1])
        if _norm(historical) <= epsilon:
            continue
        weight = max(0.0, 1.0 - source_distance / radius)
        weighted_scores.append((weight, _operator_similarity(current, historical, velocity_scale, epsilon)))
    if weighted_scores:
        weight_sum = sum(weight for weight, _ in weighted_scores)
        return (
            sum(weight * score for weight, score in weighted_scores) / max(weight_sum, epsilon),
            "RECURRENT_OPERATOR",
            len(weighted_scores),
        )
    previous = _vector(states[index - 2], states[index - 1])
    if _norm(previous) <= epsilon:
        return None, "NO_COMPARABLE_TRANSITION", 0
    return _operator_similarity(current, previous, velocity_scale, epsilon), "LOCAL_CONTINUITY", 1


def _variation_capacity(states: Sequence[Sequence[float]], start: int, index: int, epsilon: float) -> float | None:
    """Causal effective dimensionality of realized structural updates."""

    import numpy as np

    steps = [
        _vector(states[i - 1], states[i])
        for i in range(max(start + 1, 1), index + 1)
    ]
    active = [step for step in steps if _norm(step) > epsilon]
    if len(active) < 3:
        return None
    matrix = np.asarray(active, dtype=float)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = singular**2
    total = float(energy.sum())
    if total <= epsilon:
        return 0.0
    probabilities = energy[energy > epsilon] / total
    effective_rank = math.exp(-sum(float(p) * math.log(float(p)) for p in probabilities))
    dimension = matrix.shape[1]
    return min(1.0, max(0.0, (effective_rank - 1.0) / max(1, dimension - 1)))


def observe_structural_coherence_v5(
    tokens: Sequence[Mapping[str, Any]],
    trajectory: Sequence[Mapping[str, Any]],
    geometry_config: OperatorGeometryConfig,
    config: StructuralCoherenceV5Config,
) -> list[dict[str, Any]]:
    """Observe coherence as remembered, causally effective transition invariance."""

    base = observe_operator_geometry(tokens, trajectory, geometry_config)
    states = [row["state_vector"] for row in base]
    intensities: list[float] = []
    capacities: list[float | None] = []
    output: list[dict[str, Any]] = []
    coherence: float | None = None
    coherence_support_count = 0
    crystallizing_run = 0
    crystallization_onset: int | None = None

    for index, record in enumerate(base):
        start = max(0, index - geometry_config.geometry_window + 1)
        intensity, state_scale = _causal_recurrence_intensity(
            states[start : index + 1],
            geometry_config.recurrence_lag_exclusion,
            config.recurrence_relative_radius,
            config.scale_epsilon,
        )
        intensities.append(intensity)
        persistence_start = max(0, len(intensities) - config.persistence_window)
        inertial_echo = statistics.fmean(intensities[persistence_start:])
        ready = bool(record["geometry_ready"])
        movement = float(record["path_length"])
        active = ready and movement > geometry_config.activity_path_length_threshold
        recurrence_persistence = inertial_echo if active else 0.0
        evidence, evidence_source, support = _transported_inherence_evidence(
            states,
            index,
            start,
            geometry_config.recurrence_lag_exclusion,
            config.recurrence_relative_radius,
            state_scale,
            config.scale_epsilon,
        )
        if active and evidence is not None:
            coherence_support_count += support
            coherence = evidence if coherence is None else (
                config.coherence_alpha * coherence + (1.0 - config.coherence_alpha) * evidence
            )

        capacity = _variation_capacity(states, start, index, config.scale_epsilon) if ready else None
        capacities.append(capacity)
        reference_start = max(0, len(capacities) - config.variation_reference_window)
        reference_values = [value for value in capacities[reference_start:] if value is not None]
        reference_peak = max(reference_values) if reference_values else None
        variation_contraction = (
            max(0.0, 1.0 - capacity / reference_peak)
            if capacity is not None and reference_peak is not None and reference_peak > config.scale_epsilon
            else 0.0 if capacity is not None else None
        )

        coherent = coherence is not None and coherence > config.coherence_threshold
        echoing = inertial_echo > config.recurrence_threshold
        recurrent = recurrence_persistence > config.recurrence_threshold
        contracting = variation_contraction is not None and variation_contraction > config.variation_contraction_threshold
        crystallizing = bool(active and coherent and recurrent and contracting)
        crystallizing_run = crystallizing_run + 1 if crystallizing else 0
        crystallized = crystallizing and crystallizing_run >= config.crystallizing_duration_windows

        if not ready:
            state: str | None = None
        elif not active:
            state = "STAGNANT"
        elif coherence is None:
            state = "UNRESOLVED"
        elif not coherent and echoing:
            state = "IMITATIVE_ECHO"
        elif not coherent:
            state = "COHERENCE_LOSS"
        elif crystallized:
            state = "CRYSTALLIZED"
        elif crystallizing:
            state = "CRYSTALLIZING"
        elif recurrent:
            state = "RECURRENT"
        else:
            state = "VIABLE"
        if crystallized and crystallization_onset is None:
            crystallization_onset = int(record["window_index"])

        output.append({
            **record,
            "movement": movement,
            "net_transport_ratio": float(record["transport_efficiency"]) if active else None,
            "inertial_echo_recurrence": inertial_echo,
            "recurrence_persistence": recurrence_persistence,
            "transported_inherence_evidence": evidence if active else None,
            "transported_inherence_evidence_source": evidence_source if active else None,
            "transported_inherence_support": support if active else 0,
            "transport_coherence": coherence,
            "transport_coherence_support_count": coherence_support_count,
            "variation_capacity": capacity,
            "variation_reference_peak": reference_peak,
            "variation_contraction": variation_contraction,
            "crystallizing_duration_windows": crystallizing_run,
            "crystallization_onset": crystallization_onset,
            "structural_coherence_state": state,
        })
    return output
