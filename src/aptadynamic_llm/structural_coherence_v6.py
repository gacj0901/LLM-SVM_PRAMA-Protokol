"""Prospective causal observer for transported structural inherence.

Version 6 deliberately separates path openness from coherence.  Coherence is
the out-of-sample predictive fidelity of a local linear transition operator
fitted using strictly earlier states.  Recurrence, realized variation, and
regime persistence remain independent coordinates.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import statistics
from typing import Any, Mapping, Sequence

import numpy as np

from .operator_geometry import (
    MODES,
    OperatorGeometryConfig,
    _distance,
    _mode_entropy,
    _window_state,
)
from .structural_mobility_v4 import _causal_recurrence_intensity


PRIMARY_STATES = ("VIABLE", "STAGNANT", "RECURRENT", "CRYSTALLIZING", "CRYSTALLIZED")
CLASSIFICATION_STATUSES = ("PRIMARY", "DIAGNOSTIC_ONLY", "INSUFFICIENT_GEOMETRY")


@dataclass(frozen=True)
class StructuralCoherenceV6Config:
    operator_window_tau: float = 2.0
    minimum_operator_transitions: int = 8
    ridge_alpha: float = 0.001
    residual_scale_floor: float = 0.05
    recurrence_window_tau: float = 1.0
    recurrence_relative_radius: float = 0.5
    recurrence_threshold: float = 0.3
    coherence_threshold: float = 0.5
    variation_reference_tau: float = 2.0
    variation_contraction_threshold: float = 0.25
    tau_windows: int = 16
    crystallizing_duration_tau: float = 1.0
    scale_epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if self.tau_windows <= 0 or self.minimum_operator_transitions < 2:
            raise ValueError("tau_windows and minimum_operator_transitions are invalid")
        for name in (
            "operator_window_tau", "ridge_alpha", "residual_scale_floor",
            "recurrence_window_tau", "recurrence_relative_radius",
            "recurrence_threshold", "coherence_threshold",
            "variation_reference_tau", "variation_contraction_threshold",
            "crystallizing_duration_tau", "scale_epsilon",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("recurrence_threshold", "coherence_threshold", "variation_contraction_threshold"):
            if float(getattr(self, name)) >= 1.0:
                raise ValueError(f"{name} must be below one")

    @property
    def operator_window(self) -> int:
        return max(self.minimum_operator_transitions, math.ceil(self.operator_window_tau * self.tau_windows))

    @property
    def recurrence_window(self) -> int:
        return max(1, math.ceil(self.recurrence_window_tau * self.tau_windows))

    @property
    def variation_reference_window(self) -> int:
        return max(1, math.ceil(self.variation_reference_tau * self.tau_windows))

    @property
    def crystallizing_duration_windows(self) -> int:
        return max(1, math.ceil(self.crystallizing_duration_tau * self.tau_windows))

    @classmethod
    def from_contract(cls, contract: Mapping[str, Any]) -> "StructuralCoherenceV6Config":
        if contract.get("schema") != "LLM-SVM-structural-coherence-contract/6":
            raise ValueError("unexpected structural-coherence v6 contract schema")
        if contract.get("model_specific_parameters") is not False:
            raise ValueError("v6 forbids model-specific parameters")
        if contract.get("requires_external_calibration") is not False:
            raise ValueError("v6 forbids external calibration")
        channels = contract["channels"]
        machine = contract["state_machine"]
        operator = channels["transport_coherence"]
        recurrence = channels["recurrence_persistence"]
        variation = channels["variation_capacity"]
        return cls(
            operator_window_tau=float(operator["training_window_tau"]),
            minimum_operator_transitions=int(operator["minimum_training_transitions"]),
            ridge_alpha=float(operator["ridge_alpha"]),
            residual_scale_floor=float(operator["residual_scale_floor"]),
            recurrence_window_tau=float(recurrence["trailing_window_tau"]),
            recurrence_relative_radius=float(recurrence["relative_radius"]),
            recurrence_threshold=float(machine["minimum_recurrence_persistence"]),
            coherence_threshold=float(machine["minimum_transport_coherence"]),
            variation_reference_tau=float(variation["reference_window_tau"]),
            variation_contraction_threshold=float(machine["minimum_variation_contraction"]),
            tau_windows=int(machine["tau_windows"]),
            crystallizing_duration_tau=float(machine["crystallizing_duration_tau"]),
            scale_epsilon=float(operator["scale_epsilon"]),
        )


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in vector) / len(vector)) if vector else 0.0


def _ridge_prediction(
    states: Sequence[Sequence[float]], index: int, config: StructuralCoherenceV6Config
) -> tuple[float | None, float | None, int]:
    """Return (coherence, residual, support) with the current transition held out."""

    end_source = index - 1
    first_source = max(0, end_source - config.operator_window)
    sources = [states[i] for i in range(first_source, end_source)]
    targets = [states[i + 1] for i in range(first_source, end_source)]
    support = len(sources)
    if support < config.minimum_operator_transitions:
        return None, None, support
    x = np.asarray(sources, dtype=float)
    y = np.asarray(targets, dtype=float)
    dimension = x.shape[1]
    gram = x.T @ x + config.ridge_alpha * np.eye(dimension)
    try:
        coefficients = np.linalg.solve(gram, x.T @ y)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(gram) @ x.T @ y
    predicted = np.asarray(states[index - 1], dtype=float) @ coefficients
    observed = np.asarray(states[index], dtype=float)
    residual = float(np.sqrt(np.mean((observed - predicted) ** 2)))
    movements = [_distance(sources[i], targets[i]) for i in range(support)]
    positive = [value for value in movements if value > config.scale_epsilon]
    movement_scale = statistics.median(positive) if positive else config.scale_epsilon
    scale = max(config.residual_scale_floor, movement_scale, config.scale_epsilon)
    return min(1.0, max(0.0, math.exp(-residual / scale))), residual, support


def _variation_capacity(states: Sequence[Sequence[float]], start: int, index: int, epsilon: float) -> float | None:
    steps = [
        np.asarray(states[i], dtype=float) - np.asarray(states[i - 1], dtype=float)
        for i in range(max(start + 1, 1), index + 1)
    ]
    active = [step for step in steps if _norm(step.tolist()) > epsilon]
    if len(active) < 3:
        return None
    matrix = np.asarray(active, dtype=float)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    energy = np.linalg.svd(centered, compute_uv=False) ** 2
    total = float(energy.sum())
    if total <= epsilon:
        return 0.0
    probabilities = energy[energy > epsilon] / total
    effective_rank = math.exp(-sum(float(p) * math.log(float(p)) for p in probabilities))
    return min(1.0, max(0.0, (effective_rank - 1.0) / max(1, matrix.shape[1] - 1)))


def _geometry_records(
    token_windows: Sequence[Sequence[Mapping[str, Any]]],
    trajectory: Sequence[Mapping[str, Any]],
    config: OperatorGeometryConfig,
) -> list[dict[str, Any]]:
    """Operator geometry over presegmented response windows, preserving turn boundaries."""

    if len(token_windows) != len(trajectory):
        raise ValueError(f"token/trajectory window mismatch: {len(token_windows)} != {len(trajectory)}")
    states: list[list[float]] = []
    loads: list[float] = []
    modes: list[str] = []
    records: list[dict[str, Any]] = []
    recurrent_dwell = 0
    mode_dwell = 0
    for index, (members, source) in enumerate(zip(token_windows, trajectory, strict=True)):
        state, component = _window_state(members, source, config)
        states.append(state)
        loads.append(component["operator_load"])
        if index == 0:
            mode = "CONSERVATION"
        else:
            change = loads[-1] - loads[-2]
            mode = "EXPANSION" if change > config.mode_deadband else "RELEASE" if change < -config.mode_deadband else "CONSERVATION"
        mode_dwell = mode_dwell + 1 if modes and modes[-1] == mode else 1
        modes.append(mode)
        start = max(0, index - config.geometry_window + 1)
        local = states[start:index + 1]
        steps = [_distance(local[i - 1], local[i]) for i in range(1, len(local))]
        path_length = sum(steps)
        net = _distance(local[0], local[-1]) if len(local) > 1 else 0.0
        admissible = recurrent_pairs = 0
        for left in range(len(local)):
            for right in range(left + 1, len(local)):
                if right - left <= config.recurrence_lag_exclusion:
                    continue
                admissible += 1
                recurrent_pairs += _distance(local[left], local[right]) < config.recurrence_radius
        current_recurrent = any(
            _distance(local[-1], local[prior]) < config.recurrence_radius
            for prior in range(0, max(0, len(local) - config.recurrence_lag_exclusion - 1))
        )
        recurrent_dwell = recurrent_dwell + 1 if current_recurrent else 0
        local_modes = modes[start:index + 1]
        records.append({
            "turn_index": int(source.get("turn_index") or 0),
            "window_index": int(source.get("window_index") or 0),
            "absolute_window_index": index,
            "geometry_ready": len(local) >= config.minimum_geometry_points,
            "state_vector": state,
            **component,
            "path_length": path_length,
            "net_displacement": net,
            "trajectory_openness": net / path_length if path_length > 0 else None,
            "recurrence_rate": recurrent_pairs / admissible if admissible else 0.0,
            "current_state_recurrent": current_recurrent,
            "recurrent_dwell": recurrent_dwell,
            "mode": mode,
            "mode_dwell": mode_dwell,
            "mode_entropy": _mode_entropy(local_modes),
            "dominant_mode": Counter(local_modes).most_common(1)[0][0],
        })
    return records


def observe_structural_coherence_v6(
    token_windows: Sequence[Sequence[Mapping[str, Any]]],
    trajectory: Sequence[Mapping[str, Any]],
    geometry_config: OperatorGeometryConfig,
    config: StructuralCoherenceV6Config,
) -> list[dict[str, Any]]:
    base = _geometry_records(token_windows, trajectory, geometry_config)
    states = [row["state_vector"] for row in base]
    intensities: list[float] = []
    capacities: list[float | None] = []
    output: list[dict[str, Any]] = []
    crystallizing_run = 0

    for index, record in enumerate(base):
        start = max(0, index - geometry_config.geometry_window + 1)
        intensity, _ = _causal_recurrence_intensity(
            states[start:index + 1], geometry_config.recurrence_lag_exclusion,
            config.recurrence_relative_radius, config.scale_epsilon,
        )
        intensities.append(intensity)
        echo = statistics.fmean(intensities[max(0, len(intensities) - config.recurrence_window):])
        ready = bool(record["geometry_ready"])
        movement = float(record["path_length"])
        active = ready and movement > geometry_config.activity_path_length_threshold
        recurrence = echo if active else 0.0
        coherence, residual, support = _ridge_prediction(states, index, config) if active else (None, None, 0)

        capacity = _variation_capacity(states, start, index, config.scale_epsilon) if ready else None
        capacities.append(capacity)
        reference = [value for value in capacities[max(0, len(capacities) - config.variation_reference_window):] if value is not None]
        peak = max(reference) if reference else None
        contraction = (
            max(0.0, 1.0 - capacity / peak)
            if capacity is not None and peak is not None and peak > config.scale_epsilon
            else 0.0 if capacity is not None else None
        )

        coherent = coherence is not None and coherence >= config.coherence_threshold
        recurrent = recurrence >= config.recurrence_threshold
        contracting = contraction is not None and contraction >= config.variation_contraction_threshold
        crystallizing_predicate = bool(active and coherent and recurrent and contracting)
        crystallizing_run = crystallizing_run + 1 if crystallizing_predicate else 0
        persistence = min(1.0, crystallizing_run / config.crystallizing_duration_windows)

        diagnostics = {
            "imitative_echo": bool(active and recurrence >= config.recurrence_threshold and coherence is not None and not coherent),
            "coherence_loss": bool(active and coherence is not None and not coherent),
            "insufficient_geometry": bool(not ready or (active and coherence is None)),
        }
        if not ready:
            primary = None
            status = "INSUFFICIENT_GEOMETRY"
        elif not active:
            primary = "STAGNANT"
            status = "PRIMARY"
        elif coherence is None or not coherent:
            primary = None
            status = "DIAGNOSTIC_ONLY" if coherence is not None else "INSUFFICIENT_GEOMETRY"
        elif crystallizing_predicate and crystallizing_run >= config.crystallizing_duration_windows:
            primary = "CRYSTALLIZED"
            status = "PRIMARY"
        elif crystallizing_predicate:
            primary = "CRYSTALLIZING"
            status = "PRIMARY"
        elif recurrent:
            primary = "RECURRENT"
            status = "PRIMARY"
        else:
            primary = "VIABLE"
            status = "PRIMARY"

        output.append({
            **record,
            "movement": movement,
            "inertial_echo_recurrence": echo,
            "recurrence_persistence": recurrence,
            "transport_coherence": coherence,
            "operator_prediction_residual": residual,
            "operator_training_support": support,
            "variation_capacity": capacity,
            "variation_reference_peak": peak,
            "variation_contraction": contraction,
            "regime_persistence": persistence,
            "crystallizing_duration_windows": crystallizing_run,
            "primary_state": primary,
            "classification_status": status,
            "diagnostics": diagnostics,
        })
    return output
