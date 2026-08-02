"""Scale-relative structural mobility and causal crystallization state machine."""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from typing import Any, Mapping, Sequence

from .operator_geometry import OperatorGeometryConfig, _distance, observe_operator_geometry


STATES = ("MOBILE", "RECURRENT", "CRYSTALLIZING", "CRYSTALLIZED")


@dataclass(frozen=True)
class StructuralMobilityV3Config:
    """Universal dimensionless parameters for the v3 observer."""

    recurrence_persistence_window_tau: float = 1.0
    recurrence_relative_radius: float = 0.5
    recurrence_persistence_threshold: float = 0.3
    transport_efficiency_threshold: float = 0.25
    tau_windows: int = 16
    crystallizing_duration_tau: float = 1.0
    scale_epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if self.tau_windows <= 0:
            raise ValueError("tau_windows must be positive")
        for name in (
            "recurrence_persistence_window_tau",
            "recurrence_relative_radius",
            "recurrence_persistence_threshold",
            "transport_efficiency_threshold",
            "crystallizing_duration_tau",
            "scale_epsilon",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.recurrence_persistence_threshold >= 1.0:
            raise ValueError("recurrence_persistence_threshold must be below one")
        if self.transport_efficiency_threshold >= 1.0:
            raise ValueError("transport_efficiency_threshold must be below one")

    @property
    def persistence_window(self) -> int:
        return max(1, math.ceil(self.recurrence_persistence_window_tau * self.tau_windows))

    @property
    def crystallizing_duration_windows(self) -> int:
        return max(1, math.ceil(self.crystallizing_duration_tau * self.tau_windows))

    @classmethod
    def from_contract(cls, contract: Mapping[str, Any]) -> "StructuralMobilityV3Config":
        if contract.get("schema") != "LLM-SVM-CoCC-structural-mobility-contract/3":
            raise ValueError("unexpected structural-mobility v3 contract schema")
        if contract.get("model_specific_parameters") is not False:
            raise ValueError("structural mobility must forbid model-specific parameters")
        if contract.get("requires_external_calibration") is not False:
            raise ValueError("structural mobility must not require calibration")
        recurrence = contract["channels"]["recurrence_persistence"]
        classification = contract["state_machine"]
        return cls(
            recurrence_persistence_window_tau=float(recurrence["trailing_window_tau"]),
            recurrence_relative_radius=float(recurrence["relative_radius"]),
            recurrence_persistence_threshold=float(classification["minimum_recurrence_persistence"]),
            transport_efficiency_threshold=float(classification["maximum_transport_efficiency"]),
            tau_windows=int(classification["tau_windows"]),
            crystallizing_duration_tau=float(classification["crystallizing_duration_tau"]),
            scale_epsilon=float(recurrence["scale_epsilon"]),
        )


def _causal_recurrence_intensity(
    states: Sequence[Sequence[float]], lag_exclusion: int, relative_radius: float, epsilon: float
) -> tuple[float, float]:
    """Return current-state recurrence using a causal, local, scale-relative radius."""

    admissible_distances = [
        _distance(states[left], states[right])
        for left in range(len(states))
        for right in range(left + 1, len(states))
        if right - left > lag_exclusion
    ]
    if not admissible_distances:
        return 0.0, 0.0
    positive = [distance for distance in admissible_distances if distance > epsilon]
    local_scale = statistics.median(positive) if positive else epsilon
    current_distances = [
        _distance(states[-1], states[prior])
        for prior in range(max(0, len(states) - lag_exclusion - 1))
    ]
    if not current_distances:
        return 0.0, local_scale
    radius = max(epsilon, relative_radius * local_scale)
    intensity = max(max(0.0, 1.0 - distance / radius) for distance in current_distances)
    return intensity, local_scale


def observe_structural_mobility_v3(
    tokens: Sequence[Mapping[str, Any]],
    trajectory: Sequence[Mapping[str, Any]],
    geometry_config: OperatorGeometryConfig,
    mobility_config: StructuralMobilityV3Config,
) -> list[dict[str, Any]]:
    """Emit four causal coordinates and a dwell-free structural state machine."""

    base = observe_operator_geometry(tokens, trajectory, geometry_config)
    states = [record["state_vector"] for record in base]
    intensities: list[float] = []
    output: list[dict[str, Any]] = []
    crystallizing_run = 0
    ready_count = 0
    crystallized_count = 0
    crystallization_onset: int | None = None

    for index, record in enumerate(base):
        start = max(0, index - geometry_config.geometry_window + 1)
        intensity, local_scale = _causal_recurrence_intensity(
            states[start : index + 1],
            geometry_config.recurrence_lag_exclusion,
            mobility_config.recurrence_relative_radius,
            mobility_config.scale_epsilon,
        )
        intensities.append(intensity)
        persistence_start = max(0, len(intensities) - mobility_config.persistence_window)
        recurrence_persistence = statistics.fmean(intensities[persistence_start:])
        ready = bool(record["geometry_ready"])
        recurrent = ready and recurrence_persistence > mobility_config.recurrence_persistence_threshold
        crystallizing = recurrent and float(record["transport_efficiency"]) < mobility_config.transport_efficiency_threshold
        crystallizing_run = crystallizing_run + 1 if crystallizing else 0
        crystallized = crystallizing and crystallizing_run >= mobility_config.crystallizing_duration_windows

        if crystallized:
            state = "CRYSTALLIZED"
        elif crystallizing:
            state = "CRYSTALLIZING"
        elif recurrent:
            state = "RECURRENT"
        else:
            state = "MOBILE"
        if ready:
            ready_count += 1
            crystallized_count += int(crystallized)
        if crystallized and crystallization_onset is None:
            crystallization_onset = int(record["window_index"])

        output.append(
            {
                **record,
                "local_recurrence_scale": local_scale,
                "distance_weighted_recurrence": intensity,
                "recurrence_persistence": recurrence_persistence,
                "crystallized_fraction": crystallized_count / ready_count if ready_count else 0.0,
                "crystallization_onset": crystallization_onset,
                "crystallizing_duration_windows": crystallizing_run,
                "crystallizing_duration_tau": crystallizing_run / mobility_config.tau_windows,
                "structural_mobility_state": state,
            }
        )
    return output
