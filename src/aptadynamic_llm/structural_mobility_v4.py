"""Movement-conditioned structural mobility and stagnation diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from typing import Any, Mapping, Sequence

from .operator_geometry import OperatorGeometryConfig, _distance, observe_operator_geometry


STATES = ("VIABLE", "STAGNANT", "RECURRENT", "CRYSTALLIZING", "CRYSTALLIZED")


@dataclass(frozen=True)
class StructuralMobilityV4Config:
    """Universal dimensionless parameters for the movement-conditioned observer."""

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
    def from_contract(cls, contract: Mapping[str, Any]) -> "StructuralMobilityV4Config":
        if contract.get("schema") != "LLM-SVM-CoCC-structural-mobility-contract/4":
            raise ValueError("unexpected structural-mobility v4 contract schema")
        if contract.get("model_specific_parameters") is not False:
            raise ValueError("structural mobility must forbid model-specific parameters")
        if contract.get("requires_external_calibration") is not False:
            raise ValueError("structural mobility must not require calibration")
        recurrence = contract["channels"]["recurrence_persistence_raw"]
        classification = contract["state_machine"]
        return cls(
            recurrence_persistence_window_tau=float(recurrence["trailing_window_tau"]),
            recurrence_relative_radius=float(recurrence["relative_radius"]),
            recurrence_persistence_threshold=float(classification["minimum_mobile_recurrence_persistence"]),
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


def observe_structural_mobility_v4(
    tokens: Sequence[Mapping[str, Any]],
    trajectory: Sequence[Mapping[str, Any]],
    geometry_config: OperatorGeometryConfig,
    mobility_config: StructuralMobilityV4Config,
) -> list[dict[str, Any]]:
    """Separate static stagnation from recurrence sustained by structural movement."""

    base = observe_operator_geometry(tokens, trajectory, geometry_config)
    states = [record["state_vector"] for record in base]
    intensities: list[float] = []
    output: list[dict[str, Any]] = []
    crystallizing_run = 0
    ready_count = 0
    crystallized_count = 0
    stagnant_count = 0
    crystallization_onset: int | None = None
    stagnation_onset: int | None = None

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
        recurrence_persistence_raw = statistics.fmean(intensities[persistence_start:])
        ready = bool(record["geometry_ready"])
        path_length = float(record["path_length"])
        active = ready and path_length > geometry_config.activity_path_length_threshold
        activity_indicator = 1.0 if active else 0.0
        mobile_persistence = activity_indicator * recurrence_persistence_raw
        transport_efficiency_active = float(record["transport_efficiency"]) if active else None
        recurrent = bool(
            active
            and mobile_persistence > mobility_config.recurrence_persistence_threshold
            and transport_efficiency_active is not None
            and transport_efficiency_active >= mobility_config.transport_efficiency_threshold
        )
        crystallizing = bool(
            active
            and mobile_persistence > mobility_config.recurrence_persistence_threshold
            and transport_efficiency_active is not None
            and transport_efficiency_active < mobility_config.transport_efficiency_threshold
        )
        crystallizing_run = crystallizing_run + 1 if crystallizing else 0
        crystallized = crystallizing and crystallizing_run >= mobility_config.crystallizing_duration_windows
        stagnant = ready and not active

        if not ready:
            state: str | None = None
        elif stagnant:
            state = "STAGNANT"
        elif crystallized:
            state = "CRYSTALLIZED"
        elif crystallizing:
            state = "CRYSTALLIZING"
        elif recurrent:
            state = "RECURRENT"
        else:
            # Active trajectories without persistent recurrence remain viable;
            # low instantaneous transport alone is not called crystallization.
            state = "VIABLE"

        if ready:
            ready_count += 1
            stagnant_count += int(stagnant)
            crystallized_count += int(crystallized)
        if stagnant and stagnation_onset is None:
            stagnation_onset = int(record["window_index"])
        if crystallized and crystallization_onset is None:
            crystallization_onset = int(record["window_index"])

        output.append(
            {
                **record,
                "local_recurrence_scale": local_scale,
                "distance_weighted_recurrence": intensity,
                "recurrence_persistence_raw": recurrence_persistence_raw,
                "activity_indicator": activity_indicator,
                "mobile_recurrence_persistence": mobile_persistence,
                "transport_efficiency_active": transport_efficiency_active,
                "transport_efficiency_valid": active,
                "stagnant_fraction": stagnant_count / ready_count if ready_count else 0.0,
                "stagnation_onset": stagnation_onset,
                "crystallized_fraction": crystallized_count / ready_count if ready_count else 0.0,
                "crystallization_onset": crystallization_onset,
                "crystallizing_duration_windows": crystallizing_run,
                "crystallizing_duration_tau": crystallizing_run / mobility_config.tau_windows,
                "structural_mobility_state": state,
            }
        )
    return output
