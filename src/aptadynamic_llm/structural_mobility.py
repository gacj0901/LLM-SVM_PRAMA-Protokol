"""Causal structural-mobility channels layered over operator geometry v1."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from .operator_geometry import (
    OperatorGeometryConfig,
    _distance,
    observe_operator_geometry,
)


@dataclass(frozen=True)
class StructuralMobilityConfig:
    """Universal logical thresholds; no model- or outcome-specific fitting."""

    persistence_window: int = 16
    activity_path_length_threshold: float = 0.50
    transport_efficiency_threshold: float = 0.25
    recurrence_persistence_threshold: float = 0.30
    recurrent_dwell_fraction_threshold: float = 0.50
    early_crystallization_window: int = 32

    def __post_init__(self) -> None:
        if self.persistence_window < 2 or self.early_crystallization_window <= 0:
            raise ValueError("mobility windows must be positive")
        for name in (
            "activity_path_length_threshold",
            "transport_efficiency_threshold",
            "recurrence_persistence_threshold",
            "recurrent_dwell_fraction_threshold",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be finite and in (0,1)")

    @classmethod
    def from_contract(cls, contract: Mapping[str, Any]) -> "StructuralMobilityConfig":
        if contract.get("schema") != "LLM-SVM-CoCC-structural-mobility-contract/2":
            raise ValueError("unexpected structural-mobility contract schema")
        if contract.get("model_specific_parameters") is not False:
            raise ValueError("structural mobility must forbid model-specific parameters")
        if contract.get("requires_external_calibration") is not False:
            raise ValueError("structural mobility must not require calibration")
        channels = contract["channels"]
        logic = contract["classification"]
        return cls(
            persistence_window=int(channels["recurrence_persistence"]["trailing_window"]),
            activity_path_length_threshold=float(logic["activity_gate"]),
            transport_efficiency_threshold=float(logic["maximum_transport_efficiency"]),
            recurrence_persistence_threshold=float(logic["minimum_recurrence_persistence"]),
            recurrent_dwell_fraction_threshold=float(logic["minimum_recurrent_dwell_fraction"]),
            early_crystallization_window=int(logic["early_crystallization_before_window"]),
        )


def _distance_weighted_recurrence(
    states: Sequence[Sequence[float]], lag_exclusion: int, radius: float
) -> float:
    similarities: list[float] = []
    for left in range(len(states)):
        for right in range(left + 1, len(states)):
            if right - left <= lag_exclusion:
                continue
            distance = _distance(states[left], states[right])
            similarities.append(max(0.0, 1.0 - distance / radius))
    return sum(similarities) / len(similarities) if similarities else 0.0


def observe_structural_mobility(
    tokens: Sequence[Mapping[str, Any]],
    trajectory: Sequence[Mapping[str, Any]],
    geometry_config: OperatorGeometryConfig,
    mobility_config: StructuralMobilityConfig,
) -> list[dict[str, Any]]:
    """Add causal mobility channels without modifying any PRAMA coordinate."""

    base = observe_operator_geometry(tokens, trajectory, geometry_config)
    states = [record["state_vector"] for record in base]
    recurrence_intensity: list[float] = []
    output: list[dict[str, Any]] = []
    recirculation_onset: int | None = None
    crystallization_onset: int | None = None
    for index, record in enumerate(base):
        start = max(0, index - geometry_config.geometry_window + 1)
        local_states = states[start : index + 1]
        intensity = _distance_weighted_recurrence(
            local_states,
            geometry_config.recurrence_lag_exclusion,
            geometry_config.recurrence_radius,
        )
        recurrence_intensity.append(intensity)
        persistence_start = max(0, len(recurrence_intensity) - mobility_config.persistence_window)
        persistence = sum(recurrence_intensity[persistence_start:]) / (
            len(recurrence_intensity) - persistence_start
        )
        local_size = min(index + 1, geometry_config.geometry_window)
        dwell_fraction = min(int(record["recurrent_dwell"]), local_size) / local_size
        recirculating = bool(
            record["geometry_ready"]
            and float(record["path_length"]) > mobility_config.activity_path_length_threshold
            and float(record["transport_efficiency"])
            < mobility_config.transport_efficiency_threshold
            and persistence > mobility_config.recurrence_persistence_threshold
        )
        if recirculating and recirculation_onset is None:
            recirculation_onset = int(record["window_index"])
        crystallized = bool(
            recirculating
            and dwell_fraction > mobility_config.recurrent_dwell_fraction_threshold
        )
        if crystallized and crystallization_onset is None:
            crystallization_onset = int(record["window_index"])
        early = bool(
            crystallized
            and crystallization_onset is not None
            and crystallization_onset < mobility_config.early_crystallization_window
        )
        if early:
            condition = "EARLY_CRYSTALLIZATION"
        elif crystallized:
            condition = "CRYSTALLIZED"
        elif recirculating:
            condition = "RECIRCULATING"
        else:
            condition = "MOBILE_OR_UNCLASSIFIED"
        output.append(
            {
                **record,
                "distance_weighted_recurrence": intensity,
                "recurrence_persistence": persistence,
                "recurrent_dwell_fraction": dwell_fraction,
                "recirculating": recirculating,
                "crystallized": crystallized,
                "early_crystallization": early,
                "recirculation_onset": recirculation_onset,
                "crystallization_onset": crystallization_onset,
                "structural_mobility_condition": condition,
            }
        )
    return output
