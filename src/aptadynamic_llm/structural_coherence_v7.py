"""Exhaustive offline state layer for structural coherence v6 trajectories.

V7 does not recompute numeric channels or modify the PRAMA kernel. It assigns
exactly one primary state to every geometry-ready window, makes transport
diagnostics subordinate, and exposes persistence episodes explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from typing import Any, Mapping, Sequence


PRIMARY_STATES_V7 = ("VIABLE", "STAGNANT", "RECURRENT", "CRYSTALLIZING", "CRYSTALLIZED")
DIAGNOSTICS_V7 = (
    "IMITATIVE_ECHO",
    "LOCAL_TRANSPORT_DISCONTINUITY",
    "PERSISTENT_TRANSPORT_DISRUPTION",
    "INSUFFICIENT_GEOMETRY",
)


@dataclass(frozen=True)
class StructuralCoherenceV7Config:
    activity_path_length_threshold: float = 0.5
    recurrence_threshold: float = 0.3
    coherence_threshold: float = 0.5
    variation_contraction_threshold: float = 0.25
    tau_windows: int = 16
    minimum_session_transport_windows: int = 8

    def __post_init__(self) -> None:
        if self.tau_windows <= 0 or self.minimum_session_transport_windows <= 0:
            raise ValueError("window thresholds must be positive")
        if not math.isfinite(float(self.activity_path_length_threshold)) or self.activity_path_length_threshold <= 0.0:
            raise ValueError("activity_path_length_threshold must be positive")
        for name in ("recurrence_threshold", "coherence_threshold", "variation_contraction_threshold"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be in (0,1)")

    @classmethod
    def from_contract(cls, contract: Mapping[str, Any]) -> "StructuralCoherenceV7Config":
        if contract.get("schema") != "LLM-SVM-structural-coherence-contract/7":
            raise ValueError("unexpected structural-coherence v7 contract schema")
        if contract.get("model_specific_parameters") is not False:
            raise ValueError("v7 forbids model-specific parameters")
        machine = contract["state_machine"]
        return cls(
            activity_path_length_threshold=float(machine["minimum_activity_path_length"]),
            recurrence_threshold=float(machine["minimum_recurrence_persistence"]),
            coherence_threshold=float(machine["minimum_transport_coherence"]),
            variation_contraction_threshold=float(machine["minimum_variation_contraction"]),
            tau_windows=int(machine["tau_windows"]),
            minimum_session_transport_windows=int(contract["session_evaluation"]["minimum_transport_evaluable_windows"]),
        )


def classify_structural_coherence_v7(
    windows: Sequence[Mapping[str, Any]], config: StructuralCoherenceV7Config
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    crystallizing_run = 0
    disruption_run = 0
    for row in windows:
        ready = bool(row.get("geometry_ready"))
        movement = float(row.get("movement") or 0.0)
        active = ready and movement > config.activity_path_length_threshold
        coherence = row.get("transport_coherence")
        recurrent = float(row.get("recurrence_persistence") or 0.0) >= config.recurrence_threshold
        contraction = row.get("variation_contraction")
        contracting = contraction is not None and float(contraction) >= config.variation_contraction_threshold
        coherent = coherence is not None and float(coherence) >= config.coherence_threshold
        local_discontinuity = bool(ready and active and coherence is not None and not coherent)
        imitative_echo = bool(active and recurrent and local_discontinuity)
        disruption_run = disruption_run + 1 if local_discontinuity else 0

        crystallizing_predicate = bool(active and coherent and recurrent and contracting)
        crystallizing_run = crystallizing_run + 1 if crystallizing_predicate else 0
        if not ready:
            state = None
        elif not active:
            state = "STAGNANT"
        elif crystallizing_predicate and crystallizing_run >= config.tau_windows:
            state = "CRYSTALLIZED"
        elif crystallizing_predicate:
            state = "CRYSTALLIZING"
        elif coherent and recurrent:
            state = "RECURRENT"
        else:
            # Exhaustive residual mobile state. Diagnostics explicitly prevent
            # interpreting this as confirmed viability when transport is weak.
            state = "VIABLE"

        diagnostics = []
        if not ready:
            diagnostics.append("INSUFFICIENT_GEOMETRY")
        if imitative_echo:
            diagnostics.append("IMITATIVE_ECHO")
        if local_discontinuity:
            diagnostics.append("LOCAL_TRANSPORT_DISCONTINUITY")
        if disruption_run >= config.tau_windows:
            diagnostics.append("PERSISTENT_TRANSPORT_DISRUPTION")
        output.append({
            **dict(row),
            "primary_state_v7": state,
            "classification_status_v7": "PRIMARY" if ready else "INSUFFICIENT_GEOMETRY",
            "diagnostics_v7": diagnostics,
            "viability_confirmed": bool(state == "VIABLE" and coherent),
            "local_transport_discontinuity_run": disruption_run,
            "crystallizing_run_v7": crystallizing_run,
        })
    return output


def crystallizing_episode_summary(windows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    lengths: list[int] = []
    current = 0
    for row in windows:
        if row.get("primary_state_v7") in {"CRYSTALLIZING", "CRYSTALLIZED"}:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    terminal = current
    if current:
        lengths.append(current)
    return {
        "maximum_consecutive_crystallizing_windows": max(lengths, default=0),
        "number_of_crystallizing_episodes": len(lengths),
        "median_crystallizing_episode_length": statistics.median(lengths) if lengths else 0.0,
        "terminal_crystallizing_dwell": terminal,
        "episode_lengths": lengths,
    }
