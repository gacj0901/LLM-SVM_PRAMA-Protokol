"""Causal hysteresis state layer over saved structural-coherence v6 channels."""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from typing import Any, Mapping, Sequence


PRIMARY_STATES_V8 = (
    "VIABLE", "STAGNANT", "RECURRENT", "CRYSTALLIZING", "CRYSTALLIZED", "TRANSPORT_DISRUPTED"
)
DIAGNOSTICS_V8 = (
    "IMITATIVE_ECHO", "LOCAL_TRANSPORT_DISCONTINUITY", "HYSTERESIS_INHERITANCE",
    "PERSISTENT_TRANSPORT_DISRUPTION", "INSUFFICIENT_GEOMETRY",
)


@dataclass(frozen=True)
class StructuralCoherenceV8Config:
    activity_path_length_threshold: float = 0.5
    recurrence_threshold: float = 0.3
    coherence_threshold: float = 0.5
    variation_contraction_threshold: float = 0.25
    tau_windows: int = 16
    hysteresis_grace_tau: float = 0.25
    minimum_session_transport_windows: int = 8

    @property
    def hysteresis_grace_windows(self) -> int:
        return max(1, math.ceil(self.hysteresis_grace_tau * self.tau_windows))

    @classmethod
    def from_contract(cls, contract: Mapping[str, Any]) -> "StructuralCoherenceV8Config":
        if contract.get("schema") != "LLM-SVM-structural-coherence-contract/8":
            raise ValueError("unexpected structural-coherence v8 contract schema")
        if contract.get("model_specific_parameters") is not False:
            raise ValueError("v8 forbids model-specific parameters")
        machine = contract["state_machine"]
        return cls(
            activity_path_length_threshold=float(machine["minimum_activity_path_length"]),
            recurrence_threshold=float(machine["minimum_recurrence_persistence"]),
            coherence_threshold=float(machine["minimum_transport_coherence"]),
            variation_contraction_threshold=float(machine["minimum_variation_contraction"]),
            tau_windows=int(machine["tau_windows"]),
            hysteresis_grace_tau=float(machine["hysteresis_grace_tau"]),
            minimum_session_transport_windows=int(contract["session_evaluation"]["minimum_transport_evaluable_windows"]),
        )


def classify_structural_coherence_v8(
    windows: Sequence[Mapping[str, Any]], config: StructuralCoherenceV8Config
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    crystallizing_run = 0
    discontinuity_run = 0
    last_coherent_state: str | None = None
    for row in windows:
        ready = bool(row.get("geometry_ready"))
        movement = float(row.get("movement") or 0.0)
        active = ready and movement > config.activity_path_length_threshold
        coherence = row.get("transport_coherence")
        coherent = coherence is not None and float(coherence) >= config.coherence_threshold
        recurrent = float(row.get("recurrence_persistence") or 0.0) >= config.recurrence_threshold
        contraction = row.get("variation_contraction")
        contracting = contraction is not None and float(contraction) >= config.variation_contraction_threshold
        local_discontinuity = bool(ready and active and coherence is not None and not coherent)
        discontinuity_run = discontinuity_run + 1 if local_discontinuity else 0
        imitative_echo = bool(local_discontinuity and recurrent)
        crystallizing_predicate = bool(active and coherent and recurrent and contracting)
        crystallizing_run = crystallizing_run + 1 if crystallizing_predicate else 0
        inherited = False
        viability_status = "NOT_APPLICABLE"

        if not ready:
            state = None
        elif not active:
            state = "STAGNANT"
        elif coherent:
            if crystallizing_predicate and crystallizing_run >= config.tau_windows:
                state = "CRYSTALLIZED"
            elif crystallizing_predicate:
                state = "CRYSTALLIZING"
            elif recurrent:
                state = "RECURRENT"
            else:
                state = "VIABLE"
                viability_status = "CONFIRMED"
            last_coherent_state = state
        elif last_coherent_state == "VIABLE" and discontinuity_run <= config.hysteresis_grace_windows:
            state = "VIABLE"
            inherited = True
            viability_status = "PROVISIONAL"
        else:
            state = "TRANSPORT_DISRUPTED"

        diagnostics = []
        if not ready:
            diagnostics.append("INSUFFICIENT_GEOMETRY")
        if imitative_echo:
            diagnostics.append("IMITATIVE_ECHO")
        if local_discontinuity:
            diagnostics.append("LOCAL_TRANSPORT_DISCONTINUITY")
        if inherited:
            diagnostics.append("HYSTERESIS_INHERITANCE")
        if state == "TRANSPORT_DISRUPTED":
            diagnostics.append("PERSISTENT_TRANSPORT_DISRUPTION")
        output.append({
            **dict(row),
            "primary_state_v8": state,
            "classification_status_v8": "PRIMARY" if ready else "INSUFFICIENT_GEOMETRY",
            "diagnostics_v8": diagnostics,
            "viability_status_v8": viability_status,
            "hysteresis_source_state": last_coherent_state if inherited else None,
            "local_transport_discontinuity_run_v8": discontinuity_run,
            "crystallizing_run_v8": crystallizing_run,
        })
    return output


def episode_summary(windows: Sequence[Mapping[str, Any]], states: set[str]) -> dict[str, Any]:
    lengths: list[int] = []
    current = 0
    for row in windows:
        if row.get("primary_state_v8") in states:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    terminal = current
    if current:
        lengths.append(current)
    return {
        "maximum_consecutive_windows": max(lengths, default=0),
        "number_of_episodes": len(lengths),
        "median_episode_length": statistics.median(lengths) if lengths else 0.0,
        "terminal_dwell": terminal,
        "episode_lengths": lengths,
    }
