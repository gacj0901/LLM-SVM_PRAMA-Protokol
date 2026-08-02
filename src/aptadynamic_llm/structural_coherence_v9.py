"""Layered transport status and coherent-mobility regimes over v6 channels."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


TRANSPORT_STATUSES_V9 = ("UNRESOLVED", "INACTIVE", "COHERENT", "PROVISIONAL", "DISRUPTED")
MOBILITY_REGIMES_V9 = ("VIABLE", "STAGNANT", "RECURRENT", "CRYSTALLIZING", "CRYSTALLIZED")
SUMMARY_CLASSES_V9 = (*MOBILITY_REGIMES_V9, "TRANSPORT_DISRUPTED", "TRANSPORT_UNRESOLVED")
DIAGNOSTICS_V9 = (
    "IMITATIVE_ECHO", "LOCAL_TRANSPORT_DISCONTINUITY", "HYSTERESIS_INHERITANCE",
    "TRANSPORT_DISRUPTION", "PERSISTENT_TRANSPORT_DISRUPTION",
    "INSUFFICIENT_TRANSPORT_SUPPORT", "INSUFFICIENT_GEOMETRY",
)


@dataclass(frozen=True)
class StructuralCoherenceV9Config:
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
    def from_contract(cls, contract: Mapping[str, Any]) -> "StructuralCoherenceV9Config":
        if contract.get("schema") != "LLM-SVM-structural-coherence-contract/9":
            raise ValueError("unexpected structural-coherence v9 contract schema")
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


def classify_structural_coherence_v9(
    windows: Sequence[Mapping[str, Any]], config: StructuralCoherenceV9Config
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    crystallizing_run = 0
    discontinuity_run = 0
    last_coherent_regime: str | None = None
    first_transport_evaluable: int | None = None
    cumulative_deficit = 0.0
    for index, row in enumerate(windows):
        ready = bool(row.get("geometry_ready"))
        movement = float(row.get("movement") or 0.0)
        active = ready and movement > config.activity_path_length_threshold
        coherence = row.get("transport_coherence")
        recurrence = float(row.get("recurrence_persistence") or 0.0)
        recurrent = recurrence >= config.recurrence_threshold
        contraction = row.get("variation_contraction")
        contracting = contraction is not None and float(contraction) >= config.variation_contraction_threshold
        coherent = coherence is not None and float(coherence) >= config.coherence_threshold
        local_discontinuity = bool(ready and active and coherence is not None and not coherent)
        if ready and active and coherence is not None and first_transport_evaluable is None:
            first_transport_evaluable = index
        alert_eligible = bool(
            first_transport_evaluable is not None
            and index >= first_transport_evaluable + config.tau_windows
        )

        mobility: str | None = None
        inherited = False
        if not ready:
            transport = "UNRESOLVED"
            summary = None
            discontinuity_run = 0
            crystallizing_run = 0
        elif not active:
            transport = "INACTIVE"
            mobility = "STAGNANT"
            summary = mobility
            discontinuity_run = 0
            crystallizing_run = 0
            last_coherent_regime = None
        elif coherence is None:
            transport = "UNRESOLVED"
            summary = "TRANSPORT_UNRESOLVED"
            discontinuity_run = 0
            crystallizing_run = 0
        elif coherent:
            transport = "COHERENT"
            discontinuity_run = 0
            crystallizing_predicate = recurrent and contracting
            crystallizing_run = crystallizing_run + 1 if crystallizing_predicate else 0
            if crystallizing_predicate and crystallizing_run >= config.tau_windows:
                mobility = "CRYSTALLIZED"
            elif crystallizing_predicate:
                mobility = "CRYSTALLIZING"
            elif recurrent:
                mobility = "RECURRENT"
            else:
                mobility = "VIABLE"
            summary = mobility
            last_coherent_regime = mobility
        else:
            discontinuity_run += 1
            crystallizing_run = 0
            if last_coherent_regime == "VIABLE" and discontinuity_run <= config.hysteresis_grace_windows:
                transport = "PROVISIONAL"
                mobility = "VIABLE"
                summary = mobility
                inherited = True
            else:
                transport = "DISRUPTED"
                summary = "TRANSPORT_DISRUPTED"

        deficit = (
            max(0.0, (config.coherence_threshold - float(coherence)) / config.coherence_threshold)
            if ready and active and coherence is not None else None
        )
        if alert_eligible and deficit is not None:
            cumulative_deficit += deficit
        diagnostics = []
        if not ready:
            diagnostics.append("INSUFFICIENT_GEOMETRY")
        elif active and coherence is None:
            diagnostics.append("INSUFFICIENT_TRANSPORT_SUPPORT")
        if local_discontinuity:
            diagnostics.append("LOCAL_TRANSPORT_DISCONTINUITY")
            diagnostics.append("HYSTERESIS_INHERITANCE" if inherited else "TRANSPORT_DISRUPTION")
        if transport == "DISRUPTED" and discontinuity_run > config.hysteresis_grace_windows:
            diagnostics.append("PERSISTENT_TRANSPORT_DISRUPTION")
        if local_discontinuity and recurrent:
            diagnostics.append("IMITATIVE_ECHO")
        output.append({
            **dict(row),
            "transport_status_v9": transport,
            "mobility_regime_v9": mobility,
            "summary_class_v9": summary,
            "diagnostics_v9": diagnostics,
            "alert_eligible_v9": alert_eligible,
            "first_transport_evaluable_window_v9": first_transport_evaluable,
            "local_transport_discontinuity_run_v9": discontinuity_run,
            "crystallizing_run_v9": crystallizing_run,
            "transport_deficit_v9": deficit,
            "cumulative_alert_transport_deficit_v9": cumulative_deficit,
        })
    return output
