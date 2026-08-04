"""Canonical D_O v9 structural observations over causal numeric windows."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from aptadynamic_llm.artifact_schema import validate_artifact
from aptadynamic_llm.structural_coherence_v9 import (
    StructuralCoherenceV9Config,
    classify_structural_coherence_v9,
)


OBSERVER_ID = "D_O_v9"
OBSERVER_VERSION = "D_O_v9"
BASE_OBSERVER = "D_O_v6"


def _recurrence_status(
    row: Mapping[str, Any], config: StructuralCoherenceV9Config
) -> str:
    if not bool(row.get("geometry_ready")):
        return "UNRESOLVED"
    if float(row.get("movement") or 0.0) <= config.activity_path_length_threshold:
        return "INACTIVE"
    return (
        "RECURRENT"
        if float(row.get("recurrence_persistence") or 0.0)
        >= config.recurrence_threshold
        else "NON_RECURRENT"
    )


def _contraction_status(
    row: Mapping[str, Any], config: StructuralCoherenceV9Config
) -> str:
    value = row.get("variation_contraction")
    if not bool(row.get("geometry_ready")) or value is None:
        return "UNRESOLVED"
    return (
        "CONTRACTING"
        if float(value) >= config.variation_contraction_threshold
        else "NOT_CONTRACTING"
    )


def observe_structural_trajectory(
    windows: Sequence[Mapping[str, Any]],
    config: StructuralCoherenceV9Config,
) -> list[dict[str, Any]]:
    """Resolve D_O v9 without reading outcomes or provider termination metadata."""
    classified = classify_structural_coherence_v9(windows, config)
    observations: list[dict[str, Any]] = []
    for index, row in enumerate(classified):
        absolute_index = int(row.get("absolute_window_index", index))
        if absolute_index < 0:
            raise ValueError("absolute_window_index must be nonnegative")
        transport = str(row["transport_status_v9"])
        mobility = row.get("mobility_regime_v9")
        structural_state = row.get("summary_class_v9") or "TRANSPORT_UNRESOLVED"
        observations.append(
            {
                "observer": OBSERVER_ID,
                "observer_version": OBSERVER_VERSION,
                "base_observer": BASE_OBSERVER,
                "turn_index": int(row.get("turn_index") or 0),
                "window_index": int(row.get("window_index", absolute_index)),
                "absolute_window_index": absolute_index,
                "transport_status": transport,
                "recurrence_status": _recurrence_status(row, config),
                "contraction_status": _contraction_status(row, config),
                "mobility_status": mobility,
                "structural_state": structural_state,
                "movement": float(row.get("movement") or 0.0),
                "transport_coherence": (
                    None
                    if row.get("transport_coherence") is None
                    else float(row["transport_coherence"])
                ),
                "recurrence_persistence": float(
                    row.get("recurrence_persistence") or 0.0
                ),
                "variation_contraction": (
                    None
                    if row.get("variation_contraction") is None
                    else float(row["variation_contraction"])
                ),
                "diagnostics": list(row.get("diagnostics_v9") or []),
                "alert_eligible": bool(row.get("alert_eligible_v9")),
                "transport_deficit": (
                    None
                    if row.get("transport_deficit_v9") is None
                    else float(row["transport_deficit_v9"])
                ),
                "cumulative_transport_deficit": float(
                    row.get("cumulative_alert_transport_deficit_v9") or 0.0
                ),
                "evidence_window_start": 0,
                "evidence_window_end": absolute_index,
                "causal": True,
                "external_outcome_used": False,
                "provider_termination_metadata_used": False,
            }
        )
    return observations


def make_structural_observation(
    envelope: Mapping[str, Any], observation: Mapping[str, Any]
) -> dict[str, Any]:
    """Attach provenance and validate one canonical structural observation."""
    record = {**dict(envelope), **dict(observation)}
    validate_artifact(record, "structural_observation")
    return record
