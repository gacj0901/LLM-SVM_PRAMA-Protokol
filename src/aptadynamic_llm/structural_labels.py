"""Deterministic structural annotation rules with explicit precedence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from aptadynamic_llm.artifact_schema import ChannelStatus, validate_artifact


LABEL_VERSION = "structural-labels/1.0.0"


class ConfidenceStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    INDETERMINATE = "INDETERMINATE"


class Annotation(str, Enum):
    VIABLE_INTERACTION = "monitor.VIABLE_INTERACTION"
    UNRESOLVED_FRICTION = "monitor.UNRESOLVED_FRICTION"
    RECURSIVE_IMITATIVE_ITERATION = "monitor.RECURSIVE_IMITATIVE_ITERATION"
    CRYSTALLIZATION_CANDIDATE = "monitor.CRYSTALLIZATION_CANDIDATE"
    INDETERMINATE = "monitor.INDETERMINATE"


@dataclass(frozen=True)
class StructuralLabelInput:
    coupling_state: str
    external_coupling_state: str
    perturbation_response_state: str
    epistemic_channel_state: str
    structural_history_state: str
    coupling_status: str
    external_status: str
    perturbation_status: str
    epistemic_status: str
    history_status: str
    self_dependence_persistence_windows: int
    minimum_persistence_windows: int
    continued_operation: bool
    relevant_external_friction: bool
    evidence_window_start: int
    evidence_window_end: int

    def __post_init__(self) -> None:
        for name in (
            "coupling_status",
            "external_status",
            "perturbation_status",
            "epistemic_status",
            "history_status",
        ):
            ChannelStatus(getattr(self, name))
        if self.minimum_persistence_windows < 1:
            raise ValueError("minimum_persistence_windows must be positive")
        if self.evidence_window_end < self.evidence_window_start:
            raise ValueError("evidence window must be increasing")


def _observed(value: str) -> bool:
    return value == ChannelStatus.OBSERVED.value


def classify_structural_state(
    *,
    envelope: Mapping[str, Any],
    inputs: StructuralLabelInput,
    calibration_reference: str,
) -> dict[str, Any]:
    """Emit exactly one annotation under frozen precedence rules."""

    status = {
        "coupling": inputs.coupling_status,
        "external_anchor": inputs.external_status,
        "perturbation": inputs.perturbation_status,
        "epistemic": inputs.epistemic_status,
        "history": inputs.history_status,
    }
    unavailable = sorted(
        name
        for name, value in status.items()
        if value in {ChannelStatus.UNAVAILABLE.value, ChannelStatus.INVALID.value}
    )
    persistent_self = (
        inputs.coupling_state == "SELF_DOMINANT_CANDIDATE"
        and inputs.self_dependence_persistence_windows
        >= inputs.minimum_persistence_windows
    )
    crystallization_conditions = {
        "persistent_self_dominance": persistent_self,
        "external_nonintegration": (
            _observed(inputs.external_status)
            and inputs.external_coupling_state == "ANCHOR_NOT_INTEGRATED"
        ),
        "perturbation_rigidity": (
            _observed(inputs.perturbation_status)
            and inputs.perturbation_response_state in {"RIGID", "COUNTER_REACTIVE"}
        ),
        "historical_degradation": (
            _observed(inputs.history_status)
            and inputs.structural_history_state
            in {"DEGRADING_PARTIAL", "CRITICAL_PARTIAL"}
        ),
        "continued_operation": inputs.continued_operation,
    }
    recursive_conditions = {
        "persistent_self_dominance": persistent_self,
        "weak_external_uptake": (
            _observed(inputs.external_status)
            and inputs.external_coupling_state
            in {"ANCHOR_NOT_INTEGRATED", "ANCHOR_PARTIALLY_INTEGRATED"}
        ),
        "continued_operation": inputs.continued_operation,
    }
    friction_conditions = {
        "external_friction_observed": (
            _observed(inputs.external_status) and inputs.relevant_external_friction
        ),
        "partial_or_rigid_response": (
            _observed(inputs.perturbation_status)
            and inputs.perturbation_response_state in {"PARTIAL", "RIGID"}
        ),
        "history_not_recovering": (
            _observed(inputs.history_status)
            and inputs.structural_history_state
            in {"DEGRADING_PARTIAL", "CRITICAL_PARTIAL", "STABLE_PARTIAL"}
        ),
    }
    viable_conditions = {
        "functional_coupling": (
            _observed(inputs.coupling_status)
            and inputs.coupling_state in {"USER_DOMINANT", "BALANCED_COUPLING"}
        ),
        "anchor_uptake": (
            inputs.external_status == ChannelStatus.NOT_APPLICABLE.value
            or (
                _observed(inputs.external_status)
                and inputs.external_coupling_state
                in {"ANCHOR_INTEGRATED", "NO_RELEVANT_ANCHOR"}
            )
        ),
        "corrigible": (
            inputs.perturbation_status == ChannelStatus.NOT_APPLICABLE.value
            or (
                _observed(inputs.perturbation_status)
                and inputs.perturbation_response_state in {"ADAPTIVE", "PARTIAL"}
            )
        ),
        "no_persistent_degradation": (
            _observed(inputs.history_status)
            and inputs.structural_history_state
            in {"STABLE_PARTIAL", "RECOVERING_PARTIAL"}
        ),
    }

    required_channels: list[str]
    conditions: dict[str, bool]
    rule_id: str
    if all(crystallization_conditions.values()):
        label = Annotation.CRYSTALLIZATION_CANDIDATE
        rule_id = "R1_CRYSTALLIZATION_CANDIDATE"
        required_channels = ["coupling", "external_anchor", "perturbation", "history"]
        conditions = crystallization_conditions
    elif all(recursive_conditions.values()):
        label = Annotation.RECURSIVE_IMITATIVE_ITERATION
        rule_id = "R2_RECURSIVE_IMITATIVE_ITERATION"
        required_channels = ["coupling", "external_anchor"]
        conditions = recursive_conditions
    elif all(friction_conditions.values()):
        label = Annotation.UNRESOLVED_FRICTION
        rule_id = "R3_UNRESOLVED_FRICTION"
        required_channels = ["external_anchor", "perturbation", "history"]
        conditions = friction_conditions
    elif all(viable_conditions.values()):
        label = Annotation.VIABLE_INTERACTION
        rule_id = "R4_VIABLE_INTERACTION"
        required_channels = ["coupling", "history"]
        conditions = viable_conditions
    else:
        label = Annotation.INDETERMINATE
        rule_id = "R0_INDETERMINATE"
        required_channels = ["coupling", "external_anchor", "perturbation", "history"]
        conditions = {
            **crystallization_conditions,
            **{f"viable_{key}": value for key, value in viable_conditions.items()},
        }

    missing_required = sorted(
        name for name in required_channels if not _observed(status[name])
        and status[name] != ChannelStatus.NOT_APPLICABLE.value
    )
    if missing_required:
        label = Annotation.INDETERMINATE
        rule_id = "R0_REQUIRED_CHANNEL_UNAVAILABLE"

    satisfied = sorted(name for name, value in conditions.items() if value)
    failed = sorted(name for name, value in conditions.items() if not value)
    confidence = (
        ConfidenceStatus.INDETERMINATE
        if label is Annotation.INDETERMINATE
        else ConfidenceStatus.SUPPORTED
    )
    record = {
        **dict(envelope),
        "label": label.value,
        "label_version": LABEL_VERSION,
        "rule_id": rule_id,
        "required_channels": required_channels,
        "satisfied_conditions": satisfied,
        "failed_conditions": failed,
        "unavailable_conditions": sorted(set(unavailable + missing_required)),
        "calibration_reference": calibration_reference,
        "evidence_window_start": inputs.evidence_window_start,
        "evidence_window_end": inputs.evidence_window_end,
        "confidence_status": confidence.value,
        "claim_boundary": {
            "complete_viability_regime_claimed": False,
            "consciousness_claimed": False,
            "deception_claimed": False,
            "interpretive_label_is_ground_truth": False,
        },
    }
    validate_artifact(record, "structural_label")
    return record
