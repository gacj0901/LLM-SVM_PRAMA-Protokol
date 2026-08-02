"""Paired epistemic-channel comparison without a canonical scalar B_o."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from aptadynamic_llm.artifact_schema import validate_artifact


METRICS = (
    "evidence_coverage",
    "verifier_relevant_omission",
    "precision",
    "calibration",
    "response_quality",
)


@dataclass(frozen=True)
class EpistemicChannelConfig:
    competence_tolerance: float
    minimum_coordinate_effect: float

    def __post_init__(self) -> None:
        if self.competence_tolerance < 0:
            raise ValueError("competence_tolerance must be non-negative")
        if self.minimum_coordinate_effect < 0:
            raise ValueError("minimum_coordinate_effect must be non-negative")


def _metric_block(value: Mapping[str, Any], name: str) -> float:
    result = float(value[name])
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def evaluate_epistemic_pair(
    *,
    envelope: Mapping[str, Any],
    task_id: str,
    pair_id: str,
    condition_id: str,
    reference_task_state_sha256: str,
    condition_task_state_sha256: str,
    competence_reference: float,
    competence_condition: float,
    reference_metrics: Mapping[str, Any],
    condition_metrics: Mapping[str, Any],
    config: EpistemicChannelConfig,
) -> dict[str, Any]:
    """Emit a vector effect only when paired task state is byte-identical."""

    task_state_matches = (
        reference_task_state_sha256.removeprefix("sha256:")
        == condition_task_state_sha256.removeprefix("sha256:")
    )
    reference = {name: _metric_block(reference_metrics, name) for name in METRICS}
    condition = {name: _metric_block(condition_metrics, name) for name in METRICS}
    effect_vector = {
        "evidence_coverage_shift": condition["evidence_coverage"]
        - reference["evidence_coverage"],
        "verifier_relevant_omission_shift": condition["verifier_relevant_omission"]
        - reference["verifier_relevant_omission"],
        "precision_shift": condition["precision"] - reference["precision"],
        "calibration_shift": condition["calibration"] - reference["calibration"],
        "response_quality_shift": condition["response_quality"]
        - reference["response_quality"],
    }
    competence_shift = float(competence_condition) - float(competence_reference)
    competence_preserved = abs(competence_shift) <= config.competence_tolerance
    channel_valid = task_state_matches and competence_preserved
    material_coordinates = sorted(
        name
        for name, value in effect_vector.items()
        if abs(value) >= config.minimum_coordinate_effect
    )
    record = {
        **dict(envelope),
        "task_id": task_id,
        "pair_id": pair_id,
        # Backend-only. model_payload.py forbids this field at the LLM boundary.
        "condition_id": condition_id,
        "task_state_sha256": reference_task_state_sha256.removeprefix("sha256:"),
        "task_state_matches": task_state_matches,
        "competence_reference": float(competence_reference),
        "competence_condition": float(competence_condition),
        "competence_shift": competence_shift,
        "competence_preserved": competence_preserved,
        "effect_vector": effect_vector,
        "material_coordinates": material_coordinates,
        "channel_modulation_detected": bool(channel_valid and material_coordinates),
        "channel_valid": channel_valid,
        "scalar_observability_gap_claimed": False,
    }
    validate_artifact(record, "epistemic_channel")
    return record
