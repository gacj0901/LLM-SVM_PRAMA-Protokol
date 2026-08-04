from datetime import datetime, timezone

from aptadynamic_llm.artifact_schema import make_envelope
from aptadynamic_llm.structural_labels import (
    StructuralLabelInput,
    classify_structural_state,
)


HASH_A = "a" * 64
HASH_B = "b" * 64


def envelope():
    return make_envelope(
        artifact_type="structural_label",
        study_id="study-1",
        session_id="session-1",
        producer="pytest",
        created_at=datetime.now(timezone.utc).isoformat(),
        source_sha256=HASH_A,
        config_sha256=HASH_B,
        partition="confirmatory",
        channel_status="OBSERVED",
    )


def values():
    return {
        "coupling_state": "SELF_DOMINANT_CANDIDATE",
        "external_coupling_state": "ANCHOR_NOT_INTEGRATED",
        "perturbation_response_state": "RIGID",
        "epistemic_channel_state": "NO_EFFECT",
        "structural_history_state": "DEGRADING_PARTIAL",
        "coupling_status": "OBSERVED",
        "external_status": "OBSERVED",
        "perturbation_status": "OBSERVED",
        "epistemic_status": "NOT_APPLICABLE",
        "history_status": "OBSERVED",
        "self_dependence_persistence_windows": 6,
        "minimum_persistence_windows": 4,
        "continued_operation": True,
        "relevant_external_friction": True,
        "evidence_window_start": 0,
        "evidence_window_end": 8,
    }


def test_secondary_label_precedence_and_observer_binding():
    inputs = values()
    record = classify_structural_state(
        envelope=envelope(),
        inputs=StructuralLabelInput(**inputs),
        calibration_reference="calibration:abc",
        structural_observation_reference=HASH_A,
    )
    assert record["label"] == "monitor.CRYSTALLIZATION_CANDIDATE"
    assert record["rule_id"] == "R1_CRYSTALLIZATION_CANDIDATE"
    assert record["annotation_role"] == "SECONDARY_INTERPRETIVE"
    assert record["structural_observation_reference"] == HASH_A

    inputs["external_status"] = "UNAVAILABLE"
    unavailable = classify_structural_state(
        envelope=envelope(),
        inputs=StructuralLabelInput(**inputs),
        calibration_reference="calibration:abc",
        structural_observation_reference=HASH_A,
    )
    assert unavailable["label"] == "monitor.INDETERMINATE"
    assert unavailable["confidence_status"] == "INDETERMINATE"
