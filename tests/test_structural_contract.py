from datetime import datetime, timezone

import pytest

from aptadynamic_llm.artifact_schema import (
    ChannelStatus,
    ArtifactValidationError,
    make_envelope,
    sha256_value,
    validate_artifact,
)
from aptadynamic_llm.epistemic_channel import (
    EpistemicChannelConfig,
    evaluate_epistemic_pair,
)
from aptadynamic_llm.external_anchor import AnchorUptakeConfig, evaluate_anchor_uptake
from aptadynamic_llm.model_payload import (
    ModelPayloadLeakageError,
    encode_task_only_payload,
    task_only_messages,
)
from aptadynamic_llm.perturbation_response import (
    PerturbationConfig,
    evaluate_perturbation_response,
)
from aptadynamic_llm.window_prama import validate_window_kernel_declaration


HASH_A = "a" * 64
HASH_B = "b" * 64


def envelope(artifact_type: str, status: str = "OBSERVED"):
    return make_envelope(
        artifact_type=artifact_type,
        study_id="study-1",
        session_id="session-1",
        producer="pytest",
        created_at=datetime.now(timezone.utc).isoformat(),
        source_sha256=HASH_A,
        config_sha256=HASH_B,
        partition="confirmatory",
        channel_status=status,
    )


def test_coupling_contract_checks_derived_excess():
    record = {
        **envelope("coupling_observation"),
        "turn_index": 1,
        "window_index": 0,
        "token_start": 0,
        "token_end": 4,
        "self_support": 1.0,
        "user_support": 0.5,
        "interaction": 0.0,
        "support_magnitude": 1.5,
        "omega_dep": 0.4,
        "expected_omega_dep": 0.1,
        "self_dependence_excess": 0.3,
        "filler_variance": 0.01,
        "eligible": True,
        "expectation_status": "observed",
    }
    validate_artifact(record, "coupling_observation")
    record["self_dependence_excess"] = 0.9
    with pytest.raises(ArtifactValidationError, match="must equal"):
        validate_artifact(record)


def test_model_payload_is_task_only_and_fail_closed():
    assert task_only_messages([{"role": "user", "content": "Suma 2 + 2."}]) == [
        {"role": "user", "content": "Suma 2 + 2."}
    ]
    encoded = encode_task_only_payload(
        {"model": "m", "prompt": "Suma 2 + 2.", "stream": False}
    )
    assert b"condition_id" not in encoded
    with pytest.raises(ModelPayloadLeakageError):
        task_only_messages([{"role": "system", "content": "hidden"}])
    with pytest.raises(ModelPayloadLeakageError):
        encode_task_only_payload(
            {"model": "m", "prompt": "Suma 2 + 2.", "condition_id": "pressure"}
        )
    with pytest.raises(ModelPayloadLeakageError):
        encode_task_only_payload(
            {
                "model": "m",
                "prompt": "Suma 2 + 2.",
                "options": {"prama_regime_state": "hidden"},
            }
        )
    with pytest.raises(ModelPayloadLeakageError):
        encode_task_only_payload(
            {"model": "m", "prompt": "Describe la interfaz PRAMA."}
        )


def test_external_anchor_requires_independence_and_measures_first_verified_uptake():
    record = evaluate_anchor_uptake(
        envelope=envelope("external_anchor_event"),
        anchor_id="test-7",
        anchor_type="unit_test",
        introduced_at_window=3,
        anchor_state="contradiction",
        severity=0.8,
        externally_verifiable=True,
        anchor_source_sha256=HASH_A,
        source_is_evaluated_trajectory=False,
        response_windows=[
            {"window_index": 3, "uptake_score": 0.9, "verifier_passed": False},
            {"window_index": 4, "uptake_score": 0.7, "verifier_passed": True},
        ],
        config=AnchorUptakeConfig(0.6, 2),
    )
    assert record["uptake_detected"] is True
    assert record["uptake_latency_windows"] == 1
    with pytest.raises(ValueError, match="evaluated trajectory"):
        evaluate_anchor_uptake(
            envelope=envelope("external_anchor_event"),
            anchor_id="bad",
            anchor_type="model_claim",
            introduced_at_window=0,
            anchor_state="novelty",
            severity=0.2,
            externally_verifiable=True,
            anchor_source_sha256=HASH_A,
            source_is_evaluated_trajectory=True,
            response_windows=[],
            config=AnchorUptakeConfig(0.5, 1),
        )


def test_perturbation_rules_distinguish_adaptation_and_counter_reaction():
    config = PerturbationConfig(3, 0.2, 0.1, 0.6, 0.15)
    adaptive = evaluate_perturbation_response(
        envelope=envelope("perturbation_response"),
        perturbation_id="p1",
        introduced_at_window=2,
        pre_self_dependence_excess=0.3,
        post_self_dependence_excess=0.2,
        pre_anchor_uptake=0.2,
        post_anchor_uptake=0.8,
        trajectory_changed=True,
        recovery_detected=True,
        recovery_latency_windows=1,
        anchor_externally_verified=True,
        config=config,
    )
    assert adaptive["response_class"] == "adaptive_integration"
    counter = evaluate_perturbation_response(
        envelope=envelope("perturbation_response"),
        perturbation_id="p2",
        introduced_at_window=2,
        pre_self_dependence_excess=0.1,
        post_self_dependence_excess=0.4,
        pre_anchor_uptake=0.8,
        post_anchor_uptake=0.2,
        trajectory_changed=False,
        recovery_detected=False,
        recovery_latency_windows=None,
        anchor_externally_verified=True,
        config=config,
    )
    assert counter["response_class"] == "counterfactual_rejection"


def test_epistemic_channel_is_vector_and_requires_matched_task_state():
    metrics = {
        "evidence_coverage": 0.8,
        "verifier_relevant_omission": 0.1,
        "precision": 0.9,
        "calibration": 0.8,
        "response_quality": 0.9,
    }
    condition = dict(metrics, evidence_coverage=0.5, verifier_relevant_omission=0.4)
    record = evaluate_epistemic_pair(
        envelope=envelope("epistemic_channel"),
        task_id="t1",
        pair_id="pair-1",
        condition_id="pressure",
        reference_task_state_sha256=HASH_A,
        condition_task_state_sha256=HASH_A,
        competence_reference=0.9,
        competence_condition=0.88,
        reference_metrics=metrics,
        condition_metrics=condition,
        config=EpistemicChannelConfig(0.05, 0.1),
    )
    assert record["channel_valid"] is True
    assert record["channel_modulation_detected"] is True
    assert record["scalar_observability_gap_claimed"] is False


def test_window_prama_declaration_requires_exact_identity_and_hashes():
    config = {"tau": 16, "g_smooth": 8}
    input_transform = {
        "name": "signed_unit_affine_v1",
        "source_min": -1.0,
        "source_max": 1.0,
        "target_min": 0.0,
        "target_max": 1.0,
    }
    declaration = {
        "kernel_identity": {
            "package": "prama-protokol",
            "version": "0.3.0",
            "source_tree_sha256": HASH_B,
            "commit": "abcdef1",
            "kernel_api": "project_v3",
            "config_sha256": sha256_value(
                {"kernel_config": config, "input_transform": input_transform}
            ),
            "recertification_sha256": HASH_A,
            "bin_scale": "window",
        },
        "kernel_config": config,
        "input_transform": input_transform,
        "column_map": {
            "delta": "delta",
            "xi": "xi",
            "accumulated_excess": "A",
            "capacity": "lambda",
            "theta": "theta",
            "balance": "M",
            "trend": "G",
            "valid": "valid",
        },
    }
    identity, parsed, transform, _ = validate_window_kernel_declaration(
        declaration,
        actual_version="0.3.0",
        actual_source_tree_sha256=HASH_B,
        actual_commit="abcdef1",
        recertification_sha256=HASH_A,
    )
    assert identity.bin_scale == "window"
    assert parsed == config
    assert transform == input_transform
    with pytest.raises(ValueError, match="identity differs"):
        validate_window_kernel_declaration(
            declaration,
            actual_version="0.3.1",
            actual_source_tree_sha256=HASH_B,
            actual_commit="abcdef1",
            recertification_sha256=HASH_A,
        )
