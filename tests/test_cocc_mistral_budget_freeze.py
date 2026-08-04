from hashlib import sha256
import json
from pathlib import Path

from aptadynamic_llm.repository_artifact import matches_frozen_sha256


ROOT = Path(__file__).resolve().parents[1]


def _canonical_sha256(value):
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def test_mistral_budget_plan_is_frozen_and_calibration_only():
    plan_path = (
        ROOT / "config" / "cocc_mistral_medium_3_5_budget_calibration_plan_v2.json"
    )
    freeze_path = plan_path.with_suffix(".freeze.json")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))

    assert plan["status"] == "FROZEN_BEFORE_BUDGET_CALIBRATION"
    assert plan["outcome_labels_allowed"] is False
    assert plan["holdout_access_allowed"] is False
    assert plan["provider_binding"]["requested_model"] == (
        "mistralai/mistral-medium-3.5-128b"
    )
    assert plan["provider_binding"]["resolved_model"] == (
        "mistralai/mistral-medium-3.5-128b"
    )
    assert plan["provider_binding"]["token_logprobs_supported"] is True
    dataset = plan["dataset_binding"]
    assert dataset["partition"] == "calibration"
    assert dataset["required_perturbation_type"] == "clean_control"
    assert dataset["expected_session_count"] == 20
    assert dataset["holdout_rows_present"] is False
    assert dataset["outcome_labels_present"] is False
    assert plan["frozen_candidate_rule"]["candidate_max_tokens"] == [
        512,
        1024,
        2048,
        4096,
    ]
    assert plan["frozen_candidate_rule"]["maximum_observed_length_rate"] == 0.05

    for name, digest_field in (
        ("normalized_dataset", "normalized_dataset_sha256"),
        ("normalized_manifest", "normalized_manifest_sha256"),
    ):
        path = ROOT / dataset[name]
        assert matches_frozen_sha256(path, dataset[digest_field])
    provider = plan["provider_binding"]
    preflight = ROOT / provider["preflight_artifact"]
    assert sha256(preflight.read_bytes()).hexdigest() == provider[
        "preflight_artifact_sha256"
    ]

    assert _canonical_sha256(plan) == freeze["canonical_plan_sha256"]
    assert sha256(plan_path.read_bytes()).hexdigest() == freeze["raw_file_sha256"]
    assert freeze["holdout_accessed"] is False
    assert freeze["outcome_labels_accessed"] is False
    assert freeze["confirmatory_design_frozen"] is False
    assert plan["pre_acquisition_amendment"]["provider_calls_before_amendment"] == 0
    assert plan["pre_acquisition_amendment"]["scientific_rule_changed"] is False
    assert plan["implementation"]["calibration_only_runner_flag_required"] is True


def test_mistral_timeout_amendment_preserves_scientific_design():
    plan_path = (
        ROOT / "config" / "cocc_mistral_medium_3_5_budget_calibration_plan_v3.json"
    )
    freeze_path = plan_path.with_suffix(".freeze.json")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    amendment = plan["mid_calibration_operational_amendment"]

    assert plan["status"] == "FROZEN_MID_CALIBRATION_OPERATIONAL_AMENDMENT"
    assert amendment["completed_session_count_before_amendment"] == 3
    assert amendment["old_timeout_seconds"] == 600
    assert amendment["new_timeout_seconds"] == 120
    assert amendment["resume_required"] is True
    assert amendment["completed_sessions_may_be_replaced"] is False
    assert amendment["model_payload_changed"] is False
    assert amendment["generation_parameters_changed"] is False
    assert amendment["scientific_rule_changed"] is False
    assert plan["generation_parameter_set_except_budget"]["timeout_seconds"] == 120
    assert _canonical_sha256(plan) == freeze["canonical_plan_sha256"]
    assert sha256(plan_path.read_bytes()).hexdigest() == freeze["raw_file_sha256"]


def test_mistral_partial_resume_amendment_preserves_completed_acquisitions():
    plan_path = (
        ROOT / "config" / "cocc_mistral_medium_3_5_budget_calibration_plan_v4.json"
    )
    freeze_path = plan_path.with_suffix(".freeze.json")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    amendment = plan["resume_implementation_amendment"]

    assert amendment["failure_detected_before_new_provider_call"] is True
    assert amendment["completed_session_count_preserved"] == 3
    assert amendment["completed_sessions_may_be_replaced"] is False
    assert amendment["generation_parameters_changed"] is False
    assert amendment["budget_selection_rule_changed"] is False
    assert amendment["scientific_rule_changed"] is False
    assert amendment["old_runner_sha256"] == (
        "f02a6292c4aa243a2af8c9a0e3c5fb2aa484fbcab2e2afff3ad0882b47fb509f"
    )
    assert amendment["new_runner_sha256"] == (
        "fe46c6820cc32378e4264510a6184501ed7958324ebccba1b5505b1fa02f6148"
    )
    assert _canonical_sha256(plan) == freeze["canonical_plan_sha256"]
    assert sha256(plan_path.read_bytes()).hexdigest() == freeze["raw_file_sha256"]


def test_mistral_retry_control_amendment_freezes_next_candidate():
    plan_path = (
        ROOT / "config" / "cocc_mistral_medium_3_5_budget_calibration_plan_v5.json"
    )
    freeze_path = plan_path.with_suffix(".freeze.json")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    completed = plan["completed_512_candidate"]
    amendment = plan["retry_control_amendment"]

    assert completed["session_count"] == 20
    assert completed["observed_length_rate"] == 1.0
    assert completed["candidate_verdict"] == "REJECTED_GENERATION_CAP_INSUFFICIENT"
    assert completed["next_candidate_max_tokens"] == 1024
    assert amendment["sdk_internal_retries_after"] == 0
    assert amendment["explicit_outer_max_attempts_unchanged"] == 3
    assert amendment["generation_parameters_changed"] is False
    assert amendment["scientific_rule_changed"] is False
    # The SHA records the runner used at freeze time. The live runner may evolve;
    # treating its current bytes as part of this historical freeze would silently
    # rewrite provenance whenever runtime hardening is added.
    assert len(amendment["new_runner_sha256"]) == 64
    assert all(character in "0123456789abcdef" for character in amendment["new_runner_sha256"])
    assert _canonical_sha256(plan) == freeze["canonical_plan_sha256"]
    assert sha256(plan_path.read_bytes()).hexdigest() == freeze["raw_file_sha256"]
