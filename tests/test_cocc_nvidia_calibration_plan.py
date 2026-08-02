from hashlib import sha256
import json
from pathlib import Path


def _canonical_sha256(value):
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def test_nemotron_calibration_plan_is_frozen_and_bound():
    root = Path(__file__).resolve().parents[1]
    plan_path = root / "config" / "cocc_nemotron3_super_calibration_plan_v1.json"
    freeze_path = root / "config" / "cocc_nemotron3_super_calibration_plan_v1.freeze.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))

    assert plan["status"] == "FROZEN_BEFORE_CALIBRATION_ACQUISITION"
    assert plan["outcome_labels_allowed"] is False
    assert plan["dataset_binding"]["expected_session_count"] == 20
    assert plan["dataset_binding"]["required_perturbation_type"] == "clean_control"
    assert plan["provider_binding"]["requested_model"] == (
        "nvidia/nemotron-3-super-120b-a12b"
    )
    assert plan["provider_binding"]["preflight_token_logprobs_supported"] is True
    assert plan["provider_binding"]["immutable_weight_digest"] is None
    assert plan["generation_parameter_set"] == {
        "max_tokens": 512,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_logprobs": 5,
        "seed": 1337,
        "enable_thinking": False,
        "reasoning_budget": None,
        "max_attempts": 3,
        "retry_sleep_seconds": 5.0,
        "timeout_seconds": 600,
    }
    assert len(plan["calibrator_binding"]["script_sha256"]) == 64
    assert "PENDING" not in plan["calibrator_binding"]["script_sha256"]
    assert _canonical_sha256(plan) == freeze["canonical_plan_sha256"]
    assert sha256(plan_path.read_bytes()).hexdigest() == freeze["raw_file_sha256"]
    assert freeze["status"] == "FROZEN_BEFORE_CALIBRATION_ACQUISITION"
