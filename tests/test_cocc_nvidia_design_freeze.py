from hashlib import sha256
import json
from pathlib import Path

import pytest

from scripts.evaluate_break_the_chain_prama import (
    _canonical_sha256,
    _validate_design,
    _verify_run_binding,
)


DESIGN_NAME = "cocc_confirmatory_design_v4_nvidia_nemotron3_super.json"
FREEZE_NAME = "cocc_confirmatory_design_v4_nvidia_nemotron3_super.freeze.json"


def test_nvidia_v4_design_freeze_and_local_bindings():
    root = Path(__file__).resolve().parents[1]
    design_path = root / "config" / DESIGN_NAME
    freeze_path = root / "config" / FREEZE_NAME
    design = json.loads(design_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))

    _validate_design(design)
    plan = design["sampling_plan"]
    model = plan["model_version"]
    calibration = plan["projector_calibration"]
    assert model["provider"] == "nvidia_nim"
    assert model["provider_endpoint"] == "https://integrate.api.nvidia.com/v1"
    assert model["resolved_models"] == [
        "nvidia/nemotron-3-super-120b-a12b"
    ]
    assert model["model_blob_sha256"] is None
    assert calibration["artifact_sha256"] == (
        "b9ea07856fd5a8097cb37ea8b582d4c9bb404563022facca1e943c95a60bea4f"
    )
    assert plan["parameter_set"]["generation"]["enable_thinking"] is False
    assert _canonical_sha256(design) == freeze["canonical_design_sha256"]
    assert sha256(design_path.read_bytes()).hexdigest() == freeze[
        "raw_file_sha256"
    ]
    assert freeze["status"] == "FROZEN_BEFORE_CONFIRMATORY_ACQUISITION"

    artifact_path = root / calibration["artifact"]
    calibration_freeze = root / calibration["freeze"]
    if artifact_path.exists():
        assert sha256(artifact_path.read_bytes()).hexdigest() == calibration[
            "artifact_sha256"
        ]
    assert sha256(calibration_freeze.read_bytes()).hexdigest() == calibration[
        "freeze_sha256"
    ]


def _bound_run(design):
    plan = design["sampling_plan"]
    model = plan["model_version"]
    observation = plan["observation_interface_version"]
    kernel = plan["prama_kernel_version"]
    return {
        "dataset_sha256": plan["normalized_dataset_sha256"],
        "dataset_manifest_sha256": plan["normalized_manifest_sha256"],
        "provider": model["provider"],
        "provider_endpoint": model["provider_endpoint"],
        "model": plan["model"],
        "model_blob_sha256": None,
        "confirmatory_design_sha256": _canonical_sha256(design),
        "observation_interface": {
            "projector_request_schema": observation["projector_request_schema"],
            "runner_sha256": observation["runner_sha256"],
            "model_payload_sha256": observation["model_payload_sha256"],
        },
        "generation_parameter_set": plan["parameter_set"]["generation"],
        "provider_response_identity": {
            "resolved_models": model["resolved_models"],
            "system_fingerprints": model["system_fingerprints"],
        },
        "projector_kernel_identity": {
            "version": kernel["package_version"],
            "source_tree_sha256": kernel["source_tree_sha256"],
            "recertification_sha256": kernel["recertification_sha256"],
        },
        "projector_calibration_sha256": plan["projector_calibration"][
            "artifact_sha256"
        ],
    }


def test_nvidia_run_binding_checks_endpoint_identity_and_calibration(tmp_path):
    root = Path(__file__).resolve().parents[1]
    design = json.loads((root / "config" / DESIGN_NAME).read_text(encoding="utf-8"))
    manifest_path = tmp_path / "manifest.json"
    run = _bound_run(design)
    manifest_path.write_text(json.dumps(run), encoding="utf-8")
    result = _verify_run_binding(design, manifest_path)
    assert result["matches_frozen_design"] is True
    assert "provider_endpoint" in result["validated_fields"]
    assert "projector_calibration_sha256" in result["validated_fields"]

    run["provider_endpoint"] = "https://example.invalid/v1"
    manifest_path.write_text(json.dumps(run), encoding="utf-8")
    with pytest.raises(ValueError, match="provider_endpoint"):
        _verify_run_binding(design, manifest_path)

    run = _bound_run(design)
    run["projector_calibration_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(run), encoding="utf-8")
    with pytest.raises(ValueError, match="projector_calibration_sha256"):
        _verify_run_binding(design, manifest_path)
