from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from scripts import analyze_ep1
from scripts.collect_ep1_nvidia import collect, parse_args


ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "config" / "ep1_nvidia_replication_v1.json"
MODEL = "nvidia/nemotron-3-super-120b-a12b"


def test_dry_collection_resume_keeps_canonical_binding_stable(tmp_path: Path) -> None:
    out = tmp_path / "pilot"
    first = collect(
        parse_args(
            [
                "--design",
                str(DESIGN),
                "--mode",
                "pilot",
                "--model",
                MODEL,
                "--out",
                str(out),
                "--n",
                "2",
                "--dry-run",
            ]
        )
    )
    second = collect(
        parse_args(
            [
                "--design",
                str(DESIGN),
                "--mode",
                "pilot",
                "--model",
                MODEL,
                "--out",
                str(out),
                "--n",
                "2",
                "--dry-run",
                "--resume",
            ]
        )
    )
    assert first["created_at_utc"] == second["created_at_utc"]
    assert first["collection_content_sha256"] == second["collection_content_sha256"]
    assert second["completed_n"] == 2


def _replication_fixture(tmp_path: Path) -> tuple[Path, Path]:
    design_hash = sha256(DESIGN.read_bytes()).hexdigest()
    freeze = {
        "schema": "LLM-SVM-E-P1-NVIDIA-model-freeze/1",
        "status": "CONFIRMATORY_FROZEN",
        "study_id": "E-P1-NVIDIA-R1",
        "design": str(DESIGN),
        "design_sha256": design_hash,
        "model": MODEL,
        "provider": "nvidia_nim",
        "provider_endpoint": "https://integrate.api.nvidia.com/v1",
        "prompt_suite_sha256": "65080c2e2c641dce0b9eb0f1fdbb096c7a5a61881da882ea326345f7727e9b90",
        "selected_max_tokens": 256,
        "sampling": {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_logprobs": 5,
            "base_seed": 1337,
            "seed_per_index": True,
            "enable_thinking": False,
            "reasoning_effort": None,
            "stream": False,
        },
    }
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    freeze_hash = sha256(freeze_path.read_bytes()).hexdigest()
    collection = tmp_path / "collection"
    collection.mkdir()
    manifest = {
        "schema": "LLM-SVM-E-P1-NVIDIA-collection/1",
        "study_id": "E-P1-NVIDIA-R1",
        "mode": "confirmatory",
        "provider": "nvidia_nim",
        "provider_endpoint": "https://integrate.api.nvidia.com/v1",
        "model": MODEL,
        "resolved_models": [MODEL],
        "design_sha256": design_hash,
        "model_freeze_sha256": freeze_hash,
        "prompt_suite_sha256": freeze["prompt_suite_sha256"],
        "temperature": 0.7,
        "top_p": 0.9,
        "top_logprobs": 5,
        "seed": 1337,
        "seed_per_index": True,
        "max_tokens": 256,
        "enable_thinking": False,
        "reasoning_effort": None,
        "stream": False,
        "n": 400,
        "completed_n": 400,
        "complete": True,
        "collection_content_sha256": "a" * 64,
    }
    (collection / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return collection, freeze_path


def test_nvidia_replication_manifest_is_bound_to_model_freeze(tmp_path: Path) -> None:
    collection, freeze = _replication_fixture(tmp_path)
    manifest, context = analyze_ep1._validate_nvidia_replication_manifest(
        collection, freeze
    )
    assert manifest["completed_n"] == 400
    assert context["expected_model"] == MODEL
    assert context["program"] == "E-P1-NVIDIA-R1"

    path = collection / "manifest.json"
    corrupted = json.loads(path.read_text(encoding="utf-8"))
    corrupted["max_tokens"] = 512
    path.write_text(json.dumps(corrupted), encoding="utf-8")
    with pytest.raises(ValueError, match="violates its model freeze"):
        analyze_ep1._validate_nvidia_replication_manifest(collection, freeze)
