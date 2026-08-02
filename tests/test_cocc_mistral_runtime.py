import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.preflight_nvidia_mistral_medium import run_preflight
from scripts.run_break_the_chain_prama_eval_nvidia_mistral import (
    NVIDIA_BASE_URL,
    NVIDIA_MODEL,
    NVIDIA_PROVIDER,
    _call_backend,
    parse_args,
    run,
)


def test_mistral_runner_has_frozen_model_profile(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    args = parse_args(
        [
            "--dataset",
            str(dataset),
            "--output-dir",
            str(tmp_path / "out"),
            "--dry-run",
        ]
    )
    assert args.model == "mistralai/mistral-medium-3.5-128b"
    assert args.temperature == 0.7
    assert args.top_p == 1.0
    assert args.reasoning_effort == "high"


def test_mistral_calibration_only_mode_accepts_no_holdout(tmp_path):
    dataset = tmp_path / "calibration.jsonl"
    row = {
        "benchmark_name": "chain_of_code_collapse",
        "benchmark_alias": "break_the_chain_code_generation",
        "problem_id": "p1",
        "item_id": "p1:clean_control",
        "perturbation_type": "clean_control",
        "perturbed_prompt": "Write a Python solution.",
        "verifier_ref": "test:p1",
        "split": "calibration",
    }
    dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")
    args = parse_args(
        [
            "--dataset",
            str(dataset),
            "--output-dir",
            str(tmp_path / "out"),
            "--dry-run",
            "--queue-only",
            "--calibration-only",
        ]
    )
    manifest = run(args)
    assert manifest["session_count"] == 1
    assert manifest["verified_count"] == 0


def test_mistral_holdout_only_skips_calibration_and_preserves_source_ids(tmp_path):
    dataset = tmp_path / "full.jsonl"
    rows = [
        {
            "benchmark_name": "chain_of_code_collapse",
            "benchmark_alias": "break_the_chain_code_generation",
            "problem_id": f"p{index}",
            "item_id": f"p{index}:{condition}",
            "perturbation_type": condition,
            "perturbed_prompt": "Write a Python solution.",
            "verifier_ref": f"test:p{index}",
            "split": split,
        }
        for index, condition, split in (
            (0, "clean_control", "calibration"),
            (1, "clean_control", "test"),
            (2, "negation_objective", "test"),
        )
    ]
    dataset.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    manifest = run(
        parse_args(
            [
                "--dataset",
                str(dataset),
                "--output-dir",
                str(tmp_path / "out"),
                "--dry-run",
                "--queue-only",
                "--holdout-only",
            ]
        )
    )
    raw_paths = sorted((tmp_path / "out" / "sessions").rglob("raw.json"))
    assert manifest["acquisition_scope"] == "holdout_only"
    assert manifest["session_count"] == 2
    assert [path.parent.name.split("-")[1] for path in raw_paths] == ["00001", "00002"]


def test_mistral_partial_resume_preserves_existing_raw_and_acquires_missing(tmp_path):
    dataset = tmp_path / "calibration.jsonl"
    rows = [
        {
            "benchmark_name": "chain_of_code_collapse",
            "benchmark_alias": "break_the_chain_code_generation",
            "problem_id": f"p{index}",
            "item_id": f"p{index}:clean_control",
            "perturbation_type": "clean_control",
            "perturbed_prompt": "Write a Python solution.",
            "verifier_ref": f"test:p{index}",
            "split": "calibration",
        }
        for index in (1, 2)
    ]
    dataset.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    output_dir = tmp_path / "out"
    common = [
        "--dataset",
        str(dataset),
        "--output-dir",
        str(output_dir),
        "--dry-run",
        "--queue-only",
        "--calibration-only",
    ]
    first_manifest = run(parse_args([*common, "--n", "1"]))
    assert first_manifest["session_count"] == 1
    first_raw_path = next((output_dir / "sessions").rglob("raw.json"))
    first_raw_bytes = first_raw_path.read_bytes()

    resumed_manifest = run(parse_args([*common, "--n", "2", "--resume"]))

    assert resumed_manifest["session_count"] == 2
    assert first_raw_path.read_bytes() == first_raw_bytes
    assert len(list((output_dir / "sessions").rglob("raw.json"))) == 2
    assert len(list((output_dir / "verification" / "requests").glob("*.json"))) == 2
    assert len(list((output_dir / "projection" / "requests").glob("*.json"))) == 2


def test_mistral_runner_rejects_profile_drift(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--dataset",
                str(dataset),
                "--output-dir",
                str(tmp_path / "out"),
                "--temperature",
                "1.0",
            ]
        )


def test_mistral_runner_accepts_no_reasoning_profile(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    args = parse_args(
        [
            "--dataset",
            str(dataset),
            "--output-dir",
            str(tmp_path / "out"),
            "--reasoning-effort",
            "none",
        ]
    )
    assert args.reasoning_effort == "none"


def test_mistral_runner_accepts_frozen_dynamic_observer_without_calibration(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    digest = "a" * 64
    args = parse_args(
        [
            "--dataset",
            str(dataset),
            "--output-dir",
            str(tmp_path / "out"),
            "--projector-command",
            "projector.exe",
            "--projector-observer-sha256",
            digest,
        ]
    )
    assert args.projector_observer_sha256 == digest
    assert args.projector_calibration_sha256 == ""


def test_mistral_runner_rejects_observer_and_calibration_together(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--dataset",
                str(dataset),
                "--output-dir",
                str(tmp_path / "out"),
                "--projector-command",
                "projector.exe",
                "--projector-observer-sha256",
                "a" * 64,
                "--projector-calibration-sha256",
                "b" * 64,
            ]
        )


def test_mistral_preflight_is_local_without_execute(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-not-real")
    report = run_preflight(False, timeout=1, top_logprobs=5)
    assert report["requested_model"] == NVIDIA_MODEL
    assert report["api_key_present"] is True
    assert report["remote_call_executed"] is False
    assert "nvapi-test-not-real" not in json.dumps(report)


def test_mistral_backend_uses_reasoning_profile_and_logprobs(monkeypatch):
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured["request"] = kwargs
            alternatives = [
                SimpleNamespace(logprob=-0.1),
                SimpleNamespace(logprob=-2.1),
            ]
            entry = SimpleNamespace(
                token="OK", logprob=-0.1, top_logprobs=alternatives
            )
            choice = SimpleNamespace(
                logprobs=SimpleNamespace(content=[entry]),
                delta=SimpleNamespace(content="OK"),
                finish_reason="stop",
            )
            return [
                SimpleNamespace(
                    choices=[choice],
                    model=NVIDIA_MODEL,
                    id="response-id",
                    system_fingerprint="fingerprint",
                )
            ]

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.chat = SimpleNamespace(completions=FakeCompletions())

    import sys

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-not-real")
    args = SimpleNamespace(
        dry_run=False,
        provider=NVIDIA_PROVIDER,
        model=NVIDIA_MODEL,
        base_url=NVIDIA_BASE_URL,
        timeout=30,
        temperature=0.7,
        top_p=1.0,
        max_tokens=32,
        seed=1337,
        top_logprobs=5,
        reasoning_effort="high",
    )
    turn, resolved = _call_backend(SimpleNamespace(prompt="Return OK"), args)
    assert captured["request"]["extra_body"] == {"reasoning_effort": "high"}
    assert captured["request"]["logprobs"] is True
    assert captured["request"]["top_logprobs"] == 5
    assert captured["request"]["stream"] is True
    assert captured["client"]["max_retries"] == 0
    assert resolved == NVIDIA_MODEL
    assert turn["token_count"] == 1
