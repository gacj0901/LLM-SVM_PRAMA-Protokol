from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import scripts.preflight_nvidia_nemotron as preflight_module
from scripts.preflight_nvidia_nemotron import run_preflight
from scripts.run_break_the_chain_prama_eval_nvidia import (
    NVIDIA_BASE_URL,
    NVIDIA_MODEL,
    NVIDIA_PROVIDER,
    _call_backend,
    parse_args,
    projection_input_from_raw,
    run,
)


def test_nvidia_parser_freezes_endpoint_model_and_sampling(tmp_path):
    fixture = Path(__file__).parent / "fixtures" / "cocc_smoke.jsonl"
    args = parse_args(
        [
            "--dataset",
            str(fixture),
            "--output-dir",
            str(tmp_path / "run"),
            "--dry-run",
            "--queue-only",
        ]
    )
    assert args.provider == NVIDIA_PROVIDER
    assert args.model == NVIDIA_MODEL
    assert args.base_url == NVIDIA_BASE_URL
    assert args.temperature == 1.0
    assert args.top_p == 0.95
    assert args.enable_thinking is False


def test_nvidia_parser_rejects_endpoint_or_reasoning_drift(tmp_path):
    fixture = Path(__file__).parent / "fixtures" / "cocc_smoke.jsonl"
    common = [
        "--dataset",
        str(fixture),
        "--output-dir",
        str(tmp_path / "run"),
        "--dry-run",
        "--queue-only",
    ]
    with pytest.raises(SystemExit):
        parse_args(common + ["--base-url", "https://example.invalid/v1"])
    with pytest.raises(SystemExit):
        parse_args(common + ["--reasoning-budget", "8"])
    with pytest.raises(SystemExit):
        parse_args(common + ["--enable-thinking"])


def test_nvidia_dry_run_preserves_model_isolation_and_manifest(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-not-real")
    fixture = Path(__file__).parent / "fixtures" / "cocc_smoke.jsonl"
    output = tmp_path / "run"
    args = parse_args(
        [
            "--dataset",
            str(fixture),
            "--output-dir",
            str(output),
            "--dry-run",
            "--queue-only",
        ]
    )
    manifest = run(args)
    assert manifest["provider"] == NVIDIA_PROVIDER
    assert manifest["provider_endpoint"] == NVIDIA_BASE_URL
    assert manifest["api_key_environment_variable"] == "NVIDIA_API_KEY"
    assert "nvapi-test-not-real" not in json.dumps(manifest)
    raw_path = next((output / "sessions").glob("*/raw.json"))
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    request = projection_input_from_raw(raw)
    serialized = json.dumps(request).casefold()
    assert "nvidia_api_key" not in serialized
    assert "assistant_message" not in serialized


def test_nvidia_backend_requires_key_before_constructing_client(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    item = SimpleNamespace(prompt="Return OK")
    args = SimpleNamespace(
        dry_run=False,
        provider=NVIDIA_PROVIDER,
        model=NVIDIA_MODEL,
    )
    with pytest.raises(RuntimeError, match="NVIDIA_API_KEY is not set"):
        _call_backend(item, args)


def test_preflight_is_local_only_without_execute(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-not-real")
    report = run_preflight(False, timeout=1, top_logprobs=5)
    assert report["api_key_present"] is True
    assert report["remote_call_executed"] is False
    assert "nvapi-test-not-real" not in json.dumps(report)


def test_nvidia_backend_uses_explicit_key_endpoint_and_logprobs(monkeypatch):
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured["request"] = kwargs
            candidate = SimpleNamespace(logprob=-0.1)
            entry = SimpleNamespace(
                token="OK", logprob=-0.1, top_logprobs=[candidate]
            )
            choice = SimpleNamespace(
                logprobs=SimpleNamespace(content=[entry]),
                message=SimpleNamespace(content="OK"),
                finish_reason="stop",
            )
            return SimpleNamespace(
                choices=[choice],
                model=NVIDIA_MODEL,
                id="response-id",
                system_fingerprint="fingerprint",
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-not-real")
    item = SimpleNamespace(prompt="Return OK")
    args = SimpleNamespace(
        dry_run=False,
        provider=NVIDIA_PROVIDER,
        model=NVIDIA_MODEL,
        base_url=NVIDIA_BASE_URL,
        timeout=30,
        temperature=1.0,
        top_p=0.95,
        max_tokens=8,
        seed=1337,
        top_logprobs=5,
        enable_thinking=False,
        reasoning_budget=None,
    )
    turn, resolved = _call_backend(item, args)
    assert captured["client"] == {
        "api_key": "nvapi-test-not-real",
        "base_url": NVIDIA_BASE_URL,
        "timeout": 30,
    }
    assert captured["request"]["logprobs"] is True
    assert captured["request"]["top_logprobs"] == 5
    assert captured["request"]["extra_body"] == {
        "chat_template_kwargs": {
            "enable_thinking": False,
            "force_nonempty_content": True,
        }
    }
    assert resolved == NVIDIA_MODEL
    assert turn["token_count"] == 1
    assert turn["system_fingerprint"] == "fingerprint"


def test_preflight_reports_401_without_traceback(monkeypatch, capsys):
    class Unauthorized(Exception):
        status_code = 401

    def fail(*_args, **_kwargs):
        raise Unauthorized("secret provider detail")

    monkeypatch.setattr(preflight_module, "run_preflight", fail)
    assert preflight_module.main(["--execute"]) == 1
    output = capsys.readouterr().out
    assert "401 Unauthorized" in output
    assert "Build/Endpoints key" in output
    assert "Traceback" not in output
