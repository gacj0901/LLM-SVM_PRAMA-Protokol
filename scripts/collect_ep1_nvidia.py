#!/usr/bin/env python
"""Collect resumable E-P1 replication sessions from NVIDIA NIM.

The evaluated model receives one task prompt and frozen generation parameters.
PRAMA state, experiment labels, timing, and outcome interpretation are attached
only after the provider response has completed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter, sleep
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for candidate in (REPO_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from aptadynamic_llm.artifact_schema import (  # noqa: E402
    ChannelStatus,
    make_envelope,
    sha256_value,
    write_jsonl_atomic,
)
from aptadynamic_llm.model_payload import task_only_messages  # noqa: E402
from scripts.collect_ollama import load_prompts  # noqa: E402


DEFAULT_DESIGN = REPO_ROOT / "config" / "ep1_nvidia_replication_v1.json"
KEY_ENV = "NVIDIA_API_KEY"


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError("NVIDIA response contains non-finite token logprobs")
    return result


def _normalized_entropy(logprobs: list[float]) -> float:
    if len(logprobs) < 2:
        return 0.0
    maximum = max(logprobs)
    weights = [math.exp(value - maximum) for value in logprobs]
    total = sum(weights)
    probabilities = [weight / total for weight in weights]
    entropy = -sum(p * math.log(p + 1e-15) for p in probabilities)
    return max(0.0, min(1.0, entropy / math.log(len(probabilities))))


def _token_rows(content: Iterable[Any], top_logprobs: int) -> list[dict[str, Any]]:
    tokens: list[dict[str, Any]] = []
    for entry in content:
        chosen = _finite(entry.logprob)
        candidates = [_finite(item.logprob) for item in list(entry.top_logprobs or [])]
        if chosen not in candidates:
            candidates.append(chosen)
        candidates = sorted(candidates, reverse=True)[:top_logprobs]
        tokens.append(
            {
                "token": str(entry.token),
                "bytes": [],
                "top1_logprob": chosen,
                "top_logprobs": candidates,
                "gap": candidates[0] - candidates[1] if len(candidates) > 1 else 0.0,
                "entropy": _normalized_entropy(candidates),
            }
        )
    return tokens


def _nonstream_turn(response: Any, prompt: str, top_logprobs: int) -> tuple[dict[str, Any], str]:
    choice = response.choices[0]
    content = list(getattr(getattr(choice, "logprobs", None), "content", None) or [])
    tokens = _token_rows(content, top_logprobs)
    if not tokens:
        raise RuntimeError("NVIDIA returned no usable token logprobs")
    return (
        {
            "turn_index": 0,
            "user_message": prompt,
            "assistant_message": str(choice.message.content or ""),
            "finish_reason": str(choice.finish_reason or ""),
            "token_count": len(tokens),
            "tokens": tokens,
            "provider_response_id": str(getattr(response, "id", "") or ""),
            "system_fingerprint": str(getattr(response, "system_fingerprint", "") or ""),
        },
        str(getattr(response, "model", "") or ""),
    )


def _stream_turn(
    stream: Iterable[Any], prompt: str, requested_model: str, top_logprobs: int
) -> tuple[dict[str, Any], str]:
    message_parts: list[str] = []
    tokens: list[dict[str, Any]] = []
    finish_reason = ""
    response_id = ""
    fingerprint = ""
    resolved_model = requested_model
    next_progress = 512
    for chunk in stream:
        response_id = str(getattr(chunk, "id", "") or response_id)
        fingerprint = str(getattr(chunk, "system_fingerprint", "") or fingerprint)
        resolved_model = str(getattr(chunk, "model", "") or resolved_model)
        choices = list(getattr(chunk, "choices", None) or [])
        if not choices:
            continue
        choice = choices[0]
        delta = getattr(choice, "delta", None)
        piece = getattr(delta, "content", None)
        if piece:
            message_parts.append(str(piece))
        entries = list(getattr(getattr(choice, "logprobs", None), "content", None) or [])
        tokens.extend(_token_rows(entries, top_logprobs))
        if len(tokens) >= next_progress:
            print(f"  stream progress: {len(tokens)} tokens", flush=True)
            next_progress = ((len(tokens) // 512) + 1) * 512
        if choice.finish_reason:
            finish_reason = str(choice.finish_reason)
    if not tokens:
        raise RuntimeError("NVIDIA streaming response contained no usable token logprobs")
    return (
        {
            "turn_index": 0,
            "user_message": prompt,
            "assistant_message": "".join(message_parts),
            "finish_reason": finish_reason,
            "token_count": len(tokens),
            "tokens": tokens,
            "provider_response_id": response_id,
            "system_fingerprint": fingerprint,
        },
        resolved_model,
    )


def load_design(path: Path) -> dict[str, Any]:
    design = json.loads(path.read_text(encoding="utf-8"))
    if design.get("schema") != "LLM-SVM-E-P1-NVIDIA-replication-design/1":
        raise ValueError("unsupported E-P1 NVIDIA design schema")
    prompt_path = REPO_ROOT / str(design["prompt_suite"])
    if file_sha256(prompt_path) != design.get("prompt_suite_sha256"):
        raise ValueError("prompt suite does not match the replication design")
    return design


def model_profile(design: dict[str, Any], model: str) -> dict[str, Any]:
    matches = [row for row in design.get("models") or [] if row.get("model") == model]
    if len(matches) != 1:
        raise ValueError(f"model is not uniquely declared by the design: {model!r}")
    return matches[0]


def _load_freeze(path: Path, design_sha256: str, model: str) -> dict[str, Any]:
    freeze = json.loads(path.read_text(encoding="utf-8"))
    if freeze.get("schema") != "LLM-SVM-E-P1-NVIDIA-model-freeze/1":
        raise ValueError("unsupported model freeze schema")
    expected = {
        "design_sha256": design_sha256,
        "model": model,
        "status": "CONFIRMATORY_FROZEN",
    }
    mismatch = {
        key: {"expected": value, "observed": freeze.get(key)}
        for key, value in expected.items()
        if freeze.get(key) != value
    }
    if mismatch:
        raise ValueError(f"model freeze mismatch: {mismatch}")
    return freeze


def _identity(
    *,
    design_sha256: str,
    mode: str,
    model: str,
    prompt_sha256: str,
    seed: int,
    max_tokens: int,
    profile: dict[str, Any],
    freeze_sha256: str | None,
) -> dict[str, Any]:
    return {
        "design_sha256": design_sha256,
        "freeze_sha256": freeze_sha256,
        "mode": mode,
        "model": model,
        "prompt_sha256": prompt_sha256,
        "seed": seed,
        "max_tokens": max_tokens,
        "temperature": profile["temperature"],
        "top_p": profile["top_p"],
        "top_logprobs": profile["top_logprobs"],
        "enable_thinking": profile.get("enable_thinking"),
        "reasoning_effort": profile.get("reasoning_effort"),
        "stream": bool(profile.get("stream")),
    }


def _dry_turn(prompt: str, model: str, max_tokens: int, top_logprobs: int) -> tuple[dict[str, Any], str]:
    count = min(max_tokens, 64)
    tokens = []
    for index in range(count):
        chosen = -0.2 - 0.01 * (index % 7)
        candidates = [chosen, chosen - 1.0][:top_logprobs]
        tokens.append(
            {
                "token": f"dry{index}",
                "bytes": [],
                "top1_logprob": chosen,
                "top_logprobs": candidates,
                "gap": 1.0,
                "entropy": _normalized_entropy(candidates),
            }
        )
    return (
        {
            "turn_index": 0,
            "user_message": prompt,
            "assistant_message": "DRY RUN: no provider request executed.",
            "finish_reason": "stop",
            "token_count": len(tokens),
            "tokens": tokens,
            "provider_response_id": "",
            "system_fingerprint": "",
        },
        model,
    )


def _call(
    *,
    prompt: str,
    model: str,
    profile: dict[str, Any],
    max_tokens: int,
    seed: int,
    endpoint: str,
    timeout: int,
    dry_run: bool,
) -> tuple[dict[str, Any], str]:
    if dry_run:
        return _dry_turn(prompt, model, max_tokens, int(profile["top_logprobs"]))
    api_key = os.environ.get(KEY_ENV, "").strip()
    if not api_key:
        raise RuntimeError(f"{KEY_ENV} is not set in this PowerShell session")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("OpenAI SDK is not installed; install the frontend extra") from exc
    extra_body: dict[str, Any] = {}
    if profile.get("enable_thinking") is not None:
        extra_body["chat_template_kwargs"] = {
            "enable_thinking": bool(profile["enable_thinking"]),
            "force_nonempty_content": True,
        }
    if profile.get("reasoning_effort") is not None:
        extra_body["reasoning_effort"] = profile["reasoning_effort"]
    request: dict[str, Any] = {
        "model": model,
        "messages": task_only_messages([{"role": "user", "content": prompt}]),
        "temperature": float(profile["temperature"]),
        "top_p": float(profile["top_p"]),
        "max_tokens": max_tokens,
        "seed": seed,
        "logprobs": True,
        "top_logprobs": int(profile["top_logprobs"]),
        "stream": bool(profile.get("stream")),
    }
    if extra_body:
        request["extra_body"] = extra_body
    response = OpenAI(
        api_key=api_key,
        base_url=endpoint,
        timeout=timeout,
        max_retries=0,
    ).chat.completions.create(**request)
    if profile.get("stream"):
        return _stream_turn(response, prompt, model, int(profile["top_logprobs"]))
    return _nonstream_turn(response, prompt, int(profile["top_logprobs"]))


def _write_state(
    *,
    out: Path,
    design: dict[str, Any],
    design_path: Path,
    design_sha256: str,
    profile: dict[str, Any],
    mode: str,
    max_tokens: int,
    freeze_path: Path | None,
    freeze_sha256: str | None,
    expected_n: int,
) -> dict[str, Any]:
    raws: list[dict[str, Any]] = []
    for path in sorted(out.glob(f"{mode}_*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("turns"):
            raws.append(value)
    sessions = []
    artifacts = []
    resolved_models: set[str] = set()
    fingerprints: set[str] = set()
    for raw in raws:
        turn = raw["turns"][0]
        resolved_models.add(str(raw.get("resolved_model") or raw.get("model") or ""))
        fingerprints.add(str(turn.get("system_fingerprint") or ""))
        sessions.append(
            {
                "session_id": raw["session_id"],
                "prompt_id": raw["prompt_id"],
                "finish_reason": turn["finish_reason"],
                "token_count": turn["token_count"],
                "response_time_seconds": raw["response_time_seconds"],
                "attempts": raw["attempts"],
            }
        )
        artifacts.append(
            {
                **make_envelope(
                    artifact_type="generation_observation",
                    study_id=f"{design['study_id']}-{profile['slug']}-{mode}",
                    session_id=raw["session_id"],
                    producer="scripts.collect_ep1_nvidia/1",
                    created_at=raw["created_at"],
                    source_sha256=sha256_value(raw),
                    config_sha256=raw["acquisition_identity_sha256"],
                    partition="calibration" if mode == "pilot" else "confirmatory",
                    channel_status=ChannelStatus.OBSERVED,
                ),
                "model_id": str(raw.get("resolved_model") or raw["model"]),
                "prompt_sha256": raw["prompt_sha256"],
                "response_sha256": sha256(turn["assistant_message"].encode("utf-8")).hexdigest(),
                "response_time_seconds": raw["response_time_seconds"],
                "finish_reason": turn["finish_reason"],
                "token_count": turn["token_count"],
            }
        )
    write_jsonl_atomic(out / "generation_observation.jsonl", artifacts)
    manifest = {
        "schema": "LLM-SVM-E-P1-NVIDIA-collection/1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "study_id": design["study_id"],
        "mode": mode,
        "provider": design["provider"],
        "provider_endpoint": design["provider_endpoint"],
        "model": profile["model"],
        "model_slug": profile["slug"],
        "resolved_models": sorted(resolved_models),
        "system_fingerprints": sorted(fingerprints),
        "design": str(design_path),
        "design_sha256": design_sha256,
        "model_freeze": str(freeze_path) if freeze_path else None,
        "model_freeze_sha256": freeze_sha256,
        "prompt_suite": design["prompt_suite"],
        "prompt_suite_sha256": design["prompt_suite_sha256"],
        "prompt_count": 40,
        "n": expected_n,
        "completed_n": len(sessions),
        "complete": len(sessions) == expected_n,
        "temperature": profile["temperature"],
        "top_p": profile["top_p"],
        "top_logprobs": profile["top_logprobs"],
        "seed": design["sampling"]["base_seed"],
        "seed_per_index": design["sampling"]["seed_per_index"],
        "max_tokens": max_tokens,
        "enable_thinking": profile.get("enable_thinking"),
        "reasoning_effort": profile.get("reasoning_effort"),
        "stream": bool(profile.get("stream")),
        "model_payload_boundary": design["model_payload_boundary"],
        "latency_role": design["latency_role"],
        "sessions": sessions,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def collect(args: argparse.Namespace) -> dict[str, Any]:
    design_path = args.design.resolve()
    design = load_design(design_path)
    design_hash = file_sha256(design_path)
    profile = model_profile(design, args.model)
    prompt_path = REPO_ROOT / design["prompt_suite"]
    prompts = load_prompts(prompt_path)
    mode_spec = design[args.mode]
    expected_n = int(args.n or mode_spec.get("n") or mode_spec.get("initial_n"))
    freeze_path: Path | None = None
    freeze_hash: str | None = None
    if args.mode == "pilot":
        max_tokens = int(design["pilot"]["max_tokens"])
        if args.freeze is not None:
            raise ValueError("pilot acquisition must not receive a confirmatory freeze")
    else:
        if args.freeze is None:
            raise ValueError("confirmatory acquisition requires --freeze")
        freeze_path = args.freeze.resolve()
        freeze_hash = file_sha256(freeze_path)
        freeze = _load_freeze(freeze_path, design_hash, args.model)
        max_tokens = int(freeze["selected_max_tokens"])
        if freeze.get("prompt_suite_sha256") != design["prompt_suite_sha256"]:
            raise ValueError("model freeze does not bind the designed prompt suite")
    if expected_n <= 0:
        raise ValueError("n must be positive")
    if args.out.exists() and not args.resume:
        raise FileExistsError(f"output directory already exists; use --resume: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)

    for index in range(expected_n):
        prompt = prompts[index % len(prompts)]
        session_id = f"{args.mode}_{profile['slug']}_{index:04d}"
        path = args.out / f"{session_id}.json"
        seed = int(design["sampling"]["base_seed"]) + index
        prompt_hash = sha256(prompt["prompt"].encode("utf-8")).hexdigest()
        identity = _identity(
            design_sha256=design_hash,
            freeze_sha256=freeze_hash,
            mode=args.mode,
            model=args.model,
            prompt_sha256=prompt_hash,
            seed=seed,
            max_tokens=max_tokens,
            profile=profile,
        )
        identity_hash = sha256_value(identity)
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            if raw.get("acquisition_identity_sha256") != identity_hash:
                raise RuntimeError(f"resume identity mismatch: {path}")
            turn = raw["turns"][0]
            print(
                f"[{index + 1}/{expected_n}] reusing {session_id}: "
                f"finish={turn['finish_reason']}, tokens={turn['token_count']}",
                flush=True,
            )
            continue

        print(f"[{index + 1}/{expected_n}] acquiring {session_id}", flush=True)
        last_error: Exception | None = None
        for attempt in range(1, args.max_attempts + 1):
            started = perf_counter()
            try:
                turn, resolved_model = _call(
                    prompt=prompt["prompt"],
                    model=args.model,
                    profile=profile,
                    max_tokens=max_tokens,
                    seed=seed,
                    endpoint=design["provider_endpoint"],
                    timeout=args.timeout,
                    dry_run=args.dry_run,
                )
                elapsed = 0.0 if args.dry_run else perf_counter() - started
                raw = {
                    "session_id": session_id,
                    "provider": design["provider"],
                    "provider_endpoint": design["provider_endpoint"],
                    "model": args.model,
                    "resolved_model": resolved_model or args.model,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "collection_mode": args.mode,
                    "prompt_id": prompt["prompt_id"],
                    "prompt_family": prompt["family"],
                    "prompt": prompt["prompt"],
                    "prompt_sha256": prompt_hash,
                    "seed": seed,
                    "temperature": profile["temperature"],
                    "top_p": profile["top_p"],
                    "top_logprobs": profile["top_logprobs"],
                    "max_tokens": max_tokens,
                    "response_time_seconds": elapsed,
                    "attempts": attempt,
                    "acquisition_identity": identity,
                    "acquisition_identity_sha256": identity_hash,
                    "turns": [turn],
                }
                path.write_text(
                    json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                print(
                    f"[{index + 1}/{expected_n}] completed {session_id}: "
                    f"finish={turn['finish_reason']}, tokens={turn['token_count']}, "
                    f"attempts={attempt}, seconds={elapsed:.2f}",
                    flush=True,
                )
                break
            except Exception as exc:
                last_error = exc
                print(
                    f"attempt {attempt}/{args.max_attempts} failed for {session_id}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                if attempt < args.max_attempts and args.retry_sleep_seconds > 0:
                    sleep(args.retry_sleep_seconds)
        else:
            _write_state(
                out=args.out,
                design=design,
                design_path=design_path,
                design_sha256=design_hash,
                profile=profile,
                mode=args.mode,
                max_tokens=max_tokens,
                freeze_path=freeze_path,
                freeze_sha256=freeze_hash,
                expected_n=expected_n,
            )
            raise RuntimeError(
                f"generation failed after {args.max_attempts} attempts: {last_error}"
            )
        _write_state(
            out=args.out,
            design=design,
            design_path=design_path,
            design_sha256=design_hash,
            profile=profile,
            mode=args.mode,
            max_tokens=max_tokens,
            freeze_path=freeze_path,
            freeze_sha256=freeze_hash,
            expected_n=expected_n,
        )

    return _write_state(
        out=args.out,
        design=design,
        design_path=design_path,
        design_sha256=design_hash,
        profile=profile,
        mode=args.mode,
        max_tokens=max_tokens,
        freeze_path=freeze_path,
        freeze_sha256=freeze_hash,
        expected_n=expected_n,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--mode", choices=("pilot", "confirmatory"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--n", type=int)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-sleep-seconds", type=float, default=5.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        manifest = collect(parse_args(argv))
    except (FileExistsError, RuntimeError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"E-P1 NVIDIA collection failed: {exc}")
        return 1
    print(
        json.dumps(
            {
                "output": str(manifest.get("model_slug")),
                "mode": manifest["mode"],
                "completed_n": manifest["completed_n"],
                "complete": manifest["complete"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
