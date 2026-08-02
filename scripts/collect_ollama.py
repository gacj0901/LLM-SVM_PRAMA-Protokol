#!/usr/bin/env python
"""Collect reproducible Ollama sessions with native token logprobs for E-P1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any

from aptadynamic_llm.model_payload import (
    encode_task_only_payload,
    task_only_prompt,
)
from aptadynamic_llm.artifact_schema import (
    ChannelStatus,
    make_envelope,
    sha256_value,
    write_jsonl_atomic,
)
from aptadynamic_llm.ep1_config import (
    BASE_SEED,
    MIN_OLLAMA_VERSION,
    MODEL,
    MODEL_ID_PREFIX,
    SEED_PER_INDEX,
    TEMPERATURE,
    TOP_LOGPROBS,
    TOP_P,
)

DEFAULT_URL = "http://localhost:11434"
DEFAULT_MODEL = MODEL
EXPECTED_MODEL_ID_PREFIX = MODEL_ID_PREFIX
FROZEN_TEMPERATURE = TEMPERATURE
FROZEN_TOP_P = TOP_P
FROZEN_TOP_LOGPROBS = TOP_LOGPROBS
FROZEN_SEED = BASE_SEED


def _request(base_url: str, endpoint: str, payload: dict[str, Any] | None = None, timeout: int = 600) -> dict[str, Any]:
    url = base_url.rstrip("/") + endpoint
    data = None if payload is None else encode_task_only_payload(payload)
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama request failed at {url}: {exc}") from exc


def _valid_logprob(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _candidate_logprobs(item: dict[str, Any]) -> list[float]:
    values = [
        value
        for candidate in item.get("top_logprobs") or []
        if (value := _valid_logprob(candidate.get("logprob"))) is not None
    ]
    chosen = _valid_logprob(item.get("logprob"))
    if chosen is not None and chosen not in values:
        values.append(chosen)
    return sorted(values, reverse=True)[:FROZEN_TOP_LOGPROBS]


def _normalized_entropy(logprobs: list[float]) -> float:
    if len(logprobs) < 2:
        return 0.0
    maximum = max(logprobs)
    weights = [math.exp(value - maximum) for value in logprobs]
    total = sum(weights)
    probabilities = [weight / total for weight in weights]
    entropy = -sum(probability * math.log(probability + 1e-15) for probability in probabilities)
    return max(0.0, min(1.0, entropy / math.log(len(probabilities))))


def _gap(logprobs: list[float]) -> float:
    ordered = sorted(logprobs, reverse=True)
    return float(ordered[0] - ordered[1]) if len(ordered) >= 2 else 0.0


def response_tokens(response: dict[str, Any]) -> list[dict[str, Any]]:
    tokens: list[dict[str, Any]] = []
    for item in response.get("logprobs") or []:
        chosen = _valid_logprob(item.get("logprob"))
        if chosen is None:
            continue
        candidates = _candidate_logprobs(item)
        tokens.append(
            {
                "token": str(item.get("token") or ""),
                "bytes": item.get("bytes") or [],
                "top1_logprob": chosen,
                "top_logprobs": candidates,
                "gap": _gap(candidates),
                "entropy": _normalized_entropy(candidates),
            }
        )
    if not tokens:
        raise RuntimeError("Ollama returned no usable token logprobs")
    return tokens


def load_prompts(path: Path) -> list[dict[str, str]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif suffix == ".json":
        values = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(values, dict):
            values = values.get("prompts") or []
    else:
        values = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    prompts: list[dict[str, str]] = []
    for index, value in enumerate(values):
        if isinstance(value, str):
            prompts.append({"prompt_id": f"p{index:04d}", "family": "unspecified", "prompt": value})
        elif isinstance(value, dict) and value.get("prompt"):
            prompts.append(
                {
                    "prompt_id": str(value.get("prompt_id") or value.get("id") or f"p{index:04d}"),
                    "family": str(value.get("family") or "unspecified"),
                    "prompt": str(value["prompt"]),
                }
            )
    if not prompts:
        raise ValueError(f"prompt suite is empty or invalid: {path}")
    prompt_ids = [prompt["prompt_id"] for prompt in prompts]
    if len(prompts) < 40:
        raise ValueError("E-P1 v2 prompt suite requires at least 40 distinct prompts")
    if len(prompt_ids) != len(set(prompt_ids)):
        raise ValueError("E-P1 v2 prompt suite contains duplicate prompt_id values")
    invalid_families = sorted({prompt["family"] for prompt in prompts if prompt["family"] != "structured_analysis"})
    if invalid_families:
        raise ValueError(
            "E-P1 v2 permits only bounded structured_analysis prompts; "
            f"found {invalid_families!r}"
        )
    return prompts


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_digest(base_url: str, model: str, timeout: int) -> str | None:
    tags = _request(base_url, "/api/tags", timeout=timeout)
    for item in tags.get("models") or []:
        if item.get("name") == model or item.get("model") == model:
            return item.get("digest")
    return None


def _version_tuple(value: str) -> tuple[int, int, int]:
    parts = value.split(".")
    try:
        return tuple(int(part.split("-")[0]) for part in (parts + ["0", "0"])[:3])
    except ValueError as exc:
        raise RuntimeError(f"cannot parse Ollama version {value!r}") from exc


def collect(args: argparse.Namespace) -> dict[str, Any]:
    if args.out.exists():
        raise FileExistsError(f"output directory already exists: {args.out}")
    prompts = load_prompts(args.prompts)
    version = _request(args.base_url, "/api/version", timeout=args.timeout).get("version", "unknown")
    digest = _model_digest(args.base_url, args.model, args.timeout)
    if args.model != DEFAULT_MODEL:
        raise ValueError(f"E-P1 model is frozen as {DEFAULT_MODEL!r}")
    if _version_tuple(str(version)) < MIN_OLLAMA_VERSION:
        raise RuntimeError("E-P1 requires local Ollama >= 0.30.11")
    if digest is None or not str(digest).removeprefix("sha256:").startswith(EXPECTED_MODEL_ID_PREFIX):
        raise RuntimeError(
            f"model identity mismatch: expected {EXPECTED_MODEL_ID_PREFIX}, received {digest!r}"
        )
    if args.temperature != FROZEN_TEMPERATURE or args.top_p != FROZEN_TOP_P:
        raise ValueError(
            f"E-P1 sampling is frozen at temperature={FROZEN_TEMPERATURE}, top_p={FROZEN_TOP_P}"
        )
    if (
        args.top_logprobs != FROZEN_TOP_LOGPROBS
        or args.seed != FROZEN_SEED
        or args.seed_per_index is not SEED_PER_INDEX
    ):
        raise ValueError(
            "E-P1 requires top_logprobs=5, seed=1337, and --seed-per-index"
        )
    count = args.n if args.n is not None else (40 if args.pilot else 400)
    if count <= 0 or args.num_predict <= 0 or args.top_logprobs < 2:
        raise ValueError("n and num-predict must be positive; top-logprobs must be at least 2")

    args.out.mkdir(parents=True)
    completed: list[dict[str, Any]] = []
    generation_artifacts: list[dict[str, Any]] = []
    for index in range(count):
        prompt = prompts[index % len(prompts)]
        seed = args.seed + index if args.seed_per_index else args.seed
        payload = {
            "model": args.model,
            "prompt": task_only_prompt(prompt["prompt"]),
            "stream": False,
            "logprobs": True,
            "top_logprobs": args.top_logprobs,
            "options": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "num_predict": args.num_predict,
                "seed": seed,
            },
        }
        response_started = monotonic()
        response = _request(args.base_url, "/api/generate", payload, timeout=args.timeout)
        response_time_seconds = monotonic() - response_started
        tokens = response_tokens(response)
        session_id = f"{args.mode}_{index:04d}"
        raw = {
            "session_id": session_id,
            "provider": "ollama",
            "model": str(response.get("model") or args.model),
            "model_digest": digest,
            "ollama_version": version,
            "created_at": str(response.get("created_at") or datetime.now(timezone.utc).isoformat()),
            "collection_mode": args.mode,
            "prompt_id": prompt["prompt_id"],
            "prompt_family": prompt["family"],
            "prompt": prompt["prompt"],
            "seed": seed,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "num_predict": args.num_predict,
            # Execution metadata only; never included in the model payload.
            "response_time_seconds": response_time_seconds,
            "turns": [
                {
                    "turn_index": 0,
                    "assistant_message": str(response.get("response") or ""),
                    "finish_reason": str(response.get("done_reason") or ""),
                    "token_count": len(tokens),
                    "tokens": tokens,
                }
            ],
        }
        generation_artifacts.append(
            {
                **make_envelope(
                    artifact_type="generation_observation",
                    study_id=f"E-P1-{args.mode}",
                    session_id=session_id,
                    producer="scripts.collect_ollama/2",
                    created_at=raw["created_at"],
                    source_sha256=sha256_value(response),
                    config_sha256=sha256_value(
                        {
                            "model": args.model,
                            "logprobs": True,
                            "top_logprobs": args.top_logprobs,
                            "options": payload["options"],
                        }
                    ),
                    partition=(
                        "calibration" if args.mode == "pilot" else "confirmatory"
                    ),
                    channel_status=ChannelStatus.OBSERVED,
                ),
                "model_id": f"{args.model}@{digest}",
                "prompt_sha256": hashlib.sha256(
                    prompt["prompt"].encode("utf-8")
                ).hexdigest(),
                "response_sha256": hashlib.sha256(
                    raw["turns"][0]["assistant_message"].encode("utf-8")
                ).hexdigest(),
                "response_time_seconds": response_time_seconds,
                "finish_reason": raw["turns"][0]["finish_reason"],
                "token_count": len(tokens),
            }
        )
        path = args.out / f"{session_id}.json"
        path.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
        completed.append(
            {
                "session_id": session_id,
                "prompt_id": prompt["prompt_id"],
                "finish_reason": raw["turns"][0]["finish_reason"],
                "token_count": len(tokens),
                "response_time_seconds": response_time_seconds,
            }
        )
        print(f"[{index + 1}/{count}] {session_id}: {len(tokens)} tokens, {raw['turns'][0]['finish_reason']}")

    write_jsonl_atomic(
        args.out / "generation_observation.jsonl", generation_artifacts
    )
    manifest = {
        "schema": "LLM-SVM-Ollama-collection/1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "ollama_version": version,
        "model": args.model,
        "model_digest": digest,
        "prompt_suite": str(args.prompts),
        "prompt_suite_sha256": _sha256(args.prompts),
        "prompt_count": len(prompts),
        "n": count,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "num_predict": args.num_predict,
        "top_logprobs": args.top_logprobs,
        "seed": args.seed,
        "seed_per_index": args.seed_per_index,
        "sessions": completed,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", action="store_true")
    mode.add_argument("--confirmatory", action="store_true")
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--base-url", default=DEFAULT_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--n", type=int)
    parser.add_argument("--num-predict", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=FROZEN_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=FROZEN_TOP_P)
    parser.add_argument("--top-logprobs", type=int, default=FROZEN_TOP_LOGPROBS)
    parser.add_argument("--seed", type=int, default=FROZEN_SEED)
    parser.add_argument("--seed-per-index", action="store_true")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args(argv)
    args.mode = "pilot" if args.pilot else "confirmatory"
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        manifest = collect(parse_args(argv))
    except (FileExistsError, RuntimeError, ValueError, OSError) as exc:
        print(f"collection failed: {exc}")
        return 1
    print(json.dumps({key: manifest[key] for key in ("mode", "n", "model", "model_digest")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
