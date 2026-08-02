#!/usr/bin/env python
"""Run minimal structural perturbations with the universal dynamic observer and mobility v4."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for candidate in (REPO_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from aptadynamic_llm.model_payload import task_only_messages  # noqa: E402
from scripts.project_cocc_operator_geometry import file_sha256  # noqa: E402
from scripts.project_cocc_prama import validate_identity  # noqa: E402
from scripts.project_cocc_prama_dynamic import project as project_dynamic  # noqa: E402
from scripts.project_cocc_structural_mobility_v4 import build_artifact  # noqa: E402
from scripts.reproject_minimal_structural_mobility_v4 import (  # noqa: E402
    ARMS,
    atomic_json,
    contrasts,
    endpoints,
    summarize,
)
from scripts.run_minimal_structural_perturbations_nvidia import (  # noqa: E402
    NVIDIA_API_KEY_ENV,
    NVIDIA_BASE_URL,
    load_protocol,
    trials,
    write_jsonl,
)


def entropy(logprobs: Sequence[float]) -> float:
    values = [float(value) for value in logprobs if math.isfinite(float(value))]
    if not values:
        return 0.0
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    total = sum(weights)
    return -sum((weight / total) * math.log(weight / total) for weight in weights if weight > 0.0)


def response_turn(response: Any, turn_index: int, user_message: str, elapsed: float) -> dict[str, Any]:
    choice = response.choices[0]
    content = list(getattr(getattr(choice, "logprobs", None), "content", None) or [])
    tokens = []
    for entry in content:
        chosen = float(entry.logprob)
        candidates = sorted(
            {chosen, *[float(candidate.logprob) for candidate in list(entry.top_logprobs or [])]},
            reverse=True,
        )
        if not math.isfinite(chosen) or any(not math.isfinite(value) for value in candidates):
            raise RuntimeError("provider returned non-finite token logprobs")
        tokens.append({
            "token": str(entry.token),
            "top1_logprob": chosen,
            "top_logprobs": candidates,
            "gap": candidates[0] - candidates[1] if len(candidates) > 1 else 0.0,
            "entropy": entropy(candidates),
        })
    if not tokens:
        raise RuntimeError("provider returned no token logprobs")
    return {
        "turn_index": turn_index,
        "user_message": user_message,
        "assistant_message": str(choice.message.content or ""),
        "finish_reason": str(choice.finish_reason or ""),
        "token_count": len(tokens),
        "response_seconds": elapsed,
        "tokens": tokens,
        "provider_response_id": str(getattr(response, "id", "") or ""),
        "system_fingerprint": str(getattr(response, "system_fingerprint", "") or ""),
    }


def call_model(messages: Sequence[Mapping[str, str]], args: argparse.Namespace, turn_index: int) -> tuple[dict[str, Any], str]:
    key = os.environ.get(NVIDIA_API_KEY_ENV, "").strip()
    if not key:
        raise RuntimeError(f"{NVIDIA_API_KEY_ENV} is not set in this PowerShell session")
    from openai import OpenAI

    clean_messages = task_only_messages(messages)
    extra_body: dict[str, Any]
    if "mistral" in args.model.lower():
        extra_body = {"reasoning_effort": args.reasoning_effort}
    else:
        extra_body = {"chat_template_kwargs": {"enable_thinking": False, "force_nonempty_content": True}}
    last_error: Exception | None = None
    for attempt in range(1, args.max_attempts + 1):
        started = time.perf_counter()
        try:
            response = OpenAI(
                api_key=key,
                base_url=args.base_url,
                timeout=args.timeout,
                max_retries=0,
            ).chat.completions.create(
                model=args.model,
                messages=clean_messages,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                seed=args.seed,
                logprobs=True,
                top_logprobs=args.top_logprobs,
                stream=False,
                extra_body=extra_body,
            )
            elapsed = time.perf_counter() - started
            return response_turn(response, turn_index, clean_messages[-1]["content"], elapsed), str(response.model or args.model)
        except Exception as exc:
            last_error = exc
            print(f"attempt {attempt}/{args.max_attempts} failed: {type(exc).__name__}: {exc}", flush=True)
            if attempt < args.max_attempts:
                time.sleep(args.retry_sleep)
    raise RuntimeError(f"generation failed after {args.max_attempts} attempts: {last_error}")


def numeric_request(raw: Mapping[str, Any], observer_hash: str) -> dict[str, Any]:
    turn = raw["turns"][-1]
    tokens = [
        {
            "top1_logprob": float(token["top1_logprob"]),
            "top_logprobs": [float(value) for value in token["top_logprobs"]],
            "gap": float(token["gap"]),
            "entropy": float(token["entropy"]),
        }
        for token in turn["tokens"]
    ]
    request = {
        "schema": "LLM-SVM-CoCC-projector-request/1",
        "session_id": str(raw["session_id"]),
        "model_id": str(raw["model"]),
        "source_session_sha256": sha256(json.dumps(raw, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest(),
        "input_channel_status": "OBSERVED",
        "observer_contract_sha256": observer_hash,
        "turns": [{"turn_index": 1, "token_count": len(tokens), "tokens": tokens}],
    }
    serialized = json.dumps(request, sort_keys=True)
    for forbidden in ("arm", "topic", "prompt", "assistant_message", "tracked_commitment", "perturbation"):
        if f'"{forbidden}"' in serialized:
            raise ValueError(f"numeric request leaked {forbidden}")
    return request


def model_matches(raw: Mapping[str, Any], requested: str) -> bool:
    return str(raw.get("model") or "") == requested


def copy_or_acquire_baseline(item: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    item_id = str(item["item_id"])
    target = args.output_dir / "baselines" / item_id / "raw.json"
    source = args.reuse_responses_from / "baselines" / item_id / "raw.json" if args.reuse_responses_from else None
    if target.exists():
        raw = json.loads(target.read_text(encoding="utf-8"))
        print(f"reusing baseline {item_id}: finish={raw['turn']['finish_reason']}, tokens={raw['turn']['token_count']}", flush=True)
        return raw
    if source and source.exists():
        raw = json.loads(source.read_text(encoding="utf-8"))
        if not model_matches(raw, args.model):
            raise ValueError(f"baseline reuse model mismatch for {item_id}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        print(f"imported baseline {item_id}: finish={raw['turn']['finish_reason']}, tokens={raw['turn']['token_count']}", flush=True)
        return raw
    print(f"acquiring baseline {item_id}", flush=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    turn, resolved = call_model([{"role": "user", "content": str(item["baseline_prompt"])}], args, 0)
    raw = {"schema": "LLM-SVM-minimal-structural-baseline/2", "item_id": item_id, "model": resolved, "turn": turn}
    atomic_json(target, raw)
    print(f"completed baseline {item_id}: finish={turn['finish_reason']}, tokens={turn['token_count']}, seconds={turn['response_seconds']:.2f}", flush=True)
    return raw


def copy_or_acquire_session(trial: Mapping[str, Any], baseline: Mapping[str, Any], args: argparse.Namespace, index: int, total: int) -> dict[str, Any]:
    session_id = f"{trial['item_id']}--{trial['arm']}"
    target = args.output_dir / "sessions" / session_id / "raw.json"
    source = args.reuse_responses_from / "sessions" / session_id / "raw.json" if args.reuse_responses_from else None
    if target.exists():
        raw = json.loads(target.read_text(encoding="utf-8"))
        turn = raw["turns"][-1]
        print(f"[{index}/{total}] reusing {session_id}: finish={turn['finish_reason']}, tokens={turn['token_count']}", flush=True)
        return raw
    if source and source.exists():
        raw = json.loads(source.read_text(encoding="utf-8"))
        if not model_matches(raw, args.model):
            raise ValueError(f"session reuse model mismatch for {session_id}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        turn = raw["turns"][-1]
        print(f"[{index}/{total}] imported {session_id}: finish={turn['finish_reason']}, tokens={turn['token_count']}", flush=True)
        return raw
    print(f"[{index}/{total}] acquiring {session_id}", flush=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    turn, resolved = call_model(
        [
            {"role": "user", "content": str(trial["baseline_prompt"])},
            {"role": "assistant", "content": str(baseline["turn"]["assistant_message"])},
            {"role": "user", "content": str(trial["perturbation"])},
        ],
        args,
        1,
    )
    raw = {
        "schema": "LLM-SVM-minimal-structural-session/2",
        "session_id": session_id,
        "item_id": str(trial["item_id"]),
        "topic": str(trial["topic"]),
        "arm": str(trial["arm"]),
        "model": resolved,
        "tracked_commitment": str(trial["tracked_commitment"]),
        "turns": [baseline["turn"], turn],
    }
    atomic_json(target, raw)
    print(f"[{index}/{total}] completed {session_id}: finish={turn['finish_reason']}, tokens={turn['token_count']}, seconds={turn['response_seconds']:.2f}", flush=True)
    return raw


def run(args: argparse.Namespace) -> Path:
    protocol = load_protocol(args.protocol)
    planned = trials(protocol)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    observer_hash = file_sha256(args.observer_contract)
    mobility_hash = file_sha256(args.mobility_contract)
    freeze = json.loads(args.mobility_freeze.read_text(encoding="utf-8"))
    if freeze["observer_contract_sha256"] != mobility_hash:
        raise ValueError("mobility v4 contract differs from freeze")
    observer = json.loads(args.observer_contract.read_text(encoding="utf-8"))
    if observer.get("model_specific_parameters") is not False or observer.get("requires_external_calibration") is not False:
        raise ValueError("dynamic observer must be universal and calibration-free")
    kernel_config, columns, identity = validate_identity(args.declaration, args.recertification)

    baselines = {str(item["item_id"]): copy_or_acquire_baseline(item, args) for item in protocol["diagnostic_items"]}
    rows = []
    for index, trial in enumerate(planned, start=1):
        raw = copy_or_acquire_session(trial, baselines[str(trial["item_id"])], args, index, len(planned))
        session_dir = args.output_dir / "sessions" / str(raw["session_id"])
        request = numeric_request(raw, observer_hash)
        request_path = session_dir / "projection_request.json"
        atomic_json(request_path, request)
        trajectory = project_dynamic(request, observer, observer_hash, kernel_config, columns, identity)
        trajectory_path = session_dir / "trajectory.jsonl"
        write_jsonl(trajectory_path, trajectory)
        artifact = build_artifact(request_path, trajectory_path, args.geometry_contract, args.mobility_contract)
        atomic_json(session_dir / "structural_mobility_v4.json", artifact)
        full = endpoints(artifact["windows"])
        fixed = endpoints(artifact["windows"], args.fixed_ready_horizon)
        turn = raw["turns"][-1]
        rows.append({
            "session_id": str(raw["session_id"]),
            "item_id": str(raw["item_id"]),
            "arm": str(raw["arm"]),
            "finish_reason": str(turn["finish_reason"]),
            "token_count": int(turn["token_count"]),
            "response_seconds": float(turn["response_seconds"]),
            "max_delta": max((float(row["delta"]) for row in trajectory if row.get("delta") is not None), default=0.0),
            "max_xi": max((float(row["xi"]) for row in trajectory if row.get("xi") is not None), default=0.0),
            **full,
            **{f"{key}_h{args.fixed_ready_horizon}": value for key, value in fixed.items()},
        })

    suffix = f"_h{args.fixed_ready_horizon}"
    report = {
        "schema": "LLM-SVM-minimal-structural-dynamic-v4-report/1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "study_id": protocol["study_id"],
        "model": args.model,
        "generation_profile": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "top_logprobs": args.top_logprobs,
            "seed": args.seed,
            "reasoning_effort": args.reasoning_effort if "mistral" in args.model.lower() else None,
            "thinking_enabled": False,
        },
        "protocol_sha256": file_sha256(args.protocol),
        "dynamic_observer_contract_sha256": observer_hash,
        "mobility_contract_sha256": mobility_hash,
        "kernel_identity": identity,
        "model_payload_boundary": "baseline prompt, shared baseline answer, and arm instruction only",
        "observer_boundary": "perturbation-turn numeric token observations only",
        "observer_receives_arm_or_item_labels": False,
        "reused_response_source": str(args.reuse_responses_from) if args.reuse_responses_from else None,
        "fixed_geometry_ready_horizon": args.fixed_ready_horizon,
        "summary_full": summarize(rows),
        "summary_fixed_horizon": summarize(rows, suffix),
        "minimal_vs_abstract_full": contrasts(rows, "abstract_content"),
        "minimal_vs_abstract_fixed_horizon": contrasts(rows, "abstract_content", suffix),
        "minimal_vs_concrete_full": contrasts(rows, "concrete_content"),
        "minimal_vs_concrete_fixed_horizon": contrasts(rows, "concrete_content", suffix),
        "session_endpoints": rows,
    }
    output = args.output_dir / "report.json"
    atomic_json(output, report)
    print(json.dumps({"output": str(output), "sha256": file_sha256(output), "sessions": len(rows)}))
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reuse-responses-from", type=Path)
    parser.add_argument("--base-url", default=NVIDIA_BASE_URL)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--top-logprobs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--reasoning-effort", default="none")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--fixed-ready-horizon", type=int, default=16)
    parser.add_argument("--observer-contract", type=Path, default=Path("config/cocc_dynamic_observer_contract_v1.json"))
    parser.add_argument("--declaration", type=Path, default=Path("config/window_prama_kernel_declaration.json"))
    parser.add_argument("--recertification", type=Path, default=Path("run_outputs/window_prama_recertification_v030_20260730.json"))
    parser.add_argument("--geometry-contract", type=Path, default=Path("config/cocc_operator_geometry_observer_v1.json"))
    parser.add_argument("--mobility-contract", type=Path, default=Path("config/cocc_structural_mobility_observer_v4.json"))
    parser.add_argument("--mobility-freeze", type=Path, default=Path("config/cocc_structural_mobility_observer_v4.freeze.json"))
    args = parser.parse_args()
    try:
        run(args)
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"minimal structural dynamic run failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
