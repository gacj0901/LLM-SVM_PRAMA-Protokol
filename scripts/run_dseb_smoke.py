#!/usr/bin/env python3
"""Acquire the 12-turn DSEB-v0 smoke conversation with causal identities.

This stage captures raw token logprobs and independent verifier previews. It
does not claim that PRAMA, D_O v9, or ODCE ran. Verifier previews remain
ineligible as ODCE domain outcomes until a later structural projection closes
the declared causal stage order.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from time import monotonic, perf_counter, sleep
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from aptadynamic_llm.model_payload import task_only_messages, task_only_prompt  # noqa: E402
from aptadynamic_llm.ollama_observations import (  # noqa: E402
    request_json as request_ollama_json,
    response_tokens as ollama_response_tokens,
)
from benchmarks.dseb_v0.generator import DSEBGenerator, TurnState  # noqa: E402
from benchmarks.dseb_v0.interaction import parse_order_response, render_turn  # noqa: E402
from benchmarks.dseb_v0.preflight import (  # noqa: E402
    canonical_sha256,
    file_sha256,
    run_offline_preflight,
)
from benchmarks.dseb_v0.protocol import load_protocol  # noqa: E402
from benchmarks.dseb_v0.schemas import CanonicalWindowSequence  # noqa: E402
from benchmarks.dseb_v0.verifier import verify_order  # noqa: E402
from scripts.run_odce_microbenchmarks import (  # noqa: E402
    MODEL_PROFILES,
    NVIDIA_BASE_URL,
    generation_profile,
)


RAW_SCHEMA = "LLM-SVM-DSEB-smoke-raw/1"
REPORT_SCHEMA = "LLM-SVM-DSEB-smoke-report/1"
VERIFIER_SCHEMA = "LLM-SVM-DSEB-verifier-preview/1"
DEFAULT_PROTOCOL = ROOT / "benchmarks" / "dseb_v0" / "configs" / "dseb_v0_smoke.json"
NVIDIA_CONVERSATION_ADAPTER = "native_chat_messages_v1"
OLLAMA_CONVERSATION_ADAPTER = "plain_transcript_with_stop_v1"
AcquireFunction = Callable[[Sequence[Mapping[str, str]], argparse.Namespace, int], tuple[dict[str, Any], str, float, int]]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _entropy(logprobs: Sequence[float]) -> float:
    finite = [float(value) for value in logprobs if math.isfinite(float(value))]
    if len(finite) < 2:
        return 0.0
    maximum = max(finite)
    weights = [math.exp(value - maximum) for value in finite]
    total = sum(weights)
    probabilities = [weight / total for weight in weights]
    return -sum(value * math.log(value) for value in probabilities if value > 0.0)


def _nvidia_turn(response: Any, user_message: str, turn_index: int) -> dict[str, Any]:
    choice = response.choices[0]
    content = list(getattr(getattr(choice, "logprobs", None), "content", None) or [])
    tokens: list[dict[str, Any]] = []
    for entry in content:
        chosen = float(entry.logprob)
        candidates = [float(value.logprob) for value in list(entry.top_logprobs or [])]
        if not math.isfinite(chosen) or any(not math.isfinite(value) for value in candidates):
            raise RuntimeError("NVIDIA response contains non-finite token logprobs")
        if chosen not in candidates:
            candidates.append(chosen)
        candidates.sort(reverse=True)
        tokens.append(
            {
                "token": str(entry.token),
                "top1_logprob": chosen,
                "top_logprobs": candidates,
                "gap": candidates[0] - candidates[1] if len(candidates) > 1 else 0.0,
                "entropy": _entropy(candidates),
            }
        )
    if not tokens:
        raise RuntimeError("NVIDIA response returned no token logprobs")
    return {
        "turn_index": turn_index,
        "user_message": user_message,
        "assistant_message": str(choice.message.content or ""),
        "finish_reason": str(choice.finish_reason or ""),
        "token_count": len(tokens),
        "tokens": tokens,
        "provider_response_id": str(getattr(response, "id", "") or ""),
        "system_fingerprint": str(getattr(response, "system_fingerprint", "") or ""),
    }


def _ollama_prompt(messages: Sequence[Mapping[str, str]]) -> str:
    if not messages or messages[-1]["role"] != "user":
        raise ValueError("Ollama transcript must end with the current user turn")
    sections = [
        "Continue the recorded exchange. Produce only the answer to the final "
        "user turn. Do not emit role labels or continue with another turn."
    ]
    completed = messages[:-1]
    for index in range(0, len(completed), 2):
        user = completed[index]
        assistant = completed[index + 1]
        if user["role"] != "user" or assistant["role"] != "assistant":
            raise ValueError("Ollama transcript roles must alternate user/assistant")
        turn_index = index // 2
        sections.extend(
            (
                f"[USER {turn_index}]\n{user['content']}",
                f"[ASSISTANT {turn_index}]\n{assistant['content']}",
            )
        )
    sections.extend(
        (
            f"[CURRENT USER]\n{messages[-1]['content']}",
            "[CURRENT ASSISTANT]",
        )
    )
    return task_only_prompt("\n".join(sections))


def acquire_messages(
    messages: Sequence[Mapping[str, str]], args: argparse.Namespace, turn_index: int
) -> tuple[dict[str, Any], str, float, int]:
    profile = MODEL_PROFILES[args.model]
    safe_messages = task_only_messages(messages)
    last_error: Exception | None = None
    for attempt in range(1, args.max_attempts + 1):
        started = perf_counter()
        try:
            if profile["provider"] == "nvidia_nim":
                api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
                if not api_key:
                    raise RuntimeError("NVIDIA_API_KEY is not set in this PowerShell session")
                try:
                    from openai import OpenAI
                except ImportError as exc:
                    raise RuntimeError("OpenAI SDK is not installed") from exc
                response = OpenAI(
                    api_key=api_key,
                    base_url=NVIDIA_BASE_URL,
                    timeout=args.timeout,
                ).chat.completions.create(
                    model=args.model,
                    messages=safe_messages,
                    temperature=profile["temperature"],
                    top_p=profile["top_p"],
                    max_tokens=args.max_tokens,
                    seed=args.seed,
                    logprobs=True,
                    top_logprobs=args.top_logprobs,
                    stream=False,
                    extra_body={
                        "chat_template_kwargs": {
                            "enable_thinking": bool(profile.get("enable_thinking", False)),
                            "force_nonempty_content": True,
                        }
                    },
                )
                turn = _nvidia_turn(response, safe_messages[-1]["content"], turn_index)
                return turn, str(response.model or args.model), perf_counter() - started, attempt
            prompt = _ollama_prompt(safe_messages)
            response = request_ollama_json(
                args.ollama_base_url,
                "/api/generate",
                {
                    "model": args.model,
                    "prompt": prompt,
                    "stream": False,
                    "logprobs": True,
                    "top_logprobs": args.top_logprobs,
                    "options": {
                        "temperature": profile["temperature"],
                        "top_p": profile["top_p"],
                        "num_predict": args.max_tokens,
                        "seed": args.seed,
                        "stop": [
                            "\n[USER ",
                            "\n[CURRENT USER]",
                            "\nUSER:",
                            "</assistant>",
                            "</user>",
                            "<user>",
                        ],
                    },
                },
                timeout=args.timeout,
            )
            tokens = ollama_response_tokens(response, args.top_logprobs)
            turn = {
                "turn_index": turn_index,
                "user_message": safe_messages[-1]["content"],
                "assistant_message": str(response.get("response") or ""),
                "finish_reason": str(response.get("done_reason") or ""),
                "token_count": len(tokens),
                "tokens": tokens,
            }
            return turn, str(response.get("model") or args.model), perf_counter() - started, attempt
        except Exception as exc:
            last_error = exc
            if attempt < args.max_attempts and args.retry_sleep_seconds > 0:
                sleep(args.retry_sleep_seconds)
    raise RuntimeError(f"generation failed after {args.max_attempts} attempts: {last_error}")


def _profile(args: argparse.Namespace) -> dict[str, Any]:
    profile = generation_profile(args)
    profile["conversation_adapter"] = (
        NVIDIA_CONVERSATION_ADAPTER
        if profile["provider"] == "nvidia_nim"
        else OLLAMA_CONVERSATION_ADAPTER
    )
    return profile


def _validate_existing_raw(
    raw: Mapping[str, Any], *, protocol_hash: str, profile_hash: str, session_id: str
) -> list[dict[str, Any]]:
    if (
        raw.get("schema") != RAW_SCHEMA
        or raw.get("session_id") != session_id
        or raw.get("protocol_source_sha256") != protocol_hash
        or raw.get("model_profile_sha256") != profile_hash
    ):
        raise ValueError("resume identity mismatch")
    turns = list(raw.get("turns") or [])
    if [int(turn.get("turn_index", -1)) for turn in turns] != list(range(len(turns))):
        raise ValueError("resume raw turns are not contiguous")
    return [dict(turn) for turn in turns]


def run(
    args: argparse.Namespace,
    *,
    acquire_fn: AcquireFunction | None = None,
) -> dict[str, Any]:
    protocol = load_protocol(args.protocol)
    if protocol.profile != "smoke" or len(protocol.turns) != 12:
        raise ValueError("run_dseb_smoke requires the 12-turn smoke profile")
    generated = DSEBGenerator(protocol, args.seed).generate()
    protocol_hash = file_sha256(args.protocol)
    generated_hash = canonical_sha256(generated.to_dict())
    profile = _profile(args)
    profile_hash = canonical_sha256(profile)
    session_id = f"dseb-v0-smoke-{args.model.replace('/', '--').replace(':', '--')}-seed-{args.seed:04d}"
    if args.dry_run:
        return {
            "schema": REPORT_SCHEMA,
            "mode": "dry_run",
            "benchmark_id": protocol.benchmark_id,
            "benchmark_version": protocol.benchmark_version,
            "profile": protocol.profile,
            "partition": "exploratory",
            "session_id": session_id,
            "turn_count": len(generated.turns),
            "protocol_source_sha256": protocol_hash,
            "generated_protocol_sha256": generated_hash,
            "model_profile": profile,
            "offline_gate": "VALIDATED_IN_MEMORY",
            "model_call_executed": False,
            "claim_boundary": "No files written and no provider contacted.",
        }

    availability = MODEL_PROFILES[args.model].get("availability_status")
    if availability == "EOL":
        raise ValueError(f"{args.model} is EOL and cannot be acquired")
    output_dir = args.output_dir.resolve()
    raw_path = output_dir / "sessions" / session_id / "raw.json"
    report_path = output_dir / "report.json"
    if report_path.exists() and not args.resume:
        raise ValueError(f"{report_path} exists; use --resume or a new --output-dir")
    gate = run_offline_preflight(
        protocol_path=args.protocol,
        seed=args.seed,
        output_dir=output_dir / "offline_preflight",
    )
    if gate["status"] != "PASS":
        raise RuntimeError("offline preflight gate failed; model acquisition refused")

    turns: list[dict[str, Any]] = []
    if raw_path.exists():
        if not args.resume:
            raise ValueError(f"raw artifact exists: {raw_path}")
        turns = _validate_existing_raw(
            json.loads(raw_path.read_text(encoding="utf-8")),
            protocol_hash=protocol_hash,
            profile_hash=profile_hash,
            session_id=session_id,
        )
    messages: list[dict[str, str]] = []
    for state, turn in zip(generated.turns, turns, strict=False):
        prompt = render_turn(generated, state)
        if turn.get("user_message") != prompt:
            raise ValueError(f"resume prompt mismatch at turn {state.target.turn_index}")
        messages.extend(
            (
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": str(turn["assistant_message"])},
            )
        )

    acquire_fn = acquire_fn or acquire_messages
    started_at = utc_now()
    started_clock = monotonic()
    resolved_model = str(turns[-1].get("resolved_model") or args.model) if turns else args.model
    for state in generated.turns[len(turns) :]:
        if monotonic() - started_clock >= args.run_deadline_minutes * 60:
            break
        turn_index = state.target.turn_index
        prompt = render_turn(generated, state)
        request_messages = [*messages, {"role": "user", "content": prompt}]
        print(f"[{turn_index + 1}/12] turn-{turn_index:02d} {state.target.phase}", flush=True)
        turn, resolved_model, elapsed, attempts = acquire_fn(request_messages, args, turn_index)
        turn = dict(turn)
        turn["turn_index"] = turn_index
        turn["user_message"] = prompt
        turn["response_time_seconds"] = float(elapsed)
        turn["attempts"] = int(attempts)
        turn["resolved_model"] = resolved_model
        turns.append(turn)
        messages.extend(
            (
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": str(turn["assistant_message"])},
            )
        )
        raw = {
            "schema": RAW_SCHEMA,
            "session_id": session_id,
            "provider": profile["provider"],
            "provider_endpoint": NVIDIA_BASE_URL if profile["provider"] == "nvidia_nim" else args.ollama_base_url,
            "model": args.model,
            "resolved_model": resolved_model,
            "created_at": started_at,
            "updated_at": utc_now(),
            "measurement_status": "OBSERVED",
            "benchmark_id": protocol.benchmark_id,
            "benchmark_version": protocol.benchmark_version,
            "profile": protocol.profile,
            "partition": "exploratory",
            "contract_freeze_sha256": None,
            "protocol_source_sha256": protocol_hash,
            "generated_protocol_sha256": generated_hash,
            "model_profile_sha256": profile_hash,
            "protocol_source": str(args.protocol.resolve()),
            "seed": args.seed,
            "window_size_tokens": args.window_size,
            "turns": turns,
        }
        atomic_json(raw_path, raw)
        print(f"  acquired: {turn['token_count']} tokens in {elapsed:.1f}s", flush=True)

    sequence = CanonicalWindowSequence()
    turn_rows: list[dict[str, Any]] = []
    verifier_rows: list[dict[str, Any]] = []
    for state, turn in zip(generated.turns, turns, strict=False):
        window_count = max(1, math.ceil(int(turn["token_count"]) / args.window_size))
        terminal = sequence.append_turn(state.target.turn_index, window_count)
        event_index = sequence.ordinal(terminal)
        parsed = parse_order_response(str(turn["assistant_message"]), generated.symbols)
        verification = verify_order(
            symbols=generated.symbols,
            order=parsed.order,
            persistent=state.active_constraints,
            temporary=state.temporary_constraints,
            new_constraint_ids=state.new_constraint_ids,
        )
        turn_rows.append(
            {
                "schema": "LLM-SVM-DSEB-benchmark-turn/1",
                "benchmark_id": protocol.benchmark_id,
                "benchmark_version": protocol.benchmark_version,
                "profile": protocol.profile,
                "session_id": session_id,
                **state.to_dict(),
                "closed_window_count": window_count,
                "terminal_window": terminal.to_dict(),
                "terminal_event_index": event_index,
            }
        )
        verifier_rows.append(
            {
                "schema": VERIFIER_SCHEMA,
                "benchmark_id": protocol.benchmark_id,
                "benchmark_version": protocol.benchmark_version,
                "profile": protocol.profile,
                "partition": "exploratory",
                "session_id": session_id,
                "benchmark_turn_index": state.target.turn_index,
                "event_window": terminal.to_dict(),
                "event_index": event_index,
                "available_at_window": None,
                "available_at_index": None,
                "causal_registration_status": "PENDING_D_O_V9",
                "eligible_for_odce": False,
                "retrospective_backfill": False,
                "pipeline_execution_mode": "MODEL_ACQUISITION_ONLY",
                **parsed.to_dict(),
                **verification.to_dict(),
            }
        )
    atomic_jsonl(output_dir / "benchmark_turns.jsonl", turn_rows)
    atomic_jsonl(output_dir / "verifier_previews.jsonl", verifier_rows)
    complete = len(turns) == len(generated.turns)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "ACQUIRED_PENDING_STRUCTURAL_PROJECTION" if complete else "INCOMPLETE",
        "benchmark_id": protocol.benchmark_id,
        "benchmark_version": protocol.benchmark_version,
        "profile": protocol.profile,
        "partition": "exploratory",
        "contract_freeze_sha256": None,
        "session_id": session_id,
        "seed": args.seed,
        "protocol_source_sha256": protocol_hash,
        "generated_protocol_sha256": generated_hash,
        "model_profile": profile,
        "model_profile_sha256": profile_hash,
        "offline_preflight_status": gate["status"],
        "selected_turn_count": len(generated.turns),
        "completed_turn_count": len(turns),
        "canonical_window_count": len(sequence.identities),
        "window_size_tokens": args.window_size,
        "response_contract_pass_count": sum(int(row["response_contract_valid"]) for row in verifier_rows),
        "formal_pass_count": sum(int(row["verified_outcome"]) for row in verifier_rows),
        "prama_executed": False,
        "structural_observer_executed": False,
        "odce_executed": False,
        "verifier_previews_eligible_for_odce": False,
        "claim_boundary": (
            "Raw conversational logprobs and verifier previews only; causal domain "
            "outcomes require PRAMA and D_O v9 projection first."
        ),
        "raw_file": str(raw_path.resolve()),
        "turns_file": str((output_dir / "benchmark_turns.jsonl").resolve()),
        "verifier_previews_file": str((output_dir / "verifier_previews.jsonl").resolve()),
    }
    atomic_json(report_path, report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--model", required=True, choices=tuple(MODEL_PROFILES))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--top-logprobs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--retry-sleep-seconds", type=float, default=3.0)
    parser.add_argument("--run-deadline-minutes", type=float, default=30.0)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--ollama-base-url", default="http://localhost:11434")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if (
        args.max_tokens <= 0
        or not 2 <= args.top_logprobs <= 20
        or args.seed < 0
        or args.timeout <= 0
        or args.max_attempts <= 0
        or args.retry_sleep_seconds < 0
        or args.run_deadline_minutes <= 0
        or args.window_size <= 0
    ):
        parser.error("invalid numeric argument")
    args.ollama_base_url = args.ollama_base_url.rstrip("/")
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        report = run(parse_args(argv))
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"DSEB smoke failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report.get("mode") == "dry_run" or report.get("status") == "ACQUIRED_PENDING_STRUCTURAL_PROJECTION" else 2


if __name__ == "__main__":
    raise SystemExit(main())
