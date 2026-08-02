#!/usr/bin/env python
"""Run isolated CoCC acquisition through NVIDIA-hosted Nemotron 3 Super.

The evaluated model receives only the perturbed task prompt and generation
parameters. External correctness and PRAMA projection run in separate phases
and receive disjoint JSON views. Their outputs are joined only after both
channels have completed.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from time import perf_counter, sleep
from typing import Any, Iterable, Sequence

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
from aptadynamic_llm.model_payload import task_only_messages, task_only_prompt  # noqa: E402


CANONICAL_BENCHMARK = "chain_of_code_collapse"
BENCHMARK_ALIAS = "break_the_chain_code_generation"
PROMPT_FIELDS = ("perturbed_prompt", "prompt_perturbed", "btc_prompt", "cocc_prompt")
ID_FIELDS = ("problem_id", "item_id", "id", "question_id")
VERIFIER_FIELDS = (
    "verifier_ref",
    "test_ref",
    "lcb_problem_ref",
    "public_test_cases",
    "test_cases",
    "expected_output",
)
FORBIDDEN_PROJECTOR_KEYS = {
    "assistant_message",
    "benchmark_alias",
    "benchmark_name",
    "clean_prompt",
    "expected_answer",
    "item_id",
    "label",
    "observed_answer",
    "passed",
    "perturbation_type",
    "problem_id",
    "prompt",
    "provider",
    "response_time_seconds",
    "source_label",
    "token",
    "user_message",
    "verifier_ref",
}

NVIDIA_PROVIDER = "nvidia_nim"
NVIDIA_API_KEY_ENV = "NVIDIA_API_KEY"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "nvidia/nemotron-3-super-120b-a12b"
NVIDIA_MODELS = (
    NVIDIA_MODEL,
    "nvidia/nemotron-3-ultra-550b-a55b",
)


@dataclass(frozen=True)
class CoccItem:
    problem_id: str
    item_id: str
    prompt: str
    perturbation_type: str
    split: str
    verifier_ref: Any
    expected_answer: str
    source_label: Any


@dataclass(frozen=True)
class VerificationOutcome:
    label: int
    passed: bool
    payload: dict[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ollama_model_blob_sha256(model: str) -> str:
    completed = subprocess.run(
        ["ollama", "show", "--modelfile", model],
        check=True,
        capture_output=True,
        text=True,
    )
    match = re.search(r"sha256-([0-9a-fA-F]{64})", completed.stdout)
    if match is None:
        raise ValueError("could not resolve immutable Ollama model blob SHA-256")
    return match.group(1).lower()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _write_csv(path: Path, fields: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        delete=False,
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    temporary.replace(path)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ValueError(f"dataset does not exist: {path}")
    if path.suffix.lower() == ".jsonl":
        values = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif path.suffix.lower() == ".json":
        decoded = json.loads(path.read_text(encoding="utf-8"))
        values = decoded if isinstance(decoded, list) else decoded.get("items", [])
    elif path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            values = list(csv.DictReader(handle))
    else:
        raise ValueError("dataset must be JSONL, JSON, or CSV")
    if not values or not all(isinstance(row, dict) for row in values):
        raise ValueError("dataset contains no object rows")
    return values


def _first(row: dict[str, Any], fields: Sequence[str]) -> Any:
    for field in fields:
        value = row.get(field)
        if value not in (None, ""):
            return value
    return None


def load_cocc_items(path: Path, *, prospective_only: bool = False) -> list[CoccItem]:
    rows = _load_rows(path)
    items: list[CoccItem] = []
    problem_splits: dict[str, str] = {}
    for index, row in enumerate(rows):
        if row.get("benchmark_name") != CANONICAL_BENCHMARK:
            raise ValueError(f"row {index}: invalid benchmark_name")
        if row.get("benchmark_alias") != BENCHMARK_ALIAS:
            raise ValueError(f"row {index}: invalid benchmark_alias")
        prompt = _first(row, PROMPT_FIELDS)
        perturbation = str(row.get("perturbation_type") or "").strip()
        verifier_ref = _first(row, VERIFIER_FIELDS)
        if not prompt or not perturbation or verifier_ref is None:
            raise ValueError(f"row {index}: prompt, perturbation_type, and verifier mapping are required")
        problem_id = str(_first(row, ID_FIELDS) or f"item-{index}")
        item_id = str(row.get("item_id") or problem_id)
        split = str(row.get("split") or "").strip().lower()
        if split not in {"calibration", "train", "test"}:
            raise ValueError(f"row {index}: split must be calibration, train, or test")
        normalized_split = "train" if split in {"calibration", "train"} else "test"
        previous = problem_splits.setdefault(problem_id, normalized_split)
        if previous != normalized_split:
            raise ValueError(f"problem {problem_id!r} crosses train/test partitions")
        items.append(
            CoccItem(
                problem_id=problem_id,
                item_id=item_id,
                prompt=str(prompt),
                perturbation_type=perturbation,
                split=normalized_split,
                verifier_ref=verifier_ref,
                expected_answer=str(row.get("expected_answer") or row.get("expected_output") or ""),
                source_label=row.get("label", row.get("passed")),
            )
        )
    has_train = any(item.split == "train" for item in items)
    has_test = any(item.split == "test" for item in items)
    if prospective_only:
        if has_train or not has_test:
            raise ValueError("--prospective-only requires test rows and forbids train rows")
    elif not has_train or not has_test:
        raise ValueError("dataset requires non-overlapping train and test items")
    return items


def _entropy(logprobs: Sequence[float]) -> float:
    finite = [float(value) for value in logprobs if math.isfinite(float(value))]
    if not finite:
        return 0.0
    maximum = max(finite)
    weights = [math.exp(value - maximum) for value in finite]
    total = sum(weights)
    probabilities = [weight / total for weight in weights]
    return -sum(value * math.log(value) for value in probabilities if value > 0.0)


def _dry_turn(prompt: str) -> dict[str, Any]:
    tokens = []
    for index, token in enumerate(prompt.split()[:32] or ["dry"]):
        top1 = -0.2 - 0.03 * (index % 5)
        candidates = [top1, top1 - 1.0]
        tokens.append(
            {
                "token": token,
                "top1_logprob": top1,
                "top_logprobs": candidates,
                "gap": 1.0,
                "entropy": _entropy(candidates),
            }
        )
    return {
        "turn_index": 0,
        "user_message": prompt,
        "assistant_message": "DRY RUN: no model request executed.",
        "finish_reason": "dry_run",
        "token_count": len(tokens),
        "tokens": tokens,
    }


def _openai_turn(response: Any, prompt: str) -> dict[str, Any]:
    choice = response.choices[0]
    content = list(getattr(getattr(choice, "logprobs", None), "content", None) or [])
    tokens = []
    for entry in content:
        chosen = float(entry.logprob)
        candidates = [float(value.logprob) for value in list(entry.top_logprobs or [])]
        if not math.isfinite(chosen) or any(
            not math.isfinite(value) for value in candidates
        ):
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
        raise RuntimeError(
            "NVIDIA hosted endpoint returned no token logprobs; PRAMA cannot "
            "project this response"
        )
    return {
        "turn_index": 0,
        "user_message": prompt,
        "assistant_message": str(choice.message.content or ""),
        "finish_reason": str(choice.finish_reason or ""),
        "token_count": len(tokens),
        "tokens": tokens,
        "provider_response_id": str(getattr(response, "id", "") or ""),
        "system_fingerprint": str(
            getattr(response, "system_fingerprint", "") or ""
        ),
    }


def _call_backend(item: CoccItem, args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    if args.dry_run:
        return _dry_turn(item.prompt), args.model
    if args.provider == NVIDIA_PROVIDER:
        api_key = os.environ.get(NVIDIA_API_KEY_ENV, "").strip()
        if not api_key:
            raise RuntimeError(
                f"{NVIDIA_API_KEY_ENV} is not set in this PowerShell session"
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("OpenAI SDK is not installed") from exc
        extra_body: dict[str, Any] = {
            "chat_template_kwargs": {
                "enable_thinking": bool(args.enable_thinking),
                "force_nonempty_content": True,
            }
        }
        if args.enable_thinking and args.reasoning_budget is not None:
            extra_body["reasoning_budget"] = args.reasoning_budget
        response = OpenAI(
            api_key=api_key,
            base_url=args.base_url,
            timeout=args.timeout,
        ).chat.completions.create(
            model=args.model,
            messages=task_only_messages([{"role": "user", "content": item.prompt}]),
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            seed=args.seed,
            logprobs=True,
            top_logprobs=args.top_logprobs,
            stream=False,
            extra_body=extra_body,
        )
        return _openai_turn(response, item.prompt), str(response.model or args.model)
    raise ValueError(f"unsupported provider: {args.provider}")


def _acquire(item: CoccItem, args: argparse.Namespace) -> tuple[dict[str, Any], float, int]:
    last_error: Exception | None = None
    for attempt in range(1, args.max_attempts + 1):
        started = perf_counter()
        try:
            turn, resolved_model = _call_backend(item, args)
            elapsed = 0.0 if args.dry_run else perf_counter() - started
            return {"turn": turn, "resolved_model": resolved_model}, elapsed, attempt
        except Exception as exc:
            last_error = exc
            if attempt < args.max_attempts and args.retry_sleep_seconds > 0:
                sleep(args.retry_sleep_seconds)
    raise RuntimeError(f"generation failed after {args.max_attempts} attempts: {last_error}")


def projection_input_from_raw(
    raw: dict[str, Any],
    calibration_reference_sha256: str | None = None,
) -> dict[str, Any]:
    """Return a numeric-only view with no task, condition, answer, or label."""

    turns = []
    for turn in raw.get("turns") or []:
        numeric_tokens = []
        for token in turn.get("tokens") or []:
            numeric_tokens.append(
                {
                    "top1_logprob": float(token["top1_logprob"]),
                    "top_logprobs": [float(value) for value in token.get("top_logprobs") or []],
                    "gap": float(token.get("gap") or 0.0),
                    "entropy": float(token.get("entropy") or 0.0),
                }
            )
        turns.append(
            {
                "turn_index": int(turn.get("turn_index") or 0),
                "token_count": len(numeric_tokens),
                "tokens": numeric_tokens,
            }
        )
    request = {
        "schema": "LLM-SVM-CoCC-projector-request/1",
        "session_id": str(raw["session_id"]),
        "model_id": str(raw.get("model") or ""),
        "source_session_sha256": sha256_value(raw),
        "input_channel_status": "OBSERVED",
        "turns": turns,
    }
    if calibration_reference_sha256 is not None:
        request["calibration_reference_sha256"] = calibration_reference_sha256
    serialized = json.dumps(request, sort_keys=True)
    leaked = sorted(key for key in FORBIDDEN_PROJECTOR_KEYS if f'"{key}"' in serialized)
    if leaked:
        raise ValueError(f"projector request leaked forbidden keys: {leaked}")
    return request


def verification_request(item: CoccItem, raw: dict[str, Any]) -> dict[str, Any]:
    turn = raw["turns"][-1]
    return {
        "schema": "LLM-SVM-CoCC-verifier-request/1",
        "session_id": raw["session_id"],
        "problem_id": item.problem_id,
        "item_id": item.item_id,
        "verifier_ref": item.verifier_ref,
        "expected_answer": item.expected_answer,
        "observed_answer": turn.get("assistant_message", ""),
        "event_token": int(turn.get("token_count") or 0),
        "event_turn": int(turn.get("turn_index") or 0),
    }


def _run_json_command(command: Sequence[str], request: dict[str, Any], role: str) -> str:
    completed = subprocess.run(
        list(command),
        input=json.dumps(request, ensure_ascii=False),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{role} failed with exit code {completed.returncode}: {completed.stderr.strip()}"
        )
    if not completed.stdout.strip():
        raise RuntimeError(f"{role} returned empty stdout")
    return completed.stdout.strip()


def _verification(text: str) -> VerificationOutcome:
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("verifier must return one JSON object")
    if isinstance(value.get("passed"), bool):
        passed = value["passed"]
    elif value.get("label") in (0, 1, "0", "1"):
        passed = int(value["label"]) == 0
    else:
        status = str(value.get("status") or "").strip().lower()
        if status not in {"pass", "passed", "correct", "fail", "failed", "incorrect"}:
            raise ValueError("verifier result lacks an explicit PASS/FAIL value")
        passed = status in {"pass", "passed", "correct"}
    return VerificationOutcome(label=0 if passed else 1, passed=passed, payload=value)


def _trajectory(text: str) -> list[dict[str, Any]]:
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        decoded = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(decoded, dict):
        decoded = decoded.get("trajectory", [decoded])
    if not isinstance(decoded, list) or not decoded or not all(
        isinstance(row, dict) for row in decoded
    ):
        raise ValueError("projector must return a non-empty JSON/JSONL trajectory")
    return decoded


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    if args.resume and not args.output_dir.is_dir():
        raise FileNotFoundError(
            f"resume output directory does not exist: {args.output_dir}"
        )
    dataset_sha256 = _file_sha256(args.dataset)
    dataset_manifest_sha256 = (
        _file_sha256(args.dataset_manifest) if args.dataset_manifest else None
    )
    if args.dataset_manifest:
        normalized_manifest = json.loads(
            args.dataset_manifest.read_text(encoding="utf-8")
        )
        if (
            str(normalized_manifest.get("output_sha256") or "").lower()
            != dataset_sha256
        ):
            raise ValueError("normalization manifest does not bind --dataset")
    observed_model_blob_sha256 = None
    if args.provider == "ollama" and not args.dry_run:
        observed_model_blob_sha256 = _ollama_model_blob_sha256(args.model)
        if (
            args.expected_model_blob_sha256
            and observed_model_blob_sha256
            != args.expected_model_blob_sha256.lower()
        ):
            raise ValueError("Ollama model blob differs from the frozen design")
    items = load_cocc_items(args.dataset, prospective_only=args.prospective_only)
    if args.n:
        items = items[: args.n]
    if not items:
        raise ValueError("selection contains no items")
    if not args.queue_only and (not args.verifier_command or not args.projector_command):
        raise ValueError("a complete run requires both verifier and projector commands")

    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    sessions_dir = args.output_dir / "sessions"
    verification_dir = args.output_dir / "verification"
    projection_dir = args.output_dir / "projection"
    evaluation_dir = args.output_dir / "evaluation"
    acquisitions: list[dict[str, Any]] = []
    generation_artifacts = []
    projector_kernel_identity: dict[str, Any] | None = None

    # Phase 1: acquisition. No verifier/projector result exists during this phase.
    for index, item in enumerate(items):
        session_id = f"cocc-{index:05d}-{sha256(item.item_id.encode()).hexdigest()[:10]}"
        raw_path = sessions_dir / session_id / "raw.json"
        reuse_existing = args.resume and raw_path.is_file()
        if reuse_existing:
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            if (
                raw.get("session_id") != session_id
                or raw.get("problem_id") != item.problem_id
                or raw.get("item_id") != item.item_id
                or raw.get("provider") != args.provider
                or raw.get("model") != args.model
            ):
                raise ValueError(f"resume acquisition identity mismatch: {raw_path}")
            elapsed = float(raw["response_time_seconds"])
            attempts = int(raw["attempts"])
            print(
                f"[{index + 1}/{len(items)}] reusing {session_id}: "
                f"finish={raw['turns'][0]['finish_reason']}, "
                f"tokens={raw['turns'][0]['token_count']}",
                flush=True,
            )
        else:
            print(f"[{index + 1}/{len(items)}] acquiring {session_id}", flush=True)
            acquired, elapsed, attempts = _acquire(item, args)
            raw = {
                "session_id": session_id,
                "provider": args.provider,
                "model": acquired["resolved_model"],
                "created_at": _utc_now(),
                "response_time_seconds": elapsed,
                "measurement_status": "NOT_APPLICABLE" if args.dry_run else "OBSERVED",
                "attempts": attempts,
                "problem_id": item.problem_id,
                "item_id": item.item_id,
                "benchmark_name": CANONICAL_BENCHMARK,
                "perturbation_type": item.perturbation_type,
                "turns": [acquired["turn"]],
            }
            _atomic_json(raw_path, raw)
            print(
                f"[{index + 1}/{len(items)}] completed {session_id}: "
                f"finish={raw['turns'][0]['finish_reason']}, "
                f"tokens={raw['turns'][0]['token_count']}, attempts={attempts}, "
                f"response_seconds={elapsed:.2f}",
                flush=True,
            )
        verify_request = verification_request(item, raw)
        project_request = projection_input_from_raw(
            raw,
            (
                args.projector_calibration_sha256 or None
                if not args.projector_observer_sha256
                else None
            ),
        )
        verify_request_path = verification_dir / "requests" / f"{session_id}.json"
        project_request_path = projection_dir / "requests" / f"{session_id}.json"
        for path, expected in (
            (verify_request_path, verify_request),
            (project_request_path, project_request),
        ):
            if args.resume and path.is_file():
                if json.loads(path.read_text(encoding="utf-8")) != expected:
                    raise ValueError(f"resume request differs from reconstruction: {path}")
            else:
                _atomic_json(path, expected)
        prompt_hash = sha256(item.prompt.encode("utf-8")).hexdigest()
        answer = raw["turns"][0]["assistant_message"]
        generation_artifacts.append(
            {
                **make_envelope(
                    artifact_type="generation_observation",
                    study_id="CoCC-PRAMA",
                    session_id=session_id,
                    producer="scripts.run_break_the_chain_prama_eval_nvidia/1",
                    created_at=raw["created_at"],
                    source_sha256=sha256_value(raw),
                    config_sha256=sha256_value(
                        {
                            "provider": args.provider,
                            "model": args.model,
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                            "max_tokens": args.max_tokens,
                            "top_logprobs": args.top_logprobs,
                            "seed": args.seed,
                            "enable_thinking": args.enable_thinking,
                            "reasoning_budget": args.reasoning_budget,
                            "base_url": args.base_url,
                        }
                    ),
                    partition=("calibration" if item.split == "train" else "confirmatory"),
                    channel_status=(
                        ChannelStatus.NOT_APPLICABLE if args.dry_run else ChannelStatus.OBSERVED
                    ),
                ),
                "model_id": raw["model"],
                "prompt_sha256": prompt_hash,
                "response_sha256": sha256(answer.encode("utf-8")).hexdigest(),
                "response_time_seconds": elapsed,
                "finish_reason": raw["turns"][0]["finish_reason"],
                "token_count": raw["turns"][0]["token_count"],
            }
        )
        acquisitions.append(
            {
                "item": item,
                "raw": raw,
                "raw_path": raw_path,
                "verify_request": verify_request,
                "verify_request_path": verify_request_path,
                "project_request": project_request,
                "project_request_path": project_request_path,
            }
        )
    write_jsonl_atomic(args.output_dir / "generation_observation.jsonl", generation_artifacts)

    # Phase 2: verification. This channel never receives projection coordinates.
    verifications: dict[str, dict[str, Any]] = {}
    if args.verifier_command:
        for phase_index, row in enumerate(acquisitions, 1):
            session_id = row["raw"]["session_id"]
            result_path = verification_dir / "results" / f"{session_id}.json"
            if args.resume and result_path.is_file():
                result = json.loads(result_path.read_text(encoding="utf-8"))
                outcome = _verification(json.dumps(result))
                print(
                    f"[verify {phase_index}/{len(acquisitions)}] reusing {session_id}",
                    flush=True,
                )
            else:
                outcome = _verification(
                    _run_json_command(
                        args.verifier_command, row["verify_request"], "verifier"
                    )
                )
                result = {
                    **outcome.payload,
                    "session_id": session_id,
                    "label": outcome.label,
                    "passed": outcome.passed,
                    "label_semantics": "0=PASS,1=FAIL",
                    "generated_at": _utc_now(),
                }
                _atomic_json(result_path, result)
            verifications[session_id] = {"outcome": outcome, "path": result_path}

    # Phase 3: projection. This channel receives numeric token observations only.
    projections: dict[str, Path] = {}
    if args.projector_command:
        for phase_index, row in enumerate(acquisitions, 1):
            session_id = row["raw"]["session_id"]
            trajectory_path = projection_dir / session_id / "trajectory.jsonl"
            if args.resume and trajectory_path.is_file():
                projected = _trajectory(trajectory_path.read_text(encoding="utf-8"))
                print(
                    f"[project {phase_index}/{len(acquisitions)}] reusing {session_id}",
                    flush=True,
                )
            else:
                projected = _trajectory(
                    _run_json_command(
                        args.projector_command, row["project_request"], "projector"
                    )
                )
                write_jsonl_atomic(trajectory_path, projected)
                print(
                    f"[project {phase_index}/{len(acquisitions)}] completed {session_id}",
                    flush=True,
                )
            for point in projected:
                if point.get("session_id") != session_id:
                    raise ValueError("projector trajectory session_id mismatch")
                if point.get("coordinate_origin") != "DERIVED_KERNEL_STATE":
                    raise ValueError("projector trajectory lacks derived-state provenance")
                if point.get("input_channel_status") != "OBSERVED":
                    raise ValueError("projector trajectory lacks observed-input provenance")
                if args.projector_observer_sha256:
                    if (
                        point.get("observer_contract_sha256")
                        != args.projector_observer_sha256
                    ):
                        raise ValueError("projector trajectory observer hash mismatch")
                    if point.get("observer_model_specific_parameters") is not False:
                        raise ValueError("dynamic observer contains model-specific parameters")
                    if point.get("observer_external_calibration") is not False:
                        raise ValueError("dynamic observer unexpectedly uses calibration")
                elif (
                    point.get("calibration_reference_sha256")
                    != args.projector_calibration_sha256
                ):
                    raise ValueError("projector trajectory calibration hash mismatch")
                if not isinstance(point.get("kernel_identity"), dict):
                    raise ValueError("projector trajectory lacks kernel_identity")
                if projector_kernel_identity is None:
                    projector_kernel_identity = dict(point["kernel_identity"])
                elif point["kernel_identity"] != projector_kernel_identity:
                    raise ValueError(
                        "projector kernel_identity changed within the run"
                    )
            projections[session_id] = trajectory_path

    # Phase 4: blind join after both independent channels are immutable on disk.
    join_rows = []
    label_rows = []
    source_audit = []
    for row in acquisitions:
        item = row["item"]
        raw = row["raw"]
        session_id = raw["session_id"]
        source_audit.append(
            {
                "session_id": session_id,
                "problem_id": item.problem_id,
                "item_id": item.item_id,
                "source_label_raw": "" if item.source_label is None else str(item.source_label),
                "note": "audit_only_not_run_truth",
            }
        )
        if session_id not in verifications:
            continue
        outcome = verifications[session_id]["outcome"]
        turn = raw["turns"][0]
        label_rows.append(
            {
                "session_id": session_id,
                "label": outcome.label,
                "event_token": turn["token_count"],
                "event_turn": 0,
                "event_type": "external_verifier_failure" if outcome.label else "external_verifier_pass",
                "split": item.split,
                "group_sha256": sha256(item.problem_id.encode()).hexdigest(),
            }
        )
        if session_id in projections:
            join_rows.append(
                {
                    "session_id": session_id,
                    "label": outcome.label,
                    "split": item.split,
                    "group_sha256": sha256(item.problem_id.encode()).hexdigest(),
                    "event_token": turn["token_count"],
                    "verification_result_path": str(verifications[session_id]["path"]),
                    "trajectory_path": str(projections[session_id]),
                    "projection_request_path": str(row["project_request_path"]),
                }
            )

    _write_csv(
        args.output_dir / "labels.csv",
        ("session_id", "label", "event_token", "event_turn", "event_type", "split", "group_sha256"),
        label_rows,
    )
    _write_csv(
        args.output_dir / "source_labels_audit.csv",
        ("session_id", "problem_id", "item_id", "source_label_raw", "note"),
        source_audit,
    )
    _write_csv(
        evaluation_dir / "blind_join.csv",
        (
            "session_id",
            "label",
            "split",
            "group_sha256",
            "event_token",
            "verification_result_path",
            "trajectory_path",
            "projection_request_path",
        ),
        join_rows,
    )
    manifest = {
        "schema": "LLM-SVM-CoCC-PRAMA-run/1",
        "generated_at": _utc_now(),
        "benchmark_name": CANONICAL_BENCHMARK,
        "benchmark_alias": BENCHMARK_ALIAS,
        "dry_run": args.dry_run,
        "queue_only": args.queue_only,
        "provider": args.provider,
        "provider_endpoint": args.base_url,
        "api_key_environment_variable": NVIDIA_API_KEY_ENV,
        "model": args.model,
        "model_blob_sha256": observed_model_blob_sha256,
        "expected_model_blob_sha256": (
            args.expected_model_blob_sha256.lower()
            if args.expected_model_blob_sha256
            else None
        ),
        "dataset_sha256": dataset_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "confirmatory_design_sha256": args.confirmatory_design_sha256 or None,
        "observation_interface": {
            "projector_request_schema": "LLM-SVM-CoCC-projector-request/1",
            "runner_sha256": _file_sha256(Path(__file__)),
            "model_payload_sha256": _file_sha256(
                SRC_ROOT / "aptadynamic_llm" / "model_payload.py"
            ),
        },
        "generation_parameter_set": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_logprobs": args.top_logprobs,
            "seed": args.seed,
            "enable_thinking": args.enable_thinking,
            "reasoning_budget": args.reasoning_budget,
        },
        "provider_response_identity": {
            "resolved_models": sorted(
                {
                    str(entry["raw"].get("model") or "")
                    for entry in acquisitions
                }
            ),
            "system_fingerprints": sorted(
                {
                    str(
                        entry["raw"]["turns"][0].get(
                            "system_fingerprint"
                        )
                        or ""
                    )
                    for entry in acquisitions
                }
            ),
        },
        "projector_kernel_identity": projector_kernel_identity,
        "session_count": len(acquisitions),
        "verified_count": len(verifications),
        "projected_count": len(projections),
        "blind_join_count": len(join_rows),
        "model_payload_boundary": "task_prompt_and_generation_parameters_only",
        "projector_boundary": "numeric_token_observations_only",
        "projector_calibration_sha256": args.projector_calibration_sha256 or None,
        "projector_observer_sha256": args.projector_observer_sha256 or None,
        "source_labels_are_run_truth": False,
    }
    _atomic_json(args.output_dir / "manifest.json", manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--provider", choices=(NVIDIA_PROVIDER,), default=NVIDIA_PROVIDER
    )
    parser.add_argument("--model", default=NVIDIA_MODEL)
    parser.add_argument("--base-url", default=NVIDIA_BASE_URL)
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-logprobs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Nemotron reasoning mode; frozen studies must set this explicitly.",
    )
    parser.add_argument(
        "--reasoning-budget",
        type=int,
        help="Reasoning token budget; valid only with --enable-thinking.",
    )
    parser.add_argument("--confirmatory-design-sha256", default="")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-sleep-seconds", type=float, default=5.0)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--queue-only", action="store_true")
    parser.add_argument(
        "--prospective-only",
        action="store_true",
        help="Require an already frozen test-only prospective dataset.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse immutable raw/request artifacts from an interrupted output directory.",
    )
    parser.add_argument("--verifier-command", nargs="+")
    parser.add_argument("--projector-command", nargs="+")
    parser.add_argument(
        "--verifier-command-json",
        help="JSON array command form; use this when the command itself has options.",
    )
    parser.add_argument(
        "--projector-command-json",
        help="JSON array command form; use this when the command itself has options.",
    )
    parser.add_argument(
        "--projector-calibration-sha256",
        default="",
        help="Frozen causal-calibration digest required by a complete projector run.",
    )
    parser.add_argument(
        "--projector-observer-sha256",
        default="",
        help="Frozen universal dynamic-observer digest; mutually exclusive with calibration.",
    )
    args = parser.parse_args(argv)
    args.expected_model_blob_sha256 = ""
    for role in ("verifier", "projector"):
        command = getattr(args, f"{role}_command")
        encoded = getattr(args, f"{role}_command_json")
        if command and encoded:
            parser.error(
                f"--{role}-command and --{role}-command-json are mutually exclusive"
            )
        if encoded:
            try:
                decoded = json.loads(encoded)
            except json.JSONDecodeError as exc:
                parser.error(f"--{role}-command-json is not valid JSON: {exc}")
            if (
                not isinstance(decoded, list)
                or not decoded
                or not all(isinstance(value, str) and value for value in decoded)
            ):
                parser.error(f"--{role}-command-json must be a non-empty string array")
            setattr(args, f"{role}_command", decoded)
    if (
        args.max_attempts <= 0
        or args.max_tokens <= 0
        or not 2 <= args.top_logprobs <= 20
    ):
        parser.error(
            "max-attempts/max-tokens must be positive and top-logprobs must be 2..20"
        )
    if args.model not in NVIDIA_MODELS:
        parser.error(f"--model must be one of {NVIDIA_MODELS}")
    args.base_url = args.base_url.rstrip("/")
    if args.base_url != NVIDIA_BASE_URL:
        parser.error(f"this runner is frozen to --base-url {NVIDIA_BASE_URL}")
    if args.temperature != 1.0 or args.top_p != 0.95:
        parser.error("Nemotron 3 requires temperature=1.0 and top_p=0.95")
    if args.enable_thinking:
        if args.reasoning_budget is None or not 0 < args.reasoning_budget <= args.max_tokens:
            parser.error(
                "--enable-thinking requires --reasoning-budget in 1..max-tokens"
            )
    elif args.reasoning_budget is not None:
        parser.error("--reasoning-budget requires --enable-thinking")
    for field in ("confirmatory_design_sha256",):
        value = getattr(args, field).removeprefix("sha256:").lower()
        if value and (
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            parser.error(f"--{field.replace('_', '-')} must be a SHA-256 digest")
        setattr(args, field, value)
    if args.projector_command:
        if bool(args.projector_calibration_sha256) == bool(args.projector_observer_sha256):
            parser.error(
                "a projector requires exactly one of --projector-calibration-sha256 "
                "or --projector-observer-sha256"
            )
        for field in ("projector_calibration_sha256", "projector_observer_sha256"):
            digest = getattr(args, field).removeprefix("sha256:").lower()
            if digest and (
                len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                parser.error(
                    f"--{field.replace('_', '-')} must be a SHA-256 digest"
                )
            setattr(args, field, digest)
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        manifest = run(parse_args(argv))
    except (FileExistsError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"CoCC PRAMA run failed: {exc}")
        return 1
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
