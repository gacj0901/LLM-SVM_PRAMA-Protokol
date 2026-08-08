#!/usr/bin/env python3
"""Run a bounded, resumable exploratory microbenchmark for ODCE acquisition.

The model receives only the task prompt. Verification happens after generation
and is emitted as a causally available domain-return observation at the final
observed token window. The script intentionally does not freeze an ODCE
contract or alter PRAMA/D_O state.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import sys
import tempfile
from time import monotonic
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aptadynamic_llm.artifact_schema import validate_artifact  # noqa: E402
from aptadynamic_llm.model_payload import task_only_prompt  # noqa: E402


SCHEMA = "LLM-SVM-odce-microbenchmark-suite/1"
RAW_SCHEMA = "LLM-SVM-odce-microbenchmark-raw/1"
REPORT_SCHEMA = "LLM-SVM-odce-microbenchmark-report/1"
DEFAULT_SUITE = ROOT / "data" / "odce_microbenchmarks_v1.json"
STUDY_ID = "odce-microbenchmarks-v1"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
GSM8K_SUFFIX = (
    "\n\nSolve the problem. End your response with exactly one line in the form "
    "`Final answer: <number>`."
)
TIER_ORDER = {"smoke": 1, "pilot": 2, "full": 3}
MODEL_PROFILES: dict[str, dict[str, Any]] = {
    "nvidia/nemotron-3-super-120b-a12b": {
        "provider": "nvidia_nim",
        "adapter": "nemotron",
        "temperature": 1.0,
        "top_p": 0.95,
        "enable_thinking": False,
    },
    "mistralai/mistral-medium-3.5-128b": {
        "provider": "nvidia_nim",
        "adapter": "mistral",
        "availability_status": "EOL",
        "eol_at": "2026-08-07T09:00:00Z",
        "temperature": 0.7,
        "top_p": 1.0,
        "reasoning_effort": "none",
    },
    "nvidia/nemotron-3-ultra-550b-a55b": {
        "provider": "nvidia_nim",
        "adapter": "nemotron",
        "temperature": 1.0,
        "top_p": 0.95,
        "enable_thinking": False,
    },
    "hermes3:8b": {
        "provider": "ollama",
        "adapter": "ollama",
        "temperature": 0.2,
        "top_p": 0.95,
    },
}
SUPPORTED_IFEVAL = {
    "change_case:english_lowercase",
    "change_case:english_capital",
    "detectable_format:json_format",
    "detectable_format:number_bullet_lists",
    "keywords:forbidden_words",
    "keywords:existence",
    "length_constraints:number_words",
    "startend:end_checker",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def load_suite(path: Path) -> dict[str, Any]:
    decoded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict) or decoded.get("schema") != SCHEMA:
        raise ValueError(f"unsupported suite schema in {path}")
    items = decoded.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("suite requires a non-empty items array")
    seen: set[str] = set()
    ranks: dict[str, set[int]] = {}
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"item {index} is not an object")
        item_id = str(item.get("item_id", ""))
        family = str(item.get("family", ""))
        rank = item.get("tier_rank")
        verifier = item.get("verifier")
        if not item_id or item_id in seen:
            raise ValueError(f"item {index} has a missing or duplicate item_id")
        seen.add(item_id)
        if not family or not isinstance(rank, int) or rank < 1:
            raise ValueError(f"{item_id}: invalid family or tier_rank")
        ranks.setdefault(family, set()).add(rank)
        task_only_prompt(str(item.get("prompt", "")))
        if not isinstance(verifier, dict):
            raise ValueError(f"{item_id}: verifier is required")
        if verifier.get("type") == "ifeval_constraints":
            constraints = verifier.get("constraints")
            if not isinstance(constraints, list) or not constraints:
                raise ValueError(f"{item_id}: constraints are required")
            unknown = {
                str(constraint.get("id"))
                for constraint in constraints
                if str(constraint.get("id")) not in SUPPORTED_IFEVAL
            }
            if unknown:
                raise ValueError(f"{item_id}: unsupported constraints {sorted(unknown)}")
            external_constraints = verifier.get("external_integration_constraints")
            if external_constraints is not None:
                if not isinstance(external_constraints, list) or not external_constraints:
                    raise ValueError(
                        f"{item_id}: external_integration_constraints must be a nonempty list"
                    )
                unknown_external = {
                    str(constraint.get("id"))
                    for constraint in external_constraints
                    if str(constraint.get("id")) not in SUPPORTED_IFEVAL
                }
                if unknown_external:
                    raise ValueError(
                        f"{item_id}: unsupported external integration constraints "
                        f"{sorted(unknown_external)}"
                    )
        elif verifier.get("type") != "numeric_exact":
            raise ValueError(f"{item_id}: unsupported verifier type")
    full_size = int(decoded["tier_sizes_per_family"]["full"])
    expected = set(range(1, full_size + 1))
    if any(value != expected for value in ranks.values()):
        raise ValueError("each family must contain contiguous full-tier ranks")
    return decoded


def select_items(
    suite: Mapping[str, Any], tier: str, family: str
) -> list[dict[str, Any]]:
    maximum = int(suite["tier_sizes_per_family"][tier])
    selected = [
        dict(item)
        for item in suite["items"]
        if int(item["tier_rank"]) <= maximum
        and (family == "all" or item["family"] == family)
    ]
    return sorted(selected, key=lambda item: (int(item["tier_rank"]), item["family"]))


def effective_prompt(item: Mapping[str, Any]) -> str:
    prompt = str(item["prompt"])
    if item["family"] == "gsm8k":
        prompt += GSM8K_SUFFIX
    return task_only_prompt(prompt)


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", stripped, re.IGNORECASE)
    return match.group(1).strip() if match else stripped


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[^\W_]+(?:['’-][^\W_]+)*\b", text, re.UNICODE))


def _contains_word(text: str, word: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(word)}(?!\w)", text, re.IGNORECASE) is not None


def verify_ifeval_constraint(text: str, constraint: Mapping[str, Any]) -> dict[str, Any]:
    constraint_id = str(constraint["id"])
    kwargs = dict(constraint.get("kwargs") or {})
    detail: dict[str, Any] = {"constraint_id": constraint_id}
    if constraint_id == "change_case:english_lowercase":
        detail["uppercase_ascii_count"] = len(re.findall(r"[A-Z]", text))
        passed = detail["uppercase_ascii_count"] == 0
    elif constraint_id == "change_case:english_capital":
        detail["lowercase_ascii_count"] = len(re.findall(r"[a-z]", text))
        passed = detail["lowercase_ascii_count"] == 0
    elif constraint_id == "detectable_format:json_format":
        try:
            json.loads(_strip_json_fence(text))
            passed = True
        except json.JSONDecodeError:
            passed = False
    elif constraint_id == "detectable_format:number_bullet_lists":
        # CommonMark unordered list markers are '-', '+' and '*', optionally
        # preceded by up to three spaces and followed by whitespace or EOL.
        count = len(re.findall(r"(?m)^[ ]{0,3}[-+*](?:[ \t]+|$)", text))
        detail.update({"observed_bullets": count, "expected_bullets": kwargs["num_bullets"]})
        passed = count == int(kwargs["num_bullets"])
    elif constraint_id == "keywords:forbidden_words":
        present = [word for word in kwargs["forbidden_words"] if _contains_word(text, word)]
        detail["present_forbidden_words"] = present
        passed = not present
    elif constraint_id == "keywords:existence":
        missing = [word for word in kwargs["keywords"] if not _contains_word(text, word)]
        detail["missing_keywords"] = missing
        passed = not missing
    elif constraint_id == "length_constraints:number_words":
        count = _word_count(text)
        target = int(kwargs["num_words"])
        relation = str(kwargs["relation"])
        detail.update({"observed_words": count, "relation": relation, "target_words": target})
        if relation == "less than":
            passed = count < target
        elif relation == "at least":
            passed = count >= target
        elif relation == "exactly":
            passed = count == target
        else:
            raise ValueError(f"unsupported word-count relation {relation!r}")
    elif constraint_id == "startend:end_checker":
        phrase = str(kwargs["end_phrase"])
        passed = text.rstrip().endswith(phrase)
        detail["end_phrase"] = phrase
    else:  # guarded by suite validation
        raise ValueError(f"unsupported IFEval constraint {constraint_id!r}")
    detail["passed"] = bool(passed)
    return detail


def _decimal_from_response(text: str) -> Decimal | None:
    marker = re.findall(
        r"final\s+answer\s*:\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    )
    candidates = marker or re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    if not candidates:
        return None
    try:
        return Decimal(candidates[-1].replace(",", ""))
    except InvalidOperation:
        return None


def verify_response(item: Mapping[str, Any], text: str) -> dict[str, Any]:
    verifier = dict(item["verifier"])
    if verifier["type"] == "numeric_exact":
        expected = Decimal(str(verifier["expected"]))
        observed = _decimal_from_response(text)
        passed = observed == expected
        checks = [{
            "constraint_id": "gsm8k:numeric_exact",
            "passed": passed,
            "expected": str(expected),
            "observed": str(observed) if observed is not None else None,
        }]
    else:
        checks = [
            verify_ifeval_constraint(text, constraint)
            for constraint in verifier["constraints"]
        ]
        external_checks = [
            verify_ifeval_constraint(text, constraint)
            for constraint in verifier.get("external_integration_constraints", [])
        ]
        passed = all(check["passed"] for check in checks + external_checks)
    score = sum(bool(check["passed"]) for check in checks) / len(checks)
    external_score = (
        sum(bool(check["passed"]) for check in external_checks)
        / len(external_checks)
        if verifier["type"] != "numeric_exact" and external_checks
        else None
    )
    return {
        "verifier_type": verifier["type"],
        "passed": bool(passed),
        "functional_gain": float(score),
        "external_integration": (
            float(external_score) if external_score is not None else None
        ),
        "verified_outcome": 1.0 if passed else 0.0,
        "checks": checks,
        "external_integration_checks": (
            external_checks if verifier["type"] != "numeric_exact" else []
        ),
    }


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "--", value).strip("-")


def generation_profile(args: argparse.Namespace) -> dict[str, Any]:
    profile = dict(MODEL_PROFILES[args.model])
    return {
        "provider": profile["provider"],
        "model": args.model,
        "temperature": profile["temperature"],
        "top_p": profile["top_p"],
        "max_tokens": args.max_tokens,
        "top_logprobs": args.top_logprobs,
        "seed": args.seed,
        "thinking_enabled": bool(profile.get("enable_thinking", False)),
        "reasoning_effort": profile.get("reasoning_effort"),
    }


def acquire(item: Mapping[str, Any], prompt: str, args: argparse.Namespace) -> tuple[dict[str, Any], float, int]:
    profile = MODEL_PROFILES[args.model]
    common = dict(
        problem_id=str(item["source_item_id"]),
        item_id=str(item["item_id"]),
        prompt=prompt,
        perturbation_type="clean_control",
        split="test",
        verifier_ref="post_generation_local_verifier",
        expected_answer="",
        source_label=None,
    )
    namespace = argparse.Namespace(
        dry_run=False,
        provider=profile["provider"],
        model=args.model,
        base_url=(NVIDIA_BASE_URL if profile["provider"] == "nvidia_nim" else args.ollama_base_url),
        max_tokens=args.max_tokens,
        temperature=profile["temperature"],
        top_p=profile["top_p"],
        top_logprobs=args.top_logprobs,
        seed=args.seed,
        timeout=args.timeout,
        max_attempts=args.max_attempts,
        retry_sleep_seconds=args.retry_sleep_seconds,
        enable_thinking=profile.get("enable_thinking", False),
        reasoning_budget=None,
        reasoning_effort=profile.get("reasoning_effort", "none"),
    )
    if profile["adapter"] == "mistral":
        from scripts import run_break_the_chain_prama_eval_nvidia_mistral as adapter
    elif profile["adapter"] == "nemotron":
        from scripts import run_break_the_chain_prama_eval_nvidia as adapter
    else:
        from scripts import run_break_the_chain_prama_eval as adapter
    adapted_item = adapter.CoccItem(**common)
    acquired, elapsed, attempts = adapter._acquire(adapted_item, namespace)
    return dict(acquired), float(elapsed), int(attempts)


def make_domain_outcome(
    *,
    study_id: str,
    raw_path: Path,
    raw: Mapping[str, Any],
    verification: Mapping[str, Any],
    suite_sha256: str,
    verifier_sha256: str,
    window_size: int,
) -> dict[str, Any]:
    token_count = int(raw["turns"][0]["token_count"])
    terminal_window = max(0, math.ceil(token_count / window_size) - 1)
    values = {
        "functional_gain": float(verification["functional_gain"]),
        "external_integration": (
            float(verification["external_integration"])
            if verification.get("external_integration") is not None
            else None
        ),
        "verified_outcome": float(verification["verified_outcome"]),
    }
    outcome = {
        "contract_version": "0.2.0",
        "artifact_type": "domain_return_observation",
        "artifact_version": "1.0.0",
        "study_id": study_id,
        "session_id": str(raw["session_id"]),
        "producer": "scripts/run_odce_microbenchmarks.py",
        "created_at": utc_now(),
        "source_sha256": file_sha256(raw_path),
        "config_sha256": suite_sha256,
        "partition": "exploratory",
        "channel_status": "OBSERVED",
        "event_index": terminal_window,
        "available_at_index": terminal_window,
        "event_window": {"turn_index": 0, "window_index": terminal_window},
        "available_at_window": {"turn_index": 0, "window_index": terminal_window},
        "benefit_vector": values,
        "component_status": {
            "functional_gain": "OBSERVED",
            "external_integration": (
                "OBSERVED"
                if values["external_integration"] is not None
                else "UNAVAILABLE"
            ),
            "verified_outcome": "OBSERVED",
        },
        "verifier_reference_sha256": verifier_sha256,
        "retrospective_backfill": False,
        "causal_availability_declared": True,
        "provider_termination_metadata_used": False,
    }
    validate_artifact(outcome, "domain_return_observation")
    return outcome


def _is_timeout_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".casefold()
    return "timeout" in text or "timed out" in text


def _is_provider_server_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".casefold()
    return (
        "error code: 500" in text
        or "status code: 500" in text
        or "internal server error" in text
    )


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _rebuild_outcome_jsonl(output_dir: Path) -> int:
    outcomes = [
        _load_json(path)
        for path in sorted((output_dir / "outcomes").glob("*.json"))
    ]
    atomic_jsonl(output_dir / "domain_return_observations.jsonl", outcomes)
    return len(outcomes)


def run(args: argparse.Namespace) -> dict[str, Any]:
    suite = load_suite(args.suite)
    suite_hash = canonical_sha256(suite)
    items = select_items(suite, args.tier, args.family)
    if args.item_id:
        requested = set(args.item_id)
        known = {str(item["item_id"]) for item in suite["items"]}
        unknown = requested - known
        if unknown:
            raise ValueError(f"unknown --item-id values: {sorted(unknown)}")
        items = [item for item in items if str(item["item_id"]) in requested]
    if not items:
        raise ValueError("selection contains no items; check tier/family/item-id")
    profile = generation_profile(args)
    profile_hash = canonical_sha256(profile)
    if args.dry_run:
        return {
            "schema": REPORT_SCHEMA,
            "mode": "dry_run",
            "suite_id": suite["suite_id"],
            "suite_sha256": suite_hash,
            "tier": args.tier,
            "family": args.family,
            "requested_item_ids": args.item_id or None,
            "selected_count": len(items),
            "model_profile": profile,
            "item_ids": [item["item_id"] for item in items],
        }
    availability = MODEL_PROFILES[args.model].get("availability_status")
    if availability == "EOL":
        raise ValueError(
            f"{args.model} reached end of life at "
            f"{MODEL_PROFILES[args.model]['eol_at']} and cannot be acquired"
        )

    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not args.resume:
        raise ValueError(f"{manifest_path} exists; use --resume or a new --output-dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    started_clock = monotonic()
    completed: list[str] = []
    failures: list[dict[str, Any]] = []
    timeout_count = 0
    provider_server_error_count = 0
    halted_reason: str | None = None

    def write_manifest() -> dict[str, Any]:
        manifest = {
            "schema": REPORT_SCHEMA,
            "suite_id": suite["suite_id"],
            "suite_sha256": suite_hash,
            "suite_path": str(args.suite.resolve()),
            "partition": "exploratory",
            "contract_freeze_sha256": None,
            "tier": args.tier,
            "family": args.family,
            "requested_item_ids": args.item_id or None,
            "model_profile": profile,
            "model_profile_sha256": profile_hash,
            "started_at": started_at,
            "updated_at": utc_now(),
            "selected_count": len(items),
            "completed_count": len(completed),
            "failed_count": len(failures),
            "timeout_count": timeout_count,
            "provider_server_error_count": provider_server_error_count,
            "halted_reason": halted_reason,
            "completed_item_ids": completed,
            "failures": failures,
            "outcomes_file": str((output_dir / "domain_return_observations.jsonl").resolve()),
        }
        atomic_json(manifest_path, manifest)
        return manifest

    for ordinal, item in enumerate(items, start=1):
        if monotonic() - started_clock >= args.run_deadline_minutes * 60:
            halted_reason = "run_deadline_exceeded"
            break
        item_id = str(item["item_id"])
        session_id = f"{_slug(args.model)}--{item_id}"
        session_dir = output_dir / "sessions" / session_id
        raw_path = session_dir / "raw.json"
        verification_path = session_dir / "verification.json"
        outcome_path = output_dir / "outcomes" / f"{session_id}.json"
        prompt = effective_prompt(item)
        prompt_hash = sha256(prompt.encode("utf-8")).hexdigest()
        print(f"[{ordinal}/{len(items)}] {item_id}", flush=True)
        try:
            if raw_path.exists():
                if not args.resume:
                    raise ValueError(f"raw artifact already exists: {raw_path}")
                raw = _load_json(raw_path)
                expected_identity = (
                    raw.get("suite_sha256") == suite_hash
                    and raw.get("model_profile_sha256") == profile_hash
                    and raw.get("benchmark_item_id") == item_id
                    and raw.get("prompt_sha256") == prompt_hash
                )
                if not expected_identity:
                    raise ValueError(f"resume identity mismatch for {item_id}")
                print("  resume: raw reused", flush=True)
            else:
                acquired, elapsed, attempts = acquire(item, prompt, args)
                turn = dict(acquired["turn"])
                raw = {
                    "schema": RAW_SCHEMA,
                    "session_id": session_id,
                    "provider": profile["provider"],
                    "provider_endpoint": (
                        NVIDIA_BASE_URL if profile["provider"] == "nvidia_nim" else args.ollama_base_url
                    ),
                    "model": args.model,
                    "resolved_model": acquired["resolved_model"],
                    "created_at": utc_now(),
                    "response_time_seconds": elapsed,
                    "measurement_status": "OBSERVED",
                    "attempts": attempts,
                    "benchmark_name": item["family"],
                    "benchmark_item_id": item_id,
                    "source_item_id": str(item["source_item_id"]),
                    "suite_sha256": suite_hash,
                    "model_profile_sha256": profile_hash,
                    "prompt_sha256": prompt_hash,
                    "turns": [turn],
                }
                atomic_json(raw_path, raw)
                print(
                    f"  acquired: {turn['token_count']} tokens in {elapsed:.1f}s",
                    flush=True,
                )
            answer = str(raw["turns"][0]["assistant_message"])
            score = verify_response(item, answer)
            verifier_hash = canonical_sha256(item["verifier"])
            verification = {
                "schema": "LLM-SVM-odce-microbenchmark-verification/1",
                "session_id": session_id,
                "benchmark_name": item["family"],
                "benchmark_item_id": item_id,
                "source_item_id": str(item["source_item_id"]),
                "created_at": utc_now(),
                "verifier_reference_sha256": verifier_hash,
                **score,
            }
            atomic_json(verification_path, verification)
            outcome = make_domain_outcome(
                study_id=str(suite["suite_id"]).replace("_", "-"),
                raw_path=raw_path,
                raw=raw,
                verification=verification,
                suite_sha256=suite_hash,
                verifier_sha256=verifier_hash,
                window_size=args.window_size,
            )
            atomic_json(outcome_path, outcome)
            completed.append(item_id)
            print(
                f"  verified: pass={verification['passed']} gain={verification['functional_gain']:.3f}",
                flush=True,
            )
        except Exception as exc:
            timed_out = _is_timeout_error(exc)
            provider_server_error = _is_provider_server_error(exc)
            if timed_out:
                timeout_count += 1
            if provider_server_error:
                provider_server_error_count += 1
            failure = {
                "item_id": item_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "timeout": timed_out,
                "provider_server_error": provider_server_error,
                "created_at": utc_now(),
            }
            failures.append(failure)
            atomic_json(output_dir / "failures" / f"{session_id}.json", failure)
            print(f"  failed: {type(exc).__name__}: {exc}", flush=True)
            if timeout_count >= args.max_timeouts:
                halted_reason = "timeout_budget_exhausted"
            elif provider_server_error_count >= args.max_provider_server_errors:
                halted_reason = "provider_server_error_budget_exhausted"
        _rebuild_outcome_jsonl(output_dir)
        write_manifest()
        if halted_reason:
            break
    manifest = write_manifest()
    manifest["status"] = (
        "COMPLETE"
        if len(completed) == len(items) and not failures
        else "INCOMPLETE"
    )
    manifest["updated_at"] = utc_now()
    atomic_json(manifest_path, manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--model", required=True, choices=tuple(MODEL_PROFILES))
    parser.add_argument("--tier", choices=tuple(TIER_ORDER), default="pilot")
    parser.add_argument(
        "--family",
        default="all",
        help="Select one family declared by the suite, or all families.",
    )
    parser.add_argument(
        "--item-id",
        action="append",
        help="Run only this selected item; repeat the option for multiple items.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--top-logprobs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--retry-sleep-seconds", type=float, default=3.0)
    parser.add_argument("--run-deadline-minutes", type=float, default=30.0)
    parser.add_argument("--max-timeouts", type=int, default=2)
    parser.add_argument(
        "--max-provider-server-errors",
        type=int,
        default=1,
        help="Stop after this many failed items caused by provider HTTP 500 errors.",
    )
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--ollama-base-url", default="http://localhost:11434")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the plan without contacting a model or writing output.",
    )
    args = parser.parse_args(argv)
    if (
        args.max_tokens <= 0
        or not 2 <= args.top_logprobs <= 20
        or args.timeout <= 0
        or args.max_attempts <= 0
        or args.run_deadline_minutes <= 0
        or args.max_timeouts <= 0
        or args.max_provider_server_errors <= 0
        or args.window_size <= 0
    ):
        parser.error("numeric limits must be positive and top-logprobs must be 2..20")
    args.ollama_base_url = args.ollama_base_url.rstrip("/")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run(args)
    except (ImportError, KeyError, OSError, TypeError, ValueError) as exc:
        print(f"microbenchmark failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report.get("mode") == "dry_run" or report.get("status") == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
