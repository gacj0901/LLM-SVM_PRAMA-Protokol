#!/usr/bin/env python
"""Verify NVIDIA authentication and token-logprob support with one tiny call."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_break_the_chain_prama_eval_nvidia import (
    CoccItem,
    NVIDIA_API_KEY_ENV,
    NVIDIA_BASE_URL,
    NVIDIA_MODEL,
    NVIDIA_MODELS,
    NVIDIA_PROVIDER,
    _call_backend,
)


def run_preflight(
    execute: bool,
    timeout: int,
    top_logprobs: int,
    model: str = NVIDIA_MODEL,
) -> dict[str, object]:
    key_present = bool(os.environ.get(NVIDIA_API_KEY_ENV, "").strip())
    result: dict[str, object] = {
        "schema": "LLM-SVM-NVIDIA-preflight/1",
        "provider": NVIDIA_PROVIDER,
        "endpoint": NVIDIA_BASE_URL,
        "requested_model": model,
        "api_key_environment_variable": NVIDIA_API_KEY_ENV,
        "api_key_present": key_present,
        "remote_call_executed": False,
    }
    if not execute:
        return result
    if not key_present:
        raise RuntimeError(
            f"{NVIDIA_API_KEY_ENV} is not set in this PowerShell session"
        )
    item = CoccItem(
        problem_id="preflight",
        item_id="preflight",
        prompt="Respond with exactly: OK",
        perturbation_type="preflight",
        split="train",
        verifier_ref=None,
        expected_answer="OK",
        source_label=None,
    )
    args = SimpleNamespace(
        dry_run=False,
        provider=NVIDIA_PROVIDER,
        model=model,
        base_url=NVIDIA_BASE_URL,
        timeout=timeout,
        temperature=1.0,
        top_p=0.95,
        max_tokens=8,
        seed=1337,
        top_logprobs=top_logprobs,
        enable_thinking=False,
        reasoning_budget=None,
    )
    turn, resolved_model = _call_backend(item, args)
    result.update(
        {
            "remote_call_executed": True,
            "resolved_model": resolved_model,
            "finish_reason": turn["finish_reason"],
            "token_logprobs_supported": bool(turn["tokens"]),
            "observed_token_count": turn["token_count"],
            "system_fingerprint": turn["system_fingerprint"] or None,
        }
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform one remote request that may consume a small number of credits.",
    )
    parser.add_argument("--model", choices=NVIDIA_MODELS, default=NVIDIA_MODEL)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--top-logprobs", type=int, default=5)
    args = parser.parse_args(argv)
    if args.timeout <= 0 or not 2 <= args.top_logprobs <= 20:
        parser.error("timeout must be positive and top-logprobs must be 2..20")
    try:
        report = run_preflight(
            args.execute,
            args.timeout,
            args.top_logprobs,
            model=args.model,
        )
    except Exception as exc:
        status_code = getattr(exc, "status_code", None)
        if status_code == 401:
            print(
                "NVIDIA preflight failed: 401 Unauthorized. Generate a Build/"
                "Endpoints key (nvapi-...), not an NGC registry key; re-enter "
                "it in NVIDIA_API_KEY and retry."
            )
        else:
            print(
                "NVIDIA preflight failed: "
                f"{type(exc).__name__}: {str(exc)[:500]}"
            )
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
