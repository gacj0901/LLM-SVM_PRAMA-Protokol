#!/usr/bin/env python
"""Derive an immutable E-P1 NVIDIA model freeze from its completed pilot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.collect_ep1_nvidia import file_sha256, load_design, model_profile  # noqa: E402


def derive_freeze(
    design_path: Path, model: str, pilot_dir: Path
) -> dict[str, Any]:
    design_path = design_path.resolve()
    design = load_design(design_path)
    design_hash = file_sha256(design_path)
    profile = model_profile(design, model)
    manifest_path = pilot_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "schema": "LLM-SVM-E-P1-NVIDIA-collection/1",
        "study_id": design["study_id"],
        "mode": "pilot",
        "provider": design["provider"],
        "provider_endpoint": design["provider_endpoint"],
        "model": model,
        "model_slug": profile["slug"],
        "design_sha256": design_hash,
        "prompt_suite_sha256": design["prompt_suite_sha256"],
        "n": int(design["pilot"]["n"]),
        "completed_n": int(design["pilot"]["n"]),
        "complete": True,
        "max_tokens": int(design["pilot"]["max_tokens"]),
        "temperature": profile["temperature"],
        "top_p": profile["top_p"],
        "top_logprobs": profile["top_logprobs"],
        "seed": int(design["sampling"]["base_seed"]),
        "seed_per_index": True,
    }
    mismatches = {
        key: {"expected": value, "observed": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"pilot manifest violates the frozen design: {mismatches}")
    sessions = list(manifest.get("sessions") or [])
    reasons = {str(row.get("finish_reason") or "") for row in sessions}
    unsupported = sorted(reasons - {"stop", "length"})
    if unsupported:
        raise ValueError(f"pilot contains unsupported finish reasons: {unsupported}")
    stop_lengths = [
        int(row["token_count"])
        for row in sessions
        if row.get("finish_reason") == "stop"
    ]
    if not stop_lengths:
        raise ValueError("pilot contains no naturally stopped sessions")
    rule = design["pilot"]
    p95 = float(
        np.quantile(
            np.asarray(stop_lengths, dtype=float),
            float(rule["quantile"]),
            method=str(rule["quantile_method"]),
        )
    )
    selected = int(math.ceil((p95 + 1.0) / 64.0) * 64)
    selected = max(int(rule["minimum_selected_max_tokens"]), selected)
    maximum = int(rule["maximum_selected_max_tokens"])
    if selected > maximum:
        raise ValueError(
            f"selected cap {selected} exceeds pilot ceiling {maximum}; a new pilot design is required"
        )
    if not selected > p95:
        raise AssertionError("selected max_tokens is not strictly above stop-session P95")
    return {
        "schema": "LLM-SVM-E-P1-NVIDIA-model-freeze/1",
        "status": "CONFIRMATORY_FROZEN",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "study_id": design["study_id"],
        "design": str(design_path),
        "design_sha256": design_hash,
        "model": model,
        "model_slug": profile["slug"],
        "provider": design["provider"],
        "provider_endpoint": design["provider_endpoint"],
        "prompt_suite_sha256": design["prompt_suite_sha256"],
        "pilot_manifest": str(manifest_path.resolve()),
        "pilot_manifest_sha256": file_sha256(manifest_path),
        "pilot_n": len(sessions),
        "pilot_stop_n": len(stop_lengths),
        "pilot_length_n": len(sessions) - len(stop_lengths),
        "stop_token_p95": p95,
        "quantile": rule["quantile"],
        "quantile_method": rule["quantile_method"],
        "selection_rule": rule["selection_rule"],
        "selected_max_tokens": selected,
        "sampling": {
            "temperature": profile["temperature"],
            "top_p": profile["top_p"],
            "top_logprobs": profile["top_logprobs"],
            "base_seed": design["sampling"]["base_seed"],
            "seed_per_index": design["sampling"]["seed_per_index"],
            "enable_thinking": profile.get("enable_thinking"),
            "reasoning_effort": profile.get("reasoning_effort"),
            "stream": bool(profile.get("stream")),
        },
        "pilot_disposition": "discarded_from_confirmatory_analysis",
        "score_inspection_before_freeze": False,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--design",
        type=Path,
        default=REPO_ROOT / "config" / "ep1_nvidia_replication_v1.json",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--pilot-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.out.exists():
            raise FileExistsError(f"freeze already exists: {args.out}")
        freeze = derive_freeze(args.design, args.model, args.pilot_dir)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(freeze, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except (FileExistsError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"E-P1 NVIDIA freeze failed: {exc}")
        return 1
    print(
        json.dumps(
            {
                "output": str(args.out),
                "sha256": sha256(args.out.read_bytes()).hexdigest(),
                "model": freeze["model"],
                "selected_max_tokens": freeze["selected_max_tokens"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
