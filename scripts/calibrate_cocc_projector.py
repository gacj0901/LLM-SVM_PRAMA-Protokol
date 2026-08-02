#!/usr/bin/env python
"""Freeze position-wise surprisal expectations from calibration projector requests."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics
import tempfile
from typing import Any


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def load_request(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "LLM-SVM-CoCC-projector-request/1":
        raise ValueError(f"{path}: unexpected schema")
    if value.get("input_channel_status") != "OBSERVED":
        raise ValueError(f"{path}: calibration input must be OBSERVED")
    return value


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile requires observations")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def build(paths: list[Path], window_size: int, min_sessions: int) -> dict[str, Any]:
    requests = [load_request(path) for path in paths]
    sessions = {str(request["session_id"]) for request in requests}
    if len(sessions) < min_sessions:
        raise ValueError(
            f"requires at least {min_sessions} distinct calibration sessions; "
            f"got {len(sessions)}"
        )
    models = {str(request.get("model_id") or "") for request in requests}
    if len(models) != 1 or not next(iter(models)):
        raise ValueError("calibration requests must have one non-empty model_id")
    by_window: dict[int, list[float]] = {}
    token_surprisals: list[float] = []
    for request in requests:
        for turn in request.get("turns") or []:
            tokens = turn.get("tokens") or []
            for start in range(0, len(tokens), window_size):
                members = tokens[start : start + window_size]
                if not members:
                    continue
                values = [
                    max(0.0, -float(token["top1_logprob"])) for token in members
                ]
                if not all(math.isfinite(value) for value in values):
                    raise ValueError("non-finite surprisal in calibration input")
                index = start // window_size
                by_window.setdefault(index, []).append(statistics.fmean(values))
                token_surprisals.extend(values)
    if not by_window:
        raise ValueError("calibration requests contain no token observations")
    scale = max(quantile(token_surprisals, 0.99), 1e-12)
    expected = {
        str(index): {
            "mean_surprisal": statistics.fmean(values),
            "session_count": len(values),
        }
        for index, values in sorted(by_window.items())
    }
    source_files = [
        {"name": path.name, "sha256": file_sha256(path)}
        for path in sorted(paths, key=lambda value: value.name)
    ]
    return {
        "schema": "LLM-SVM-CoCC-frozen-calibration/1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN",
        "partition": "calibration",
        "model_id": next(iter(models)),
        "session_ids": sorted(sessions),
        "session_count": len(sessions),
        "window_size": window_size,
        "window_stride": window_size,
        "expected_by_window": expected,
        "global_expected_mean_surprisal": statistics.fmean(
            item["mean_surprisal"] for item in expected.values()
        ),
        "surprisal_scale": {
            "method": "calibration_token_surprisal_quantile",
            "quantile": 0.99,
            "value": scale,
            "clip_min": 0.0,
            "clip_max": 1.0,
        },
        "source_requests": source_files,
        "source_set_sha256": sha256(
            json.dumps(
                source_files, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
        "contains_prompt_or_answer": False,
        "contains_outcome_labels": False,
    }


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--min-sessions", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        if args.window_size <= 0 or args.min_sessions <= 0:
            raise ValueError("window-size and min-sessions must be positive")
        paths = sorted(args.requests_dir.glob("*.json"))
        if not paths:
            raise ValueError("requests directory contains no JSON requests")
        artifact = build(paths, args.window_size, args.min_sessions)
        atomic_json(args.output, artifact)
    except (OSError, TypeError, ValueError) as exc:
        print(f"CoCC projector calibration failed: {exc}")
        return 1
    print(json.dumps({"output": str(args.output), "sha256": file_sha256(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
