#!/usr/bin/env python
"""Build frozen factorial-ablation observations from precomputed score JSONL.

This command stops at the pre-kernel boundary. ``kernel_inputs.jsonl`` contains
only eligible confirmatory rows with an observed frozen expectation; it is not
projected until a window-scale PRAMA configuration is pinned and recertified.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

from aptadynamic_llm.factorial_ablation import AblationConfig
from aptadynamic_llm.factorial_pipeline import (
    PIPELINE_SCHEMA,
    run_factorial_pipeline,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: each row must be an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"no JSONL rows found in {path}")
    return rows


def _write_jsonl(path: Path, rows: tuple[dict[str, Any], ...]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.out.exists():
        raise FileExistsError(f"output directory already exists: {args.out}")
    source_bytes = args.input.read_bytes()
    rows = _read_jsonl(args.input)
    config = AblationConfig(
        min_support_magnitude=args.min_support_magnitude,
        max_filler_variance=args.max_filler_variance,
        epsilon=args.epsilon,
        min_fillers=args.min_fillers,
        min_assistant_turn_index=args.min_assistant_turn_index,
        natural_score_tolerance=args.natural_score_tolerance,
    )
    result = run_factorial_pipeline(
        rows,
        config=config,
        estimator_id=args.estimator_id,
        min_context_sessions=args.min_context_sessions,
    )
    manifest = {
        "schema": PIPELINE_SCHEMA,
        "source_sha256": sha256(source_bytes).hexdigest(),
        "config": asdict(config),
        "estimator": {
            "id": result.estimator.estimator_id,
            "statistics_sha256": result.estimator.statistics_sha256,
            "min_context_sessions": result.estimator.min_context_sessions,
            "calibration_max_session_order": (
                result.estimator.calibration_max_session_order
            ),
        },
        "counts": {
            "input_rows": len(rows),
            "observation_rows": len(result.observations),
            "eligible_rows": sum(
                bool(row["eligible"]) for row in result.observations
            ),
            "kernel_ready_confirmatory_rows": len(result.kernel_inputs),
        },
        "kernel_status": "not_invoked_pending_window_scale_recertification",
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".factorial-", dir=args.out.parent) as tmp:
        staged = Path(tmp) / args.out.name
        staged.mkdir()
        _write_jsonl(staged / "observations.jsonl", result.observations)
        _write_jsonl(staged / "kernel_inputs.jsonl", result.kernel_inputs)
        (staged / "manifest.json").write_text(
            json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        shutil.move(str(staged), str(args.out))
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--estimator-id", required=True)
    parser.add_argument("--min-context-sessions", required=True, type=int)
    parser.add_argument("--min-support-magnitude", required=True, type=float)
    parser.add_argument("--max-filler-variance", required=True, type=float)
    parser.add_argument("--epsilon", type=float, default=1e-12)
    parser.add_argument("--min-fillers", type=int, default=3)
    parser.add_argument("--min-assistant-turn-index", type=int, default=1)
    parser.add_argument("--natural-score-tolerance", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        manifest = run(parse_args(argv))
    except (FileExistsError, OSError, TypeError, ValueError) as exc:
        print(f"factorial evaluation failed: {exc}")
        return 1
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
