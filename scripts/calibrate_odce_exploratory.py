#!/usr/bin/env python
"""Fit an explicitly non-frozen exploratory ODCE-v0.1.0 normalization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from typing import Any

from aptadynamic_llm.artifact_schema import canonical_json, read_jsonl
from aptadynamic_llm.odce_calibration import calibrate_exploratory_contract


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(canonical_json(value) + "\n")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, nargs="+", type=Path)
    parser.add_argument(
        "--base-contract",
        type=Path,
        default=Path("config/odce_v0_1_exploratory.json"),
    )
    parser.add_argument("--out-contract", required=True, type=Path)
    parser.add_argument("--out-report", required=True, type=Path)
    parser.add_argument("--min-observations", type=int, default=20)
    parser.add_argument("--minimum-observed-fraction", type=float, default=0.10)
    parser.add_argument("--minimum-session-count", type=int, default=5)
    parser.add_argument("--minimum-session-coverage", type=float, default=0.10)
    parser.add_argument("--minimum-scale", type=float, default=1e-12)
    parser.add_argument("--allow-exploratory-input", action="store_true")
    args = parser.parse_args(argv)
    try:
        base_contract = json.loads(
            args.base_contract.read_text(encoding="utf-8")
        )
        rows = tuple(
            row for path in args.input for row in read_jsonl(path)
        )
        calibrated, report = calibrate_exploratory_contract(
            rows,
            base_contract,
            min_observations=args.min_observations,
            minimum_observed_fraction=args.minimum_observed_fraction,
            minimum_session_count=args.minimum_session_count,
            minimum_session_coverage=args.minimum_session_coverage,
            minimum_scale=args.minimum_scale,
            allow_exploratory_input=args.allow_exploratory_input,
        )
        _write_json_atomic(args.out_contract, calibrated)
        _write_json_atomic(args.out_report, report)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"ODCE exploratory calibration failed: {exc}")
        return 1
    print(
        f"calibrated {report['artifact_count']} ODCE artifacts; "
        f"{len(report['blocking_channels'])} channels remain blocking"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
