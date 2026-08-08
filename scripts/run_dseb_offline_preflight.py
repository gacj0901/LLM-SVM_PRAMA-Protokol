#!/usr/bin/env python3
"""Run the deterministic DSEB-v0 offline preflight without an LLM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.dseb_v0.preflight import run_offline_preflight  # noqa: E402


DEFAULT_PROTOCOL = ROOT / "benchmarks" / "dseb_v0" / "configs" / "dseb_v0.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_offline_preflight(
        protocol_path=args.protocol,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    visible = report if args.verbose else {
        key: report[key]
        for key in (
            "schema",
            "status",
            "benchmark_id",
            "benchmark_version",
            "partition",
            "session_id",
            "seed",
            "model_call_executed",
            "turn_count",
            "canonical_window_count",
            "outcome_count",
            "protocol_source_sha256",
            "generated_protocol_sha256",
            "checks_passed",
            "checks_failed",
        )
    }
    visible["report_file"] = str((args.output_dir / "report.json").resolve())
    print(json.dumps(visible, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
