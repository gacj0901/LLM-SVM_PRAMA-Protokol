#!/usr/bin/env python
"""Check monitor/outcome namespace separation before empirical validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aptadynamic_llm.artifact_schema import read_jsonl, validate_artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monitor", required=True, type=Path)
    parser.add_argument("--outcomes", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        monitor = read_jsonl(args.monitor)
        outcomes = read_jsonl(args.outcomes)
        for row in monitor:
            validate_artifact(row, "structural_label")
            if not str(row["label"]).startswith("monitor."):
                raise ValueError("monitor label lacks monitor. namespace")
        outcome_by_session = {}
        for row in outcomes:
            if not str(row.get("label", "")).startswith("outcome."):
                raise ValueError("outcome label lacks outcome. namespace")
            session_id = str(row["session_id"])
            if session_id in outcome_by_session:
                raise ValueError(f"duplicate outcome session_id: {session_id}")
            outcome_by_session[session_id] = row["label"]
        paired = sum(row["session_id"] in outcome_by_session for row in monitor)
    except (KeyError, OSError, ValueError) as exc:
        print(f"structural-label validation failed: {exc}")
        return 1
    print(json.dumps({"monitor_rows": len(monitor), "outcome_rows": len(outcomes), "paired_rows": paired}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
