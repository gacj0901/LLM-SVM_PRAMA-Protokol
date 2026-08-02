#!/usr/bin/env python
"""Validate JSONL files against the runtime structural artifact contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aptadynamic_llm.artifact_schema import read_jsonl, validate_artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--expected-type")
    args = parser.parse_args(argv)
    counts = {}
    try:
        for path in args.paths:
            rows = read_jsonl(path)
            for row in rows:
                validate_artifact(row, args.expected_type)
            counts[str(path)] = len(rows)
    except (OSError, ValueError) as exc:
        print(f"artifact validation failed: {exc}")
        return 1
    print(json.dumps({"valid": True, "counts": counts}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
