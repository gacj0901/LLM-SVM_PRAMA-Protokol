#!/usr/bin/env python3
"""Materialize an outcome-free CoCC calibration-only JSONL artifact."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


FORBIDDEN_OUTCOME_KEYS = {
    "label",
    "outcome",
    "passed",
    "verification",
    "verification_result",
    "verifier_result",
}


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def extract(source: Path, expected_count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number}: expected object")
            if row.get("split") != "calibration":
                continue
            if row.get("perturbation_type") != "clean_control":
                raise ValueError(
                    f"line {line_number}: calibration row is not clean_control"
                )
            forbidden = sorted(FORBIDDEN_OUTCOME_KEYS.intersection(row))
            if forbidden:
                raise ValueError(
                    f"line {line_number}: forbidden outcome fields: {forbidden}"
                )
            rows.append(row)
    if len(rows) != expected_count:
        raise ValueError(
            f"expected {expected_count} calibration controls, found {len(rows)}"
        )
    item_ids = [str(row.get("item_id") or "") for row in rows]
    if any(not value for value in item_ids) or len(set(item_ids)) != len(item_ids):
        raise ValueError("calibration item_id values must be non-empty and unique")
    return rows


def write_artifacts(
    source: Path,
    output: Path,
    manifest_path: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    temporary.replace(output)
    item_ids = sorted(str(row["item_id"]) for row in rows)
    problem_ids = sorted(str(row["problem_id"]) for row in rows)
    manifest = {
        "schema": "LLM-SVM-CoCC-calibration-partition/1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(source),
        "source_dataset_sha256": file_sha256(source),
        "output_dataset": str(output),
        "output_dataset_sha256": file_sha256(output),
        "output_sha256": file_sha256(output),
        "partition": "calibration",
        "perturbation_type": "clean_control",
        "session_count": len(rows),
        "item_ids_sha256": canonical_sha256(item_ids),
        "problem_ids_sha256": canonical_sha256(problem_ids),
        "outcome_labels_present": False,
        "holdout_rows_present": False,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--expected-count", type=int, default=20)
    args = parser.parse_args(argv)
    try:
        if args.expected_count <= 0:
            raise ValueError("expected-count must be positive")
        rows = extract(args.source, args.expected_count)
        manifest = write_artifacts(
            args.source, args.output, args.manifest, rows
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"CoCC calibration extraction failed: {exc}")
        return 1
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
