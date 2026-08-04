#!/usr/bin/env python3
"""Build a paired audit between historical CoCC verifier reports."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def row_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row["model"]), str(row["session_id"])


def classification(row: Mapping[str, Any]) -> tuple[str, str | None]:
    return str(row["outcome"]), row.get("failure_kind")


def reason_for_change(old: Mapping[str, Any], new: Mapping[str, Any]) -> str:
    status = (new.get("source_extraction") or {}).get("status")
    matches = (new.get("source_extraction") or {}).get("matching_block_count")
    if status == "ambiguous_extraction":
        return (
            "v1 selected the longest code block; v2 found "
            f"{matches} blocks declaring the required callable and rejected the "
            "non-unique contractual extraction"
        )
    if status == "callable_absent":
        return (
            "v2 parsed every eligible code block and found no top-level function or "
            "Solution method matching the required callable; no candidate code was executed"
        )
    if old.get("outcome") != new.get("outcome"):
        return "outcome changed under the corrected extraction and exception audit"
    return "failure taxonomy changed under the corrected extraction and exception audit"


def build_audit(old_report: Mapping[str, Any], new_report: Mapping[str, Any]) -> dict[str, Any]:
    old_rows = {row_key(row): row for row in old_report["items"]}
    new_rows = {row_key(row): row for row in new_report["items"]}
    if len(old_rows) != len(old_report["items"]) or len(new_rows) != len(new_report["items"]):
        raise ValueError("report contains duplicate (model, session_id) keys")
    if old_rows.keys() != new_rows.keys():
        missing_old = sorted(new_rows.keys() - old_rows.keys())
        missing_new = sorted(old_rows.keys() - new_rows.keys())
        raise ValueError(
            f"paired report key mismatch: absent_from_v1={missing_old}, absent_from_v2={missing_new}"
        )

    changed: list[dict[str, Any]] = []
    transition_counts: Counter[str] = Counter()
    for key in sorted(old_rows):
        old = old_rows[key]
        new = new_rows[key]
        old_label = old.get("failure_kind") or "PASS"
        new_label = new.get("failure_kind") or "PASS"
        transition_counts[f"{old_label} -> {new_label}"] += 1
        if classification(old) == classification(new):
            continue
        extraction = new.get("source_extraction") or {}
        changed.append(
            {
                "model": new["model"],
                "problem_id": new["problem_id"],
                "session_id": new["session_id"],
                "old_outcome": old["outcome"],
                "new_outcome": new["outcome"],
                "old_failure_kind": old.get("failure_kind"),
                "new_failure_kind": new.get("failure_kind"),
                "source_extraction_status": extraction.get("status"),
                "candidate_block_count": extraction.get("candidate_block_count"),
                "matching_block_count": extraction.get("matching_block_count"),
                "required_callable": extraction.get("required_callable"),
                "reason_for_change": reason_for_change(old, new),
            }
        )

    outcome_changes = [row for row in changed if row["old_outcome"] != row["new_outcome"]]
    return {
        "schema": "LLM-SVM-historical-CoCC-verifier-transition-audit/1",
        "pairing_key": ["model", "session_id"],
        "paired_response_count": len(old_rows),
        "classification_changed_count": len(changed),
        "outcome_changed_count": len(outcome_changes),
        "transition_counts": dict(sorted(transition_counts.items())),
        "outcome_changes": outcome_changes,
        "classification_changes": changed,
        "contains_prompt_or_answer": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-report", type=Path, required=True)
    parser.add_argument("--v2-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    audit = build_audit(load_json(args.v1_report), load_json(args.v2_report))
    audit["bound_reports"] = {
        "v1_report_sha256": file_sha256(args.v1_report),
        "v2_report_sha256": file_sha256(args.v2_report),
    }
    atomic_json(args.output, audit)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "classification_changed_count": audit["classification_changed_count"],
                "outcome_changed_count": audit["outcome_changed_count"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
