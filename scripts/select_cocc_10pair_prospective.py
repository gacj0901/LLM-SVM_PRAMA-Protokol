#!/usr/bin/env python
"""Materialize the frozen, outcome-blind 10-pair prospective CoCC subset."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import tempfile
from typing import Any


SELECTED_PROBLEM_IDS = (
    "2730",
    "2755",
    "2808",
    "2827",
    "2839",
    "2848",
    "2921",
    "3032",
    "3184",
    "3235",
)
REQUIRED_CONDITIONS = ("clean_control", "negation_objective")


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def build(source: Path, source_manifest: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_hash = file_sha256(source)
    normalized = json.loads(source_manifest.read_text(encoding="utf-8"))
    if normalized.get("output_sha256") != source_hash:
        raise ValueError("source normalization manifest does not bind source dataset")
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_problem: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        problem_id = str(row.get("problem_id"))
        if problem_id not in SELECTED_PROBLEM_IDS:
            continue
        condition = str(row.get("perturbation_type"))
        if condition in by_problem.setdefault(problem_id, {}):
            raise ValueError(f"duplicate {problem_id}:{condition}")
        by_problem[problem_id][condition] = row
    output = []
    pair_order = []
    for pair_index, problem_id in enumerate(SELECTED_PROBLEM_IDS):
        conditions = by_problem.get(problem_id, {})
        if set(conditions) != set(REQUIRED_CONDITIONS):
            raise ValueError(f"{problem_id}: expected exactly {REQUIRED_CONDITIONS}")
        order = (
            REQUIRED_CONDITIONS
            if pair_index % 2 == 0
            else tuple(reversed(REQUIRED_CONDITIONS))
        )
        pair_order.append({"problem_id": problem_id, "condition_order": list(order)})
        for within_pair_order, condition in enumerate(order):
            row = dict(conditions[condition])
            if row.get("split") != "test":
                raise ValueError(f"{problem_id}:{condition} is not holdout/test")
            row["prospective_pair_index"] = pair_index
            row["within_pair_order"] = within_pair_order
            output.append(row)
    manifest = {
        "schema": "LLM-SVM-CoCC-prospective-paired-subset/1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selection_status": "FROZEN_BEFORE_MODEL_ACQUISITION",
        "selection_rule": "ten prespecified diverse problem_id clusters; both conditions retained; no outcome used",
        "selection_used_external_outcomes": False,
        "source_dataset": str(source),
        "source_dataset_sha256": source_hash,
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": file_sha256(source_manifest),
        "problem_ids": list(SELECTED_PROBLEM_IDS),
        "problem_ids_sha256": canonical_sha256(list(SELECTED_PROBLEM_IDS)),
        "pair_count": len(SELECTED_PROBLEM_IDS),
        "session_count": len(output),
        "conditions": list(REQUIRED_CONDITIONS),
        "pair_order": pair_order,
        "condition_order_balance": {
            "clean_first": 5,
            "negation_first": 5,
        },
        "independent_unit": "problem_id",
    }
    return output, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--out-manifest", required=True, type=Path)
    args = parser.parse_args()
    rows, manifest = build(args.source, args.source_manifest)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )
    atomic_text(args.out, payload)
    manifest["output_sha256"] = file_sha256(args.out)
    atomic_text(args.out_manifest, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "dataset": str(args.out),
                "dataset_sha256": manifest["output_sha256"],
                "manifest": str(args.out_manifest),
                "manifest_sha256": file_sha256(args.out_manifest),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
