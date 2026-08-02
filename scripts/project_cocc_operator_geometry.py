#!/usr/bin/env python
"""Project saved numeric observations into causal operator geometry."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for candidate in (REPO_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from aptadynamic_llm.operator_geometry import (  # noqa: E402
    OperatorGeometryConfig,
    observe_operator_geometry,
)


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected one JSON object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path}: expected nonempty JSONL objects")
    return rows


def build_artifact(
    request_path: Path, trajectory_path: Path, contract_path: Path
) -> dict[str, Any]:
    request = load_json(request_path)
    trajectory = load_jsonl(trajectory_path)
    contract = load_json(contract_path)
    config = OperatorGeometryConfig.from_contract(contract)
    if request.get("schema") != "LLM-SVM-CoCC-projector-request/1":
        raise ValueError("unexpected numeric request schema")
    session_id = str(request["session_id"])
    if any(str(row.get("session_id")) != session_id for row in trajectory):
        raise ValueError("trajectory session identity mismatch")
    tokens = [token for turn in request.get("turns") or [] for token in turn.get("tokens") or []]
    windows = observe_operator_geometry(tokens, trajectory, config)
    return {
        "schema": "LLM-SVM-CoCC-operator-geometry-artifact/1",
        "session_id": session_id,
        "model": str(request.get("model_id") or ""),
        "observer_contract_sha256": file_sha256(contract_path),
        "source_projection_request_sha256": file_sha256(request_path),
        "source_prama_trajectory_sha256": file_sha256(trajectory_path),
        "model_specific_parameters": False,
        "contains_outcome_label": False,
        "kernel_modified": False,
        "window_count": len(windows),
        "windows": windows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    try:
        artifact = build_artifact(args.request, args.trajectory, args.contract)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"operator-geometry projection failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"output": str(args.out), "windows": artifact["window_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
