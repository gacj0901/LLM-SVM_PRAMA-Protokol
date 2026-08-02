#!/usr/bin/env python
"""Project saved numeric trajectories into structural coherence v5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for candidate in (REPO_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from aptadynamic_llm.operator_geometry import OperatorGeometryConfig  # noqa: E402
from aptadynamic_llm.structural_coherence_v5 import (  # noqa: E402
    StructuralCoherenceV5Config,
    observe_structural_coherence_v5,
)
from scripts.project_cocc_operator_geometry import file_sha256, load_json, load_jsonl  # noqa: E402


def build_artifact(request_path: Path, trajectory_path: Path, geometry_path: Path, coherence_path: Path) -> dict[str, Any]:
    request = load_json(request_path)
    trajectory = load_jsonl(trajectory_path)
    geometry_contract = load_json(geometry_path)
    coherence_contract = load_json(coherence_path)
    geometry_hash = file_sha256(geometry_path)
    if coherence_contract["base_geometry_contract_sha256"] != geometry_hash:
        raise ValueError("coherence contract is not bound to supplied geometry contract")
    if request.get("schema") != "LLM-SVM-CoCC-projector-request/1":
        raise ValueError("unexpected numeric request schema")
    session_id = str(request["session_id"])
    if any(str(row.get("session_id")) != session_id for row in trajectory):
        raise ValueError("trajectory session identity mismatch")
    tokens = [token for turn in request.get("turns") or [] for token in turn.get("tokens") or []]
    windows = observe_structural_coherence_v5(
        tokens,
        trajectory,
        OperatorGeometryConfig.from_contract(geometry_contract),
        StructuralCoherenceV5Config.from_contract(coherence_contract),
    )
    return {
        "schema": "LLM-SVM-CoCC-structural-coherence-artifact/5",
        "session_id": session_id,
        "model": str(request.get("model_id") or ""),
        "coherence_contract_sha256": file_sha256(coherence_path),
        "base_geometry_contract_sha256": geometry_hash,
        "source_projection_request_sha256": file_sha256(request_path),
        "source_prama_trajectory_sha256": file_sha256(trajectory_path),
        "contains_outcome_label": False,
        "kernel_modified": False,
        "window_count": len(windows),
        "windows": windows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--geometry-contract", required=True, type=Path)
    parser.add_argument("--coherence-contract", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    try:
        artifact = build_artifact(args.request, args.trajectory, args.geometry_contract, args.coherence_contract)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"structural-coherence v5 projection failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"output": str(args.out), "windows": artifact["window_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
