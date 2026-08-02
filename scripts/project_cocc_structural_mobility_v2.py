#!/usr/bin/env python
"""Project saved CoCC numeric observations into structural mobility v2."""

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
from aptadynamic_llm.structural_mobility import (  # noqa: E402
    StructuralMobilityConfig,
    observe_structural_mobility,
)
from scripts.project_cocc_operator_geometry import (  # noqa: E402
    file_sha256,
    load_json,
    load_jsonl,
)


def build_artifact(
    request_path: Path,
    trajectory_path: Path,
    geometry_contract_path: Path,
    mobility_contract_path: Path,
) -> dict[str, Any]:
    request = load_json(request_path)
    trajectory = load_jsonl(trajectory_path)
    geometry_contract = load_json(geometry_contract_path)
    mobility_contract = load_json(mobility_contract_path)
    geometry_hash = file_sha256(geometry_contract_path)
    if mobility_contract["base_geometry_contract_sha256"] != geometry_hash:
        raise ValueError("mobility contract is not bound to the supplied geometry contract")
    geometry_config = OperatorGeometryConfig.from_contract(geometry_contract)
    mobility_config = StructuralMobilityConfig.from_contract(mobility_contract)
    if request.get("schema") != "LLM-SVM-CoCC-projector-request/1":
        raise ValueError("unexpected numeric request schema")
    session_id = str(request["session_id"])
    if any(str(row.get("session_id")) != session_id for row in trajectory):
        raise ValueError("trajectory session identity mismatch")
    tokens = [token for turn in request.get("turns") or [] for token in turn.get("tokens") or []]
    windows = observe_structural_mobility(
        tokens, trajectory, geometry_config, mobility_config
    )
    return {
        "schema": "LLM-SVM-CoCC-structural-mobility-artifact/2",
        "session_id": session_id,
        "model": str(request.get("model_id") or ""),
        "mobility_contract_sha256": file_sha256(mobility_contract_path),
        "base_geometry_contract_sha256": geometry_hash,
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
    parser.add_argument("--geometry-contract", required=True, type=Path)
    parser.add_argument("--mobility-contract", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    try:
        artifact = build_artifact(
            args.request,
            args.trajectory,
            args.geometry_contract,
            args.mobility_contract,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"structural-mobility projection failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"output": str(args.out), "windows": artifact["window_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
