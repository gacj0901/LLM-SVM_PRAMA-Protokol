#!/usr/bin/env python
"""Project numeric CoCC token observations using frozen calibration and pinned PRAMA."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for candidate in (REPO_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from aptadynamic_llm.window_prama import (  # noqa: E402
    discover_git_commit,
    signed_unit_affine_v1,
    source_tree_sha256,
    validate_window_kernel_declaration,
)
from aptadynamic_llm.artifact_schema import make_envelope, sha256_value  # noqa: E402


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected one JSON object")
    return value


def normalized_surprisal(value: float, scale: float) -> float:
    unit = min(1.0, max(0.0, float(value) / scale))
    return signed_unit_affine_v1(2.0 * unit - 1.0)


def validate_identity(
    declaration_path: Path, recertification_path: Path
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    import prama_protokol

    package_root = Path(prama_protokol.__file__).resolve().parent
    declaration = load_json(declaration_path)
    recertification = load_json(recertification_path)
    if recertification.get("status") != "PASS":
        raise RuntimeError("PRAMA recertification status is not PASS")
    tests = recertification.get("tests")
    if not isinstance(tests, list) or not tests or not all(
        isinstance(test, dict) and test.get("passed") is True for test in tests
    ):
        raise RuntimeError("PRAMA recertification tests are incomplete")
    identity, config, _input_transform, column_map = (
        validate_window_kernel_declaration(
            declaration,
            actual_version=str(getattr(prama_protokol, "__version__", "")),
            actual_source_tree_sha256=source_tree_sha256(package_root),
            actual_commit=discover_git_commit(package_root),
            recertification_sha256=file_sha256(recertification_path),
        )
    )
    certified_identity = recertification.get("kernel_identity", {})
    expected_identity = {
        "package": identity.package,
        "version": identity.version,
        "source_tree_sha256": identity.source_tree_sha256,
        "commit": identity.commit,
        "kernel_api": identity.kernel_api,
        "bin_scale": identity.bin_scale,
    }
    if certified_identity != expected_identity:
        raise RuntimeError("recertification kernel identity differs from declaration")
    if recertification.get("config_sha256") != identity.config_sha256:
        raise RuntimeError("recertification config hash differs from declaration")
    if recertification.get("kernel_config") != config:
        raise RuntimeError("recertification kernel config differs from declaration")
    return config, column_map, identity.to_dict()


def project(
    request: dict[str, Any],
    calibration: dict[str, Any],
    calibration_hash: str,
    config: dict[str, Any],
    columns: dict[str, str],
    identity: dict[str, Any],
) -> list[dict[str, Any]]:
    import numpy as np
    from prama_protokol import KernelConfigV3, project_v3

    if request.get("schema") != "LLM-SVM-CoCC-projector-request/1":
        raise ValueError("unexpected projector request schema")
    if request.get("input_channel_status") != "OBSERVED":
        raise ValueError("projector accepts only OBSERVED inputs")
    if request.get("calibration_reference_sha256") != calibration_hash:
        raise ValueError("projector request calibration digest mismatch")
    if calibration.get("schema") != "LLM-SVM-CoCC-frozen-calibration/1":
        raise ValueError("unexpected calibration schema")
    if calibration.get("status") != "FROZEN":
        raise ValueError("calibration status must be FROZEN")
    if str(request.get("model_id") or "") != str(calibration.get("model_id") or ""):
        raise ValueError("projector request model differs from frozen calibration")
    window_size = int(calibration["window_size"])
    scale = float(calibration["surprisal_scale"]["value"])
    if window_size <= 0 or not math.isfinite(scale) or scale <= 0:
        raise ValueError("invalid frozen calibration scale/window")
    observed: list[float] = []
    expected: list[float] = []
    metadata: list[tuple[int, int, float, float]] = []
    frozen = calibration["expected_by_window"]
    fallback = float(calibration["global_expected_mean_surprisal"])
    for turn in request.get("turns") or []:
        tokens = turn.get("tokens") or []
        for start in range(0, len(tokens), window_size):
            members = tokens[start : start + window_size]
            if not members:
                continue
            mean_surprisal = statistics.fmean(
                max(0.0, -float(token["top1_logprob"])) for token in members
            )
            window_index = start // window_size
            expected_surprisal = float(
                frozen.get(str(window_index), {}).get("mean_surprisal", fallback)
            )
            observed.append(normalized_surprisal(mean_surprisal, scale))
            expected.append(normalized_surprisal(expected_surprisal, scale))
            metadata.append(
                (
                    int(turn.get("turn_index") or 0),
                    window_index,
                    mean_surprisal,
                    expected_surprisal,
                )
            )
    if not observed:
        raise ValueError("projector request contains no token observations")
    gamma = project_v3(
        np.asarray(observed, dtype=float),
        np.asarray(expected, dtype=float),
        KernelConfigV3(**config),
    )
    values = gamma.as_dict()
    session_id = str(request["session_id"])
    trajectory = []
    for index, (
        turn_index,
        window_index,
        raw_observed,
        raw_expected,
    ) in enumerate(metadata):
        valid = bool(values[columns["valid"]][index])

        def coordinate(name: str) -> float | None:
            return float(values[columns[name]][index]) if valid else None

        trajectory.append(
            {
                **make_envelope(
                    artifact_type="prama_trajectory",
                    study_id="CoCC-PRAMA",
                    session_id=session_id,
                    producer="scripts.project_cocc_prama/1",
                    created_at=datetime.now(timezone.utc).isoformat(),
                    source_sha256=sha256_value(request),
                    config_sha256=identity["config_sha256"],
                    partition="exploratory",
                    channel_status="OBSERVED" if valid else "INVALID",
                ),
                "session_id": session_id,
                "turn_index": turn_index,
                "window_index": window_index,
                "delta": coordinate("delta"),
                "xi": coordinate("xi"),
                "accumulated_excess": coordinate("accumulated_excess"),
                "capacity": coordinate("capacity"),
                "theta": coordinate("theta"),
                "balance": coordinate("balance"),
                "trend": coordinate("trend"),
                "observed_mean_surprisal": raw_observed,
                "expected_mean_surprisal": raw_expected,
                "normalized_observed": observed[index],
                "normalized_expected": expected[index],
                "input_transform": "signed_unit_affine_v1",
                "upstream_input_transform": "frozen_p99_surprisal_scale_clip_v1",
                "input_channel_status": "OBSERVED",
                "coordinate_origin": "DERIVED_KERNEL_STATE",
                "calibration_reference_sha256": calibration_hash,
                "kernel_identity": identity,
                "valid": valid,
            }
        )
    return trajectory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--declaration", required=True, type=Path)
    parser.add_argument("--recertification", required=True, type=Path)
    parser.add_argument("--prama-source-root", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.prama_source_root:
            source = args.prama_source_root.resolve()
            import_root = source / "src" if (source / "src").is_dir() else source
            sys.path.insert(0, str(import_root))
        calibration_hash = file_sha256(args.calibration)
        calibration = load_json(args.calibration)
        config, columns, identity = validate_identity(
            args.declaration, args.recertification
        )
        result = project(
            json.load(sys.stdin),
            calibration,
            calibration_hash,
            config,
            columns,
            identity,
        )
    except (
        ImportError,
        json.JSONDecodeError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"CoCC PRAMA projection failed: {exc}", file=sys.stderr)
        return 1
    sys.stdout.write(json.dumps({"trajectory": result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
