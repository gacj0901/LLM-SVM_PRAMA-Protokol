#!/usr/bin/env python
"""Project coupling windows only with a pinned, recertified PRAMA identity."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from aptadynamic_llm.artifact_schema import (
    ChannelStatus,
    make_envelope,
    sha256_value,
    validate_artifact,
    write_jsonl_atomic,
)
from aptadynamic_llm.window_prama import (
    discover_git_commit,
    signed_unit_affine_v1,
    source_tree_sha256,
    validate_window_kernel_declaration,
)
from _structural_artifact_cli import read_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--declaration", required=True, type=Path)
    parser.add_argument("--recertification", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--producer", required=True)
    parser.add_argument("--partition", choices=("confirmatory", "exploratory"), required=True)
    args = parser.parse_args(argv)
    try:
        import numpy as np
        import prama_protokol
        from prama_protokol import KernelConfigV3, project_v3

        package_root = Path(prama_protokol.__file__).resolve().parent
        actual_source_hash = source_tree_sha256(package_root)
        actual_commit = discover_git_commit(package_root)
        declaration = json.loads(args.declaration.read_text(encoding="utf-8"))
        recertification = json.loads(args.recertification.read_text(encoding="utf-8"))
        if recertification.get("status") != "PASS":
            raise RuntimeError("window-scale recertification status is not PASS")
        tests = recertification.get("tests")
        if not isinstance(tests, list) or not tests or not all(
            test.get("passed") is True for test in tests if isinstance(test, dict)
        ) or not all(isinstance(test, dict) for test in tests):
            raise RuntimeError("window-scale recertification tests are incomplete")
        recertification_hash = sha256(args.recertification.read_bytes()).hexdigest()
        identity, config, input_transform, column_map = validate_window_kernel_declaration(
            declaration,
            actual_version=str(getattr(prama_protokol, "__version__", "")),
            actual_source_tree_sha256=actual_source_hash,
            actual_commit=actual_commit,
            recertification_sha256=recertification_hash,
        )
        certified_identity = recertification.get("kernel_identity", {})
        certified_config_hash = recertification.get("config_sha256")
        expected_certified_identity = {
            "package": identity.package,
            "version": identity.version,
            "source_tree_sha256": identity.source_tree_sha256,
            "commit": identity.commit,
            "kernel_api": identity.kernel_api,
            "bin_scale": identity.bin_scale,
        }
        if certified_identity != expected_certified_identity:
            raise RuntimeError(
                "recertification kernel identity differs from declaration"
            )
        if certified_config_hash != identity.config_sha256:
            raise RuntimeError(
                "recertification config hash differs from declaration"
            )
        if (
            recertification.get("kernel_config") != config
            or recertification.get("input_transform") != input_transform
        ):
            raise RuntimeError(
                "recertification projection configuration differs from declaration"
            )
        rows = read_rows(args.input)
        grouped = {}
        seen_windows = set()
        for row in rows:
            validate_artifact(row, expected_type="coupling_observation")
            if row["study_id"] != args.study_id:
                raise ValueError("input study_id differs from --study-id")
            if row["partition"] != args.partition:
                raise ValueError("input partition differs from --partition")
            if row["channel_status"] != ChannelStatus.OBSERVED.value:
                raise ValueError("PRAMA accepts only OBSERVED coupling inputs")
            if row["eligible"] is not True:
                raise ValueError("PRAMA accepts only eligible coupling inputs")
            if row["expectation_status"] != "FROZEN":
                raise ValueError("PRAMA requires expectation_status='FROZEN'")
            if row["expected_omega_dep"] is None:
                raise ValueError("PRAMA requires a frozen expected_omega_dep")
            key = (
                str(row["session_id"]),
                int(row["turn_index"]),
                int(row["window_index"]),
            )
            if key in seen_windows:
                raise ValueError(f"duplicate coupling window identity: {key}")
            seen_windows.add(key)
            grouped.setdefault(str(row["session_id"]), []).append(row)
        outputs = []
        source_hash = sha256(args.input.read_bytes()).hexdigest()
        for session_id, members in sorted(grouped.items()):
            members.sort(key=lambda row: (int(row["turn_index"]), int(row["window_index"])))
            omega = np.asarray(
                [
                    signed_unit_affine_v1(
                        row["omega_dep"] if "omega_dep" in row else row["omega"]
                    )
                    for row in members
                ],
                dtype=float,
            )
            expected = np.asarray(
                [
                    signed_unit_affine_v1(
                        row["expected_omega_dep"]
                        if "expected_omega_dep" in row
                        else row["expected_omega"]
                    )
                    for row in members
                ],
                dtype=float,
            )
            if not np.isfinite(omega).all() or not np.isfinite(expected).all():
                raise ValueError(f"{session_id}: PRAMA inputs must be finite")
            gamma = project_v3(omega, expected, KernelConfigV3(**config))
            columns = gamma.as_dict()
            if len(gamma) != len(members):
                raise RuntimeError(
                    f"{session_id}: PRAMA emitted {len(gamma)} rows for "
                    f"{len(members)} windows"
                )
            for index, source in enumerate(members):
                valid = bool(columns[column_map["valid"]][index])
                def coordinate(name: str):
                    value = columns[column_map[name]][index]
                    return float(value) if valid else None
                xi = coordinate("xi")
                theta = coordinate("theta")
                outputs.append(
                    {
                        **make_envelope(
                            artifact_type="prama_trajectory",
                            study_id=args.study_id,
                            session_id=session_id,
                            producer=args.producer,
                            created_at=datetime.now(timezone.utc).isoformat(),
                            source_sha256=source_hash,
                            config_sha256=identity.config_sha256,
                            partition=args.partition,
                            channel_status=(
                                ChannelStatus.OBSERVED
                                if valid
                                else ChannelStatus.INVALID
                            ),
                        ),
                        "turn_index": int(source["turn_index"]),
                        "window_index": int(source["window_index"]),
                        "delta": coordinate("delta"),
                        "xi": xi,
                        "accumulated_excess": coordinate("accumulated_excess"),
                        "capacity": coordinate("capacity"),
                        "theta": theta,
                        "balance": coordinate("balance"),
                        "trend": coordinate("trend"),
                        "input_transform": input_transform["name"],
                        "input_channel_status": source["channel_status"],
                        "coordinate_origin": "DERIVED_KERNEL_STATE",
                        "kernel_identity": identity.to_dict(),
                        "valid": valid,
                    }
                )
        write_jsonl_atomic(args.out, outputs)
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"window PRAMA projection failed: {exc}")
        return 1
    print(f"wrote {len(outputs)} recertified PRAMA window artifacts to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
