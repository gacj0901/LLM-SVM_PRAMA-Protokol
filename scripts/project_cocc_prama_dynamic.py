#!/usr/bin/env python
"""Project numeric CoCC observations through the universal causal observer."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import statistics
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for candidate in (REPO_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from aptadynamic_llm.artifact_schema import make_envelope, sha256_value  # noqa: E402
from aptadynamic_llm.dynamic_observer import (  # noqa: E402
    DynamicObserverConfig,
    observe,
)
from scripts.project_cocc_prama import (  # noqa: E402
    file_sha256,
    load_json,
    validate_identity,
)


def project(
    request: dict[str, Any],
    contract: dict[str, Any],
    contract_hash: str,
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
    observer_config = DynamicObserverConfig.from_contract(contract)
    window_means: list[float] = []
    metadata: list[tuple[int, int, int]] = []
    for turn in request.get("turns") or []:
        tokens = turn.get("tokens") or []
        for start in range(0, len(tokens), observer_config.window_size_tokens):
            members = tokens[start : start + observer_config.window_size_tokens]
            if not members:
                continue
            window_means.append(
                statistics.fmean(
                    max(0.0, -float(token["top1_logprob"])) for token in members
                )
            )
            metadata.append(
                (int(turn.get("turn_index") or 0), start // observer_config.window_size_tokens, len(members))
            )
    observations = observe(window_means, observer_config)
    observer_delta = np.asarray([item["delta"] for item in observations], dtype=float)
    gamma = project_v3(
        observer_delta,
        np.zeros(len(observer_delta), dtype=float),
        KernelConfigV3(**config),
    )
    values = gamma.as_dict()
    session_id = str(request["session_id"])
    source_hash = sha256_value(request)
    created_at = datetime.now(timezone.utc).isoformat()
    trajectory = []
    for index, ((turn_index, window_index, member_count), observer) in enumerate(
        zip(metadata, observations, strict=True)
    ):
        valid = bool(values[columns["valid"]][index])

        def coordinate(name: str) -> float | None:
            return float(values[columns[name]][index]) if valid else None

        kernel_delta = coordinate("delta")
        if valid and abs(float(kernel_delta) - float(observer["delta"])) > 1e-12:
            raise RuntimeError("kernel adapter invariant failed: kernel Δ != observer Δ")
        trajectory.append(
            {
                **make_envelope(
                    artifact_type="prama_trajectory",
                    study_id="CoCC-PRAMA",
                    session_id=session_id,
                    producer="scripts.project_cocc_prama_dynamic/1",
                    created_at=created_at,
                    source_sha256=source_hash,
                    config_sha256=identity["config_sha256"],
                    partition="exploratory",
                    channel_status="OBSERVED" if valid else "INVALID",
                ),
                "session_id": session_id,
                "turn_index": turn_index,
                "window_index": window_index,
                "n_tokens_in_window": member_count,
                "delta": kernel_delta,
                "xi": coordinate("xi"),
                "accumulated_excess": coordinate("accumulated_excess"),
                "capacity": coordinate("capacity"),
                "theta": coordinate("theta"),
                "balance": coordinate("balance"),
                "trend": coordinate("trend"),
                "observed_mean_surprisal": window_means[index],
                **observer,
                "input_transform": "signed_unit_affine_v1",
                "upstream_input_transform": "causal_robust_surprisal_v1",
                "observer_source_transform": "mean_surprisal_then_log1p_same_window_units",
                "input_channel_status": "OBSERVED",
                "coordinate_origin": "DERIVED_KERNEL_STATE",
                "observer_contract_sha256": contract_hash,
                "observer_model_specific_parameters": False,
                "observer_external_calibration": False,
                "kernel_identity": identity,
                "valid": valid,
            }
        )
    return trajectory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observer-contract", required=True, type=Path)
    parser.add_argument("--declaration", required=True, type=Path)
    parser.add_argument("--recertification", required=True, type=Path)
    parser.add_argument("--prama-source-root", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.prama_source_root:
            source = args.prama_source_root.resolve()
            import_root = source / "src" if (source / "src").is_dir() else source
            sys.path.insert(0, str(import_root))
        contract = load_json(args.observer_contract)
        contract_hash = file_sha256(args.observer_contract)
        config, columns, identity = validate_identity(
            args.declaration, args.recertification
        )
        result = project(
            json.load(sys.stdin),
            contract,
            contract_hash,
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
        print(f"dynamic CoCC PRAMA projection failed: {exc}", file=sys.stderr)
        return 1
    sys.stdout.write(json.dumps({"trajectory": result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
