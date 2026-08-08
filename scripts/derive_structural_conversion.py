#!/usr/bin/env python
"""Derive causal ODCE-v0 artifacts from PRAMA and D_O v9 JSONL records."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from aptadynamic_llm.artifact_schema import (
    ChannelStatus,
    make_envelope,
    read_jsonl,
    sha256_value,
    validate_artifact,
    write_jsonl_atomic,
)
from aptadynamic_llm.structural_conversion import (
    ODCEConfig,
    compute_structural_conversion_trajectory,
    make_structural_conversion_differential,
    validate_contract_freeze,
)


def _group(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        session_id = str(row.get("session_id", "")).strip()
        if not session_id:
            raise ValueError("every input row requires session_id")
        grouped[session_id].append(dict(row))
    return dict(grouped)


def _window_key(row: Mapping[str, Any]) -> tuple[int, int]:
    return int(row["turn_index"]), int(row["window_index"])


def _combined_source_sha256(paths: Iterable[Path]) -> str:
    digest = sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prama", required=True, type=Path)
    parser.add_argument("--structural-observations", type=Path)
    parser.add_argument("--outcomes", type=Path)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("config/odce_v0_1_exploratory.json"),
    )
    parser.add_argument(
        "--contract-freeze",
        type=Path,
        help="Prospective freeze manifest required for confirmatory output.",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--producer", required=True)
    parser.add_argument(
        "--partition",
        choices=("calibration", "confirmatory", "exploratory"),
        required=True,
    )
    args = parser.parse_args(argv)
    try:
        contract = json.loads(args.contract.read_text(encoding="utf-8"))
        config = ODCEConfig.from_contract(contract)
        contract_freeze_hash: str | None = None
        if args.partition == "confirmatory":
            if contract.get("status") != "FROZEN_PROSPECTIVE":
                raise ValueError(
                    "confirmatory output requires an ODCE contract with status FROZEN_PROSPECTIVE"
                )
            if args.contract_freeze is None:
                raise ValueError(
                    "confirmatory output requires --contract-freeze"
                )
            freeze = json.loads(
                args.contract_freeze.read_text(encoding="utf-8")
            )
            validate_contract_freeze(contract, freeze)
            contract_freeze_hash = sha256_value(freeze)
        elif args.contract_freeze is not None:
            raise ValueError(
                "--contract-freeze is valid only with --partition confirmatory"
            )
        prama_rows = read_jsonl(args.prama)
        for row in prama_rows:
            validate_artifact(row, "prama_trajectory")
        structural_rows = (
            read_jsonl(args.structural_observations)
            if args.structural_observations
            else ()
        )
        for row in structural_rows:
            validate_artifact(row, "structural_observation")
        outcome_rows = read_jsonl(args.outcomes) if args.outcomes else ()
        for row in outcome_rows:
            validate_artifact(row, "domain_return_observation")

        prama_by_session = _group(prama_rows)
        structural_by_session = _group(structural_rows)
        outcomes_by_session = _group(outcome_rows)
        unknown_structural = set(structural_by_session) - set(prama_by_session)
        unknown_outcomes = set(outcomes_by_session) - set(prama_by_session)
        if unknown_structural or unknown_outcomes:
            raise ValueError(
                "secondary inputs reference unknown PRAMA sessions: "
                f"{sorted(unknown_structural | unknown_outcomes)}"
            )

        source_paths = [args.prama]
        if args.structural_observations:
            source_paths.append(args.structural_observations)
        if args.outcomes:
            source_paths.append(args.outcomes)
        source_hash = _combined_source_sha256(source_paths)
        config_hash = sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        created_at = datetime.now(timezone.utc).isoformat()
        outputs: list[dict[str, Any]] = []
        for session_id, session_prama in prama_by_session.items():
            ordered_prama = sorted(session_prama, key=_window_key)
            keys = [_window_key(row) for row in ordered_prama]
            if len(keys) != len(set(keys)):
                raise ValueError(f"{session_id}: duplicate PRAMA window key")

            observations = compute_structural_conversion_trajectory(
                ordered_prama,
                structural_by_session.get(session_id, []),
                outcomes_by_session.get(session_id, []),
                config,
                normalization_contract=contract["normalization"],
                correspondence_contract=contract["correspondence"],
                contract_freeze_sha256=contract_freeze_hash,
            )
            for observation in observations:
                envelope = make_envelope(
                    artifact_type="structural_conversion_differential",
                    study_id=args.study_id,
                    session_id=session_id,
                    producer=args.producer,
                    created_at=created_at,
                    source_sha256=source_hash,
                    config_sha256=config_hash,
                    partition=args.partition,
                    channel_status=ChannelStatus.OBSERVED,
                )
                outputs.append(
                    make_structural_conversion_differential(envelope, observation)
                )
        write_jsonl_atomic(args.out, outputs)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"ODCE derivation failed: {exc}")
        return 1
    print(f"wrote {len(outputs)} ODCE-v0 records to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
