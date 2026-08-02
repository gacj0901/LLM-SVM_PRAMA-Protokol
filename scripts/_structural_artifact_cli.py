"""Shared CLI helpers for structural artifact builders."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from aptadynamic_llm.artifact_schema import ChannelStatus, make_envelope


def read_rows(path: Path) -> tuple[dict[str, Any], ...]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: row must be an object")
        rows.append(value)
    if not rows:
        raise ValueError(f"{path}: no rows")
    return tuple(rows)


def source_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def envelope_for(
    args: argparse.Namespace,
    *,
    artifact_type: str,
    session_id: str,
    config: Any,
) -> dict[str, Any]:
    return make_envelope(
        artifact_type=artifact_type,
        study_id=args.study_id,
        session_id=session_id,
        producer=args.producer,
        created_at=datetime.now(timezone.utc).isoformat(),
        source_sha256=source_sha256(args.input),
        config_sha256=sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        partition=args.partition,
        channel_status=ChannelStatus.OBSERVED,
    )


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--producer", required=True)
    parser.add_argument(
        "--partition",
        choices=("calibration", "confirmatory", "exploratory"),
        required=True,
    )
