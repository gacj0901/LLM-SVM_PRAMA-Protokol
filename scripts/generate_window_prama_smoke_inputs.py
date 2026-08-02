#!/usr/bin/env python
"""Generate deterministic subcritical and supercritical coupling smoke inputs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from aptadynamic_llm.artifact_schema import (
    ChannelStatus,
    make_envelope,
    sha256_value,
    validate_artifact,
    write_jsonl_atomic,
)


SUBCRITICAL = (-0.2, -0.1, 0.1, 0.7, 0.8, 0.4, 0.0, -0.1)
SUPERCRITICAL = (-1.0,) * 8 + (1.0,) * 24 + (-1.0,) * 40


def _rows(
    *,
    session_id: str,
    omega_values: tuple[float, ...],
    expected: float,
    source_sha256: str,
    config_sha256: str,
) -> list[dict[str, object]]:
    created_at = datetime(2026, 7, 30, tzinfo=timezone.utc).isoformat()
    output = []
    for index, omega in enumerate(omega_values):
        self_support = (omega + 1.0) / 2.0
        user_support = (1.0 - omega) / 2.0
        record = {
            **make_envelope(
                artifact_type="coupling_observation",
                study_id="window-prama-smoke",
                session_id=session_id,
                producer="offline-smoke-generator",
                created_at=created_at,
                source_sha256=source_sha256,
                config_sha256=config_sha256,
                partition="exploratory",
                channel_status=ChannelStatus.OBSERVED,
            ),
            "turn_index": 0,
            "window_index": index,
            "token_start": index * 16,
            "token_end": (index + 1) * 16,
            "self_support": self_support,
            "user_support": user_support,
            "interaction": 0.0,
            "support_magnitude": self_support + user_support,
            "omega_dep": omega,
            "expected_omega_dep": expected,
            "self_dependence_excess": omega - expected,
            "filler_variance": 0.0,
            "eligible": True,
            "expectation_status": "FROZEN",
        }
        validate_artifact(record, expected_type="coupling_observation")
        output.append(record)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    profile = {
        "subcritical": SUBCRITICAL,
        "supercritical": {
            "baseline_windows": 8,
            "pulse_windows": 24,
            "recovery_windows": 40,
        },
    }
    source_hash = sha256_value(profile)
    config_hash = sha256_value(
        {
            "window_tokens": 16,
            "profiles": ("subcritical", "supercritical"),
            "offline_only": True,
        }
    )
    rows = _rows(
        session_id="smoke-subcritical",
        omega_values=SUBCRITICAL,
        expected=-0.2,
        source_sha256=source_hash,
        config_sha256=config_hash,
    )
    rows.extend(
        _rows(
            session_id="smoke-supercritical",
            omega_values=SUPERCRITICAL,
            expected=-1.0,
            source_sha256=source_hash,
            config_sha256=config_hash,
        )
    )
    write_jsonl_atomic(args.out, rows)
    print(
        f"wrote {len(rows)} offline coupling smoke inputs "
        f"(subcritical={len(SUBCRITICAL)}, supercritical={len(SUPERCRITICAL)}) "
        f"to {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
