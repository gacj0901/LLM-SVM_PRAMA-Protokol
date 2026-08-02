"""Auditable external-anchor uptake measurements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from aptadynamic_llm.artifact_schema import (
    ChannelStatus,
    make_envelope,
    validate_artifact,
)


@dataclass(frozen=True)
class AnchorUptakeConfig:
    uptake_threshold: float
    response_horizon_windows: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.uptake_threshold <= 1.0:
            raise ValueError("uptake_threshold must be within [0, 1]")
        if self.response_horizon_windows < 1:
            raise ValueError("response_horizon_windows must be positive")


def evaluate_anchor_uptake(
    *,
    envelope: Mapping[str, Any],
    anchor_id: str,
    anchor_type: str,
    introduced_at_window: int,
    anchor_state: str,
    severity: float,
    externally_verifiable: bool,
    anchor_source_sha256: str,
    source_is_evaluated_trajectory: bool,
    response_windows: Iterable[Mapping[str, Any]],
    config: AnchorUptakeConfig,
) -> dict[str, Any]:
    """Measure first independently verified uptake within a frozen horizon.

    Each response row must provide ``window_index``, ``uptake_score`` in
    ``[0, 1]`` and ``verifier_passed``.  Text similarity alone is insufficient.
    """

    if source_is_evaluated_trajectory:
        raise ValueError("an anchor cannot be generated solely by the evaluated trajectory")
    if not externally_verifiable:
        raise ValueError("uptake evaluation requires an externally verifiable anchor")
    if introduced_at_window < 0:
        raise ValueError("introduced_at_window cannot be negative")
    if not 0.0 <= severity <= 1.0:
        raise ValueError("severity must be within [0, 1]")

    horizon_end = introduced_at_window + config.response_horizon_windows
    eligible = []
    for row in response_windows:
        index = int(row["window_index"])
        score = float(row["uptake_score"])
        if not 0.0 <= score <= 1.0:
            raise ValueError("uptake_score must be within [0, 1]")
        if introduced_at_window <= index <= horizon_end:
            eligible.append((index, score, bool(row["verifier_passed"])))
    eligible.sort()

    detected_at = next(
        (
            index
            for index, score, verifier_passed in eligible
            if verifier_passed and score >= config.uptake_threshold
        ),
        None,
    )
    record = {
        **dict(envelope),
        "anchor_id": anchor_id,
        "anchor_type": anchor_type,
        "introduced_at_window": introduced_at_window,
        "anchor_state": anchor_state,
        "severity": float(severity),
        "externally_verifiable": True,
        "anchor_source_sha256": anchor_source_sha256.removeprefix("sha256:"),
        "uptake_detected": detected_at is not None,
        "uptake_latency_windows": (
            None if detected_at is None else detected_at - introduced_at_window
        ),
        "peak_uptake_score": max((score for _, score, _ in eligible), default=None),
        "evaluated_response_windows": len(eligible),
        "response_horizon_windows": config.response_horizon_windows,
    }
    validate_artifact(record, "external_anchor_event")
    return record


def unavailable_anchor_artifact(
    *,
    study_id: str,
    session_id: str,
    producer: str,
    created_at: str,
    source_sha256: str,
    config_sha256: str,
    partition: str,
    reason: str,
) -> dict[str, Any]:
    """Represent absence explicitly; do not silently fabricate a negative anchor."""

    envelope = make_envelope(
        artifact_type="external_anchor_event",
        study_id=study_id,
        session_id=session_id,
        producer=producer,
        created_at=created_at,
        source_sha256=source_sha256,
        config_sha256=config_sha256,
        partition=partition,
        channel_status=ChannelStatus.UNAVAILABLE,
    )
    return {
        **envelope,
        "anchor_id": "unavailable",
        "anchor_type": "unavailable",
        "introduced_at_window": 0,
        "anchor_state": "unavailable",
        "severity": 0.0,
        "externally_verifiable": False,
        "anchor_source_sha256": source_sha256.removeprefix("sha256:"),
        "uptake_detected": False,
        "uptake_latency_windows": None,
        "unavailable_reason": reason,
    }
