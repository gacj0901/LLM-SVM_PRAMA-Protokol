"""Typed temporal identities for DSEB-v0 artifacts."""

from __future__ import annotations

from dataclasses import dataclass


PROTOCOL_ARTIFACT_SCHEMA = "LLM-SVM-DSEB-benchmark-protocol/1"
TURN_ARTIFACT_SCHEMA = "LLM-SVM-DSEB-benchmark-turn/1"
OUTCOME_ARTIFACT_SCHEMA = "LLM-SVM-DSEB-verifier-outcome/1"
PREFLIGHT_REPORT_SCHEMA = "LLM-SVM-DSEB-offline-preflight/1"


@dataclass(frozen=True, order=True, slots=True)
class WindowIdentity:
    turn_index: int
    window_index: int

    def __post_init__(self) -> None:
        if self.turn_index < 0 or self.window_index < 0:
            raise ValueError("window identity indices must be nonnegative")

    def to_dict(self) -> dict[str, int]:
        return {
            "turn_index": self.turn_index,
            "window_index": self.window_index,
        }


class CanonicalWindowSequence:
    """Materialized canonical order of (turn_index, window_index) pairs."""

    def __init__(self) -> None:
        self._identities: list[WindowIdentity] = []
        self._ordinals: dict[WindowIdentity, int] = {}
        self._last_turn = -1

    def append_turn(self, turn_index: int, window_count: int) -> WindowIdentity:
        if turn_index != self._last_turn + 1:
            raise ValueError("turns must be appended contiguously")
        if window_count <= 0:
            raise ValueError("each turn requires at least one closed window")
        terminal: WindowIdentity | None = None
        for window_index in range(window_count):
            identity = WindowIdentity(turn_index, window_index)
            self._ordinals[identity] = len(self._identities)
            self._identities.append(identity)
            terminal = identity
        self._last_turn = turn_index
        assert terminal is not None
        return terminal

    def ordinal(self, identity: WindowIdentity) -> int:
        try:
            return self._ordinals[identity]
        except KeyError as exc:
            raise ValueError(f"window identity is not materialized: {identity}") from exc

    @property
    def identities(self) -> tuple[WindowIdentity, ...]:
        return tuple(self._identities)


@dataclass(frozen=True, slots=True)
class CausalOutcomeIdentity:
    benchmark_turn_index: int
    event_window: WindowIdentity
    event_index: int
    available_at_window: WindowIdentity
    available_at_index: int

    def validate(self, sequence: CanonicalWindowSequence) -> None:
        if self.benchmark_turn_index != self.event_window.turn_index:
            raise ValueError("benchmark turn and event window disagree")
        if sequence.ordinal(self.event_window) != self.event_index:
            raise ValueError("event_index is not the canonical window ordinal")
        if sequence.ordinal(self.available_at_window) != self.available_at_index:
            raise ValueError("available_at_index is not the canonical window ordinal")
        if self.available_at_index < self.event_index:
            raise ValueError("outcome availability precedes its event")

    def to_dict(self) -> dict[str, object]:
        return {
            "benchmark_turn_index": self.benchmark_turn_index,
            "event_window": self.event_window.to_dict(),
            "event_index": self.event_index,
            "available_at_window": self.available_at_window.to_dict(),
            "available_at_index": self.available_at_index,
        }
