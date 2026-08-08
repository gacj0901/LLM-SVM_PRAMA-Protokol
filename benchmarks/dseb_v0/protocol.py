"""Versioned DSEB-v0 protocol schedule and validation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping


PROTOCOL_SCHEMA = "LLM-SVM-DSEB-protocol/1"
CAUSAL_STAGE_ORDER = (
    "terminal_window_closed",
    "O_D_observed",
    "PRAMA_projected",
    "D_O_v9_observed",
    "outcome_verified",
    "outcome_registered",
    "ODCE_executed",
)


@dataclass(frozen=True, slots=True)
class TurnTarget:
    turn_index: int
    phase: str
    constraint_load: int
    context_span: int
    revision_pressure: int
    perturbation_pressure: int
    checkpoint_transition: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "turn_index": self.turn_index,
            "phase": self.phase,
            "constraint_load": self.constraint_load,
            "context_span": self.context_span,
            "revision_pressure": self.revision_pressure,
            "perturbation_pressure": self.perturbation_pressure,
            "checkpoint_transition": self.checkpoint_transition,
        }


@dataclass(frozen=True, slots=True)
class DSEBProtocol:
    benchmark_id: str
    benchmark_version: str
    status: str
    profile: str
    symbol_count: int
    turns: tuple[TurnTarget, ...]
    checkpoint_turn: int
    recovery_context_path: tuple[int, ...]
    anchor_refresh_turns: frozenset[int]
    causal_stage_order: tuple[str, ...]
    raw: Mapping[str, Any]

    def target(self, turn_index: int) -> TurnTarget:
        return self.turns[turn_index]


def _integer_list(phase: Mapping[str, Any], name: str, length: int) -> list[int]:
    values = phase.get(name)
    if not isinstance(values, list) or len(values) != length:
        raise ValueError(f"phase {phase.get('name')!r}: {name} must contain {length} values")
    if any(not isinstance(value, int) or value < 0 for value in values):
        raise ValueError(f"phase {phase.get('name')!r}: {name} must be nonnegative integers")
    return values


def load_protocol(path: Path) -> DSEBProtocol:
    decoded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict) or decoded.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError(f"unsupported DSEB protocol schema in {path}")
    if decoded.get("benchmark_id") != "DSEB_v0":
        raise ValueError("protocol benchmark_id must remain DSEB_v0")
    if decoded.get("benchmark_version") != "DSEB-v0":
        raise ValueError("protocol benchmark_version must remain DSEB-v0")
    if decoded.get("status") != "EXPLORATORY":
        raise ValueError("DSEB-v0 implementation is exploratory")
    symbol_count = decoded.get("symbol_count")
    if not isinstance(symbol_count, int) or symbol_count < 4:
        raise ValueError("symbol_count must be an integer >= 4")
    phases = decoded.get("phases")
    if not isinstance(phases, list) or not phases:
        raise ValueError("protocol requires phases")

    targets: list[TurnTarget] = []
    for phase in phases:
        if not isinstance(phase, dict):
            raise ValueError("each phase must be an object")
        name = str(phase.get("name") or "")
        start = phase.get("start_turn")
        end = phase.get("end_turn")
        if not name or not isinstance(start, int) or not isinstance(end, int) or end < start:
            raise ValueError("invalid phase name or turn interval")
        length = end - start + 1
        load = _integer_list(phase, "constraint_load", length)
        span = _integer_list(phase, "context_span", length)
        revision = _integer_list(phase, "revision_pressure", length)
        perturbation = _integer_list(phase, "perturbation_pressure", length)
        checkpoint_turns = set(phase.get("checkpoint_turns") or [])
        if any(not isinstance(turn, int) or turn < start or turn > end for turn in checkpoint_turns):
            raise ValueError(f"phase {name!r}: invalid checkpoint_turns")
        for offset, turn_index in enumerate(range(start, end + 1)):
            targets.append(
                TurnTarget(
                    turn_index=turn_index,
                    phase=name,
                    constraint_load=load[offset],
                    context_span=span[offset],
                    revision_pressure=revision[offset],
                    perturbation_pressure=perturbation[offset],
                    checkpoint_transition=turn_index in checkpoint_turns,
                )
            )

    profile = decoded.get("profile", "full")
    if profile not in {"full", "smoke"}:
        raise ValueError("DSEB-v0 profile must be full or smoke")
    declared_turns = decoded.get("turn_count")
    expected_turns = 36 if profile == "full" else 12
    if not isinstance(declared_turns, int) or declared_turns != expected_turns:
        raise ValueError(
            f"DSEB-v0 {profile} profile requires exactly {expected_turns} turns"
        )
    if len(targets) != declared_turns:
        raise ValueError("phase schedules do not cover the declared turn_count")
    if [target.turn_index for target in targets] != list(range(declared_turns)):
        raise ValueError("phase schedules must cover contiguous turns exactly once")

    checkpoints = [target.turn_index for target in targets if target.checkpoint_transition]
    checkpoint_turn = decoded.get("checkpoint_turn")
    if not isinstance(checkpoint_turn, int) or checkpoints != [checkpoint_turn]:
        raise ValueError("checkpoint_turn must identify the only checkpoint transition")
    last_checkpoint: int | None = None
    previous_load = 0
    for target in targets:
        if target.constraint_load <= 0:
            raise ValueError("constraint_load must remain positive")
        if target.checkpoint_transition:
            if target.context_span != 0 or target.revision_pressure != 0:
                raise ValueError("checkpoint must have H_t=0 and ordinary R_t=0")
            last_checkpoint = target.turn_index
        else:
            ordinary_drop = max(0, previous_load - target.constraint_load)
            if ordinary_drop > target.revision_pressure:
                raise ValueError(
                    f"turn {target.turn_index}: load drop requires at least "
                    f"{ordinary_drop} ordinary retirements"
                )
        causal_age_limit = target.turn_index - (last_checkpoint or 0)
        if target.context_span > causal_age_limit:
            raise ValueError(
                f"turn {target.turn_index}: context_span exceeds causal age limit"
            )
        previous_load = target.constraint_load

    recovery_context_path = decoded.get("recovery_context_path")
    if (
        not isinstance(recovery_context_path, list)
        or not recovery_context_path
        or any(not isinstance(value, int) or value < 0 for value in recovery_context_path)
    ):
        raise ValueError("recovery_context_path must be a nonempty integer list")
    recovery_end = checkpoint_turn + len(recovery_context_path)
    recovery = [
        target.context_span for target in targets[checkpoint_turn:recovery_end]
    ]
    if recovery != recovery_context_path:
        raise ValueError("recovery context schedule differs from recovery_context_path")
    causal_order = tuple(decoded.get("causal_stage_order") or ())
    if causal_order != CAUSAL_STAGE_ORDER:
        raise ValueError("causal_stage_order differs from the DSEB-v0 contract")
    anchor_refresh = frozenset(decoded.get("anchor_refresh_turns") or [])
    if any(not isinstance(turn, int) or turn < 0 or turn >= declared_turns for turn in anchor_refresh):
        raise ValueError("anchor_refresh_turns contains an invalid turn")
    return DSEBProtocol(
        benchmark_id="DSEB_v0",
        benchmark_version="DSEB-v0",
        status="EXPLORATORY",
        profile=profile,
        symbol_count=symbol_count,
        turns=tuple(targets),
        checkpoint_turn=checkpoint_turn,
        recovery_context_path=tuple(recovery_context_path),
        anchor_refresh_turns=anchor_refresh,
        causal_stage_order=causal_order,
        raw=decoded,
    )
