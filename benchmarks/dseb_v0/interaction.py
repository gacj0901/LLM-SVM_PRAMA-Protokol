"""Render DSEB-v0 turns and parse the model's constrained response."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Mapping, Sequence

from .constraints import Constraint
from .generator import GeneratedProtocol, TurnState


FIRST_TURN_PREAMBLE = (
    "We will maintain an ordering of the same ten symbols across several turns. "
    "Persistent rules remain active until I explicitly retire or replace them. "
    "Turn-local rules expire after the current response. Return only one JSON "
    "object of the form {\"order\":[\"A\",...,\"J\"]}; include every symbol "
    "exactly once and no other keys or prose."
)


@dataclass(frozen=True, slots=True)
class ParsedOrder:
    order: tuple[str, ...]
    response_contract_valid: bool
    error: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "order": list(self.order),
            "response_contract_valid": self.response_contract_valid,
            "parse_error": self.error,
        }


def _line(constraint: Constraint) -> str:
    return f"- {constraint.constraint_id}: {constraint.to_text()}"


def _by_id(state: TurnState) -> dict[str, Constraint]:
    return {
        constraint.constraint_id: constraint
        for constraint in (*state.active_constraints, *state.temporary_constraints)
    }


def render_turn(generated: GeneratedProtocol, state: TurnState) -> str:
    """Render only causal updates; unchanged rules are not silently refreshed."""

    turn_index = state.target.turn_index
    constraints = _by_id(state)
    temporary_ids = {
        constraint.constraint_id for constraint in state.temporary_constraints
    }
    new_persistent = [
        constraints[constraint_id]
        for constraint_id in state.new_constraint_ids
        if constraint_id not in temporary_ids
    ]
    refreshed = [
        constraints[constraint_id]
        for constraint_id in state.re_presented_constraint_ids
    ]
    parts: list[str] = []
    if turn_index == 0:
        parts.append(FIRST_TURN_PREAMBLE)
        parts.append("Symbols: " + ", ".join(generated.symbols) + ".")
    parts.append(f"Turn {turn_index}. Phase: {state.target.phase}.")
    if state.checkpoint_transition:
        retired = ", ".join(state.checkpoint_retired_constraint_ids) or "none"
        parts.append(
            "Checkpoint transition. Retire these persistent rule IDs: " + retired + "."
        )
        parts.append("The complete retained persistent rule set is now:")
        parts.extend(_line(constraint) for constraint in state.active_constraints)
    else:
        if state.ordinary_removed_constraint_ids:
            parts.append(
                "Retire these persistent rule IDs: "
                + ", ".join(state.ordinary_removed_constraint_ids)
                + "."
            )
        if new_persistent:
            parts.append("Add these persistent rules:")
            parts.extend(_line(constraint) for constraint in new_persistent)
        if refreshed:
            parts.append("Reminder of still-active persistent rules:")
            parts.extend(_line(constraint) for constraint in refreshed)
    if state.temporary_constraints:
        parts.append("Apply these turn-local rules only for this response:")
        parts.extend(_line(constraint) for constraint in state.temporary_constraints)
    parts.append(
        "Return a satisfying order now as exactly {\"order\":[\"...\"]}."
    )
    return "\n".join(parts)


def parse_order_response(text: str, symbols: Sequence[str]) -> ParsedOrder:
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", stripped, re.I)
    if fenced:
        stripped = fenced.group(1).strip()
    try:
        decoded: Any = json.loads(stripped)
    except json.JSONDecodeError as exc:
        return ParsedOrder((), False, f"invalid JSON: {exc.msg}")
    if not isinstance(decoded, Mapping) or set(decoded) != {"order"}:
        return ParsedOrder((), False, "response must contain only the order key")
    order = decoded.get("order")
    if not isinstance(order, list) or any(not isinstance(value, str) for value in order):
        return ParsedOrder((), False, "order must be an array of strings")
    candidate = tuple(order)
    expected = tuple(symbols)
    if (
        len(candidate) != len(expected)
        or len(set(candidate)) != len(candidate)
        or set(candidate) != set(expected)
    ):
        return ParsedOrder(candidate, False, "order is not an exact symbol permutation")
    return ParsedOrder(candidate, True, None)
