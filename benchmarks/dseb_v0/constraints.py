"""Constraint language and deterministic CSP solver for DSEB-v0."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Mapping, Sequence


SUPPORTED_KINDS = {
    "before",
    "adjacent",
    "distance",
    "bounded_before",
    "conditional_before",
}


@dataclass(frozen=True, slots=True)
class Constraint:
    """One persistent or turn-local ordering constraint."""

    constraint_id: str
    kind: str
    symbols: tuple[str, ...]
    parameter: int | None
    introduced_at: int
    last_presented_at: int
    revision_of: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in SUPPORTED_KINDS:
            raise ValueError(f"unsupported constraint kind {self.kind!r}")
        expected_arity = 4 if self.kind == "conditional_before" else 2
        if len(self.symbols) != expected_arity:
            raise ValueError(f"{self.kind} requires {expected_arity} symbols")
        if len(set(self.symbols)) < 2:
            raise ValueError("a constraint requires at least two distinct symbols")
        if self.kind in {"distance", "bounded_before"}:
            if self.parameter is None or self.parameter < 0:
                raise ValueError(f"{self.kind} requires a nonnegative parameter")
            if self.kind == "bounded_before" and self.parameter < 1:
                raise ValueError("bounded_before requires parameter >= 1")
        elif self.parameter is not None:
            raise ValueError(f"{self.kind} does not accept a parameter")
        if self.introduced_at < 0 or self.last_presented_at < self.introduced_at:
            raise ValueError("invalid constraint presentation chronology")

    @property
    def signature(self) -> tuple[object, ...]:
        return (self.kind, self.symbols, self.parameter)

    def refreshed(self, turn_index: int) -> "Constraint":
        if turn_index < self.last_presented_at:
            raise ValueError("cannot refresh a constraint in the past")
        return replace(self, last_presented_at=turn_index)

    def evaluate(self, order: Sequence[str]) -> bool:
        if len(set(order)) != len(order):
            return False
        positions = {symbol: index for index, symbol in enumerate(order)}
        if any(symbol not in positions for symbol in self.symbols):
            return False
        return self._evaluate_positions(positions)

    def _evaluate_positions(self, positions: Mapping[str, int]) -> bool:
        if self.kind == "before":
            left, right = self.symbols
            return positions[left] < positions[right]
        if self.kind == "adjacent":
            left, right = self.symbols
            return abs(positions[left] - positions[right]) == 1
        if self.kind == "distance":
            left, right = self.symbols
            return abs(positions[left] - positions[right]) - 1 == self.parameter
        if self.kind == "bounded_before":
            left, right = self.symbols
            distance = positions[right] - positions[left]
            return 1 <= distance <= int(self.parameter)
        antecedent_left, antecedent_right, consequent_left, consequent_right = self.symbols
        antecedent = positions[antecedent_left] < positions[antecedent_right]
        consequent = positions[consequent_left] < positions[consequent_right]
        return (not antecedent) or consequent

    def to_text(self) -> str:
        if self.kind == "before":
            return f"{self.symbols[0]} precedes {self.symbols[1]}"
        if self.kind == "adjacent":
            return f"{self.symbols[0]} is adjacent to {self.symbols[1]}"
        if self.kind == "distance":
            return (
                f"exactly {self.parameter} symbols occur between "
                f"{self.symbols[0]} and {self.symbols[1]}"
            )
        if self.kind == "bounded_before":
            return (
                f"{self.symbols[0]} precedes {self.symbols[1]} by at most "
                f"{self.parameter} positions"
            )
        a, b, c, d = self.symbols
        return f"if {a} precedes {b}, then {c} precedes {d}"

    def to_dict(self) -> dict[str, object]:
        return {
            "constraint_id": self.constraint_id,
            "kind": self.kind,
            "symbols": list(self.symbols),
            "parameter": self.parameter,
            "introduced_at": self.introduced_at,
            "last_presented_at": self.last_presented_at,
            "revision_of": self.revision_of,
            "text": self.to_text(),
        }


def all_satisfied(order: Sequence[str], constraints: Iterable[Constraint]) -> bool:
    return all(constraint.evaluate(order) for constraint in constraints)


class PermutationSolver:
    """Find a satisfying permutation with deterministic backtracking and pruning."""

    def __init__(self, symbols: Sequence[str]) -> None:
        canonical = tuple(str(symbol) for symbol in symbols)
        if not canonical or len(set(canonical)) != len(canonical):
            raise ValueError("solver symbols must be nonempty and unique")
        self.symbols = canonical

    def solve(
        self,
        constraints: Sequence[Constraint],
        *,
        preferred_order: Sequence[str] | None = None,
    ) -> tuple[str, ...] | None:
        constraints = tuple(constraints)
        known = set(self.symbols)
        if any(not set(constraint.symbols).issubset(known) for constraint in constraints):
            raise ValueError("constraint references a symbol outside the solver world")
        preferred = tuple(preferred_order or self.symbols)
        if set(preferred) != known or len(preferred) != len(self.symbols):
            raise ValueError("preferred_order must be a permutation of solver symbols")
        rank = {symbol: index for index, symbol in enumerate(preferred)}
        degree = {symbol: 0 for symbol in self.symbols}
        for constraint in constraints:
            for symbol in set(constraint.symbols):
                degree[symbol] += 1

        order: list[str] = []
        remaining = set(self.symbols)
        positions: dict[str, int] = {}

        def search() -> tuple[str, ...] | None:
            if not remaining:
                candidate = tuple(order)
                return candidate if all_satisfied(candidate, constraints) else None
            candidates = sorted(
                remaining,
                key=lambda symbol: (-degree[symbol], rank[symbol], symbol),
            )
            position = len(order)
            for symbol in candidates:
                order.append(symbol)
                remaining.remove(symbol)
                positions[symbol] = position
                if self._partial_consistent(constraints, positions, remaining):
                    solved = search()
                    if solved is not None:
                        return solved
                del positions[symbol]
                remaining.add(symbol)
                order.pop()
            return None

        return search()

    def _partial_consistent(
        self,
        constraints: Sequence[Constraint],
        positions: Mapping[str, int],
        remaining: set[str],
    ) -> bool:
        next_position = len(positions)
        size = len(self.symbols)
        for constraint in constraints:
            assigned = [symbol in positions for symbol in constraint.symbols]
            if all(assigned):
                if not constraint._evaluate_positions(positions):
                    return False
                continue
            if constraint.kind == "conditional_before":
                a, b, c, d = constraint.symbols
                if a in positions and b in positions and positions[a] < positions[b]:
                    if not self._partial_before(c, d, positions, remaining):
                        return False
                continue
            left, right = constraint.symbols
            if constraint.kind == "before":
                if not self._partial_before(left, right, positions, remaining):
                    return False
            elif constraint.kind == "bounded_before":
                if not self._partial_bounded_before(
                    left,
                    right,
                    int(constraint.parameter),
                    positions,
                    remaining,
                    next_position,
                ):
                    return False
            elif constraint.kind in {"adjacent", "distance"}:
                distance = 1 if constraint.kind == "adjacent" else int(constraint.parameter) + 1
                if not self._partial_exact_distance(
                    left,
                    right,
                    distance,
                    positions,
                    remaining,
                    next_position,
                    size,
                ):
                    return False
        return True

    @staticmethod
    def _partial_before(
        left: str,
        right: str,
        positions: Mapping[str, int],
        remaining: set[str],
    ) -> bool:
        if left in positions and right in positions:
            return positions[left] < positions[right]
        if right in positions and left in remaining:
            return False
        return True

    @staticmethod
    def _partial_bounded_before(
        left: str,
        right: str,
        limit: int,
        positions: Mapping[str, int],
        remaining: set[str],
        next_position: int,
    ) -> bool:
        if left in positions and right in positions:
            distance = positions[right] - positions[left]
            return 1 <= distance <= limit
        if right in positions and left in remaining:
            return False
        if left in positions and right in remaining:
            return next_position <= positions[left] + limit
        return True

    @staticmethod
    def _partial_exact_distance(
        left: str,
        right: str,
        distance: int,
        positions: Mapping[str, int],
        remaining: set[str],
        next_position: int,
        size: int,
    ) -> bool:
        if left in positions and right in positions:
            return abs(positions[left] - positions[right]) == distance
        assigned_symbol = left if left in positions else right if right in positions else None
        if assigned_symbol is None:
            return True
        other = right if assigned_symbol == left else left
        if other not in remaining:
            return False
        position = positions[assigned_symbol]
        required_future = position + distance
        return next_position <= required_future < size
