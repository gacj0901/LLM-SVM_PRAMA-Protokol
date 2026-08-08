"""Deterministic DSEB-v0 protocol generator."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable, Sequence

from .constraints import Constraint, PermutationSolver, all_satisfied
from .protocol import DSEBProtocol, TurnTarget


@dataclass(frozen=True, slots=True)
class TurnState:
    target: TurnTarget
    active_constraints: tuple[Constraint, ...]
    temporary_constraints: tuple[Constraint, ...]
    oracle_order: tuple[str, ...]
    new_constraint_ids: tuple[str, ...]
    re_presented_constraint_ids: tuple[str, ...]
    ordinary_removed_constraint_ids: tuple[str, ...]
    ordinary_replacements: tuple[tuple[str, str], ...]
    checkpoint_retired_constraint_ids: tuple[str, ...]
    observed_context_span: int

    @property
    def constraint_load(self) -> int:
        return len(self.active_constraints)

    @property
    def perturbation_pressure(self) -> int:
        return len(self.temporary_constraints)

    @property
    def revision_pressure(self) -> int:
        return len(self.ordinary_removed_constraint_ids)

    @property
    def checkpoint_transition(self) -> bool:
        return self.target.checkpoint_transition

    @property
    def retired_constraint_count(self) -> int:
        return len(self.checkpoint_retired_constraint_ids)

    def to_dict(self) -> dict[str, object]:
        return {
            "turn_index": self.target.turn_index,
            "phase": self.target.phase,
            "targets": self.target.to_dict(),
            "controls": {
                "constraint_load": self.constraint_load,
                "context_span": self.observed_context_span,
                "revision_pressure": self.revision_pressure,
                "perturbation_pressure": self.perturbation_pressure,
                "checkpoint_transition": self.checkpoint_transition,
                "retired_constraint_count": self.retired_constraint_count,
            },
            "active_constraints": [constraint.to_dict() for constraint in self.active_constraints],
            "temporary_constraints": [
                constraint.to_dict() for constraint in self.temporary_constraints
            ],
            "oracle_order": list(self.oracle_order),
            "new_constraint_ids": list(self.new_constraint_ids),
            "re_presented_constraint_ids": list(self.re_presented_constraint_ids),
            "ordinary_removed_constraint_ids": list(self.ordinary_removed_constraint_ids),
            "ordinary_replacements": [
                {"retired": retired, "replacement": replacement}
                for retired, replacement in self.ordinary_replacements
            ],
            "checkpoint_retired_constraint_ids": list(
                self.checkpoint_retired_constraint_ids
            ),
        }


@dataclass(frozen=True, slots=True)
class GeneratedProtocol:
    seed: int
    symbols: tuple[str, ...]
    initial_witness: tuple[str, ...]
    turns: tuple[TurnState, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "symbols": list(self.symbols),
            "initial_witness": list(self.initial_witness),
            "turn_count": len(self.turns),
            "turns": [turn.to_dict() for turn in self.turns],
        }


class DSEBGenerator:
    """Generate one satisfiable, provenance-complete DSEB-v0 trajectory."""

    def __init__(self, protocol: DSEBProtocol, seed: int) -> None:
        if seed < 0:
            raise ValueError("seed must be nonnegative")
        self.protocol = protocol
        self.seed = seed
        self.rng = random.Random(seed)
        self.symbols = tuple(chr(ord("A") + index) for index in range(protocol.symbol_count))
        witness = list(self.symbols)
        self.rng.shuffle(witness)
        self.initial_witness = tuple(witness)
        self.solver = PermutationSolver(self.symbols)
        self._persistent_counter = 0
        self._temporary_counter = 0

    def generate(self) -> GeneratedProtocol:
        active: list[Constraint] = []
        preferred = self.initial_witness
        turns: list[TurnState] = []
        for target in self.protocol.turns:
            checkpoint_retired: list[str] = []
            ordinary_removed: list[str] = []
            replacements: list[tuple[str, str]] = []
            new_ids: list[str] = []
            re_presented: list[str] = []

            if target.checkpoint_transition:
                active, checkpoint_retired = self._apply_checkpoint(
                    active, target.constraint_load, target.turn_index
                )
                re_presented.extend(constraint.constraint_id for constraint in active)
            else:
                active, removed, replacement_pairs, replacement_new = self._apply_revisions(
                    active,
                    target,
                    preferred,
                )
                ordinary_removed.extend(removed)
                replacements.extend(replacement_pairs)
                new_ids.extend(replacement_new)

            while len(active) < target.constraint_load:
                constraint = self._new_constraint(
                    preferred,
                    target.turn_index,
                    existing=active,
                )
                active.append(constraint)
                new_ids.append(constraint.constraint_id)
            if len(active) != target.constraint_load:
                raise RuntimeError(f"turn {target.turn_index}: failed to reach target C_t")

            active, refreshed = self._align_context_span(
                active, target.turn_index, target.context_span
            )
            re_presented.extend(refreshed)
            if target.turn_index in self.protocol.anchor_refresh_turns:
                active, anchor_id = self._refresh_future_anchor(active, target.turn_index)
                re_presented.append(anchor_id)

            temporary: list[Constraint] = []
            for _ in range(target.perturbation_pressure):
                temporary.append(
                    self._new_constraint(
                        preferred,
                        target.turn_index,
                        existing=(*active, *temporary),
                        temporary=True,
                    )
                )
            new_ids.extend(constraint.constraint_id for constraint in temporary)

            oracle = self.solver.solve(
                (*active, *temporary), preferred_order=preferred
            )
            if oracle is None:
                raise RuntimeError(
                    f"turn {target.turn_index}: CSP state is unsatisfiable"
                )
            if not all_satisfied(oracle, (*active, *temporary)):
                raise AssertionError("solver returned a non-satisfying oracle")
            preferred = oracle
            observed_h = self._context_span(active, target.turn_index)
            state = TurnState(
                target=target,
                active_constraints=tuple(active),
                temporary_constraints=tuple(temporary),
                oracle_order=oracle,
                new_constraint_ids=tuple(dict.fromkeys(new_ids)),
                re_presented_constraint_ids=tuple(dict.fromkeys(re_presented)),
                ordinary_removed_constraint_ids=tuple(ordinary_removed),
                ordinary_replacements=tuple(replacements),
                checkpoint_retired_constraint_ids=tuple(checkpoint_retired),
                observed_context_span=observed_h,
            )
            self._validate_state(state)
            turns.append(state)
        return GeneratedProtocol(
            seed=self.seed,
            symbols=self.symbols,
            initial_witness=self.initial_witness,
            turns=tuple(turns),
        )

    def _apply_checkpoint(
        self,
        active: list[Constraint],
        target_load: int,
        turn_index: int,
    ) -> tuple[list[Constraint], list[str]]:
        if target_load > len(active):
            raise RuntimeError("checkpoint cannot retain more constraints than are active")
        retained = sorted(
            active,
            key=lambda constraint: (
                constraint.last_presented_at,
                constraint.constraint_id,
            ),
        )[:target_load]
        retained_ids = {constraint.constraint_id for constraint in retained}
        retired = [
            constraint.constraint_id
            for constraint in active
            if constraint.constraint_id not in retained_ids
        ]
        return [constraint.refreshed(turn_index) for constraint in retained], retired

    def _apply_revisions(
        self,
        active: list[Constraint],
        target: TurnTarget,
        preferred: Sequence[str],
    ) -> tuple[list[Constraint], list[str], list[tuple[str, str]], list[str]]:
        active = list(active)
        required_removals = max(0, len(active) - target.constraint_load)
        if required_removals > target.revision_pressure:
            raise RuntimeError(
                f"turn {target.turn_index}: C_t drop exceeds ordinary R_t"
            )
        victims = self._revision_victims(active, target.revision_pressure)
        removed: list[str] = []
        replacements: list[tuple[str, str]] = []
        new_ids: list[str] = []
        for action_index, victim in enumerate(victims):
            active.remove(victim)
            removed.append(victim.constraint_id)
            if action_index >= required_removals:
                replacement = self._new_constraint(
                    preferred,
                    target.turn_index,
                    existing=active,
                    revision_of=victim.constraint_id,
                )
                active.append(replacement)
                replacements.append((victim.constraint_id, replacement.constraint_id))
                new_ids.append(replacement.constraint_id)
        return active, removed, replacements, new_ids

    @staticmethod
    def _revision_victims(
        active: Sequence[Constraint], count: int
    ) -> list[Constraint]:
        if count > len(active):
            raise RuntimeError("revision_pressure exceeds active constraint count")
        # Prefer the most recently presented constraints, preserving the oldest
        # contextual anchor whenever possible.
        return sorted(
            active,
            key=lambda constraint: (
                -constraint.last_presented_at,
                constraint.constraint_id,
            ),
        )[:count]

    def _new_constraint(
        self,
        order: Sequence[str],
        turn_index: int,
        *,
        existing: Iterable[Constraint],
        revision_of: str | None = None,
        temporary: bool = False,
    ) -> Constraint:
        existing_signatures = {constraint.signature for constraint in existing}
        for _ in range(1000):
            kind = self.rng.choice(
                (
                    "before",
                    "adjacent",
                    "distance",
                    "bounded_before",
                    "conditional_before",
                )
            )
            positions = list(range(len(order)))
            if kind == "adjacent":
                left_position = self.rng.randrange(len(order) - 1)
                symbols = (order[left_position], order[left_position + 1])
                parameter = None
            elif kind == "conditional_before":
                selected = self.rng.sample(positions, 4)
                a, b = sorted(selected[:2])
                c, d = sorted(selected[2:])
                symbols = (order[a], order[b], order[c], order[d])
                parameter = None
            else:
                left_position, right_position = sorted(self.rng.sample(positions, 2))
                symbols = (order[left_position], order[right_position])
                if kind == "distance":
                    parameter = right_position - left_position - 1
                elif kind == "bounded_before":
                    minimum = right_position - left_position
                    parameter = self.rng.randint(minimum, len(order) - 1)
                else:
                    parameter = None
            signature = (kind, tuple(symbols), parameter)
            if signature in existing_signatures:
                continue
            constraint_id = self._next_constraint_id(temporary=temporary)
            candidate = Constraint(
                constraint_id=constraint_id,
                kind=kind,
                symbols=tuple(symbols),
                parameter=parameter,
                introduced_at=turn_index,
                last_presented_at=turn_index,
                revision_of=revision_of,
            )
            if candidate.evaluate(order):
                return candidate
        raise RuntimeError("unable to sample a unique satisfiable constraint")

    def _next_constraint_id(self, *, temporary: bool) -> str:
        if temporary:
            self._temporary_counter += 1
            return f"DSEB-P{self._temporary_counter:04d}"
        self._persistent_counter += 1
        return f"DSEB-C{self._persistent_counter:04d}"

    @staticmethod
    def _context_span(active: Sequence[Constraint], turn_index: int) -> int:
        if not active:
            raise RuntimeError("DSEB requires at least one active constraint")
        return max(turn_index - constraint.last_presented_at for constraint in active)

    def _align_context_span(
        self,
        active: list[Constraint],
        turn_index: int,
        target_span: int,
    ) -> tuple[list[Constraint], list[str]]:
        refreshed: list[str] = []
        aligned: list[Constraint] = []
        for constraint in active:
            age = turn_index - constraint.last_presented_at
            if age > target_span:
                aligned.append(constraint.refreshed(turn_index))
                refreshed.append(constraint.constraint_id)
            else:
                aligned.append(constraint)
        observed = self._context_span(aligned, turn_index)
        if observed != target_span:
            raise RuntimeError(
                f"turn {turn_index}: context target {target_span} is causally "
                f"unreachable; observed {observed}"
            )
        return aligned, refreshed

    @staticmethod
    def _refresh_future_anchor(
        active: list[Constraint], turn_index: int
    ) -> tuple[list[Constraint], str]:
        if len(active) < 2:
            raise RuntimeError("future anchor refresh requires at least two constraints")
        oldest = min(constraint.last_presented_at for constraint in active)
        candidates = [
            (index, constraint)
            for index, constraint in enumerate(active)
            if constraint.last_presented_at > oldest
        ]
        if not candidates:
            candidates = list(enumerate(active[1:], start=1))
        index, selected = sorted(
            candidates,
            key=lambda item: (-item[1].last_presented_at, item[1].constraint_id),
        )[0]
        refreshed = list(active)
        refreshed[index] = selected.refreshed(turn_index)
        return refreshed, selected.constraint_id

    @staticmethod
    def _validate_state(state: TurnState) -> None:
        target = state.target
        observed = {
            "C_t": state.constraint_load,
            "H_t": state.observed_context_span,
            "R_t": state.revision_pressure,
            "P_t": state.perturbation_pressure,
            "K_t": int(state.checkpoint_transition),
        }
        expected = {
            "C_t": target.constraint_load,
            "H_t": target.context_span,
            "R_t": target.revision_pressure,
            "P_t": target.perturbation_pressure,
            "K_t": int(target.checkpoint_transition),
        }
        if observed != expected:
            raise RuntimeError(
                f"turn {target.turn_index}: control mismatch {observed} != {expected}"
            )
        active_ids = {constraint.constraint_id for constraint in state.active_constraints}
        temporary_ids = {
            constraint.constraint_id for constraint in state.temporary_constraints
        }
        if active_ids & temporary_ids:
            raise AssertionError("persistent and temporary IDs overlap")
        if set(state.re_presented_constraint_ids) & set(state.new_constraint_ids):
            raise AssertionError("identical re-presentation cannot become new evidence")
        if state.checkpoint_transition:
            if state.revision_pressure != 0:
                raise AssertionError("checkpoint retirements leaked into ordinary R_t")
            if set(state.re_presented_constraint_ids) != active_ids:
                raise AssertionError("checkpoint did not re-present every retained constraint")
