"""Independent deterministic outcome verifier for DSEB-v0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .constraints import Constraint


@dataclass(frozen=True, slots=True)
class VerificationResult:
    syntax_valid: bool
    active_constraint_count: int
    satisfied_constraint_count: int
    functional_gain: float
    new_constraint_count: int
    new_constraint_satisfied_count: int | None
    external_integration: float | None
    external_integration_status: str
    verified_outcome: int
    constraint_checks: tuple[tuple[str, bool], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "syntax_valid": self.syntax_valid,
            "active_constraint_count": self.active_constraint_count,
            "satisfied_constraint_count": self.satisfied_constraint_count,
            "functional_gain": self.functional_gain,
            "new_constraint_count": self.new_constraint_count,
            "new_constraint_satisfied_count": self.new_constraint_satisfied_count,
            "external_integration": self.external_integration,
            "external_integration_status": self.external_integration_status,
            "verified_outcome": self.verified_outcome,
            "constraint_checks": [
                {"constraint_id": constraint_id, "satisfied": satisfied}
                for constraint_id, satisfied in self.constraint_checks
            ],
        }


def verify_order(
    *,
    symbols: Sequence[str],
    order: Sequence[str],
    persistent: Sequence[Constraint],
    temporary: Sequence[Constraint] = (),
    new_constraint_ids: Sequence[str] = (),
    measurement_available: bool = True,
) -> VerificationResult:
    """Score only formal world constraints; never inspect PRAMA or ODCE readings."""

    expected_symbols = tuple(symbols)
    candidate = tuple(order)
    syntax_valid = (
        len(candidate) == len(expected_symbols)
        and len(set(candidate)) == len(candidate)
        and set(candidate) == set(expected_symbols)
    )
    effective = tuple(persistent) + tuple(temporary)
    checks = tuple(
        (
            constraint.constraint_id,
            bool(syntax_valid and constraint.evaluate(candidate)),
        )
        for constraint in effective
    )
    satisfied = sum(int(passed) for _, passed in checks)
    functional_gain = satisfied / len(effective) if effective else float(syntax_valid)
    effective_by_id = {constraint.constraint_id: constraint for constraint in effective}
    new_ids = tuple(dict.fromkeys(str(value) for value in new_constraint_ids))
    unknown_new = set(new_ids) - set(effective_by_id)
    if unknown_new:
        raise ValueError(f"new_constraint_ids are not effective: {sorted(unknown_new)}")
    if not new_ids:
        external_value = None
        external_status = "NOT_APPLICABLE"
        new_satisfied: int | None = None
    elif not measurement_available:
        external_value = None
        external_status = "UNAVAILABLE"
        new_satisfied = None
    else:
        new_satisfied = sum(
            int(syntax_valid and effective_by_id[constraint_id].evaluate(candidate))
            for constraint_id in new_ids
        )
        external_value = new_satisfied / len(new_ids)
        external_status = "OBSERVED"
    verified = int(syntax_valid and satisfied == len(effective))
    return VerificationResult(
        syntax_valid=syntax_valid,
        active_constraint_count=len(effective),
        satisfied_constraint_count=satisfied,
        functional_gain=functional_gain,
        new_constraint_count=len(new_ids),
        new_constraint_satisfied_count=new_satisfied,
        external_integration=external_value,
        external_integration_status=external_status,
        verified_outcome=verified,
        constraint_checks=checks,
    )
