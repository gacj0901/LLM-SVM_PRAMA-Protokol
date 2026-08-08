"""Offline preflight for DSEB-v0; never calls a model or structural instrument."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import tempfile
from typing import Any, Iterable

from .constraints import Constraint, PermutationSolver
from .generator import DSEBGenerator
from .protocol import DSEBProtocol, load_protocol
from .schemas import (
    CausalOutcomeIdentity,
    CanonicalWindowSequence,
    OUTCOME_ARTIFACT_SCHEMA,
    PREFLIGHT_REPORT_SCHEMA,
    PROTOCOL_ARTIFACT_SCHEMA,
    TURN_ARTIFACT_SCHEMA,
)
from .verifier import verify_order


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        delete=False,
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        delete=False,
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "details": details}


def run_offline_preflight(
    *,
    protocol_path: Path,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    generated = DSEBGenerator(protocol, seed).generate()
    session_id = f"dseb-v0-{protocol.profile}-offline-seed-{seed:04d}"
    protocol_hash = file_sha256(protocol_path)
    verifier_hash = file_sha256(Path(__file__).with_name("verifier.py"))
    generated_hash = canonical_sha256(generated.to_dict())
    sequence = CanonicalWindowSequence()
    turn_rows: list[dict[str, Any]] = []
    outcome_rows: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []

    previous_temporary_ids: set[str] = set()
    for state in generated.turns:
        turn_index = state.target.turn_index
        active_ids = {
            constraint.constraint_id for constraint in state.active_constraints
        }
        checks.append(
            _check(
                f"turn_{turn_index:02d}_counterfactual_scope",
                not (previous_temporary_ids & active_ids),
                leaked_ids=sorted(previous_temporary_ids & active_ids),
            )
        )
        previous_temporary_ids = {
            constraint.constraint_id for constraint in state.temporary_constraints
        }

        window_count = 1 + ((seed + turn_index) % 3)
        terminal = sequence.append_turn(turn_index, window_count)
        event_index = sequence.ordinal(terminal)
        identity = CausalOutcomeIdentity(
            benchmark_turn_index=turn_index,
            event_window=terminal,
            event_index=event_index,
            available_at_window=terminal,
            available_at_index=event_index,
        )
        identity.validate(sequence)
        verification = verify_order(
            symbols=generated.symbols,
            order=state.oracle_order,
            persistent=state.active_constraints,
            temporary=state.temporary_constraints,
            new_constraint_ids=state.new_constraint_ids,
        )
        controls_match = (
            state.constraint_load == state.target.constraint_load
            and state.observed_context_span == state.target.context_span
            and state.revision_pressure == state.target.revision_pressure
            and state.perturbation_pressure == state.target.perturbation_pressure
            and state.checkpoint_transition == state.target.checkpoint_transition
        )
        checks.append(
            _check(
                f"turn_{turn_index:02d}_controls",
                controls_match,
                observed=state.to_dict()["controls"],
                expected=state.target.to_dict(),
            )
        )
        checks.append(
            _check(
                f"turn_{turn_index:02d}_oracle",
                verification.functional_gain == 1.0
                and verification.verified_outcome == 1,
                functional_gain=verification.functional_gain,
                verified_outcome=verification.verified_outcome,
            )
        )
        checks.append(
            _check(
                f"turn_{turn_index:02d}_causal_identity",
                identity.event_index == sequence.ordinal(identity.event_window)
                and identity.available_at_index
                == sequence.ordinal(identity.available_at_window)
                and identity.available_at_index >= identity.event_index,
                **identity.to_dict(),
            )
        )
        turn_rows.append(
            {
                "schema": TURN_ARTIFACT_SCHEMA,
                "benchmark_id": protocol.benchmark_id,
                "benchmark_version": protocol.benchmark_version,
                "session_id": session_id,
                **state.to_dict(),
                "simulated_closed_window_count": window_count,
                "terminal_window": terminal.to_dict(),
                "terminal_event_index": event_index,
            }
        )
        outcome_rows.append(
            {
                "schema": OUTCOME_ARTIFACT_SCHEMA,
                "benchmark_id": protocol.benchmark_id,
                "benchmark_version": protocol.benchmark_version,
                "partition": "exploratory",
                "session_id": session_id,
                **identity.to_dict(),
                "causal": True,
                "retrospective_backfill": False,
                "pipeline_execution_mode": "SIMULATED_OFFLINE",
                "causal_stage_order_validated": True,
                "causal_stage_order": list(protocol.causal_stage_order),
                "verifier_reference_sha256": verifier_hash,
                **verification.to_dict(),
            }
        )

    checkpoint_index = protocol.checkpoint_turn
    checkpoint = generated.turns[checkpoint_index]
    checks.append(
        _check(
            "checkpoint_separation",
            checkpoint.checkpoint_transition
            and checkpoint.retired_constraint_count > 0
            and checkpoint.revision_pressure == 0
            and set(checkpoint.re_presented_constraint_ids)
            == {
                constraint.constraint_id
                for constraint in checkpoint.active_constraints
            }
            and not checkpoint.new_constraint_ids,
            retired_constraint_count=checkpoint.retired_constraint_count,
            revision_pressure=checkpoint.revision_pressure,
            new_constraint_ids=list(checkpoint.new_constraint_ids),
        )
    )
    checks.append(
        _check(
            "recovery_context_trajectory",
            [
                turn.observed_context_span
                for turn in generated.turns[
                    checkpoint_index : checkpoint_index
                    + len(protocol.recovery_context_path)
                ]
            ]
            == list(protocol.recovery_context_path),
            observed=[
                turn.observed_context_span
                for turn in generated.turns[
                    checkpoint_index : checkpoint_index
                    + len(protocol.recovery_context_path)
                ]
            ],
        )
    )
    checkpoint_outcome = outcome_rows[checkpoint_index]
    checks.append(
        _check(
            "checkpoint_re_presentation_is_not_external_evidence",
            checkpoint_outcome["external_integration_status"] == "NOT_APPLICABLE"
            and checkpoint_outcome["external_integration"] is None,
            external_integration_status=checkpoint_outcome[
                "external_integration_status"
            ],
        )
    )

    # Prove that the current-state solver is not an alias for the initial witness.
    left, right = generated.initial_witness[:2]
    drift_constraint = Constraint(
        constraint_id="DSEB-PREFLIGHT-DRIFT",
        kind="before",
        symbols=(right, left),
        parameter=None,
        introduced_at=0,
        last_presented_at=0,
    )
    drift_oracle = PermutationSolver(generated.symbols).solve(
        (drift_constraint,), preferred_order=generated.initial_witness
    )
    checks.append(
        _check(
            "state_oracle_not_initial_witness",
            drift_oracle is not None
            and drift_oracle != generated.initial_witness
            and drift_constraint.evaluate(drift_oracle),
            initial_witness=list(generated.initial_witness),
            state_oracle=list(drift_oracle or ()),
        )
    )

    unavailable_probe = verify_order(
        symbols=generated.symbols,
        order=generated.turns[0].oracle_order,
        persistent=generated.turns[0].active_constraints,
        new_constraint_ids=generated.turns[0].new_constraint_ids,
        measurement_available=False,
    )
    checks.append(
        _check(
            "external_integration_unavailable_semantics",
            unavailable_probe.external_integration is None
            and unavailable_probe.external_integration_status == "UNAVAILABLE",
            status=unavailable_probe.external_integration_status,
        )
    )
    checks.append(
        _check(
            "causal_pipeline_order",
            tuple(protocol.causal_stage_order)
            == (
                "terminal_window_closed",
                "O_D_observed",
                "PRAMA_projected",
                "D_O_v9_observed",
                "outcome_verified",
                "outcome_registered",
                "ODCE_executed",
            ),
            execution_mode="SIMULATED_OFFLINE",
            stage_order=list(protocol.causal_stage_order),
        )
    )

    all_passed = all(check["passed"] for check in checks)
    protocol_artifact = {
        "schema": PROTOCOL_ARTIFACT_SCHEMA,
        "benchmark_id": protocol.benchmark_id,
        "benchmark_version": protocol.benchmark_version,
        "profile": protocol.profile,
        "partition": "exploratory",
        "session_id": session_id,
        "seed": seed,
        "symbol_count": len(generated.symbols),
        "symbols": list(generated.symbols),
        "initial_witness": list(generated.initial_witness),
        "turn_count": len(generated.turns),
        "canonical_window_count": len(sequence.identities),
        "protocol_source": str(protocol_path.resolve()),
        "protocol_source_sha256": protocol_hash,
        "generated_protocol_sha256": generated_hash,
        "causal_stage_order": list(protocol.causal_stage_order),
        "claim_boundary": (
            "Offline protocol verification only; no LLM, PRAMA, D_O or ODCE "
            "reading is produced."
        ),
    }
    report = {
        "schema": PREFLIGHT_REPORT_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all_passed else "FAIL",
        "benchmark_id": protocol.benchmark_id,
        "benchmark_version": protocol.benchmark_version,
        "profile": protocol.profile,
        "partition": "exploratory",
        "session_id": session_id,
        "seed": seed,
        "model_call_executed": False,
        "prama_executed": False,
        "structural_observer_executed": False,
        "odce_executed": False,
        "turn_count": len(turn_rows),
        "canonical_window_count": len(sequence.identities),
        "outcome_count": len(outcome_rows),
        "protocol_source_sha256": protocol_hash,
        "generated_protocol_sha256": generated_hash,
        "checks_passed": sum(int(check["passed"]) for check in checks),
        "checks_failed": sum(int(not check["passed"]) for check in checks),
        "checks": checks,
    }
    _write_json_atomic(output_dir / "benchmark_protocol.json", protocol_artifact)
    _write_jsonl_atomic(output_dir / "benchmark_turns.jsonl", turn_rows)
    _write_jsonl_atomic(output_dir / "verifier_outcomes.jsonl", outcome_rows)
    _write_json_atomic(output_dir / "report.json", report)
    return report
