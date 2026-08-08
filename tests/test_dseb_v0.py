import json
from pathlib import Path

import pytest

from benchmarks.dseb_v0.constraints import Constraint, PermutationSolver
from benchmarks.dseb_v0.generator import DSEBGenerator
from benchmarks.dseb_v0.interaction import parse_order_response, render_turn
from benchmarks.dseb_v0.preflight import run_offline_preflight
from benchmarks.dseb_v0.protocol import CAUSAL_STAGE_ORDER, load_protocol
from benchmarks.dseb_v0.schemas import (
    CausalOutcomeIdentity,
    CanonicalWindowSequence,
)
from benchmarks.dseb_v0.verifier import verify_order
from scripts.run_dseb_smoke import (
    OLLAMA_CONVERSATION_ADAPTER,
    _ollama_prompt,
    _profile,
    parse_args as parse_smoke_args,
    run as run_smoke,
)
from scripts.project_dseb_smoke import (
    _promote_outcomes,
    parse_args as parse_projection_args,
    project as project_smoke,
)


ROOT = Path(__file__).parents[1]
PROTOCOL_PATH = ROOT / "benchmarks" / "dseb_v0" / "configs" / "dseb_v0.json"
SMOKE_PROTOCOL_PATH = (
    ROOT / "benchmarks" / "dseb_v0" / "configs" / "dseb_v0_smoke.json"
)
FROZEN_PRAMA_SOURCE = (
    ROOT / "run_outputs" / "prama_kernel_cb41d590" / "PRAMA-Protokol-py"
)


def make_constraint(
    constraint_id,
    kind,
    symbols,
    parameter=None,
    turn=0,
):
    return Constraint(
        constraint_id=constraint_id,
        kind=kind,
        symbols=tuple(symbols),
        parameter=parameter,
        introduced_at=turn,
        last_presented_at=turn,
    )


def test_constraint_language_evaluates_all_declared_kinds():
    order = ("A", "B", "C", "D")
    constraints = [
        make_constraint("c1", "before", ("A", "D")),
        make_constraint("c2", "adjacent", ("B", "C")),
        make_constraint("c3", "distance", ("A", "C"), 1),
        make_constraint("c4", "bounded_before", ("A", "D"), 3),
        make_constraint("c5", "conditional_before", ("A", "B", "C", "D")),
    ]
    assert all(constraint.evaluate(order) for constraint in constraints)
    assert not constraints[0].evaluate(tuple(reversed(order)))


def test_solver_finds_current_oracle_and_rejects_unsatisfiable_state():
    solver = PermutationSolver(("A", "B", "C"))
    reverse_initial = make_constraint("c1", "before", ("B", "A"))
    oracle = solver.solve((reverse_initial,), preferred_order=("A", "B", "C"))
    assert oracle is not None
    assert oracle != ("A", "B", "C")
    assert reverse_initial.evaluate(oracle)
    impossible = (
        make_constraint("c2", "before", ("A", "B")),
        make_constraint("c3", "before", ("B", "A")),
    )
    assert solver.solve(impossible) is None


def test_protocol_is_v0_exploratory_and_has_realizable_recovery():
    protocol = load_protocol(PROTOCOL_PATH)
    assert protocol.benchmark_id == "DSEB_v0"
    assert protocol.benchmark_version == "DSEB-v0"
    assert protocol.status == "EXPLORATORY"
    assert len(protocol.turns) == 36
    assert [turn.context_span for turn in protocol.turns[21:26]] == [0, 1, 2, 3, 4]
    assert [turn.turn_index for turn in protocol.turns if turn.checkpoint_transition] == [21]
    assert protocol.causal_stage_order == CAUSAL_STAGE_ORDER


def test_smoke_profile_remains_v0_and_has_twelve_realizable_turns():
    protocol = load_protocol(SMOKE_PROTOCOL_PATH)
    assert protocol.benchmark_id == "DSEB_v0"
    assert protocol.benchmark_version == "DSEB-v0"
    assert protocol.status == "EXPLORATORY"
    assert protocol.profile == "smoke"
    assert len(protocol.turns) == 12
    assert protocol.checkpoint_turn == 7
    assert [turn.context_span for turn in protocol.turns[7:10]] == [0, 1, 2]
    generated = DSEBGenerator(protocol, 7).generate()
    assert len(generated.turns) == 12


@pytest.mark.parametrize("seed", [0, 1, 2, 7, 1337])
def test_generator_matches_every_control_and_keeps_each_state_satisfiable(seed):
    generated = DSEBGenerator(load_protocol(PROTOCOL_PATH), seed).generate()
    assert len(generated.turns) == 36
    for state in generated.turns:
        target = state.target
        assert state.constraint_load == target.constraint_load
        assert state.observed_context_span == target.context_span
        assert state.revision_pressure == target.revision_pressure
        assert state.perturbation_pressure == target.perturbation_pressure
        result = verify_order(
            symbols=generated.symbols,
            order=state.oracle_order,
            persistent=state.active_constraints,
            temporary=state.temporary_constraints,
            new_constraint_ids=state.new_constraint_ids,
        )
        assert result.functional_gain == 1.0
        assert result.verified_outcome == 1


def test_checkpoint_retirements_are_not_ordinary_revision_or_external_evidence():
    generated = DSEBGenerator(load_protocol(PROTOCOL_PATH), 7).generate()
    checkpoint = generated.turns[21]
    assert checkpoint.retired_constraint_count == 5
    assert checkpoint.revision_pressure == 0
    assert not checkpoint.new_constraint_ids
    assert set(checkpoint.re_presented_constraint_ids) == {
        constraint.constraint_id for constraint in checkpoint.active_constraints
    }
    result = verify_order(
        symbols=generated.symbols,
        order=checkpoint.oracle_order,
        persistent=checkpoint.active_constraints,
        new_constraint_ids=checkpoint.new_constraint_ids,
    )
    assert result.external_integration is None
    assert result.external_integration_status == "NOT_APPLICABLE"


def test_counterfactual_constraints_never_persist_to_next_turn():
    generated = DSEBGenerator(load_protocol(PROTOCOL_PATH), 7).generate()
    for current, following in zip(generated.turns, generated.turns[1:]):
        temporary_ids = {
            constraint.constraint_id for constraint in current.temporary_constraints
        }
        following_active = {
            constraint.constraint_id for constraint in following.active_constraints
        }
        assert temporary_ids.isdisjoint(following_active)


def test_verifier_distinguishes_not_applicable_unavailable_and_observed():
    symbols = ("A", "B", "C")
    constraint = make_constraint("new", "before", ("A", "B"))
    not_applicable = verify_order(
        symbols=symbols,
        order=symbols,
        persistent=(constraint,),
    )
    assert not_applicable.external_integration_status == "NOT_APPLICABLE"
    unavailable = verify_order(
        symbols=symbols,
        order=symbols,
        persistent=(constraint,),
        new_constraint_ids=("new",),
        measurement_available=False,
    )
    assert unavailable.external_integration_status == "UNAVAILABLE"
    observed = verify_order(
        symbols=symbols,
        order=symbols,
        persistent=(constraint,),
        new_constraint_ids=("new",),
    )
    assert observed.external_integration_status == "OBSERVED"
    assert observed.external_integration == 1.0


def test_verifier_reports_partial_functional_gain_without_collapsing_syntax():
    symbols = ("A", "B", "C")
    constraints = (
        make_constraint("c1", "before", ("A", "B")),
        make_constraint("c2", "before", ("B", "C")),
    )
    result = verify_order(
        symbols=symbols,
        order=("B", "A", "C"),
        persistent=constraints,
    )
    assert result.syntax_valid is True
    assert result.functional_gain == 0.5
    assert result.verified_outcome == 0


def test_canonical_window_ordinal_is_not_reconstructed_from_turn():
    sequence = CanonicalWindowSequence()
    sequence.append_turn(0, 2)
    terminal = sequence.append_turn(1, 3)
    identity = CausalOutcomeIdentity(
        benchmark_turn_index=1,
        event_window=terminal,
        event_index=4,
        available_at_window=terminal,
        available_at_index=4,
    )
    identity.validate(sequence)
    assert identity.event_index != identity.benchmark_turn_index


def test_offline_preflight_writes_reproducible_causal_bundle(tmp_path):
    first = run_offline_preflight(
        protocol_path=PROTOCOL_PATH,
        seed=7,
        output_dir=tmp_path / "first",
    )
    second = run_offline_preflight(
        protocol_path=PROTOCOL_PATH,
        seed=7,
        output_dir=tmp_path / "second",
    )
    assert first["status"] == second["status"] == "PASS"
    assert first["generated_protocol_sha256"] == second["generated_protocol_sha256"]
    assert first["checks_failed"] == second["checks_failed"] == 0
    assert first["turn_count"] == 36
    assert first["canonical_window_count"] == 72
    assert first["outcome_count"] == 36
    turns = [
        json.loads(line)
        for line in (tmp_path / "first" / "benchmark_turns.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    outcomes = [
        json.loads(line)
        for line in (tmp_path / "first" / "verifier_outcomes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(turns) == len(outcomes) == 36
    assert outcomes[0]["event_index"] == outcomes[0]["available_at_index"] == 1
    assert outcomes[-1]["event_index"] == outcomes[-1]["available_at_index"] == 71
    assert all(row["retrospective_backfill"] is False for row in outcomes)
    assert all(row["pipeline_execution_mode"] == "SIMULATED_OFFLINE" for row in outcomes)
    assert all(row["causal_stage_order_validated"] is True for row in outcomes)
    assert all(row["causal_stage_order"] == list(CAUSAL_STAGE_ORDER) for row in outcomes)


def test_dseb_json_schemas_are_present_and_parseable():
    for name in (
        "dseb-benchmark-protocol.schema.json",
        "dseb-benchmark-turn.schema.json",
        "dseb-verifier-outcome.schema.json",
    ):
        schema = json.loads((ROOT / "schemas" / name).read_text(encoding="utf-8"))
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["type"] == "object"


def test_renderer_preserves_causal_presentations_and_parser_is_strict():
    generated = DSEBGenerator(load_protocol(SMOKE_PROTOCOL_PATH), 7).generate()
    first = render_turn(generated, generated.turns[0])
    assert "Symbols: A, B, C, D, E, F, G, H, I, J." in first
    checkpoint = render_turn(generated, generated.turns[7])
    assert "Checkpoint transition" in checkpoint
    assert "complete retained persistent rule set" in checkpoint
    valid = parse_order_response(
        json.dumps({"order": list(generated.turns[0].oracle_order)}),
        generated.symbols,
    )
    assert valid.response_contract_valid is True
    extra_key = parse_order_response(
        json.dumps({"order": list(generated.turns[0].oracle_order), "score": 1}),
        generated.symbols,
    )
    assert extra_key.response_contract_valid is False


def test_smoke_runner_dry_run_writes_nothing(tmp_path):
    output = tmp_path / "dry"
    args = parse_smoke_args(
        [
            "--model",
            "hermes3:8b",
            "--output-dir",
            str(output),
            "--dry-run",
        ]
    )
    report = run_smoke(args)
    assert report["mode"] == "dry_run"
    assert report["turn_count"] == 12
    assert report["model_call_executed"] is False
    assert not output.exists()


def test_ollama_transcript_does_not_invite_role_tag_continuation(tmp_path):
    messages = [
        {"role": "user", "content": "First task."},
        {"role": "assistant", "content": '{"order":["A"]}'},
        {"role": "user", "content": "Current task."},
    ]
    prompt = _ollama_prompt(messages)
    assert "<assistant>" not in prompt
    assert "</assistant>" not in prompt
    assert "<user>" not in prompt
    assert prompt.endswith("[CURRENT ASSISTANT]")
    args = parse_smoke_args(
        [
            "--model",
            "hermes3:8b",
            "--output-dir",
            str(tmp_path / "unused"),
            "--dry-run",
        ]
    )
    assert _profile(args)["conversation_adapter"] == OLLAMA_CONVERSATION_ADAPTER


def test_smoke_runner_fake_acquisition_is_resumable_and_not_prematurely_causal(
    tmp_path,
):
    protocol = load_protocol(SMOKE_PROTOCOL_PATH)
    generated = DSEBGenerator(protocol, 7).generate()

    def fake_acquire(messages, args, turn_index):
        answer = json.dumps({"order": list(generated.turns[turn_index].oracle_order)})
        tokens = [
            {
                "token": token,
                "top1_logprob": -0.2,
                "top_logprobs": [-0.2, -1.2],
                "gap": 1.0,
                "entropy": 0.5,
            }
            for token in answer.split()
        ]
        return (
            {
                "turn_index": turn_index,
                "user_message": messages[-1]["content"],
                "assistant_message": answer,
                "finish_reason": "stop",
                "token_count": len(tokens),
                "tokens": tokens,
            },
            args.model,
            0.01,
            1,
        )

    output = tmp_path / "fake"
    args = parse_smoke_args(
        [
            "--model",
            "hermes3:8b",
            "--output-dir",
            str(output),
            "--seed",
            "7",
        ]
    )
    report = run_smoke(args, acquire_fn=fake_acquire)
    assert report["status"] == "ACQUIRED_PENDING_STRUCTURAL_PROJECTION"
    assert report["completed_turn_count"] == 12
    assert report["formal_pass_count"] == 12
    assert report["response_contract_pass_count"] == 12
    assert report["prama_executed"] is False
    assert report["verifier_previews_eligible_for_odce"] is False
    previews = [
        json.loads(line)
        for line in (output / "verifier_previews.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert all(row["available_at_index"] is None for row in previews)
    assert all(row["eligible_for_odce"] is False for row in previews)
    args.resume = True
    resumed = run_smoke(args, acquire_fn=fake_acquire)
    assert resumed["completed_turn_count"] == 12

    raw_path = next((output / "sessions").glob("*/raw.json"))
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    prama_rows = []
    for preview in previews:
        window = preview["event_window"]
        prama_rows.append(
            {
                "turn_index": window["turn_index"],
                "window_index": window["window_index"],
            }
        )
    outcomes = _promote_outcomes(
        raw=raw,
        raw_path=raw_path,
        previews=previews,
        prama_rows=prama_rows,
        protocol_hash=report["protocol_source_sha256"],
    )
    assert len(outcomes) == 12
    assert all(row["event_index"] == row["available_at_index"] for row in outcomes)
    assert all(row["retrospective_backfill"] is False for row in outcomes)
    assert outcomes[0]["event_window"] == outcomes[0]["available_at_window"]


def test_smoke_causal_projection_closes_all_stages_with_frozen_prama(tmp_path):
    if not FROZEN_PRAMA_SOURCE.is_dir():
        pytest.skip("local frozen PRAMA checkout is not available")
    generated = DSEBGenerator(load_protocol(SMOKE_PROTOCOL_PATH), 7).generate()

    def fake_acquire(messages, args, turn_index):
        answer = json.dumps({"order": list(generated.turns[turn_index].oracle_order)})
        tokens = [
            {
                "token": f"t{index}",
                "top1_logprob": -0.2 - 0.01 * ((turn_index + index) % 5),
                "top_logprobs": [-0.2, -1.2],
                "gap": 1.0,
                "entropy": 0.5,
            }
            for index in range(20 + turn_index)
        ]
        return (
            {
                "assistant_message": answer,
                "finish_reason": "stop",
                "token_count": len(tokens),
                "tokens": tokens,
            },
            args.model,
            0.01,
            1,
        )

    acquisition = tmp_path / "acquisition"
    acquisition_args = parse_smoke_args(
        [
            "--model",
            "hermes3:8b",
            "--output-dir",
            str(acquisition),
            "--seed",
            "7",
        ]
    )
    run_smoke(acquisition_args, acquire_fn=fake_acquire)
    projection = tmp_path / "projection"
    projection_args = parse_projection_args(
        [
            "--acquisition-run",
            str(acquisition),
            "--output-dir",
            str(projection),
            "--prama-source-root",
            str(FROZEN_PRAMA_SOURCE),
        ]
    )
    report = project_smoke(projection_args)
    assert report["status"] == "COMPLETE_EXPLORATORY_CAUSAL"
    assert report["domain_outcome_count"] == 12
    assert report["odce_observation_count"] == report["prama_observation_count"]
    outcomes = [
        json.loads(line)
        for line in (projection / "domain_return_observations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert all(row["event_index"] == row["available_at_index"] for row in outcomes)
    odce = [
        json.loads(line)
        for line in (projection / "structural_conversion_differentials.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert all(row["causal"] is True for row in odce)
    assert all(row["future_outcome_used"] is False for row in odce)
    assert all(row["causal_availability_enforced"] is True for row in odce)
