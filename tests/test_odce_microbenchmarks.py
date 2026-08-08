import json
from pathlib import Path

import pytest

from scripts import run_odce_microbenchmarks as micro


SUITE = Path(__file__).parents[1] / "data" / "odce_microbenchmarks_v1.json"
OPERATIONAL_SUITE = (
    Path(__file__).parents[1] / "data" / "odce_operational_calibration_v1.json"
)
OPERATIONAL_SUITE_V1_1 = (
    Path(__file__).parents[1] / "data" / "odce_operational_calibration_v1_1.json"
)


def test_suite_has_staged_balanced_selection():
    suite = micro.load_suite(SUITE)
    assert len(micro.select_items(suite, "smoke", "all")) == 2
    assert len(micro.select_items(suite, "pilot", "all")) == 8
    assert len(micro.select_items(suite, "full", "all")) == 24
    assert len(micro.select_items(suite, "pilot", "gsm8k")) == 4
    assert len(micro.select_items(suite, "pilot", "ifeval")) == 4
    prompts = [micro.effective_prompt(item) for item in suite["items"]]
    assert all(prompt.strip() for prompt in prompts)
    assert all("prama" not in prompt.casefold() for prompt in prompts)


def test_operational_suite_is_small_staged_and_has_external_checks():
    suite = micro.load_suite(OPERATIONAL_SUITE)
    assert len(micro.select_items(suite, "smoke", "all")) == 2
    assert len(micro.select_items(suite, "pilot", "all")) == 4
    assert len(micro.select_items(suite, "full", "all")) == 8
    assert len(micro.select_items(suite, "pilot", "dynamic_range")) == 2
    assert len(micro.select_items(suite, "pilot", "evidence_integration")) == 2
    evidence = [
        item for item in suite["items"] if item["family"] == "evidence_integration"
    ]
    assert all(
        item["verifier"]["external_integration_constraints"] for item in evidence
    )


def test_operational_suite_v1_1_aligns_declared_word_floors_and_terminators():
    suite = micro.load_suite(OPERATIONAL_SUITE_V1_1)
    assert suite["suite_id"] == "odce_operational_calibration_v1_1"
    assert len(micro.select_items(suite, "smoke", "all")) == 2
    assert len(micro.select_items(suite, "pilot", "all")) == 4
    assert len(micro.select_items(suite, "full", "all")) == 8
    dynamic = [item for item in suite["items"] if item["family"] == "dynamic_range"]
    for item in dynamic:
        length_constraints = [
            constraint
            for constraint in item["verifier"]["constraints"]
            if constraint["id"] == "length_constraints:number_words"
        ]
        assert length_constraints == [
            {
                "id": "length_constraints:number_words",
                "kwargs": {"relation": "at least", "num_words": 480},
            }
        ]
        assert "no punctuation or text after" in item["prompt"]


@pytest.mark.parametrize(
    ("text", "constraint", "passed"),
    [
        ("only lowercase letters.", {"id": "change_case:english_lowercase"}, True),
        ("Has uppercase.", {"id": "change_case:english_lowercase"}, False),
        ("ALL CAPS!", {"id": "change_case:english_capital"}, True),
        ('{"ok": true}', {"id": "detectable_format:json_format"}, True),
        ("```json\n{\"ok\": true}\n```", {"id": "detectable_format:json_format"}, True),
        ("prefix {\"ok\": true}", {"id": "detectable_format:json_format"}, False),
        (
            "* one\n* two\n* three",
            {"id": "detectable_format:number_bullet_lists", "kwargs": {"num_bullets": 3}},
            True,
        ),
        (
            "- one\n- two\n- three",
            {"id": "detectable_format:number_bullet_lists", "kwargs": {"num_bullets": 3}},
            True,
        ),
        (
            "+ one\n + two\n  * three",
            {"id": "detectable_format:number_bullet_lists", "kwargs": {"num_bullets": 3}},
            True,
        ),
        (
            "*emphasis*\n---\nordinary prose",
            {"id": "detectable_format:number_bullet_lists", "kwargs": {"num_bullets": 0}},
            True,
        ),
        (
            "safe wording",
            {"id": "keywords:forbidden_words", "kwargs": {"forbidden_words": ["word"]}},
            True,
        ),
        (
            "one two three",
            {"id": "length_constraints:number_words", "kwargs": {"relation": "less than", "num_words": 4}},
            True,
        ),
        (
            "Done. Exact ending",
            {"id": "startend:end_checker", "kwargs": {"end_phrase": "Exact ending"}},
            True,
        ),
    ],
)
def test_ifeval_compatible_checks(text, constraint, passed):
    assert micro.verify_ifeval_constraint(text, constraint)["passed"] is passed


def test_numeric_verifier_prefers_final_answer_marker():
    item = {"verifier": {"type": "numeric_exact", "expected": "18"}}
    result = micro.verify_response(item, "16 - 7 = 9. 9 * 2 = 18.\nFinal answer: 18")
    assert result["passed"] is True
    assert result["functional_gain"] == 1.0
    assert result["external_integration"] is None


def test_external_integration_is_scored_separately_and_emitted(tmp_path):
    item = {
        "verifier": {
            "type": "ifeval_constraints",
            "constraints": [
                {"id": "keywords:existence", "kwargs": {"keywords": ["summary"]}}
            ],
            "external_integration_constraints": [
                {"id": "keywords:existence", "kwargs": {"keywords": ["amber"]}},
                {"id": "keywords:existence", "kwargs": {"keywords": ["cobalt"]}},
            ],
        }
    }
    verification = micro.verify_response(item, "summary includes amber")
    assert verification["functional_gain"] == 1.0
    assert verification["external_integration"] == 0.5
    assert verification["verified_outcome"] == 0.0
    raw_path = tmp_path / "raw.json"
    raw = {"session_id": "s1", "turns": [{"token_count": 17}]}
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    outcome = micro.make_domain_outcome(
        study_id="test-operational-study",
        raw_path=raw_path,
        raw=raw,
        verification=verification,
        suite_sha256="a" * 64,
        verifier_sha256="b" * 64,
        window_size=16,
    )
    assert outcome["benefit_vector"]["external_integration"] == 0.5
    assert outcome["study_id"] == "test-operational-study"
    assert outcome["component_status"]["external_integration"] == "OBSERVED"


def test_provider_server_error_classification():
    error = RuntimeError(
        "generation failed: Error code: 500 - Internal Server Error"
    )
    assert micro._is_provider_server_error(error) is True
    assert micro._is_timeout_error(error) is False


def test_domain_outcome_is_available_only_at_terminal_window(tmp_path):
    raw_path = tmp_path / "raw.json"
    raw = {
        "session_id": "s1",
        "turns": [{"token_count": 33}],
    }
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    outcome = micro.make_domain_outcome(
        study_id="test-microbenchmark-study",
        raw_path=raw_path,
        raw=raw,
        verification={"functional_gain": 0.5, "verified_outcome": 0.0},
        suite_sha256="a" * 64,
        verifier_sha256="b" * 64,
        window_size=16,
    )
    assert outcome["event_index"] == 2
    assert outcome["available_at_index"] == 2
    assert outcome["retrospective_backfill"] is False
    assert outcome["provider_termination_metadata_used"] is False


def test_dry_run_validates_plan_without_writing(tmp_path):
    output = tmp_path / "unused"
    args = micro.parse_args(
        [
            "--model",
            "hermes3:8b",
            "--tier",
            "pilot",
            "--output-dir",
            str(output),
            "--dry-run",
        ]
    )
    report = micro.run(args)
    assert report["selected_count"] == 8
    assert report["mode"] == "dry_run"
    assert not output.exists()


def test_dry_run_can_select_one_item_for_sensitivity_analysis(tmp_path):
    args = micro.parse_args(
        [
            "--model",
            "hermes3:8b",
            "--tier",
            "pilot",
            "--item-id",
            "ifeval-1075",
            "--output-dir",
            str(tmp_path / "unused"),
            "--dry-run",
        ]
    )
    report = micro.run(args)
    assert report["selected_count"] == 1
    assert report["item_ids"] == ["ifeval-1075"]


def test_item_id_must_exist_and_belong_to_selected_tier(tmp_path):
    args = micro.parse_args(
        [
            "--model",
            "hermes3:8b",
            "--tier",
            "smoke",
            "--item-id",
            "ifeval-1075",
            "--output-dir",
            str(tmp_path / "unused"),
            "--dry-run",
        ]
    )
    with pytest.raises(ValueError, match="selection contains no items"):
        micro.run(args)


def test_eol_model_is_visible_in_plan_but_cannot_run(tmp_path):
    args = micro.parse_args(
        [
            "--model",
            "mistralai/mistral-medium-3.5-128b",
            "--tier",
            "smoke",
            "--output-dir",
            str(tmp_path / "run"),
        ]
    )
    with pytest.raises(ValueError, match="reached end of life"):
        micro.run(args)


def test_smoke_run_is_resumable_and_writes_causal_outcomes(tmp_path, monkeypatch):
    output = tmp_path / "run"

    def fake_acquire(item, prompt, args):
        answer = (
            "Final answer: 18"
            if item["family"] == "gsm8k"
            else "are two young boys holding toy guns and horns?"
        )
        tokens = [
            {
                "token": str(index),
                "top1_logprob": -0.1,
                "top_logprobs": [-0.1, -1.1],
                "gap": 1.0,
                "entropy": 0.5,
            }
            for index in range(17)
        ]
        return (
            {
                "resolved_model": args.model,
                "turn": {
                    "turn_index": 0,
                    "user_message": prompt,
                    "assistant_message": answer,
                    "finish_reason": "stop",
                    "token_count": len(tokens),
                    "tokens": tokens,
                },
            },
            0.1,
            1,
        )

    monkeypatch.setattr(micro, "acquire", fake_acquire)
    args = micro.parse_args(
        ["--model", "hermes3:8b", "--tier", "smoke", "--output-dir", str(output)]
    )
    report = micro.run(args)
    assert report["status"] == "COMPLETE"
    assert report["completed_count"] == 2
    rows = [
        json.loads(line)
        for line in (output / "domain_return_observations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rows) == 2
    assert {row["study_id"] for row in rows} == {"odce-microbenchmarks-v1"}
    assert all(row["event_index"] == row["available_at_index"] == 1 for row in rows)
    assert all(row["benefit_vector"]["functional_gain"] == 1.0 for row in rows)

    def should_not_acquire(*_args, **_kwargs):
        raise AssertionError("resume must reuse existing raw artifacts")

    monkeypatch.setattr(micro, "acquire", should_not_acquire)
    resumed_args = micro.parse_args(
        [
            "--model",
            "hermes3:8b",
            "--tier",
            "smoke",
            "--output-dir",
            str(output),
            "--resume",
        ]
    )
    resumed = micro.run(resumed_args)
    assert resumed["status"] == "COMPLETE"
    assert resumed["completed_count"] == 2


def test_run_stops_after_provider_server_error_budget(tmp_path, monkeypatch):
    calls = []

    def server_error(item, prompt, args):
        calls.append(item["item_id"])
        raise RuntimeError("Error code: 500 - Internal Server Error")

    monkeypatch.setattr(micro, "acquire", server_error)
    args = micro.parse_args(
        [
            "--model",
            "hermes3:8b",
            "--tier",
            "smoke",
            "--output-dir",
            str(tmp_path / "run"),
        ]
    )
    report = micro.run(args)
    assert len(calls) == 1
    assert report["provider_server_error_count"] == 1
    assert report["halted_reason"] == "provider_server_error_budget_exhausted"
