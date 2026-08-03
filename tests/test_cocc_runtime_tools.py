import json
from pathlib import Path

import pytest

from scripts._cocc_verify_worker import evaluate, exception_result
from scripts.calibrate_cocc_projector import build
from scripts.cocc_external_verifier import (
    audit_worker_exception,
    extract_source,
    verify,
)
from scripts.normalize_cocc_break_chain import natural_task_prompt, split_for
from scripts.project_cocc_prama import normalized_surprisal
from scripts.verify_historical_cocc_v9 import (
    verifier_diagnostic_audit,
    verifier_diagnostic_bucket,
)


def test_natural_prompt_excludes_monitoring_context():
    prompt = natural_task_prompt("Add two integers.", "def add(a, b):\n    pass")
    lowered = prompt.lower()
    assert "add two integers" in lowered
    for forbidden in (
        "prama",
        "interface",
        "monitor",
        "trajectory",
        "entropy",
        "logprob",
        "label",
    ):
        assert forbidden not in lowered


def test_problem_split_is_deterministic_and_grouped():
    first = split_for("problem-1", "seed", 0.25)
    assert first == split_for("problem-1", "seed", 0.25)
    assert first in {"calibration", "test"}


def test_clean_control_prompt_uses_clean_problem_only():
    clean = natural_task_prompt("Return the maximum.", "def solve(values):\n    pass")
    attacked = natural_task_prompt("Return the minimum.", "def solve(values):\n    pass")
    assert "maximum" in clean and "minimum" not in clean
    assert "minimum" in attacked and "maximum" not in attacked


def test_calibration_is_numeric_and_frozen(tmp_path):
    paths = []
    for index in range(4):
        path = tmp_path / f"s{index}.json"
        path.write_text(
            json.dumps(
                {
                    "schema": "LLM-SVM-CoCC-projector-request/1",
                    "session_id": f"s{index}",
                    "model_id": "model",
                    "input_channel_status": "OBSERVED",
                    "turns": [
                        {
                            "tokens": [
                                {"top1_logprob": -0.2 - index * 0.01}
                                for _ in range(20)
                            ]
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)
    artifact = build(paths, window_size=16, min_sessions=4)
    assert artifact["status"] == "FROZEN"
    assert artifact["session_count"] == 4
    assert artifact["surprisal_scale"]["value"] > 0
    assert artifact["contains_prompt_or_answer"] is False
    assert artifact["contains_outcome_labels"] is False
    assert set(artifact["expected_by_window"]) == {"0", "1"}


def test_worker_distinguishes_correct_and_incorrect_functional_code():
    tests = [
        {"input": "2\n3", "output": "5", "testtype": "functional"},
        {"input": "-2\n1", "output": "-1", "testtype": "functional"},
    ]
    correct = evaluate(
        {
            "source": "def add(a, b):\n    return a + b",
            "tests": tests,
            "func_name": "add",
        }
    )
    incorrect = evaluate(
        {
            "source": "def add(a, b):\n    return a - b",
            "tests": tests,
            "func_name": "add",
        }
    )
    assert correct["passed"] is True
    assert incorrect["passed"] is False


def test_missing_callable_requires_attribute_error_and_exact_message():
    result = exception_result(AttributeError("callable 'add' not found"), 2)
    audited = audit_worker_exception(result)
    assert audited["failure_kind"] == "missing_callable"
    assert audited["exception_type"] == "AttributeError"
    assert audited["exception_audit"]["missing_callable_message_match"] is True


def test_internal_attribute_error_is_not_missing_callable():
    result = exception_result(
        AttributeError("'dict' object has no attribute 'value'"), 2
    )
    audited = audit_worker_exception(result)
    assert audited["failure_kind"] == "AttributeError"
    assert audited["exception_audit"]["missing_callable_message_match"] is False


def test_parent_rejects_forged_missing_callable_label():
    with pytest.raises(ValueError, match="disagrees with exception message"):
        audit_worker_exception(
            {
                "passed": False,
                "status": "failed",
                "failure_kind": "missing_callable",
                "exception_type": "AttributeError",
                "exception_message": "'dict' object has no attribute 'value'",
            }
        )


def test_extraction_selects_shorter_block_that_declares_required_callable():
    answer = """```python
irrelevant = '''this is deliberately the longest block''' * 20
```
```python
def add(a, b):
    return a + b
```"""
    source, audit = extract_source(answer, "add")
    assert source == "def add(a, b):\n    return a + b"
    assert audit["status"] == "selected"
    assert audit["matching_block_count"] == 1


def test_extraction_rejects_multiple_blocks_that_declare_callable():
    answer = """```python
def add(a, b): return a + b
```
```python
def add(a, b): return sum((a, b))
```"""
    source, audit = extract_source(answer, "add")
    assert source is None
    assert audit["status"] == "ambiguous_extraction"


def test_extraction_distinguishes_callable_absent_from_missing_callable():
    source, audit = extract_source("```python\nvalue = 3\n```", "add")
    assert source is None
    assert audit["status"] == "callable_absent"


def test_single_syntax_error_block_is_preserved_for_worker_diagnosis():
    source, audit = extract_source("```python\ndef add(:\n```", "add")
    assert source == "def add(:"
    assert audit["status"] == "selected_syntax_unresolved"


def test_verifier_transports_unicode_source_to_worker_as_utf8():
    rows = {
        "item": {
            "public_test_cases": [
                {"input": "2\n3", "output": "5", "testtype": "functional"}
            ],
            "private_test_cases": [],
            "metadata": {"func_name": "add"},
        }
    }
    result = verify(
        {
            "item_id": "item",
            "observed_answer": "```python\n# accepts values ≤ limit\ndef add(a, b):\n    return a + b\n```",
        },
        rows,
        Path(__file__).resolve().parents[1] / "scripts/_cocc_verify_worker.py",
        timeout=5,
    )
    assert result["passed"] is True


def test_historical_audit_keeps_candidate_attribute_error_separate():
    internal = {
        "model": "model-a",
        "outcome": "FAIL",
        "failure_kind": "AttributeError",
        "exception_type": "AttributeError",
    }
    absent = {
        "model": "model-a",
        "outcome": "FAIL",
        "failure_kind": "callable_absent",
        "exception_type": None,
    }
    assert verifier_diagnostic_bucket(internal) == "candidate_AttributeError"
    assert verifier_diagnostic_bucket(absent) == "callable_absent"
    audit = verifier_diagnostic_audit([internal, absent])
    assert audit["total"]["candidate_AttributeError"] == 1
    assert audit["total"]["callable_absent"] == 1
    assert audit["total"]["missing_callable"] == 0
    assert audit["total"]["worker_error"] == 0


def test_frozen_surprisal_normalization_is_bounded():
    assert normalized_surprisal(-1.0, 2.0) == 0.0
    assert normalized_surprisal(0.0, 2.0) == 0.0
    assert normalized_surprisal(1.0, 2.0) == 0.5
    assert normalized_surprisal(3.0, 2.0) == 1.0
