from hashlib import sha256
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "select_cocc_generation_budget.py"
SPEC = importlib.util.spec_from_file_location("select_cocc_generation_budget", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _protocol():
    return {
        "protocol_id": "test",
        "status": "DRAFT_NOT_FROZEN",
        "generation_budget_calibration": {
            "candidate_max_tokens": [512, 1024],
            "partition": "calibration",
            "required_perturbation_type": "clean_control",
            "outcome_labels_allowed": False,
            "minimum_clean_control_sessions": 4,
            "maximum_observed_length_rate": 0.25,
            "selection_rule": "smallest_candidate_meeting_rate_threshold",
        },
    }


def _dataset(path: Path):
    rows = [
        {
            "item_id": f"p{i}:clean_control",
            "problem_id": f"p{i}",
            "split": "calibration",
            "perturbation_type": "clean_control",
        }
        for i in range(4)
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _candidate(path: Path, budget: int, reasons):
    _write_json(
        path / "manifest.json",
        {
            "provider": "test",
            "provider_endpoint": "https://example.invalid/v1",
            "model": "model",
            "generation_parameter_set": {"max_tokens": budget},
            "provider_response_identity": {
                "resolved_models": ["model"],
                "system_fingerprints": ["fp"],
            },
        },
    )
    for index, reason in enumerate(reasons):
        _write_json(
            path / "sessions" / f"s{index}" / "raw.json",
            {
                "session_id": f"s{index}",
                "item_id": f"p{index}:clean_control",
                "perturbation_type": "clean_control",
                "turns": [
                    {
                        "finish_reason": reason,
                        "token_count": budget if reason == "length" else 10,
                    }
                ],
            },
        )


def test_selects_smallest_calibration_budget_meeting_threshold(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    _dataset(dataset)
    run_512 = tmp_path / "run512"
    run_1024 = tmp_path / "run1024"
    _candidate(run_512, 512, ["length", "length", "stop", "stop"])
    _candidate(run_1024, 1024, ["length", "stop", "stop", "stop"])

    report = MODULE.select_budget(
        _protocol(), dataset, [(512, run_512), (1024, run_1024)]
    )

    assert report["status"] == "SELECTED"
    assert report["selected_max_tokens"] == 1024
    assert report["outcome_labels_consumed"] is False
    assert report["holdout_rows_consumed"] is False
    assert report["design_freeze_allowed"] is True


def test_rejects_noncontiguous_candidate_sequence(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    _dataset(dataset)
    run_1024 = tmp_path / "run1024"
    _candidate(run_1024, 1024, ["stop"] * 4)

    with pytest.raises(ValueError, match="contiguous prefix"):
        MODULE.select_budget(_protocol(), dataset, [(1024, run_1024)])


def test_frozen_512_record_hashes_and_draft_guards():
    record = json.loads(
        (ROOT / "config" / "cocc_nemotron3_super_512_confirmatory_record.json")
        .read_text(encoding="utf-8")
    )
    assert record["confirmatory_verdict"] == "honest_null"
    assert record["verdict_is_final"] is True
    assert record["reinterpretation_prohibited"] is True
    for field in (
        "design",
        "run_manifest",
        "confirmatory_report",
        "posthoc_generation_cap_audit",
    ):
        target = ROOT / record[field]["path"]
        if target.exists():
            assert sha256(target.read_bytes()).hexdigest() == record[field].get(
                "raw_sha256", record[field].get("sha256")
            )

    draft = json.loads(
        (ROOT / "config" / "cocc_next_confirmatory_protocol.draft.json").read_text(
            encoding="utf-8"
        )
    )
    assert draft["status"] == "DRAFT_NOT_FROZEN"
    budget = draft["generation_budget_calibration"]
    assert budget["partition"] == "calibration"
    assert budget["required_perturbation_type"] == "clean_control"
    assert budget["outcome_labels_allowed"] is False
    assert budget["holdout_access_allowed"] is False
    assert budget["maximum_observed_length_rate"] == 0.05
    assert draft["primary_early_warning_analysis"][
        "allowed_absolute_window_horizons"
    ] == [1, 2, 4, 8, 16]
    assert draft["mandatory_generation_cap_audit"]["execution_order"] == (
        "before_interpreting_any_accumulative_score"
    )
