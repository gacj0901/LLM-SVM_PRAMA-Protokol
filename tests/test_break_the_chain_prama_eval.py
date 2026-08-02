import json
from hashlib import sha256
from pathlib import Path

import pytest

from scripts.evaluate_break_the_chain_prama import (
    LEGACY_DESIGN,
    _canonical_sha256,
    _cluster_bootstrap_auc_difference,
    _cluster_permutation_result,
    _validate_design,
    _verify_dataset_binding,
    evaluate,
)
from scripts.run_break_the_chain_prama_eval import (
    FORBIDDEN_PROJECTOR_KEYS,
    load_cocc_items,
    parse_args,
    projection_input_from_raw,
    run,
)


def test_projector_view_contains_only_numeric_observations():
    raw = {
        "session_id": "s1",
        "provider": "ollama",
        "model": "hermes3:8b",
        "prompt": "secret task",
        "problem_id": "p1",
        "perturbation_type": "condition-secret",
        "response_time_seconds": 1.2,
        "turns": [
            {
                "turn_index": 0,
                "user_message": "secret task",
                "assistant_message": "secret answer",
                "tokens": [
                    {
                        "token": "secret",
                        "top1_logprob": -0.2,
                        "top_logprobs": [-0.2, -1.2],
                        "gap": 1.0,
                        "entropy": 0.4,
                    }
                ],
            }
        ],
    }
    request = projection_input_from_raw(raw)
    text = json.dumps(request)
    assert request["turns"][0]["tokens"][0] == {
        "top1_logprob": -0.2,
        "top_logprobs": [-0.2, -1.2],
        "gap": 1.0,
        "entropy": 0.4,
    }
    assert not any(f'"{key}"' in text for key in FORBIDDEN_PROJECTOR_KEYS)


def test_dry_run_writes_queued_channels_and_nonobserved_timing(tmp_path):
    fixture = Path(__file__).parent / "fixtures" / "cocc_smoke.jsonl"
    output = tmp_path / "run"
    args = parse_args(
        [
            "--dataset",
            str(fixture),
            "--output-dir",
            str(output),
            "--model",
            "hermes3:8b",
            "--dry-run",
            "--queue-only",
        ]
    )
    manifest = run(args)
    assert manifest["session_count"] == 4
    assert manifest["verified_count"] == 0
    assert manifest["projected_count"] == 0
    artifacts = [
        json.loads(line)
        for line in (output / "generation_observation.jsonl").read_text().splitlines()
    ]
    assert all(row["channel_status"] == "NOT_APPLICABLE" for row in artifacts)
    assert all(row["response_time_seconds"] == 0.0 for row in artifacts)
    request = json.loads(next((output / "projection" / "requests").glob("*.json")).read_text())
    assert "perturbation_type" not in request
    assert "assistant_message" not in json.dumps(request)


def test_resume_reuses_acquisition_without_model_call(tmp_path):
    fixture = Path(__file__).parent / "fixtures" / "cocc_smoke.jsonl"
    output = tmp_path / "run"
    first = parse_args(
        [
            "--dataset",
            str(fixture),
            "--output-dir",
            str(output),
            "--model",
            "hermes3:8b",
            "--dry-run",
            "--queue-only",
        ]
    )
    run(first)
    resumed = parse_args(
        [
            "--dataset",
            str(fixture),
            "--output-dir",
            str(output),
            "--model",
            "hermes3:8b",
            "--dry-run",
            "--queue-only",
            "--resume",
        ]
    )
    manifest = run(resumed)
    assert manifest["session_count"] == 4


def test_dataset_rejects_problem_split_leakage(tmp_path):
    path = tmp_path / "bad.jsonl"
    base = {
        "benchmark_name": "chain_of_code_collapse",
        "benchmark_alias": "break_the_chain_code_generation",
        "problem_id": "same",
        "perturbation_type": "x",
        "perturbed_prompt": "task",
        "verifier_ref": {"tests": ["x"]},
    }
    path.write_text(
        json.dumps({**base, "item_id": "a", "split": "train"})
        + "\n"
        + json.dumps({**base, "item_id": "b", "split": "test"})
        + "\n"
    )
    with pytest.raises(ValueError, match="crosses train/test"):
        load_cocc_items(path)


def test_json_command_form_supports_nested_command_options(tmp_path):
    args = parse_args(
        [
            "--dataset",
            str(Path(__file__).parent / "fixtures" / "cocc_smoke.jsonl"),
            "--output-dir",
            str(tmp_path / "run"),
            "--queue-only",
            "--verifier-command-json",
            '["python","verifier.py","--dataset","data.jsonl"]',
        ]
    )
    assert args.verifier_command == [
        "python",
        "verifier.py",
        "--dataset",
        "data.jsonl",
    ]


def test_blind_evaluator_separates_prama_score_from_instantaneous_baseline(tmp_path):
    join_rows = []
    for index in range(8):
        split = "train" if index < 4 else "test"
        label = index % 2
        trajectory_path = tmp_path / f"trajectory-{index}.jsonl"
        trajectory_path.write_text(
            json.dumps(
                {
                    "delta": 0.25,
                    "xi": 0.8 if label else 0.1,
                    "balance": -0.8 if label else 0.1,
                    "theta": 0.2,
                    "accumulated_excess": 1.0 if label else 0.0,
                    "capacity": 0.5 if label else 1.0,
                }
            )
            + "\n"
        )
        request_path = tmp_path / f"request-{index}.json"
        request_path.write_text(
            json.dumps(
                {
                    "turns": [
                        {
                            "tokens": [
                                {
                                    "top1_logprob": -0.3,
                                    "entropy": 0.5,
                                    "gap": 1.0,
                                }
                            ]
                        }
                    ]
                }
            )
        )
        join_rows.append(
            {
                "session_id": f"s{index}",
                "label": label,
                "split": split,
                "group_sha256": f"g{index}",
                "event_token": 1,
                "verification_result_path": str(tmp_path / f"verify-{index}.json"),
                "trajectory_path": str(trajectory_path),
                "projection_request_path": str(request_path),
            }
        )
    join_path = tmp_path / "blind_join.csv"
    import csv

    with join_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(join_rows[0]))
        writer.writeheader()
        writer.writerows(join_rows)
    report = evaluate(join_path, "max_negative_balance", permutations=19, seed=7)
    assert report["metrics"]["max_negative_balance"]["auroc"] == 1.0
    assert report["metrics"]["max_negative_balance"]["permutation_p"] == pytest.approx(
        1 / 6
    )
    assert (
        report["metrics"]["max_negative_balance"]["permutation_test"]["method"]
        == "exact_enumeration"
    )
    assert report["metrics"]["max_delta"]["auroc"] == 0.5
    assert report["verdict"] == "honest_null"
    comparison = report["mandatory_coordinate_comparison"]
    assert comparison["metrics"]["max_delta"]["auroc"] == 0.5
    assert comparison["metrics"]["max_xi"]["auroc"] == 1.0
    assert comparison["metrics"]["max_negative_balance"]["auroc"] == 1.0
    assert report["temporal_censoring"]["absolute_window_horizons"]["1"][
        "at_risk_n"
    ] == 4
    activation = report["activation_audit"]["test"]
    assert activation["mechanisms_activated"] is True
    assert activation["accumulated_excess_activation_session_count"] == 2
    assert activation["capacity_degradation_session_count"] == 0
    assert len(report["session_horizon_table"]) == 4
    assert (
        report["temporal_censoring"]["absolute_horizon_semantics"].startswith(
            "prefix windows retained"
        )
    )


def test_frozen_class_quota_fails_closed_before_confirmatory_verdict(tmp_path):
    join_rows = []
    for index in range(8):
        split = "train" if index < 4 else "test"
        label = index % 2
        trajectory_path = tmp_path / f"trajectory-{index}.jsonl"
        trajectory_path.write_text(
            json.dumps(
                {
                    "delta": 0.5 if label else 0.1,
                    "xi": 0.4 if label else 0.05,
                    "balance": -0.2 if label else 0.15,
                    "theta": 0.2,
                    "accumulated_excess": 0.0,
                    "capacity": 1.0,
                }
            )
            + "\n"
        )
        request_path = tmp_path / f"request-{index}.json"
        request_path.write_text(
            json.dumps(
                {
                    "turns": [
                        {
                            "tokens": [
                                {
                                    "top1_logprob": -0.3,
                                    "entropy": 0.5,
                                    "gap": 1.0,
                                }
                            ]
                        }
                    ]
                }
            )
        )
        join_rows.append(
            {
                "session_id": f"s{index}",
                "label": label,
                "split": split,
                "group_sha256": f"g{index}",
                "event_token": 1,
                "verification_result_path": str(tmp_path / f"verify-{index}.json"),
                "trajectory_path": str(trajectory_path),
                "projection_request_path": str(request_path),
            }
        )
    join_path = tmp_path / "blind_join.csv"
    import csv

    with join_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(join_rows[0]))
        writer.writeheader()
        writer.writerows(join_rows)
    design = json.loads(json.dumps(LEGACY_DESIGN))
    design["design_id"] = "quota-test"
    design["min_test_fail"] = 3
    design["min_test_pass"] = 3
    report = evaluate(
        join_path,
        "max_negative_balance",
        permutations=19,
        seed=7,
        design=design,
    )
    assert report["class_support"]["observed_fail"] == 2
    assert report["class_support"]["observed_pass"] == 2
    assert report["class_support"]["valid_for_confirmatory_test"] is False
    assert report["verdict"] == "inconclusive_insufficient_class_support"


def test_frozen_dataset_binding_requires_exact_blind_join_membership(tmp_path):
    dataset_path = tmp_path / "dataset.jsonl"
    rows = [
        {"problem_id": "p1", "item_id": "p1:clean", "split": "calibration"},
        {"problem_id": "p2", "item_id": "p2:clean", "split": "test"},
    ]
    dataset_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    design = json.loads(json.dumps(LEGACY_DESIGN))
    design["sampling_plan"] = {
        "normalized_dataset_sha256": sha256(dataset_path.read_bytes()).hexdigest()
    }
    join_rows = []
    for index, row in enumerate(rows):
        item_id = row["item_id"]
        session_id = f"cocc-{index:05d}-{sha256(item_id.encode()).hexdigest()[:10]}"
        join_rows.append(
            {
                "session_id": session_id,
                "split": "train" if row["split"] == "calibration" else "test",
                "group_sha256": sha256(row["problem_id"].encode()).hexdigest(),
            }
        )
    binding = _verify_dataset_binding(design, dataset_path, join_rows)
    assert binding["exact_session_membership"] is True
    with pytest.raises(ValueError, match="complete frozen dataset"):
        _verify_dataset_binding(design, dataset_path, join_rows[:1])


def _small_cluster_design():
    design = json.loads(json.dumps(LEGACY_DESIGN))
    design.update(
        {
            "schema": "LLM-SVM-CoCC-confirmatory-design/3",
            "design_id": "small-cluster-test",
            "min_test_problem_clusters": 4,
            "min_problem_clusters_with_fail": 2,
            "min_problem_clusters_with_pass": 2,
            "independent_unit": {
                "unit": "problem_id",
                "cluster_id_field": "problem_id",
                "within_cluster_condition_field": "perturbation_type",
                "expected_sessions_per_cluster": 2,
                "required_conditions": [
                    "clean_control",
                    "negation_objective",
                ],
            },
            "cluster_inference": {
                "method": "problem_cluster_outcome_vector_permutation",
                "monte_carlo": {
                    "seed": 101,
                    "draws": 199,
                    "rng_algorithm": "python_3_12_random_mt19937_shuffle",
                    "plus_one_correction": True,
                    "exact_fallback_max_assignments": 1000,
                    "invalid_cluster_fallback": (
                        "fail_closed_without_session_level_fallback"
                    ),
                },
            },
            "incremental_auc_test": {
                "prama_score": "max_negative_balance",
                "baseline_score": "max_delta",
                "alpha": 0.01,
                "minimum_effect_of_interest": 0.05,
                "monte_carlo": {
                    "seed": 102,
                    "draws": 199,
                    "rng_algorithm": "python_3_12_random_mt19937_randrange",
                    "plus_one_correction": True,
                    "maximum_attempts": 1990,
                    "invalid_draw_fallback": (
                        "discard_and_redraw_until_maximum_attempts_then_fail_closed"
                    ),
                },
            },
        }
    )
    return _validate_design(design)


def _small_cluster_rows():
    profiles = ((0, 1), (0, 1), (1, 1), (0, 0))
    rows = []
    for cluster_index, profile in enumerate(profiles):
        for condition_index, condition in enumerate(
            ("clean_control", "negation_objective")
        ):
            label = profile[condition_index]
            rows.append(
                {
                    "problem_id": f"p{cluster_index}",
                    "perturbation_type": condition,
                    "label": label,
                    "features": {
                        "max_negative_balance": (
                            0.8 + cluster_index * 0.01
                            if label
                            else 0.1 + cluster_index * 0.01
                        ),
                        "max_delta": 0.1 + condition_index * 0.01,
                    },
                }
            )
    return rows


def test_cluster_inference_uses_problem_as_independent_unit_and_is_deterministic():
    design = _small_cluster_design()
    rows = _small_cluster_rows()
    first = _cluster_permutation_result(
        rows, "max_negative_balance", design
    )
    second = _cluster_permutation_result(
        rows, "max_negative_balance", design
    )
    assert first == second
    assert first["independent_unit"] == "problem_id"
    assert first["cluster_count"] == 4
    assert first["method"] == "exact_cluster_outcome_vector_enumeration"
    incremental_first = _cluster_bootstrap_auc_difference(rows, design)
    incremental_second = _cluster_bootstrap_auc_difference(rows, design)
    assert incremental_first == incremental_second
    assert incremental_first["statistic"] == (
        "AUROC_PRAMA_minus_AUROC_delta"
    )
    assert incremental_first["minimum_effect_of_interest"] == 0.05
    assert incremental_first["monte_carlo"]["seed"] == 102
    assert incremental_first["monte_carlo"]["draws"] == 199


def test_cluster_inference_fails_closed_on_missing_condition():
    design = _small_cluster_design()
    rows = _small_cluster_rows()[:-1]
    with pytest.raises(ValueError, match="frozen session count"):
        _cluster_permutation_result(rows, "max_negative_balance", design)


def test_v3_design_freeze_matches_canonical_and_raw_hashes():
    root = Path(__file__).resolve().parents[1]
    design_path = root / "config" / "cocc_confirmatory_design_v3.json"
    freeze_path = (
        root / "config" / "cocc_confirmatory_design_v3.freeze.json"
    )
    design = json.loads(design_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    assert _canonical_sha256(design) == freeze["canonical_design_sha256"]
    assert sha256(design_path.read_bytes()).hexdigest() == freeze[
        "raw_file_sha256"
    ]
    assert freeze["status"] == "FROZEN_BEFORE_CONFIRMATORY_ACQUISITION"
