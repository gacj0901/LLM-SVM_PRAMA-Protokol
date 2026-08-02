#!/usr/bin/env python
"""Blindly evaluate PRAMA state against external CoCC PASS/FAIL outcomes."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from hashlib import sha256
from itertools import combinations
import json
import math
from pathlib import Path
import random
from statistics import mean, median
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for candidate in (REPO_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from aptadynamic_llm.evaluation.state_discrimination_metrics import (  # noqa: E402
    auroc,
    permutation_pvalue,
)


SCORES = (
    "max_negative_balance",
    "max_xi",
    "final_accumulated_excess",
    "capacity_loss",
)
BASELINES = ("max_delta", "mean_surprisal", "mean_entropy", "negative_mean_gap")
MANDATORY_COORDINATES = ("max_delta", "max_xi", "max_negative_balance")
LEGACY_DESIGN = {
    "schema": "LLM-SVM-CoCC-confirmatory-design/2",
    "design_id": "legacy-minimum-validity",
    "primary_score": "max_negative_balance",
    "p_threshold": 0.01,
    "min_test_fail": 1,
    "min_test_pass": 1,
    "mandatory_coordinate_comparison": list(MANDATORY_COORDINATES),
    "temporal_censoring": {
        "absolute_window_horizons": [1, 2, 4, 8],
        "relative_prefix_fractions": [0.25, 0.5, 0.75, 1.0],
        "short_session_policy": "at_risk_only",
    },
    "activation_tolerance": 1e-12,
    "exact_permutation_max_assignments": 100000,
}


def _canonical_sha256(value: Any) -> str:
    return sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _average_precision(scores: Sequence[float], labels: Sequence[int]) -> float:
    positives = sum(labels)
    if positives == 0:
        raise ValueError("average precision requires a positive label")
    ranked = sorted(zip(scores, labels), key=lambda value: value[0], reverse=True)
    hits = 0
    total = 0.0
    for rank, (_, label) in enumerate(ranked, 1):
        if label:
            hits += 1
            total += hits / rank
    return total / positives


def _permutation_result(
    scores: Sequence[float],
    labels: Sequence[int],
    permutations: int,
    seed: int,
    exact_max_assignments: int,
) -> dict[str, Any]:
    observed = auroc(scores, labels)
    n = len(labels)
    positives = sum(labels)
    unique_assignments = math.comb(n, positives)
    if unique_assignments <= exact_max_assignments:
        exceedances = 0
        for positive_indices in combinations(range(n), positives):
            permuted = [0] * n
            for index in positive_indices:
                permuted[index] = 1
            exceedances += auroc(scores, permuted) >= observed
        return {
            "p": exceedances / unique_assignments,
            "method": "exact_enumeration",
            "unique_label_assignments": unique_assignments,
            "evaluated_assignments": unique_assignments,
            "plus_one_correction": False,
        }
    return {
        "p": permutation_pvalue(scores, labels, permutations, seed),
        "method": "monte_carlo_plus_one",
        "unique_label_assignments": unique_assignments,
        "evaluated_assignments": permutations,
        "plus_one_correction": True,
        "seed": seed,
    }


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        for offset in range(start, end):
            ranks[order[offset]] = average_rank
        start = end
    return ranks


def _auroc_from_ranks(ranks: Sequence[float], labels: Sequence[int]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0:
        raise ValueError("at least one positive label is required")
    if negatives == 0:
        raise ValueError("at least one negative label is required; all labels were positive")
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def _cluster_rows(
    rows: list[dict[str, Any]], design: dict[str, Any]
) -> tuple[list[str], list[str], dict[str, list[dict[str, Any]]]]:
    unit = design["independent_unit"]
    cluster_field = str(unit["cluster_id_field"])
    condition_field = str(unit["within_cluster_condition_field"])
    expected_conditions = [str(value) for value in unit["required_conditions"]]
    clusters: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        cluster_id = row.get(cluster_field)
        if cluster_id is None or str(cluster_id).strip() == "":
            raise ValueError(
                f"cluster-aware inference requires nonempty {cluster_field}"
            )
        clusters.setdefault(str(cluster_id), []).append(row)
    expected_set = set(expected_conditions)
    for cluster_id, members in clusters.items():
        conditions = [str(member.get(condition_field) or "") for member in members]
        if len(members) != int(unit["expected_sessions_per_cluster"]):
            raise ValueError(
                f"cluster {cluster_id} does not have the frozen session count"
            )
        if len(conditions) != len(set(conditions)):
            raise ValueError(f"cluster {cluster_id} repeats a condition")
        if set(conditions) != expected_set:
            raise ValueError(
                f"cluster {cluster_id} does not contain the frozen conditions"
            )
    cluster_ids = sorted(clusters)
    ordered = {
        cluster_id: sorted(
            clusters[cluster_id],
            key=lambda row: expected_conditions.index(
                str(row.get(condition_field) or "")
            ),
        )
        for cluster_id in cluster_ids
    }
    return cluster_ids, expected_conditions, ordered


def _multiset_permutation_count(values: Sequence[tuple[int, ...]]) -> int:
    counts = Counter(values)
    result = math.factorial(len(values))
    for count in counts.values():
        result //= math.factorial(count)
    return result


def _unique_multiset_permutations(
    values: Sequence[tuple[int, ...]],
) -> Sequence[tuple[tuple[int, ...], ...]]:
    counts = Counter(values)
    ordered_values = sorted(counts)
    length = len(values)

    def generate(prefix: list[tuple[int, ...]]):
        if len(prefix) == length:
            yield tuple(prefix)
            return
        for value in ordered_values:
            if counts[value] == 0:
                continue
            counts[value] -= 1
            prefix.append(value)
            yield from generate(prefix)
            prefix.pop()
            counts[value] += 1

    return generate([])


def _cluster_permutation_result(
    rows: list[dict[str, Any]],
    field: str,
    design: dict[str, Any],
) -> dict[str, Any]:
    plan = design["cluster_inference"]
    monte_carlo = plan["monte_carlo"]
    cluster_ids, conditions, clusters = _cluster_rows(rows, design)
    ordered_rows = [
        member for cluster_id in cluster_ids for member in clusters[cluster_id]
    ]
    scores = [float(row["features"][field]) for row in ordered_rows]
    labels = [int(row["label"]) for row in ordered_rows]
    ranks = _average_ranks(scores)
    observed = _auroc_from_ranks(ranks, labels)
    profiles = [
        tuple(int(member["label"]) for member in clusters[cluster_id])
        for cluster_id in cluster_ids
    ]
    unique_assignments = _multiset_permutation_count(profiles)
    exact_limit = int(monte_carlo["exact_fallback_max_assignments"])
    if unique_assignments <= exact_limit:
        assignments = _unique_multiset_permutations(profiles)
        method = "exact_cluster_outcome_vector_enumeration"
        evaluated = unique_assignments
        seed = None
    else:
        draws = int(monte_carlo["draws"])
        rng = random.Random(int(monte_carlo["seed"]))

        def shuffled_assignments():
            for _ in range(draws):
                assignment = list(profiles)
                rng.shuffle(assignment)
                yield assignment

        assignments = shuffled_assignments()
        method = "cluster_outcome_vector_monte_carlo_plus_one"
        evaluated = draws
        seed = int(monte_carlo["seed"])
    exceedances = 0
    for assignment in assignments:
        permuted_labels = [
            label for profile in assignment for label in profile
        ]
        exceedances += _auroc_from_ranks(ranks, permuted_labels) >= observed
    plus_one = method.endswith("plus_one")
    p_value = (
        (exceedances + 1) / (evaluated + 1)
        if plus_one
        else exceedances / evaluated
    )
    return {
        "p": p_value,
        "method": method,
        "independent_unit": "problem_id",
        "cluster_count": len(cluster_ids),
        "within_cluster_conditions": conditions,
        "unique_cluster_profile_assignments": unique_assignments,
        "evaluated_assignments": evaluated,
        "plus_one_correction": plus_one,
        "seed": seed,
        "draws": evaluated if plus_one else None,
        "invalid_cluster_fallback": monte_carlo["invalid_cluster_fallback"],
    }


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile requires values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _cluster_bootstrap_auc_difference(
    rows: list[dict[str, Any]], design: dict[str, Any]
) -> dict[str, Any]:
    test_plan = design["incremental_auc_test"]
    cluster_ids, conditions, clusters = _cluster_rows(rows, design)
    primary = str(test_plan["prama_score"])
    baseline = str(test_plan["baseline_score"])
    observed_primary = auroc(
        [row["features"][primary] for row in rows],
        [row["label"] for row in rows],
    )
    observed_baseline = auroc(
        [row["features"][baseline] for row in rows],
        [row["label"] for row in rows],
    )
    observed_difference = observed_primary - observed_baseline
    draws = int(test_plan["monte_carlo"]["draws"])
    rng = random.Random(int(test_plan["monte_carlo"]["seed"]))
    maximum_attempts = int(test_plan["monte_carlo"]["maximum_attempts"])
    bootstrap_differences: list[float] = []
    attempts = 0
    while len(bootstrap_differences) < draws and attempts < maximum_attempts:
        attempts += 1
        sampled_cluster_ids = [
            cluster_ids[rng.randrange(len(cluster_ids))]
            for _ in cluster_ids
        ]
        sampled_rows = [
            member
            for cluster_id in sampled_cluster_ids
            for member in clusters[cluster_id]
        ]
        labels = [row["label"] for row in sampled_rows]
        if not any(labels) or all(labels):
            continue
        primary_auc = auroc(
            [row["features"][primary] for row in sampled_rows], labels
        )
        baseline_auc = auroc(
            [row["features"][baseline] for row in sampled_rows], labels
        )
        bootstrap_differences.append(primary_auc - baseline_auc)
    if len(bootstrap_differences) != draws:
        raise ValueError(
            "cluster bootstrap exhausted the frozen maximum_attempts; fail closed"
        )
    alpha = float(test_plan["alpha"])
    centered_exceedances = sum(
        difference - observed_difference >= observed_difference
        for difference in bootstrap_differences
    )
    p_value = (
        1.0
        if observed_difference <= 0.0
        else (centered_exceedances + 1) / (draws + 1)
    )
    lower_bound = _quantile(bootstrap_differences, alpha)
    minimum_effect = float(test_plan["minimum_effect_of_interest"])
    passes = (
        observed_difference >= minimum_effect
        and p_value < alpha
        and lower_bound > 0.0
    )
    return {
        "statistic": "AUROC_PRAMA_minus_AUROC_delta",
        "prama_score": primary,
        "baseline_score": baseline,
        "null_hypothesis": "AUROC_PRAMA - AUROC_delta <= 0",
        "alternative_hypothesis": "AUROC_PRAMA - AUROC_delta > 0",
        "observed_prama_auroc": observed_primary,
        "observed_delta_auroc": observed_baseline,
        "observed_difference": observed_difference,
        "minimum_effect_of_interest": minimum_effect,
        "method": "paired_problem_cluster_bootstrap_centered_one_sided",
        "independent_unit": "problem_id",
        "cluster_count": len(cluster_ids),
        "within_cluster_conditions": conditions,
        "alpha": alpha,
        "p": p_value,
        "one_sided_percentile_lower_confidence_bound": lower_bound,
        "confidence_level": 1.0 - alpha,
        "monte_carlo": {
            "seed": int(test_plan["monte_carlo"]["seed"]),
            "draws": draws,
            "attempts": attempts,
            "maximum_attempts": maximum_attempts,
            "invalid_draw_fallback": test_plan["monte_carlo"][
                "invalid_draw_fallback"
            ],
            "plus_one_correction": True,
        },
        "passes_formal_incremental_rule": passes,
    }


def _numeric_tokens(path: Path) -> list[dict[str, Any]]:
    request = json.loads(path.read_text(encoding="utf-8"))
    return [
        token
        for turn in request.get("turns") or []
        for token in turn.get("tokens") or []
    ]


def _trajectory(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"empty trajectory: {path}")
    return rows


def _coordinate_features(trajectory: list[dict[str, Any]]) -> dict[str, float]:
    balances = [
        float(row["balance"]) for row in trajectory if row.get("balance") is not None
    ]
    xis = [float(row["xi"]) for row in trajectory if row.get("xi") is not None]
    excess = [
        float(row["accumulated_excess"])
        for row in trajectory
        if row.get("accumulated_excess") is not None
    ]
    capacities = [
        float(row["capacity"]) for row in trajectory if row.get("capacity") is not None
    ]
    deltas = [
        float(row["delta"]) for row in trajectory if row.get("delta") is not None
    ]
    if not balances or not xis or not excess or not capacities or not deltas:
        raise ValueError("incomplete trajectory")
    return {
        "max_negative_balance": max(-value for value in balances),
        "max_xi": max(xis),
        "final_accumulated_excess": excess[-1],
        "capacity_loss": capacities[0] - capacities[-1],
        "max_delta": max(deltas),
    }


def _features(
    trajectory: list[dict[str, Any]], tokens: list[dict[str, Any]]
) -> dict[str, float]:
    if not tokens:
        raise ValueError("numeric observation contains no tokens")
    surprisals = [-float(token["top1_logprob"]) for token in tokens]
    entropies = [float(token.get("entropy") or 0.0) for token in tokens]
    gaps = [float(token.get("gap") or 0.0) for token in tokens]
    return {
        **_coordinate_features(trajectory),
        "mean_surprisal": sum(surprisals) / len(surprisals),
        "mean_entropy": sum(entropies) / len(entropies),
        "negative_mean_gap": -sum(gaps) / len(gaps),
    }


def _metric(
    rows: list[dict[str, Any]],
    field: str,
    permutations: int,
    seed: int,
    exact_max_assignments: int,
    design: dict[str, Any] | None = None,
    formal_inference: bool = True,
) -> dict[str, Any]:
    scores = [row["features"][field] for row in rows]
    labels = [row["label"] for row in rows]
    try:
        result = {
            "auroc": auroc(scores, labels),
            "average_precision": _average_precision(scores, labels),
        }
        if design and design.get("cluster_inference"):
            if formal_inference and field == design["primary_score"]:
                permutation = _cluster_permutation_result(rows, field, design)
                result.update(
                    {
                        "permutation_p": permutation["p"],
                        "permutation_test": permutation,
                        "inference_status": "confirmatory_cluster_aware",
                    }
                )
            else:
                result.update(
                    {
                        "permutation_p": None,
                        "permutation_test": None,
                        "inference_status": (
                            "descriptive_only; no session-level inferential "
                            "fallback is permitted"
                        ),
                    }
                )
        else:
            permutation = _permutation_result(
                scores, labels, permutations, seed, exact_max_assignments
            )
            result.update(
                {
                    "permutation_p": permutation["p"],
                    "permutation_test": permutation,
                }
            )
        return result
    except ValueError as exc:
        return {
            "auroc": None,
            "average_precision": None,
            "permutation_p": None,
            "error": str(exc),
        }


def _validate_design(design: dict[str, Any]) -> dict[str, Any]:
    required = {
        "schema",
        "design_id",
        "primary_score",
        "p_threshold",
        "min_test_fail",
        "min_test_pass",
        "mandatory_coordinate_comparison",
        "temporal_censoring",
        "activation_tolerance",
        "exact_permutation_max_assignments",
    }
    missing = sorted(required - design.keys())
    if missing:
        raise ValueError(f"confirmatory design missing fields: {missing}")
    if design["schema"] not in {
        "LLM-SVM-CoCC-confirmatory-design/2",
        "LLM-SVM-CoCC-confirmatory-design/3",
    }:
        raise ValueError("unexpected confirmatory design schema")
    if design["primary_score"] != "max_negative_balance":
        raise ValueError("v2 primary_score must remain max_negative_balance")
    if float(design["p_threshold"]) != 0.01:
        raise ValueError("v2 p_threshold must remain exactly 0.01")
    if tuple(design["mandatory_coordinate_comparison"]) != MANDATORY_COORDINATES:
        raise ValueError("mandatory comparison must be Δ vs Ξ vs balance")
    for field in ("min_test_fail", "min_test_pass"):
        if int(design[field]) <= 0:
            raise ValueError(f"{field} must be positive")
    temporal = design["temporal_censoring"]
    horizons = [int(value) for value in temporal["absolute_window_horizons"]]
    fractions = [float(value) for value in temporal["relative_prefix_fractions"]]
    if not horizons or horizons != sorted(set(horizons)) or horizons[0] <= 0:
        raise ValueError("absolute horizons must be unique increasing positive integers")
    if (
        not fractions
        or fractions != sorted(set(fractions))
        or fractions[-1] != 1.0
        or any(not 0.0 < value <= 1.0 for value in fractions)
    ):
        raise ValueError("relative fractions must be unique, increasing and end at 1.0")
    if temporal.get("short_session_policy") not in {
        "at_risk_only",
        "cluster_complete_at_risk",
    }:
        raise ValueError("unexpected short_session_policy")
    primary_horizon = temporal.get("primary_absolute_window_horizon")
    if primary_horizon is not None:
        primary_horizon = int(primary_horizon)
        if primary_horizon not in horizons:
            raise ValueError("primary absolute horizon must be in frozen horizons")
        if temporal.get("short_session_policy") != "cluster_complete_at_risk":
            raise ValueError(
                "cluster-aware primary horizon requires cluster_complete_at_risk"
            )
        if int(temporal.get("window_size_tokens") or 0) <= 0:
            raise ValueError("primary horizon requires positive window_size_tokens")
    if float(design["activation_tolerance"]) < 0:
        raise ValueError("activation_tolerance cannot be negative")
    if int(design["exact_permutation_max_assignments"]) <= 0:
        raise ValueError("exact_permutation_max_assignments must be positive")
    cluster_fields = {
        "independent_unit",
        "cluster_inference",
        "incremental_auc_test",
    }
    present_cluster_fields = cluster_fields.intersection(design)
    if present_cluster_fields and present_cluster_fields != cluster_fields:
        raise ValueError(
            "cluster-aware designs require independent_unit, cluster_inference "
            "and incremental_auc_test together"
        )
    if present_cluster_fields:
        if design["schema"] != "LLM-SVM-CoCC-confirmatory-design/3":
            raise ValueError("cluster-aware confirmatory designs require schema /3")
        unit = design["independent_unit"]
        if unit.get("unit") != "problem_id" or unit.get(
            "cluster_id_field"
        ) != "problem_id":
            raise ValueError("problem_id must be the frozen independent unit")
        if unit.get("within_cluster_condition_field") != "perturbation_type":
            raise ValueError(
                "within-cluster condition field must be perturbation_type"
            )
        if int(unit.get("expected_sessions_per_cluster") or 0) != 2:
            raise ValueError("the frozen design requires two sessions per problem")
        if list(unit.get("required_conditions") or []) != [
            "clean_control",
            "negation_objective",
        ]:
            raise ValueError("unexpected frozen within-problem conditions")
        for field in (
            "min_test_problem_clusters",
            "min_problem_clusters_with_fail",
            "min_problem_clusters_with_pass",
        ):
            if int(design.get(field) or 0) <= 0:
                raise ValueError(f"{field} must be positive")
        cluster_plan = design["cluster_inference"]
        if (
            cluster_plan.get("method")
            != "problem_cluster_outcome_vector_permutation"
        ):
            raise ValueError("unexpected cluster inference method")
        monte_carlo = cluster_plan["monte_carlo"]
        if int(monte_carlo.get("draws") or 0) <= 0:
            raise ValueError("cluster permutation Monte Carlo draws must be positive")
        if int(monte_carlo.get("exact_fallback_max_assignments") or 0) <= 0:
            raise ValueError("cluster exact fallback threshold must be positive")
        if (
            monte_carlo.get("rng_algorithm")
            != "python_3_12_random_mt19937_shuffle"
            or monte_carlo.get("plus_one_correction") is not True
        ):
            raise ValueError("unexpected frozen cluster permutation RNG")
        if (
            monte_carlo.get("invalid_cluster_fallback")
            != "fail_closed_without_session_level_fallback"
        ):
            raise ValueError("cluster inference must fail closed")
        incremental = design["incremental_auc_test"]
        if incremental.get("prama_score") != design["primary_score"]:
            raise ValueError("incremental test must use the frozen primary score")
        if incremental.get("baseline_score") != "max_delta":
            raise ValueError("incremental test baseline must be max_delta")
        if float(incremental.get("alpha")) != float(design["p_threshold"]):
            raise ValueError("incremental test alpha must equal p_threshold")
        minimum_effect = float(incremental.get("minimum_effect_of_interest"))
        if not 0.0 < minimum_effect < 1.0:
            raise ValueError("minimum effect of interest must be between 0 and 1")
        bootstrap = incremental["monte_carlo"]
        draws = int(bootstrap.get("draws") or 0)
        if draws <= 0:
            raise ValueError("cluster bootstrap draws must be positive")
        if int(bootstrap.get("maximum_attempts") or 0) < draws:
            raise ValueError("cluster bootstrap maximum_attempts must cover draws")
        if (
            bootstrap.get("rng_algorithm")
            != "python_3_12_random_mt19937_randrange"
            or bootstrap.get("plus_one_correction") is not True
        ):
            raise ValueError("unexpected frozen cluster bootstrap RNG")
        if (
            bootstrap.get("invalid_draw_fallback")
            != "discard_and_redraw_until_maximum_attempts_then_fail_closed"
        ):
            raise ValueError("cluster bootstrap must fail closed")
    return design


def _dataset_index(dataset_path: Path) -> dict[str, dict[str, str]]:
    expected_rows: dict[str, dict[str, str]] = {}
    with dataset_path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            item_id = str(row.get("item_id") or row.get("problem_id") or f"item-{index}")
            problem_id = str(row.get("problem_id") or item_id)
            raw_split = str(row.get("split") or "").strip().lower()
            if raw_split not in {"calibration", "train", "test"}:
                raise ValueError("frozen dataset contains an invalid split")
            split = "train" if raw_split in {"calibration", "train"} else "test"
            session_id = (
                f"cocc-{index:05d}-{sha256(item_id.encode()).hexdigest()[:10]}"
            )
            if session_id in expected_rows:
                raise ValueError("frozen dataset produces duplicate session identifiers")
            expected_rows[session_id] = {
                "split": split,
                "group_sha256": sha256(problem_id.encode()).hexdigest(),
                "problem_id": problem_id,
                "item_id": item_id,
                "perturbation_type": str(row.get("perturbation_type") or ""),
                "difficulty": str(row.get("difficulty") or ""),
            }
    return expected_rows


def _verify_dataset_binding(
    design: dict[str, Any],
    dataset_path: Path | None,
    join_rows: list[dict[str, str]],
    expected_rows: dict[str, dict[str, str]] | None = None,
    dataset_manifest_path: Path | None = None,
    acquisition_scope: str = "full_dataset",
) -> dict[str, Any] | None:
    plan = design.get("sampling_plan")
    if plan is None:
        return None
    expected = str(plan.get("normalized_dataset_sha256") or "").lower()
    if len(expected) != 64:
        raise ValueError("sampling_plan.normalized_dataset_sha256 is required")
    if dataset_path is None:
        raise ValueError(
            "the frozen design binds a normalized dataset; --dataset is required"
        )
    observed = _file_sha256(dataset_path)
    if observed != expected:
        raise ValueError(
            "normalized dataset SHA-256 differs from the frozen sampling plan"
        )
    frozen_manifest_sha256 = str(
        plan.get("normalized_manifest_sha256") or ""
    ).lower()
    manifest_binding = None
    if frozen_manifest_sha256:
        if dataset_manifest_path is None:
            raise ValueError(
                "the frozen design binds a normalization manifest; "
                "--dataset-manifest is required"
            )
        observed_manifest_sha256 = _file_sha256(dataset_manifest_path)
        if observed_manifest_sha256 != frozen_manifest_sha256:
            raise ValueError(
                "normalization manifest SHA-256 differs from the frozen "
                "sampling plan"
            )
        manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
        if str(manifest.get("output_sha256") or "").lower() != observed:
            raise ValueError("normalization manifest does not bind the dataset")
        manifest_binding = {
            "path": str(dataset_manifest_path),
            "expected_sha256": frozen_manifest_sha256,
            "observed_sha256": observed_manifest_sha256,
            "dataset_sha256_matches_manifest": True,
        }
    expected_rows = expected_rows or _dataset_index(dataset_path)
    if acquisition_scope not in {"full_dataset", "holdout_only"}:
        raise ValueError("unsupported acquisition_scope in run manifest")
    required_rows = (
        {
            session_id: row
            for session_id, row in expected_rows.items()
            if row["split"] == "test"
        }
        if acquisition_scope == "holdout_only"
        else expected_rows
    )
    observed_rows = {row["session_id"]: row for row in join_rows}
    if len(observed_rows) != len(join_rows):
        raise ValueError("blind join contains duplicate session identifiers")
    if set(observed_rows) != set(required_rows):
        missing = len(set(required_rows) - set(observed_rows))
        unexpected = len(set(observed_rows) - set(required_rows))
        raise ValueError(
            "blind join is not the complete frozen dataset "
            f"(missing={missing}, unexpected={unexpected})"
        )
    for session_id, expected_row in required_rows.items():
        observed_row = observed_rows[session_id]
        if (
            observed_row["split"].strip().lower() != expected_row["split"]
            or observed_row["group_sha256"] != expected_row["group_sha256"]
        ):
            raise ValueError(
                f"blind join metadata differs from frozen dataset for {session_id}"
            )
    train_ids = sorted(
        session_id
        for session_id, row in expected_rows.items()
        if row["split"] == "train"
    )
    test_ids = sorted(
        session_id
        for session_id, row in expected_rows.items()
        if row["split"] == "test"
    )
    train_ids_sha256 = _canonical_sha256(train_ids)
    test_ids_sha256 = _canonical_sha256(test_ids)
    for field, observed_value in (
        ("train_ids_sha256", train_ids_sha256),
        ("test_ids_sha256", test_ids_sha256),
    ):
        frozen_value = str(plan.get(field) or "").lower()
        if frozen_value and frozen_value != observed_value:
            raise ValueError(f"{field} differs from the frozen sampling plan")
    return {
        "path": str(dataset_path),
        "expected_sha256": expected,
        "observed_sha256": observed,
        "matches_frozen_design": True,
        "acquisition_scope": acquisition_scope,
        "expected_session_count": len(required_rows),
        "joined_session_count": len(observed_rows),
        "exact_session_membership": True,
        "train_session_count": len(train_ids),
        "test_session_count": len(test_ids),
        "train_ids_sha256": train_ids_sha256,
        "test_ids_sha256": test_ids_sha256,
        "normalization_manifest": manifest_binding,
    }


def _class_support(test: list[dict[str, Any]], design: dict[str, Any]) -> dict[str, Any]:
    fail = sum(row["label"] for row in test)
    passed = len(test) - fail
    min_fail = int(design["min_test_fail"])
    min_pass = int(design["min_test_pass"])
    label_assignments = math.comb(len(test), fail) if fail and passed else 1
    result = {
        "label_semantics": "FAIL=1,PASS=0",
        "required_fail": min_fail,
        "required_pass": min_pass,
        "observed_fail": fail,
        "observed_pass": passed,
        "fail_quota_met": fail >= min_fail,
        "pass_quota_met": passed >= min_pass,
        "valid_for_confirmatory_test": fail >= min_fail and passed >= min_pass,
        "minimum_attainable_exact_one_sided_p_without_ties": 1.0
        / label_assignments,
    }
    if design.get("cluster_inference"):
        cluster_ids, _, clusters = _cluster_rows(test, design)
        profiles = [
            tuple(int(member["label"]) for member in clusters[cluster_id])
            for cluster_id in cluster_ids
        ]
        clusters_with_fail = sum(any(profile) for profile in profiles)
        clusters_with_pass = sum(not all(profile) for profile in profiles)
        required_clusters = int(design["min_test_problem_clusters"])
        required_fail_clusters = int(design["min_problem_clusters_with_fail"])
        required_pass_clusters = int(design["min_problem_clusters_with_pass"])
        cluster_support_valid = (
            len(cluster_ids) >= required_clusters
            and clusters_with_fail >= required_fail_clusters
            and clusters_with_pass >= required_pass_clusters
        )
        unique_profile_assignments = _multiset_permutation_count(profiles)
        result.update(
            {
                "independent_unit": "problem_id",
                "required_problem_clusters": required_clusters,
                "observed_problem_clusters": len(cluster_ids),
                "required_problem_clusters_with_fail": required_fail_clusters,
                "observed_problem_clusters_with_fail": clusters_with_fail,
                "required_problem_clusters_with_pass": required_pass_clusters,
                "observed_problem_clusters_with_pass": clusters_with_pass,
                "problem_cluster_quota_met": cluster_support_valid,
                "unique_cluster_outcome_profile_assignments": (
                    unique_profile_assignments
                ),
                "minimum_attainable_cluster_permutation_p": (
                    1.0 / unique_profile_assignments
                ),
                "valid_for_confirmatory_test": (
                    result["valid_for_confirmatory_test"]
                    and cluster_support_valid
                ),
            }
        )
    return result


def _verify_run_binding(
    design: dict[str, Any],
    run_manifest_path: Path | None,
) -> dict[str, Any] | None:
    plan = design.get("sampling_plan") or {}
    if not all(
        key in plan
        for key in (
            "model_version",
            "observation_interface_version",
            "prama_kernel_version",
            "parameter_set",
        )
    ):
        return None
    if run_manifest_path is None:
        raise ValueError(
            "the frozen design binds execution versions; --run-manifest is required"
        )
    run = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    expected_model = plan["model_version"]
    expected_observation = plan["observation_interface_version"]
    expected_kernel = plan["prama_kernel_version"]
    expected_generation = plan["parameter_set"]["generation"]
    comparisons = {
        "dataset_sha256": (
            run.get("dataset_sha256"),
            plan["normalized_dataset_sha256"],
        ),
        "dataset_manifest_sha256": (
            run.get("dataset_manifest_sha256"),
            plan["normalized_manifest_sha256"],
        ),
        "model": (run.get("model"), plan["model"]),
        "model_blob_sha256": (
            run.get("model_blob_sha256"),
            expected_model.get("model_blob_sha256"),
        ),
        "confirmatory_design_sha256": (
            run.get("confirmatory_design_sha256"),
            _canonical_sha256(design),
        ),
        "projector_request_schema": (
            (run.get("observation_interface") or {}).get(
                "projector_request_schema"
            ),
            expected_observation["projector_request_schema"],
        ),
        "runner_sha256": (
            (run.get("observation_interface") or {}).get("runner_sha256"),
            expected_observation["runner_sha256"],
        ),
        "model_payload_sha256": (
            (run.get("observation_interface") or {}).get(
                "model_payload_sha256"
            ),
            expected_observation["model_payload_sha256"],
        ),
        "generation_parameter_set": (
            run.get("generation_parameter_set"),
            expected_generation,
        ),
        "kernel_version": (
            (run.get("projector_kernel_identity") or {}).get("version"),
            expected_kernel["package_version"],
        ),
        "kernel_source_tree_sha256": (
            (run.get("projector_kernel_identity") or {}).get(
                "source_tree_sha256"
            ),
            expected_kernel["source_tree_sha256"],
        ),
        "kernel_recertification_sha256": (
            (run.get("projector_kernel_identity") or {}).get(
                "recertification_sha256"
            ),
            expected_kernel["recertification_sha256"],
        ),
    }
    if "provider" in expected_model:
        comparisons["provider"] = (
            run.get("provider"),
            expected_model["provider"],
        )
    if "provider_endpoint" in expected_model:
        comparisons["provider_endpoint"] = (
            run.get("provider_endpoint"),
            expected_model["provider_endpoint"],
        )
    provider_identity = run.get("provider_response_identity") or {}
    if "resolved_models" in expected_model:
        comparisons["provider_resolved_models"] = (
            provider_identity.get("resolved_models"),
            expected_model["resolved_models"],
        )
    if "system_fingerprints" in expected_model:
        comparisons["provider_system_fingerprints"] = (
            provider_identity.get("system_fingerprints"),
            expected_model["system_fingerprints"],
        )
    calibration = plan.get("projector_calibration") or {}
    if calibration:
        comparisons["projector_calibration_sha256"] = (
            run.get("projector_calibration_sha256"),
            calibration.get("artifact_sha256"),
        )
    mismatches = {
        field: {"observed": observed, "expected": expected}
        for field, (observed, expected) in comparisons.items()
        if observed != expected
    }
    if mismatches:
        raise ValueError(
            "run manifest differs from frozen execution binding: "
            + ", ".join(sorted(mismatches))
        )
    return {
        "path": str(run_manifest_path),
        "sha256": _file_sha256(run_manifest_path),
        "matches_frozen_design": True,
        "validated_fields": sorted(comparisons),
        "projector_calibration_sha256": run.get(
            "projector_calibration_sha256"
        ),
    }


def _activation_audit(
    rows: list[dict[str, Any]], tolerance: float
) -> dict[str, Any]:
    excess_sessions = 0
    capacity_sessions = 0
    crossing_sessions = 0
    first_excess = []
    first_capacity = []
    maximum_excess = 0.0
    minimum_capacity = 1.0
    maximum_xi_minus_theta = float("-inf")
    for row in rows:
        trajectory = row["trajectory"]
        excess = [float(point["accumulated_excess"]) for point in trajectory]
        capacities = [float(point["capacity"]) for point in trajectory]
        gaps = [
            float(point["xi"]) - float(point["theta"]) for point in trajectory
        ]
        maximum_excess = max(maximum_excess, max(excess))
        minimum_capacity = min(minimum_capacity, min(capacities))
        maximum_xi_minus_theta = max(maximum_xi_minus_theta, max(gaps))
        activated_excess = [
            index for index, value in enumerate(excess) if value > tolerance
        ]
        activated_capacity = [
            index
            for index, value in enumerate(capacities)
            if value < capacities[0] - tolerance
        ]
        if activated_excess:
            excess_sessions += 1
            first_excess.append(activated_excess[0])
        if activated_capacity:
            capacity_sessions += 1
            first_capacity.append(activated_capacity[0])
        if any(value > tolerance for value in gaps):
            crossing_sessions += 1
    mechanisms_activated = excess_sessions > 0 or capacity_sessions > 0
    return {
        "session_count": len(rows),
        "activation_tolerance": tolerance,
        "threshold_crossing_session_count": crossing_sessions,
        "accumulated_excess_activation_session_count": excess_sessions,
        "capacity_degradation_session_count": capacity_sessions,
        "maximum_accumulated_excess": maximum_excess,
        "minimum_capacity": minimum_capacity,
        "maximum_xi_minus_theta": maximum_xi_minus_theta,
        "first_excess_activation_windows": first_excess,
        "first_capacity_degradation_windows": first_capacity,
        "mechanisms_activated": mechanisms_activated,
        "interpretation": (
            "A/capacity mechanisms were reached and may be evaluated"
            if mechanisms_activated
            else "A/capacity mechanisms were not reached; non-discrimination is not "
            "evidence against those mechanisms"
        ),
    }


def _coordinate_metrics(
    rows: list[dict[str, Any]],
    permutations: int,
    seed: int,
    exact_max_assignments: int,
    design: dict[str, Any] | None = None,
    formal_inference: bool = True,
) -> dict[str, Any]:
    metrics = {
        field: _metric(
            rows,
            field,
            permutations,
            seed,
            exact_max_assignments,
            design,
            formal_inference,
        )
        for field in MANDATORY_COORDINATES
    }
    values = {field: metrics[field].get("auroc") for field in MANDATORY_COORDINATES}
    differences = {}
    pairs = (
        ("max_xi", "max_delta"),
        ("max_negative_balance", "max_delta"),
        ("max_negative_balance", "max_xi"),
    )
    for left, right in pairs:
        differences[f"{left}_minus_{right}"] = (
            None
            if values[left] is None or values[right] is None
            else values[left] - values[right]
        )
    available = {key: value for key, value in values.items() if value is not None}
    winners = (
        []
        if not available
        else [
            field
            for field, value in available.items()
            if value == max(available.values())
        ]
    )
    result = {
        "required": True,
        "coordinate_semantics": {
            "max_delta": "Δ instantaneous perturbation",
            "max_xi": "Ξ dynamic occupation with memory",
            "max_negative_balance": "negative balance θ-Ξ",
        },
        "metrics": metrics,
        "auroc_differences": differences,
        "rank_equivalence": {
            f"{left}_vs_{right}": _pairwise_rank_audit(rows, left, right)
            for left, right in pairs
        },
        "winner": winners,
        "dynamic_strictly_exceeds_delta": (
            values["max_delta"] is not None
            and (
                (values["max_xi"] is not None and values["max_xi"] > values["max_delta"])
                or (
                    values["max_negative_balance"] is not None
                    and values["max_negative_balance"] > values["max_delta"]
                )
            )
        ),
    }
    if design and design.get("incremental_auc_test") and formal_inference:
        result["incremental_auc_test"] = _cluster_bootstrap_auc_difference(
            rows, design
        )
    return result


def _pairwise_rank_audit(
    rows: list[dict[str, Any]], left: str, right: str
) -> dict[str, Any]:
    concordant = discordant = tied_both = tie_mismatch = 0
    for first, second in combinations(rows, 2):
        left_difference = (
            first["features"][left] - second["features"][left]
        )
        right_difference = (
            first["features"][right] - second["features"][right]
        )
        left_sign = (left_difference > 0) - (left_difference < 0)
        right_sign = (right_difference > 0) - (right_difference < 0)
        if left_sign == 0 and right_sign == 0:
            tied_both += 1
        elif left_sign == 0 or right_sign == 0:
            tie_mismatch += 1
        elif left_sign == right_sign:
            concordant += 1
        else:
            discordant += 1
    return {
        "left": left,
        "right": right,
        "rank_order_equivalent": discordant == 0 and tie_mismatch == 0,
        "concordant_pairs": concordant,
        "discordant_pairs": discordant,
        "tied_in_both": tied_both,
        "tie_mismatches": tie_mismatch,
    }


def _rank_equivalence_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = (
        ("max_delta", "max_xi"),
        ("max_delta", "max_negative_balance"),
        ("max_xi", "max_negative_balance"),
    )
    balance_identity_residuals = [
        abs(
            -float(point["balance"])
            - (float(point["xi"]) - float(point["theta"]))
        )
        for row in rows
        for point in row["trajectory"]
    ]
    return {
        "scope": "full-trajectory per-session maxima",
        "pairwise": {
            f"{left}_vs_{right}": _pairwise_rank_audit(rows, left, right)
            for left, right in pairs
        },
        "balance_identity": {
            "tested_identity": "-balance = xi - theta",
            "maximum_absolute_window_residual": max(balance_identity_residuals),
            "exact_within_float_representation": max(balance_identity_residuals)
            == 0.0,
        },
    }


def _length_statistics(values: list[int]) -> dict[str, Any]:
    return {
        "n": len(values),
        "minimum": min(values),
        "median": median(values),
        "mean": mean(values),
        "maximum": max(values),
    }


def _trajectory_length_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_label: dict[str, list[int]] = {}
    by_perturbation: dict[str, list[int]] = {}
    for row in rows:
        count = len(row["trajectory"])
        label = "FAIL" if row["label"] else "PASS"
        perturbation = row.get("perturbation_type") or "UNKNOWN"
        by_label.setdefault(label, []).append(count)
        by_perturbation.setdefault(perturbation, []).append(count)
    return {
        "all": _length_statistics([len(row["trajectory"]) for row in rows]),
        "by_label": {
            key: _length_statistics(values) for key, values in by_label.items()
        },
        "by_perturbation": {
            key: _length_statistics(values)
            for key, values in by_perturbation.items()
        },
    }


def _session_horizon_table(
    rows: list[dict[str, Any]], design: dict[str, Any]
) -> list[dict[str, Any]]:
    horizons = [
        int(value)
        for value in design["temporal_censoring"]["absolute_window_horizons"]
    ]
    table = []
    for row in sorted(rows, key=lambda value: value["session_id"]):
        horizon_scores = {}
        for horizon in horizons:
            horizon_scores[str(horizon)] = (
                _coordinate_features(row["trajectory"][:horizon])
                if len(row["trajectory"]) >= horizon
                else None
            )
        table.append(
            {
                "session_id": row["session_id"],
                "label": row["label"],
                "outcome": "FAIL" if row["label"] else "PASS",
                "problem_id": row.get("problem_id"),
                "item_id": row.get("item_id"),
                "perturbation_type": row.get("perturbation_type"),
                "difficulty": row.get("difficulty"),
                "n_windows": len(row["trajectory"]),
                "prefix_scores": horizon_scores,
                "full_scores": _coordinate_features(row["trajectory"]),
            }
        )
    return table


def _write_session_horizon_csv(
    path: Path, table: list[dict[str, Any]], horizons: Sequence[int]
) -> None:
    identity_fields = (
        "session_id",
        "label",
        "outcome",
        "problem_id",
        "item_id",
        "perturbation_type",
        "difficulty",
        "n_windows",
    )
    fields = list(identity_fields)
    for horizon in horizons:
        fields.extend(
            f"{coordinate}_h{horizon}" for coordinate in MANDATORY_COORDINATES
        )
    fields.extend(f"{coordinate}_full" for coordinate in MANDATORY_COORDINATES)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for source in table:
            row = {field: source.get(field) for field in identity_fields}
            for horizon in horizons:
                scores = source["prefix_scores"].get(str(horizon))
                for coordinate in MANDATORY_COORDINATES:
                    row[f"{coordinate}_h{horizon}"] = (
                        None if scores is None else scores[coordinate]
                    )
            for coordinate in MANDATORY_COORDINATES:
                row[f"{coordinate}_full"] = source["full_scores"][coordinate]
            writer.writerow(row)


def _temporal_evaluation(
    test: list[dict[str, Any]],
    design: dict[str, Any],
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    temporal = design["temporal_censoring"]
    exact_max = int(design["exact_permutation_max_assignments"])
    absolute = {}
    for horizon in temporal["absolute_window_horizons"]:
        horizon = int(horizon)
        at_risk = []
        for row in test:
            if len(row["trajectory"]) < horizon:
                continue
            at_risk.append(
                {
                    **row,
                    "features": _coordinate_features(row["trajectory"][:horizon]),
                }
            )
        absolute[str(horizon)] = {
            "window_horizon": horizon,
            "short_session_policy": "at_risk_only",
            "at_risk_n": len(at_risk),
            "fail_n": sum(row["label"] for row in at_risk),
            "pass_n": len(at_risk) - sum(row["label"] for row in at_risk),
            "comparison": _coordinate_metrics(
                at_risk,
                permutations,
                seed + horizon,
                exact_max,
                design,
                formal_inference=False,
            )
            if at_risk
            else None,
        }
    relative = {}
    for offset, fraction in enumerate(temporal["relative_prefix_fractions"]):
        fraction = float(fraction)
        censored = []
        for row in test:
            count = max(1, math.ceil(len(row["trajectory"]) * fraction))
            censored.append(
                {
                    **row,
                    "features": _coordinate_features(row["trajectory"][:count]),
                }
            )
        relative[str(fraction)] = {
            "prefix_fraction": fraction,
            "n": len(censored),
            "comparison": _coordinate_metrics(
                censored,
                permutations,
                seed + 1000 + offset,
                exact_max,
                design,
                formal_inference=False,
            ),
        }
    return {
        "purpose": "anticipatory discrimination before the final generated token",
        "absolute_horizon_semantics": (
            "prefix windows retained from generation start; h=1 is the earliest "
            "single-window prefix, not one window removed from the end"
        ),
        "final_outcome_used_only_as_external_label": True,
        "inference_status": (
            "exploratory descriptive sensitivity; no session-level p-values "
            "or confirmatory decisions"
            if design.get("cluster_inference")
            else "legacy session-level inference"
        ),
        "absolute_window_horizons": absolute,
        "relative_prefix_sensitivity": relative,
    }


def _primary_horizon_test_rows(
    test: list[dict[str, Any]], design: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    temporal = design["temporal_censoring"]
    configured = temporal.get("primary_absolute_window_horizon")
    if configured is None:
        return test, {
            "time_basis": "full_trajectory",
            "primary_absolute_window_horizon": None,
            "full_test_n": len(test),
            "at_risk_n": len(test),
        }
    horizon = int(configured)
    window_size = int(temporal["window_size_tokens"])
    expected_sessions = int(design["independent_unit"]["expected_sessions_per_cluster"])
    clusters: dict[str, list[dict[str, Any]]] = {}
    for row in test:
        problem_id = str(row.get("problem_id") or "")
        if not problem_id:
            raise ValueError("primary horizon requires problem_id on every test session")
        clusters.setdefault(problem_id, []).append(row)
    retained: list[dict[str, Any]] = []
    excluded_clusters = []
    for problem_id, cluster_rows in sorted(clusters.items()):
        if len(cluster_rows) != expected_sessions or any(
            len(row["trajectory"]) < horizon for row in cluster_rows
        ):
            excluded_clusters.append(problem_id)
            continue
        for row in cluster_rows:
            retained.append(
                {
                    **row,
                    "features": _features(
                        row["trajectory"][:horizon],
                        row["tokens"][: horizon * window_size],
                    ),
                }
            )
    if not retained:
        raise ValueError("no complete problem clusters are at risk at primary horizon")
    return retained, {
        "time_basis": "absolute_window_horizon_from_generation_start",
        "primary_absolute_window_horizon": horizon,
        "window_size_tokens": window_size,
        "nominal_token_horizon": horizon * window_size,
        "short_session_policy": "exclude_entire_problem_cluster_if_either_session_is_short",
        "full_test_n": len(test),
        "full_problem_cluster_count": len(clusters),
        "at_risk_n": len(retained),
        "at_risk_problem_cluster_count": len(retained) // expected_sessions,
        "excluded_problem_cluster_count": len(excluded_clusters),
        "excluded_problem_ids": excluded_clusters,
    }


def evaluate(
    join_path: Path,
    primary: str,
    permutations: int,
    seed: int,
    design: dict[str, Any] | None = None,
    dataset_path: Path | None = None,
    dataset_manifest_path: Path | None = None,
    run_manifest_path: Path | None = None,
) -> dict[str, Any]:
    design = _validate_design(dict(design or LEGACY_DESIGN))
    if primary != design["primary_score"]:
        raise ValueError("CLI primary score differs from frozen confirmatory design")
    with join_path.open(newline="", encoding="utf-8") as handle:
        source = list(csv.DictReader(handle))
    if not source:
        raise ValueError("blind join contains no completed examples")
    dataset_rows = _dataset_index(dataset_path) if dataset_path else {}
    run_manifest = (
        json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if run_manifest_path
        else {}
    )
    acquisition_scope = str(
        run_manifest.get("acquisition_scope") or "full_dataset"
    )
    dataset_binding = _verify_dataset_binding(
        design,
        dataset_path,
        source,
        dataset_rows or None,
        dataset_manifest_path,
        acquisition_scope,
    )
    run_binding = _verify_run_binding(design, run_manifest_path)
    group_splits: dict[str, set[str]] = {}
    rows = []
    for source_row in source:
        split = source_row["split"].strip().lower()
        group = source_row["group_sha256"]
        group_splits.setdefault(group, set()).add(split)
        trajectory = _trajectory(Path(source_row["trajectory_path"]))
        tokens = _numeric_tokens(Path(source_row["projection_request_path"]))
        rows.append(
            {
                "session_id": source_row["session_id"],
                "group_sha256": group,
                "label": int(source_row["label"]),
                "split": split,
                "problem_id": dataset_rows.get(source_row["session_id"], {}).get(
                    "problem_id"
                ),
                "item_id": dataset_rows.get(source_row["session_id"], {}).get(
                    "item_id"
                ),
                "perturbation_type": dataset_rows.get(
                    source_row["session_id"], {}
                ).get("perturbation_type"),
                "difficulty": dataset_rows.get(source_row["session_id"], {}).get(
                    "difficulty"
                ),
                "trajectory": trajectory,
                "tokens": tokens,
                "features": _features(trajectory, tokens),
            }
        )
    leaked = [group for group, splits in group_splits.items() if len(splits) > 1]
    if leaked:
        raise ValueError("problem groups cross train/test partitions")
    train = [row for row in rows if row["split"] == "train"]
    test = [row for row in rows if row["split"] == "test"]
    if not test or (not train and acquisition_scope != "holdout_only"):
        raise ValueError("blind evaluation requires the frozen test rows")
    full_test = test
    confirmatory_test, primary_horizon_audit = _primary_horizon_test_rows(
        full_test, design
    )
    support = _class_support(confirmatory_test, design)
    exact_max = int(design["exact_permutation_max_assignments"])
    test_metrics = {
        field: _metric(
            confirmatory_test, field, permutations, seed, exact_max, design
        )
        for field in SCORES + BASELINES
    }
    comparison = _coordinate_metrics(
        confirmatory_test,
        permutations,
        seed,
        exact_max,
        design,
        formal_inference=True,
    )
    primary_metric = test_metrics[primary]
    incremental_test = comparison.get("incremental_auc_test")
    baseline_aurocs = [
        test_metrics[field]["auroc"]
        for field in BASELINES
        if test_metrics[field]["auroc"] is not None
    ]
    if not support["valid_for_confirmatory_test"]:
        verdict = "inconclusive_insufficient_class_support"
    elif primary_metric["auroc"] is None or not baseline_aurocs:
        verdict = "inconclusive"
    elif (
        primary_metric["permutation_p"] is not None
        and primary_metric["permutation_p"] < float(design["p_threshold"])
        and (
            incremental_test is None
            or incremental_test["passes_formal_incremental_rule"]
        )
        and primary_metric["auroc"] > max(baseline_aurocs)
    ):
        verdict = "positive"
    else:
        verdict = "honest_null"
    activation_tolerance = float(design["activation_tolerance"])
    return {
        "schema": (
            "LLM-SVM-CoCC-PRAMA-evaluation/3"
            if design.get("cluster_inference")
            else "LLM-SVM-CoCC-PRAMA-evaluation/2"
        ),
        "design_id": design["design_id"],
        "design_sha256": _canonical_sha256(design),
        "dataset_binding": dataset_binding,
        "run_binding": run_binding,
        "primary_score": primary,
        "train_n": len(train),
        "test_n": len(confirmatory_test),
        "full_test_n": len(full_test),
        "test_positive_n": support["observed_fail"],
        "test_negative_n": support["observed_pass"],
        "class_support": support,
        "primary_horizon_audit": primary_horizon_audit,
        "independent_unit": design.get("independent_unit"),
        "cluster_inference_plan": design.get("cluster_inference"),
        "minimum_effect_of_interest": (
            (design.get("incremental_auc_test") or {}).get(
                "minimum_effect_of_interest"
            )
        ),
        "metrics": test_metrics,
        "mandatory_coordinate_comparison": comparison,
        "rank_equivalence_audit": _rank_equivalence_audit(confirmatory_test),
        "trajectory_length_audit": _trajectory_length_audit(full_test),
        "session_horizon_table": _session_horizon_table(full_test, design),
        "temporal_censoring": _temporal_evaluation(
            full_test, design, permutations, seed
        ),
        "activation_audit": {
            "train": _activation_audit(train, activation_tolerance),
            "test": _activation_audit(full_test, activation_tolerance),
            "all": _activation_audit(rows, activation_tolerance),
        },
        "verdict": verdict,
        "confirmatory_rule": (
            "problem-cluster permutation p < 0.01 for the primary score; "
            "paired problem-cluster bootstrap p < 0.01 and one-sided 99% lower "
            "bound > 0 for AUROC_PRAMA - AUROC_delta; observed AUROC difference "
            "must be at least the frozen minimum effect of interest; primary "
            "AUROC must also exceed every frozen baseline; all session and "
            "problem-cluster quotas must first be met"
            if design.get("cluster_inference")
            else (
                "primary permutation_p < 0.01 and primary AUROC strictly exceeds "
                "every instantaneous/logprob baseline on the untouched test "
                "partition; the frozen minimum PASS and FAIL quotas must first "
                "be met"
            )
        ),
        "claim_boundary": (
            "external final-answer discrimination and temporally censored "
            "anticipation; not proof of semantic identity between PRAMA "
            "coordinates and degradation"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blind-join", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--design", type=Path)
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Normalized dataset bound by the frozen confirmatory design",
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        help="Normalization manifest bound by the frozen confirmatory design",
    )
    parser.add_argument(
        "--run-manifest",
        type=Path,
        help="Run manifest attesting model, interface, parameters and kernel",
    )
    parser.add_argument(
        "--session-horizon-csv",
        type=Path,
        help="Optional output path; defaults beside --out",
    )
    parser.add_argument(
        "--primary-score", choices=SCORES, default="max_negative_balance"
    )
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args(argv)
    try:
        design = (
            json.loads(args.design.read_text(encoding="utf-8"))
            if args.design
            else None
        )
        report = evaluate(
            args.blind_join,
            args.primary_score,
            args.permutations,
            args.seed,
            design,
            args.dataset,
            args.dataset_manifest,
            args.run_manifest,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        horizon_path = args.session_horizon_csv or args.out.with_name(
            f"{args.out.stem}_session_horizons.csv"
        )
        _write_session_horizon_csv(
            horizon_path,
            report["session_horizon_table"],
            [
                int(value)
                for value in design["temporal_censoring"][
                    "absolute_window_horizons"
                ]
            ]
            if design
            else LEGACY_DESIGN["temporal_censoring"][
                "absolute_window_horizons"
            ],
        )
    except (OSError, TypeError, ValueError) as exc:
        print(f"CoCC PRAMA evaluation failed: {exc}")
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
