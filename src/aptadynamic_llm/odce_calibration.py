"""Exploratory, non-freezing calibration for ODCE-v0.1.0."""

from __future__ import annotations

from copy import deepcopy
import math
from statistics import median
from typing import Any, Mapping, Sequence

from aptadynamic_llm.artifact_schema import sha256_value, validate_artifact
from aptadynamic_llm.structural_conversion import (
    BENEFIT_CHANNELS,
    COST_CHANNELS,
    ODCEConfig,
    OPERATOR_ID,
    OPERATOR_VERSION,
)


CALIBRATION_SCHEMA = "LLM-SVM-ODCE-exploratory-calibration/0.1"
THRESHOLD_CALIBRATION_SCHEMA = (
    "LLM-SVM-ODCE-differential-noise-floor-calibration/0.1"
)
ESTIMATOR = "median_max_mad_iqr"
NORMAL_MAD_FACTOR = 1.4826
NORMAL_IQR_FACTOR = 1.349


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile requires at least one value")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _estimate(values: Sequence[float], minimum_scale: float) -> dict[str, float]:
    location = float(median(values))
    mad = float(median([abs(value - location) for value in values]))
    q25 = _quantile(values, 0.25)
    q75 = _quantile(values, 0.75)
    iqr = float(q75 - q25)
    scale = max(NORMAL_MAD_FACTOR * mad, iqr / NORMAL_IQR_FACTOR)
    return {
        "location": location,
        "mad": mad,
        "q25": q25,
        "q75": q75,
        "iqr": iqr,
        "scale": scale,
        "minimum_scale": minimum_scale,
    }


def calibrate_exploratory_differential_threshold(
    rows: Sequence[Mapping[str, Any]],
    base_contract: Mapping[str, Any],
    *,
    correspondence_names: Sequence[str],
    stable_condition_id: str,
    min_observations: int = 20,
    residual_quantile: float = 0.95,
    allow_exploratory_input: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Derive a material-positive threshold from declared stable ODCE rows.

    The estimator treats each selected stable differential as a zero-centered
    measurement with possible empirical bias.  Its per-correspondence noise
    floor is ``abs(median(D)) + Q_q(abs(D - median(D)))``.  The single ODCE
    threshold is the maximum supported floor, so no selected baseline channel
    receives a less conservative threshold than the one it measured.
    """

    if min_observations < 2:
        raise ValueError("min_observations must be at least 2")
    if not 0.5 < residual_quantile < 1.0:
        raise ValueError("residual_quantile must be within (0.5, 1)")
    if not isinstance(stable_condition_id, str) or not stable_condition_id.strip():
        raise ValueError("stable_condition_id must be nonempty")
    if not rows:
        raise ValueError("noise-floor calibration requires stable ODCE artifacts")

    config = ODCEConfig.from_contract(base_contract)
    if base_contract.get("status") != "EXPLORATORY_CAUSAL_POST_KERNEL":
        raise ValueError("noise-floor calibration base contract must remain exploratory")
    available_names = {row.name for row in config.correspondence}
    selected = tuple(dict.fromkeys(str(name) for name in correspondence_names))
    if not selected or any(name not in available_names for name in selected):
        raise ValueError(
            "correspondence_names must select known ODCE correspondences"
        )

    materialized = [dict(row) for row in rows]
    partitions: set[str] = set()
    sessions: set[str] = set()
    for row in materialized:
        validate_artifact(row, "structural_conversion_differential")
        partition = str(row["partition"])
        if partition == "confirmatory":
            raise ValueError("confirmatory artifacts cannot calibrate ODCE")
        if partition == "exploratory" and not allow_exploratory_input:
            raise ValueError(
                "exploratory input requires explicit allow_exploratory_input"
            )
        if partition not in {"calibration", "exploratory"}:
            raise ValueError(f"unsupported calibration partition: {partition}")
        partitions.add(partition)
        sessions.add(str(row["session_id"]))

    channel_report: dict[str, dict[str, Any]] = {}
    floors: list[float] = []
    for name in selected:
        values = [
            float(row["differential_vector"][name])
            for row in materialized
            if row["component_status"]["differential"][name] == "OBSERVED"
        ]
        if len(values) < min_observations:
            raise ValueError(
                f"{name} has {len(values)} stable observations; "
                f"{min_observations} required"
            )
        center = float(median(values))
        absolute_residuals = [abs(value - center) for value in values]
        residual_floor = float(
            _quantile(absolute_residuals, residual_quantile)
        )
        floor = abs(center) + residual_floor
        floors.append(floor)
        channel_report[name] = {
            "observed_count": len(values),
            "stable_center_median": center,
            "absolute_center_bias": abs(center),
            "absolute_residual_quantile": residual_floor,
            "residual_quantile_probability": residual_quantile,
            "derived_noise_floor": floor,
            "minimum": min(values),
            "maximum": max(values),
        }

    threshold = max(floors)
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError(
            "stable observations produced a degenerate differential noise floor"
        )

    calibrated = deepcopy(dict(base_contract))
    normalization_hash = sha256_value(calibrated["normalization"])
    input_hash = sha256_value(materialized)
    calibrated["differential_threshold"] = threshold
    calibrated["differential_threshold_calibration"] = {
        "schema": THRESHOLD_CALIBRATION_SCHEMA,
        "status": "EXPLORATORY_EMPIRICAL_NOISE_FLOOR",
        "stable_condition_id": stable_condition_id.strip(),
        "input_artifacts_sha256": input_hash,
        "artifact_count": len(materialized),
        "session_count": len(sessions),
        "partitions": sorted(partitions),
        "correspondence_names": list(selected),
        "estimator": "max_selected(abs(median(D)) + residual_quantile(abs(D-median(D))))",
        "residual_quantile": residual_quantile,
        "min_observations": min_observations,
        "derived_differential_threshold": threshold,
        "normalization_contract_sha256": normalization_hash,
        "normalization_modified": False,
        "frozen": False,
    }
    if sha256_value(calibrated["normalization"]) != normalization_hash:
        raise ValueError("noise-floor calibration must not modify normalization")
    ODCEConfig.from_contract(calibrated)
    report = {
        "schema": THRESHOLD_CALIBRATION_SCHEMA,
        "operator_id": OPERATOR_ID,
        "operator_version": OPERATOR_VERSION,
        "status": "EXPLORATORY_NOT_FROZEN",
        "stable_condition_id": stable_condition_id.strip(),
        "input_artifacts_sha256": input_hash,
        "base_contract_sha256": sha256_value(base_contract),
        "output_contract_sha256": sha256_value(calibrated),
        "artifact_count": len(materialized),
        "session_count": len(sessions),
        "partitions": sorted(partitions),
        "correspondence_names": list(selected),
        "estimator": {
            "center": "median(D)",
            "residual": "abs(D - median(D))",
            "residual_quantile": residual_quantile,
            "per_correspondence_floor": (
                "abs(median(D)) + residual_quantile(abs(D-median(D)))"
            ),
            "global_threshold": "maximum supported correspondence floor",
            "min_observations": min_observations,
        },
        "channels": channel_report,
        "derived_differential_threshold": threshold,
        "normalization_contract_sha256": normalization_hash,
        "normalization_modified": False,
        "calibration_ready_for_freeze": False,
        "claim_boundary": [
            "The threshold is an exploratory empirical noise floor from an explicitly declared stable condition.",
            "The calibration changes differential_threshold only and preserves every normalization rule.",
            "The result does not authorize confirmatory use or freeze ODCE.",
        ],
    }
    return calibrated, report


def calibrate_exploratory_contract(
    rows: Sequence[Mapping[str, Any]],
    base_contract: Mapping[str, Any],
    *,
    min_observations: int = 20,
    minimum_observed_fraction: float = 0.10,
    minimum_session_count: int = 5,
    minimum_session_coverage: float = 0.10,
    minimum_scale: float = 1e-12,
    allow_exploratory_input: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit robust index-level rules without authorizing confirmatory use."""

    if min_observations < 2:
        raise ValueError("min_observations must be at least 2")
    if not 0.0 <= minimum_observed_fraction <= 1.0:
        raise ValueError("minimum_observed_fraction must be within [0, 1]")
    if minimum_session_count < 1:
        raise ValueError("minimum_session_count must be positive")
    if not 0.0 <= minimum_session_coverage <= 1.0:
        raise ValueError("minimum_session_coverage must be within [0, 1]")
    if not math.isfinite(minimum_scale) or minimum_scale <= 0:
        raise ValueError("minimum_scale must be finite and positive")
    if not rows:
        raise ValueError("exploratory ODCE calibration requires input artifacts")
    ODCEConfig.from_contract(base_contract)
    if base_contract.get("status") != "EXPLORATORY_CAUSAL_POST_KERNEL":
        raise ValueError("calibration base contract must remain exploratory")
    if base_contract["normalization"].get("confirmatory_use_allowed") is not False:
        raise ValueError("calibration base contract must forbid confirmatory use")

    materialized = [dict(row) for row in rows]
    partitions: set[str] = set()
    sessions: set[str] = set()
    for row in materialized:
        validate_artifact(row, "structural_conversion_differential")
        partition = str(row["partition"])
        if partition == "confirmatory":
            raise ValueError("confirmatory artifacts cannot calibrate ODCE")
        if partition == "exploratory" and not allow_exploratory_input:
            raise ValueError(
                "exploratory input requires explicit allow_exploratory_input"
            )
        if partition not in {"calibration", "exploratory"}:
            raise ValueError(f"unsupported calibration partition: {partition}")
        partitions.add(partition)
        sessions.add(str(row["session_id"]))

    calibrated = deepcopy(dict(base_contract))
    normalization = calibrated["normalization"]
    channel_report: dict[str, dict[str, Any]] = {"cost": {}, "benefit": {}}
    blocking_channels: list[str] = []
    total = len(materialized)
    for group, channels in (
        ("cost", COST_CHANNELS),
        ("benefit", BENEFIT_CHANNELS),
    ):
        for channel in channels:
            observed_rows = [
                row
                for row in materialized
                if row["component_status"][group][channel] == "OBSERVED"
            ]
            values = [
                float(row[f"{group}_vector"][channel])
                for row in observed_rows
            ]
            observed_sessions = {
                str(row["session_id"]) for row in observed_rows
            }
            observed_fraction = len(values) / total
            session_coverage = len(observed_sessions) / len(sessions)
            base_rule = dict(normalization[group][channel])
            details: dict[str, Any] = {
                "observed_count": len(values),
                "total_artifact_count": total,
                "observed_fraction": observed_fraction,
                "observed_session_count": len(observed_sessions),
                "total_session_count": len(sessions),
                "session_coverage": session_coverage,
                "coverage_requirements": {
                    "min_observations": min_observations,
                    "minimum_observed_fraction": minimum_observed_fraction,
                    "minimum_session_count": minimum_session_count,
                    "minimum_session_coverage": minimum_session_coverage,
                },
                "sampling_unit": "ODCE_INDEX",
            }
            if len(values) < min_observations:
                status = "INSUFFICIENT_OBSERVATIONS"
                rule = base_rule
                details["estimate"] = None
            elif observed_fraction < minimum_observed_fraction:
                status = "INSUFFICIENT_OBSERVED_FRACTION"
                rule = base_rule
                details["estimate"] = None
            elif len(observed_sessions) < minimum_session_count:
                status = "INSUFFICIENT_SESSION_COUNT"
                rule = base_rule
                details["estimate"] = None
            elif session_coverage < minimum_session_coverage:
                status = "INSUFFICIENT_SESSION_COVERAGE"
                rule = base_rule
                details["estimate"] = None
            else:
                estimate = _estimate(values, minimum_scale)
                details["estimate"] = estimate
                if estimate["scale"] < minimum_scale:
                    status = "DEGENERATE_SCALE"
                    rule = base_rule
                else:
                    status = "CALIBRATED_EXPLORATORY"
                    rule = {
                        "location": estimate["location"],
                        "scale": estimate["scale"],
                        "orientation": float(base_rule.get("orientation", 1.0)),
                        "extreme_policy": "preserve",
                    }
            if status != "CALIBRATED_EXPLORATORY":
                blocking_channels.append(f"{group}.{channel}")
            normalization[group][channel] = rule
            details["status"] = status
            details["applied_rule"] = rule
            channel_report[group][channel] = details

    correspondence_report: dict[str, dict[str, Any]] = {}
    blocking_correspondences: list[str] = []
    for correspondence in calibrated["correspondence"]:
        cost_channel = str(correspondence["cost_channel"])
        benefit_channel = str(correspondence["benefit_channel"])
        cost_status = channel_report["cost"][cost_channel]["status"]
        benefit_status = channel_report["benefit"][benefit_channel]["status"]
        calibrated_sides = sum(
            status == "CALIBRATED_EXPLORATORY"
            for status in (cost_status, benefit_status)
        )
        if calibrated_sides == 2:
            correspondence_status = "CALIBRATED"
        elif calibrated_sides == 1:
            correspondence_status = "PARTIALLY_CALIBRATED"
        else:
            correspondence_status = "UNCALIBRATED"
        interpretation_allowed = correspondence_status == "CALIBRATED"
        correspondence["calibration_status"] = correspondence_status
        correspondence["instrumental_interpretation_allowed"] = (
            interpretation_allowed
        )
        name = str(correspondence["name"])
        if not interpretation_allowed:
            blocking_correspondences.append(name)
        correspondence_report[name] = {
            "cost_channel": cost_channel,
            "cost_channel_status": cost_status,
            "benefit_channel": benefit_channel,
            "benefit_channel_status": benefit_status,
            "calibration_status": correspondence_status,
            "instrumental_interpretation_allowed": interpretation_allowed,
        }

    input_sha256 = sha256_value(materialized)
    normalization["calibration_status"] = (
        "EXPLORATORY_ROBUST_COMPLETE"
        if not blocking_channels
        else "EXPLORATORY_ROBUST_PARTIAL"
    )
    normalization["confirmatory_use_allowed"] = False
    normalization["calibration_provenance"] = {
        "schema": CALIBRATION_SCHEMA,
        "input_artifacts_sha256": input_sha256,
        "artifact_count": total,
        "session_count": len(sessions),
        "partitions": sorted(partitions),
        "estimator": ESTIMATOR,
        "min_observations": min_observations,
        "minimum_observed_fraction": minimum_observed_fraction,
        "minimum_session_count": minimum_session_count,
        "minimum_session_coverage": minimum_session_coverage,
        "minimum_scale": minimum_scale,
        "sampling_unit": "ODCE_INDEX",
        "cluster_weighting": "NONE_EXPLORATORY",
        "frozen": False,
    }
    calibrated["differential_threshold"] = 0.0
    calibrated.pop("differential_threshold_calibration", None)
    calibrated["claim_boundary"] = [
        "This contract contains an exploratory robust normalization fitted at the ODCE-index level.",
        "Channels reported as insufficient or degenerate retain their base rule and are not calibrated.",
        "The contract is not domain-representative, cluster-weighted, confirmatory, or frozen.",
        "A future freeze requires full channel coverage, reviewed correspondences, and prospective domain calibration.",
    ]
    ODCEConfig.from_contract(calibrated)
    report = {
        "schema": CALIBRATION_SCHEMA,
        "operator_id": OPERATOR_ID,
        "operator_version": OPERATOR_VERSION,
        "status": "EXPLORATORY_NOT_FROZEN",
        "input_artifacts_sha256": input_sha256,
        "base_contract_sha256": sha256_value(base_contract),
        "output_contract_sha256": sha256_value(calibrated),
        "artifact_count": total,
        "session_count": len(sessions),
        "partitions": sorted(partitions),
        "estimator": {
            "name": ESTIMATOR,
            "location": "median",
            "scale": "max(1.4826*MAD, IQR/1.349)",
            "extreme_policy": "preserve",
            "min_observations": min_observations,
            "minimum_observed_fraction": minimum_observed_fraction,
            "minimum_session_count": minimum_session_count,
            "minimum_session_coverage": minimum_session_coverage,
            "minimum_scale": minimum_scale,
            "sampling_unit": "ODCE_INDEX",
            "cluster_weighting": "NONE_EXPLORATORY",
        },
        "channels": channel_report,
        "blocking_channels": blocking_channels,
        "correspondences": correspondence_report,
        "blocking_correspondences": blocking_correspondences,
        "differential_threshold_invalidated": True,
        "differential_threshold_recalibration_required": True,
        "calibration_ready_for_freeze": False,
        "claim_boundary": [
            "This is an exploratory index-level calibration.",
            "Unobserved, insufficient, or degenerate channels retain their base rule and are reported as blocking.",
            "The calibration does not authorize confirmatory output or create a freeze manifest.",
            "A future freeze requires domain-representative calibration with declared cluster weighting and full correspondence coverage.",
        ],
    }
    return calibrated, report
