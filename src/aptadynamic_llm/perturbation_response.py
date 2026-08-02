"""Frozen, externally anchored perturbation-response classification."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from aptadynamic_llm.artifact_schema import validate_artifact


RESPONSE_CLASSES = {
    "adaptive_integration",
    "partial_integration",
    "persistent_nonintegration",
    "counterfactual_rejection",
    "indeterminate",
}


@dataclass(frozen=True)
class PerturbationConfig:
    response_horizon_windows: int
    minimum_uptake_gain: float
    maximum_self_dependence_increase: float
    minimum_integrated_uptake: float
    counter_reactive_margin: float

    def __post_init__(self) -> None:
        if self.response_horizon_windows < 1:
            raise ValueError("response_horizon_windows must be positive")
        for name in (
            "minimum_uptake_gain",
            "maximum_self_dependence_increase",
            "minimum_integrated_uptake",
            "counter_reactive_margin",
        ):
            value = float(getattr(self, name))
            if value < 0 or not math.isfinite(value):
                raise ValueError(f"{name} must be finite and non-negative")


def evaluate_perturbation_response(
    *,
    envelope: Mapping[str, Any],
    perturbation_id: str,
    introduced_at_window: int,
    pre_self_dependence_excess: float,
    post_self_dependence_excess: float,
    pre_anchor_uptake: float,
    post_anchor_uptake: float,
    trajectory_changed: bool,
    recovery_detected: bool,
    recovery_latency_windows: int | None,
    anchor_externally_verified: bool,
    config: PerturbationConfig,
) -> dict[str, Any]:
    """Classify response using only preregistered numeric gates."""

    if not anchor_externally_verified:
        response_class = "indeterminate"
    else:
        uptake_gain = post_anchor_uptake - pre_anchor_uptake
        self_shift = post_self_dependence_excess - pre_self_dependence_excess
        if (
            uptake_gain <= -config.counter_reactive_margin
            and self_shift >= config.counter_reactive_margin
        ):
            response_class = "counterfactual_rejection"
        elif (
            trajectory_changed
            and uptake_gain >= config.minimum_uptake_gain
            and self_shift <= config.maximum_self_dependence_increase
            and post_anchor_uptake >= config.minimum_integrated_uptake
        ):
            response_class = "adaptive_integration"
        elif (
            not recovery_detected
            and post_anchor_uptake < config.minimum_integrated_uptake
        ):
            response_class = "persistent_nonintegration"
        else:
            response_class = "partial_integration"

    uptake_shift = post_anchor_uptake - pre_anchor_uptake
    self_shift = post_self_dependence_excess - pre_self_dependence_excess
    record = {
        **dict(envelope),
        "perturbation_id": perturbation_id,
        "introduced_at_window": introduced_at_window,
        "response_horizon_windows": config.response_horizon_windows,
        "pre_self_dependence_excess": float(pre_self_dependence_excess),
        "post_self_dependence_excess": float(post_self_dependence_excess),
        "self_dependence_shift": float(self_shift),
        "pre_anchor_uptake": float(pre_anchor_uptake),
        "post_anchor_uptake": float(post_anchor_uptake),
        "anchor_uptake_shift": float(uptake_shift),
        "trajectory_changed": bool(trajectory_changed),
        "trajectory_change_magnitude": math.hypot(self_shift, uptake_shift),
        "recovery_detected": bool(recovery_detected),
        "recovery_latency_windows": recovery_latency_windows,
        "anchor_externally_verified": bool(anchor_externally_verified),
        "response_class": response_class,
    }
    validate_artifact(record, "perturbation_response")
    return record
