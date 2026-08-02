"""End-to-end pre-kernel pipeline for precomputed factorial score windows.

The pipeline accepts already teacher-forced condition scores, evaluates
eligibility, builds a causal calibration trace, freezes the expectation, and
emits only conforming confirmatory ``(omega, expected_omega)`` pairs as kernel
inputs.  It deliberately does not invoke PRAMA until a window-scale kernel
configuration is pinned and numerically recertified.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

from aptadynamic_llm.factorial_ablation import (
    AblationConfig,
    FactorialConditionScores,
    WindowMetadata,
    evaluate_window,
)
from aptadynamic_llm.frozen_expectation import (
    CalibrationObservation,
    ExpectationStatus,
    ExpectationStratum,
    FrozenExpectation,
    causal_calibration_trace,
)


PIPELINE_SCHEMA = "LLM-SVM-FACTORIAL-OD/2"


@dataclass(frozen=True)
class FactorialPipelineResult:
    observations: tuple[dict[str, Any], ...]
    kernel_inputs: tuple[dict[str, Any], ...]
    estimator: FrozenExpectation


def _required(row: dict[str, Any], name: str) -> Any:
    if name not in row:
        raise ValueError(f"input row is missing required field {name!r}")
    return row[name]


def _parse_metadata(row: dict[str, Any]) -> WindowMetadata:
    return WindowMetadata(
        session_id=str(_required(row, "session_id")),
        turn_index=int(_required(row, "turn_index")),
        window_index=int(_required(row, "window_index")),
        token_start=int(_required(row, "token_start")),
        token_end=int(_required(row, "token_end")),
        natural_assistant_region=bool(row.get("natural_assistant_region", True)),
        exactly_aligned=bool(row.get("exactly_aligned", True)),
        generation_truncated=bool(row.get("generation_truncated", False)),
        contains_unmodeled_content=bool(row.get("contains_unmodeled_content", False)),
    )


def _parse_conditions(row: dict[str, Any]) -> tuple[FactorialConditionScores, ...]:
    values = _required(row, "conditions")
    if not isinstance(values, list):
        raise ValueError("conditions must be a JSON array")
    output = []
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("each condition realization must be an object")
        output.append(
            FactorialConditionScores(
                filler_id=str(_required(value, "filler_id")),
                l11=_required(value, "L11"),
                l10=_required(value, "L10"),
                l01=_required(value, "L01"),
                l00=_required(value, "L00"),
            )
        )
    return tuple(output)


def _parse_stratum(row: dict[str, Any]) -> ExpectationStratum:
    value = _required(row, "expectation_stratum")
    if not isinstance(value, dict):
        raise ValueError("expectation_stratum must be an object")
    return ExpectationStratum(
        generator_id=str(_required(value, "generator_id")),
        task_family=str(_required(value, "task_family")),
        turn_depth_bucket=int(_required(value, "turn_depth_bucket")),
        window_position_bucket=int(_required(value, "window_position_bucket")),
        scoring_mode=str(_required(value, "scoring_mode")),
        scoring_model_id=str(_required(value, "scoring_model_id")),
    )


def _row_key(
    session_order: int,
    session_id: str,
    turn_index: int,
    window_index: int,
    stratum: ExpectationStratum,
) -> tuple[int, str, int, int, str]:
    return session_order, session_id, turn_index, window_index, stratum.stable_id()


def run_factorial_pipeline(
    rows: Iterable[dict[str, Any]],
    *,
    config: AblationConfig,
    estimator_id: str,
    min_context_sessions: int,
) -> FactorialPipelineResult:
    """Evaluate calibration and confirmatory rows without kernel mutation."""

    parsed: list[dict[str, Any]] = []
    seen_windows: set[tuple[str, int, int]] = set()
    for source in rows:
        partition = str(_required(source, "partition")).strip().lower()
        if partition not in {"calibration", "confirmatory"}:
            raise ValueError("partition must be calibration or confirmatory")
        session_order = int(_required(source, "session_order"))
        if session_order < 0:
            raise ValueError("session_order must be non-negative")
        metadata = _parse_metadata(source)
        unique_window = (
            metadata.session_id,
            metadata.turn_index,
            metadata.window_index,
        )
        if unique_window in seen_windows:
            raise ValueError(f"duplicate window identity: {unique_window}")
        seen_windows.add(unique_window)
        stratum = _parse_stratum(source)
        record = evaluate_window(metadata, _parse_conditions(source), config)
        parsed.append(
            {
                "partition": partition,
                "session_order": session_order,
                "stratum": stratum,
                "record": record,
            }
        )

    if not parsed:
        raise ValueError("the factorial pipeline requires at least one input row")
    calibration = [item for item in parsed if item["partition"] == "calibration"]
    confirmatory = [item for item in parsed if item["partition"] == "confirmatory"]
    if not calibration or not confirmatory:
        raise ValueError("both calibration and confirmatory partitions are required")
    if max(item["session_order"] for item in calibration) >= min(
        item["session_order"] for item in confirmatory
    ):
        raise ValueError(
            "the complete calibration partition must precede the confirmatory partition"
        )

    calibration_observations = []
    for item in calibration:
        record = item["record"]
        if record.eligible and record.omega is not None:
            calibration_observations.append(
                CalibrationObservation(
                    session_order=item["session_order"],
                    session_id=record.metadata.session_id,
                    window_index=record.metadata.window_index,
                    stratum=item["stratum"],
                    omega=record.omega,
                    turn_index=record.metadata.turn_index,
                )
            )
    if not calibration_observations:
        raise ValueError("no eligible calibration windows remain")

    trace = causal_calibration_trace(
        calibration_observations,
        min_context_sessions=min_context_sessions,
    )
    trace_by_key = {
        _row_key(
            point.observation.session_order,
            point.observation.session_id,
            point.observation.turn_index,
            point.observation.window_index,
            point.observation.stratum,
        ): point
        for point in trace
    }
    estimator = FrozenExpectation.fit(
        calibration_observations,
        estimator_id=estimator_id,
        min_context_sessions=min_context_sessions,
        calibration_max_session_order=max(
            item["session_order"] for item in calibration
        ),
    )

    observations: list[dict[str, Any]] = []
    kernel_inputs: list[dict[str, Any]] = []
    for item in parsed:
        record = item["record"]
        stratum = item["stratum"]
        output = record.to_dict()
        output.update(
            {
                "schema": PIPELINE_SCHEMA,
                "partition": item["partition"],
                "session_order": item["session_order"],
                "expectation_stratum": stratum.stable_id(),
                "estimator_id": estimator.estimator_id,
                "statistics_sha256": estimator.statistics_sha256,
            }
        )
        if item["partition"] == "calibration":
            if not record.eligible:
                output.update(
                    {
                        "expected_omega": None,
                        "expectation_status": ExpectationStatus.INELIGIBLE.value,
                        "calibration_session_count": 0,
                        "calibration_window_count": 0,
                    }
                )
            else:
                point = trace_by_key[
                    _row_key(
                        item["session_order"],
                        record.metadata.session_id,
                        record.metadata.turn_index,
                        record.metadata.window_index,
                        stratum,
                    )
                ]
                output.update(
                    {
                        "expected_omega": point.expected_omega,
                        "expectation_status": point.status,
                        "calibration_session_count": point.prior_session_count,
                        "calibration_window_count": point.prior_window_count,
                    }
                )
        else:
            attached = estimator.attach(
                record,
                stratum,
                confirmatory_session_order=item["session_order"],
            )
            output.update(
                {
                    "expected_omega": attached.expected_omega,
                    "expectation_status": attached.expectation_status,
                    "calibration_session_count": attached.calibration_session_count,
                    "calibration_window_count": attached.calibration_window_count,
                }
            )
            if attached.pass_to_kernel:
                omega, expected_omega = attached.kernel_input() or (math.nan, math.nan)
                kernel_inputs.append(
                    {
                        "schema": PIPELINE_SCHEMA,
                        "session_id": record.metadata.session_id,
                        "turn_index": record.metadata.turn_index,
                        "window_index": record.metadata.window_index,
                        "omega": omega,
                        "expected_omega": expected_omega,
                        "omega_dep": omega,
                        "expected_omega_dep": expected_omega,
                        "self_dependence_excess": omega - expected_omega,
                        "expectation_stratum": stratum.stable_id(),
                        "estimator_id": estimator.estimator_id,
                        "statistics_sha256": estimator.statistics_sha256,
                    }
                )
        # Canonical 0.2 field names.  Legacy aliases remain for readers of the
        # already emitted factorial-ablation/1 artifacts.
        output["omega_dep"] = output["omega"]
        output["expected_omega_dep"] = output["expected_omega"]
        output["self_dependence_excess"] = (
            None
            if output["omega"] is None or output["expected_omega"] is None
            else output["omega"] - output["expected_omega"]
        )
        observations.append(output)

    return FactorialPipelineResult(
        observations=tuple(observations),
        kernel_inputs=tuple(kernel_inputs),
        estimator=estimator,
    )
