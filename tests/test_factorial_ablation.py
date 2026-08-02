import math

import pytest

from aptadynamic_llm.factorial_ablation import (
    AblationConfig,
    ExclusionReason,
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
from aptadynamic_llm.factorial_pipeline import run_factorial_pipeline


def _metadata(**overrides):
    values = {
        "session_id": "session-1",
        "turn_index": 1,
        "window_index": 0,
        "token_start": 0,
        "token_end": 1,
    }
    values.update(overrides)
    return WindowMetadata(**values)


def _config(**overrides):
    values = {
        "min_support_magnitude": 0.01,
        "max_filler_variance": 2.0,
    }
    values.update(overrides)
    return AblationConfig(**values)


def _condition(filler_id, *, self_support, user_support, l11=-1.0):
    # Additive cell construction with a shared natural L11 reference.
    l00 = l11 - self_support - user_support
    l10 = l11 - self_support
    l01 = l11 - user_support
    return FactorialConditionScores(
        filler_id,
        l11=[l11],
        l10=[l10],
        l01=[l01],
        l00=[l00],
    )


def _ensemble(*, self_support, user_support):
    return [
        _condition(
            f"filler-{index}",
            self_support=self_support,
            user_support=user_support,
        )
        for index in range(3)
    ]


def _eligible_record(omega_kind="self"):
    if omega_kind == "self":
        conditions = _ensemble(self_support=2.0, user_support=0.0)
    else:
        conditions = _ensemble(self_support=0.0, user_support=2.0)
    return evaluate_window(_metadata(), conditions, _config())


def test_user_only_and_self_only_signatures():
    user = evaluate_window(
        _metadata(),
        _ensemble(self_support=0.0, user_support=2.0),
        _config(),
    )
    self_history = evaluate_window(
        _metadata(),
        _ensemble(self_support=2.0, user_support=0.0),
        _config(),
    )

    assert user.eligible
    assert user.self_support == pytest.approx(0.0)
    assert user.user_support == pytest.approx(2.0)
    assert user.omega == pytest.approx(-1.0, abs=1e-10)
    assert self_history.eligible
    assert self_history.self_support == pytest.approx(2.0)
    assert self_history.user_support == pytest.approx(0.0)
    assert self_history.omega == pytest.approx(1.0, abs=1e-10)


def test_pure_interaction_has_zero_main_support_and_is_indeterminate():
    conditions = [
        FactorialConditionScores(
            f"filler-{index}",
            l11=[-3.0],
            l10=[-5.0],
            l01=[-5.0],
            l00=[-3.0],
        )
        for index in range(3)
    ]
    record = evaluate_window(_metadata(), conditions, _config())

    assert not record.eligible
    assert record.self_support == pytest.approx(0.0)
    assert record.user_support == pytest.approx(0.0)
    assert record.interaction == pytest.approx(4.0)
    assert ExclusionReason.LOW_SUPPORT_MAGNITUDE.value in record.exclusion_reasons


def test_signed_contrast_does_not_alone_establish_positive_self_support():
    conditions = _ensemble(self_support=-0.1, user_support=-10.0)
    record = evaluate_window(_metadata(), conditions, _config())

    assert record.eligible
    assert record.omega > 0.98
    assert record.intact_self_effect == pytest.approx(-0.1)
    assert not record.positive_self_dominance_candidate(omega_threshold=0.8)


def test_primary_omega_is_mean_of_per_filler_coordinates():
    conditions = [
        _condition("filler-a", self_support=9.0, user_support=1.0),
        _condition("filler-b", self_support=0.0, user_support=1.0),
        _condition("filler-c", self_support=0.0, user_support=1.0),
    ]
    record = evaluate_window(_metadata(), conditions, _config())

    assert record.eligible
    assert record.omega == pytest.approx(-0.4, abs=1e-10)
    ratio_from_mean_supports = (
        record.self_support - record.user_support
    ) / (abs(record.self_support) + abs(record.user_support))
    assert ratio_from_mean_supports == pytest.approx(0.5)
    assert record.filler_variance == pytest.approx(1.08, abs=1e-9)


def test_invalid_windows_are_returned_but_never_kernel_eligible():
    conditions = _ensemble(self_support=2.0, user_support=0.0)
    early = evaluate_window(_metadata(turn_index=0), conditions, _config())
    truncated = evaluate_window(
        _metadata(generation_truncated=True), conditions, _config()
    )
    nonfinite = list(conditions)
    nonfinite[1] = FactorialConditionScores(
        "filler-1", l11=[-1.0], l10=[math.nan], l01=[-1.0], l00=[-1.0]
    )
    missing = evaluate_window(_metadata(), nonfinite, _config())

    assert not early.eligible
    assert ExclusionReason.ASSISTANT_TURN_TOO_EARLY.value in early.exclusion_reasons
    assert not truncated.eligible
    assert ExclusionReason.GENERATION_TRUNCATED.value in truncated.exclusion_reasons
    assert not missing.eligible
    assert ExclusionReason.MISSING_SCORE.value in missing.exclusion_reasons


def test_natural_reference_must_be_shared_across_fillers():
    conditions = _ensemble(self_support=2.0, user_support=0.0)
    conditions[2] = FactorialConditionScores(
        "filler-2", l11=[-1.1], l10=[-3.0], l01=[-1.0], l00=[-3.0]
    )
    record = evaluate_window(_metadata(), conditions, _config())

    assert not record.eligible
    assert ExclusionReason.INCONSISTENT_NATURAL_SCORE.value in record.exclusion_reasons


def _stratum():
    return ExpectationStratum(
        generator_id="generator@1",
        task_family="structured_analysis",
        turn_depth_bucket=2,
        window_position_bucket=0,
        scoring_mode="observer-relative",
        scoring_model_id="observer@1",
    )


def test_causal_trace_updates_only_after_the_complete_session():
    stratum = _stratum()
    observations = [
        CalibrationObservation(0, "s0", 0, stratum, 0.1),
        CalibrationObservation(0, "s0", 1, stratum, 0.3),
        CalibrationObservation(1, "s1", 0, stratum, 0.5),
        CalibrationObservation(2, "s2", 0, stratum, 0.7),
    ]
    trace = causal_calibration_trace(observations, min_context_sessions=2)

    assert trace[0].expected_omega is None
    assert trace[1].expected_omega is None
    assert trace[1].prior_window_count == 0
    assert trace[2].expected_omega is None
    assert trace[2].prior_session_count == 1
    assert trace[3].status == ExpectationStatus.OBSERVED.value
    assert trace[3].expected_omega == pytest.approx(0.3)
    assert trace[3].prior_session_count == 2


def test_frozen_expectation_has_temporal_and_warmup_gates():
    stratum = _stratum()
    observations = [
        CalibrationObservation(0, "s0", 0, stratum, 0.2),
        CalibrationObservation(1, "s1", 0, stratum, 0.4),
    ]
    estimator = FrozenExpectation.fit(
        observations,
        estimator_id="frozen-v1",
        min_context_sessions=2,
    )
    attached = estimator.attach(
        _eligible_record(),
        stratum,
        confirmatory_session_order=2,
    )

    assert attached.pass_to_kernel
    assert attached.kernel_input() == pytest.approx((1.0, 0.3), abs=1e-10)
    assert len(attached.statistics_sha256) == 64
    with pytest.raises(ValueError, match="after the complete calibration"):
        estimator.attach(
            _eligible_record(),
            stratum,
            confirmatory_session_order=1,
        )


def test_ineligible_record_never_passes_even_with_frozen_expectation():
    stratum = _stratum()
    estimator = FrozenExpectation.fit(
        [
            CalibrationObservation(0, "s0", 0, stratum, 0.2),
            CalibrationObservation(1, "s1", 0, stratum, 0.4),
        ],
        estimator_id="frozen-v1",
        min_context_sessions=2,
    )
    record = evaluate_window(
        _metadata(turn_index=0),
        _ensemble(self_support=2.0, user_support=0.0),
        _config(),
    )
    attached = estimator.attach(record, stratum, confirmatory_session_order=2)

    assert attached.expectation_status == ExpectationStatus.INELIGIBLE.value
    assert not attached.pass_to_kernel
    assert attached.kernel_input() is None


def _pipeline_row(session_id, session_order, partition, self_support, user_support):
    conditions = _ensemble(
        self_support=self_support,
        user_support=user_support,
    )
    return {
        "session_id": session_id,
        "session_order": session_order,
        "partition": partition,
        "turn_index": 1,
        "window_index": 0,
        "token_start": 0,
        "token_end": 1,
        "expectation_stratum": {
            "generator_id": "generator@1",
            "task_family": "structured_analysis",
            "turn_depth_bucket": 2,
            "window_position_bucket": 0,
            "scoring_mode": "observer-relative",
            "scoring_model_id": "observer@1",
        },
        "conditions": [
            {
                "filler_id": item.filler_id,
                "L11": list(item.l11),
                "L10": list(item.l10),
                "L01": list(item.l01),
                "L00": list(item.l00),
            }
            for item in conditions
        ],
    }


def test_parallel_pipeline_emits_only_confirmatory_kernel_ready_rows():
    result = run_factorial_pipeline(
        [
            _pipeline_row("cal-0", 0, "calibration", 1.0, 0.0),
            _pipeline_row("cal-1", 1, "calibration", 0.0, 1.0),
            _pipeline_row("test-0", 2, "confirmatory", 2.0, 0.0),
        ],
        config=_config(),
        estimator_id="factorial-frozen-v1",
        min_context_sessions=2,
    )

    assert len(result.observations) == 3
    assert len(result.kernel_inputs) == 1
    assert result.kernel_inputs[0]["session_id"] == "test-0"
    assert result.kernel_inputs[0]["expected_omega"] == pytest.approx(0.0, abs=1e-10)


def test_parallel_pipeline_rejects_temporal_partition_overlap():
    with pytest.raises(ValueError, match="complete calibration partition"):
        run_factorial_pipeline(
            [
                _pipeline_row("cal-0", 0, "calibration", 1.0, 0.0),
                _pipeline_row("cal-1", 2, "calibration", 0.0, 1.0),
                _pipeline_row("test-0", 1, "confirmatory", 2.0, 0.0),
            ],
            config=_config(),
            estimator_id="factorial-frozen-v1",
            min_context_sessions=2,
        )
