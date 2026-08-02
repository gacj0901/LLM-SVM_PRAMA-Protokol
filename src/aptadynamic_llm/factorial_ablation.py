"""Parallel factorial-ablation Observation Interface for recorded LLM sessions.

This module is intentionally independent from :mod:`aptadynamic_llm.omega`.
It implements the model-agnostic arithmetic and validity boundary of the
factorial session-dependence operator.  Rendering and teacher-forced model
scoring live behind the contracts in ``factorial_scoring.py``.

Implementation decisions that remove ambiguities in the 0.2 draft:

* ``turn_index`` is zero-based; the default first eligible assistant turn is
  index 1 (the second assistant turn).
* the primary ensemble estimates are means of the per-filler normalized
  coordinates, not ratios reconstructed from mean supports;
* support magnitude is the mean per-filler magnitude;
* filler variance is the unbiased sample variance (``ddof=1``);
* the natural ``L11`` scores must be identical across the filler ensemble;
* a positive self-dominance candidate additionally requires a positive
  intact-context self effect, ``L11 - L10``.

No function in this module calls or mutates the PRAMA kernel.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
import json
import math
from statistics import fmean, variance
from typing import Iterable, Sequence


SPECIFICATION_VERSION = "0.2.0"
IMPLEMENTATION_CONTRACT = "factorial-ablation/1"


class ExclusionReason(str, Enum):
    """Machine-readable reasons for rejecting a window before kernel mutation."""

    ASSISTANT_TURN_TOO_EARLY = "assistant_turn_too_early"
    UNNATURAL_ASSISTANT_REGION = "unnatural_assistant_region"
    TOKEN_ALIGNMENT_FAILURE = "token_alignment_failure"
    GENERATION_TRUNCATED = "generation_truncated"
    UNMODELED_CONTENT = "unmodeled_system_or_tool_content"
    INSUFFICIENT_FILLERS = "insufficient_fillers"
    DUPLICATE_FILLER_ID = "duplicate_filler_id"
    MISSING_SCORE = "missing_or_nonfinite_score"
    INCONSISTENT_NATURAL_SCORE = "inconsistent_l11_across_fillers"
    LOW_SUPPORT_MAGNITUDE = "support_magnitude_below_minimum"
    HIGH_FILLER_SENSITIVITY = "filler_sensitivity_above_maximum"


@dataclass(frozen=True)
class AblationConfig:
    """Frozen validity and numerical parameters for one study."""

    min_support_magnitude: float
    max_filler_variance: float
    epsilon: float = 1e-12
    min_fillers: int = 3
    min_assistant_turn_index: int = 1
    natural_score_tolerance: float = 0.0

    def __post_init__(self) -> None:
        if self.min_support_magnitude < 0:
            raise ValueError("min_support_magnitude must be non-negative")
        if self.max_filler_variance < 0:
            raise ValueError("max_filler_variance must be non-negative")
        if self.epsilon <= 0 or not math.isfinite(self.epsilon):
            raise ValueError("epsilon must be positive and finite")
        if self.min_fillers < 3:
            raise ValueError("the factorial protocol requires at least three fillers")
        if self.min_assistant_turn_index < 1:
            raise ValueError("an eligible turn must have prior assistant history")
        if self.natural_score_tolerance < 0:
            raise ValueError("natural_score_tolerance must be non-negative")


@dataclass(frozen=True)
class WindowMetadata:
    """Observation identity and pre-scoring eligibility facts.

    ``turn_index`` is zero-based. ``token_end`` is exclusive.
    """

    session_id: str
    turn_index: int
    window_index: int
    token_start: int
    token_end: int
    natural_assistant_region: bool = True
    exactly_aligned: bool = True
    generation_truncated: bool = False
    contains_unmodeled_content: bool = False

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id cannot be empty")
        if self.turn_index < 0 or self.window_index < 0:
            raise ValueError("turn_index and window_index must be non-negative")
        if self.token_start < 0 or self.token_end <= self.token_start:
            raise ValueError("token range must be non-empty and increasing")

    @property
    def token_count(self) -> int:
        return self.token_end - self.token_start

    @property
    def assistant_turn_number(self) -> int:
        """Human-facing one-based assistant-turn number."""

        return self.turn_index + 1


def _float_tuple(values: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


@dataclass(frozen=True)
class FactorialConditionScores:
    """Token log-likelihoods for one coupled filler realization."""

    filler_id: str
    l11: tuple[float, ...]
    l10: tuple[float, ...]
    l01: tuple[float, ...]
    l00: tuple[float, ...]

    def __init__(
        self,
        filler_id: str,
        l11: Sequence[float],
        l10: Sequence[float],
        l01: Sequence[float],
        l00: Sequence[float],
    ) -> None:
        object.__setattr__(self, "filler_id", str(filler_id))
        object.__setattr__(self, "l11", _float_tuple(l11))
        object.__setattr__(self, "l10", _float_tuple(l10))
        object.__setattr__(self, "l01", _float_tuple(l01))
        object.__setattr__(self, "l00", _float_tuple(l00))


@dataclass(frozen=True)
class FillerWindowResult:
    """Factorial decomposition for one filler and one token window."""

    filler_id: str
    l11_sum: float
    l10_sum: float
    l01_sum: float
    l00_sum: float
    self_support: float
    user_support: float
    interaction: float
    intact_self_effect: float
    intact_user_effect: float
    support_magnitude: float
    omega: float
    interaction_ratio: float


@dataclass(frozen=True)
class AblationWindowRecord:
    """Complete pre-kernel record for one evaluated window."""

    metadata: WindowMetadata
    filler_results: tuple[FillerWindowResult, ...]
    self_support: float | None
    user_support: float | None
    interaction: float | None
    intact_self_effect: float | None
    intact_user_effect: float | None
    support_magnitude: float | None
    omega: float | None
    interaction_ratio: float | None
    filler_variance: float | None
    eligible: bool
    exclusion_reasons: tuple[str, ...]
    contract: str = IMPLEMENTATION_CONTRACT
    specification_version: str = SPECIFICATION_VERSION

    @property
    def exclusion_reason(self) -> str | None:
        return ",".join(self.exclusion_reasons) if self.exclusion_reasons else None

    def positive_self_dominance_candidate(
        self,
        omega_threshold: float,
        min_positive_intact_self_effect: float = 0.0,
    ) -> bool:
        """Return the component-level self-dominance gate, never an echo verdict."""

        return bool(
            self.eligible
            and self.omega is not None
            and self.intact_self_effect is not None
            and self.omega > omega_threshold
            and self.intact_self_effect > min_positive_intact_self_effect
        )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["session_id"] = self.metadata.session_id
        payload["turn_index"] = self.metadata.turn_index
        payload["assistant_turn_number"] = self.metadata.assistant_turn_number
        payload["window_index"] = self.metadata.window_index
        payload["token_start"] = self.metadata.token_start
        payload["token_end"] = self.metadata.token_end
        payload["filler_count"] = len(self.filler_results)
        payload["exclusion_reason"] = self.exclusion_reason
        return payload

    def deterministic_hash(self) -> str:
        encoded = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return sha256(encoded).hexdigest()


def _append_once(reasons: list[str], reason: ExclusionReason) -> None:
    if reason.value not in reasons:
        reasons.append(reason.value)


def _all_finite(condition: FactorialConditionScores) -> bool:
    return all(
        math.isfinite(value)
        for values in (condition.l11, condition.l10, condition.l01, condition.l00)
        for value in values
    )


def _same_natural_scores(
    left: Sequence[float],
    right: Sequence[float],
    tolerance: float,
) -> bool:
    return len(left) == len(right) and all(
        math.isclose(a, b, rel_tol=0.0, abs_tol=tolerance)
        for a, b in zip(left, right)
    )


def decompose_filler_window(
    condition: FactorialConditionScores,
    epsilon: float,
) -> FillerWindowResult:
    """Apply the symmetric 2x2 factorial decomposition after token summation."""

    lengths = {len(condition.l11), len(condition.l10), len(condition.l01), len(condition.l00)}
    if lengths == {0} or len(lengths) != 1:
        raise ValueError("all four condition arrays must have the same non-zero length")
    if not _all_finite(condition):
        raise ValueError("all condition log-likelihoods must be finite")

    l11_sum = math.fsum(condition.l11)
    l10_sum = math.fsum(condition.l10)
    l01_sum = math.fsum(condition.l01)
    l00_sum = math.fsum(condition.l00)
    self_support = 0.5 * ((l01_sum - l00_sum) + (l11_sum - l10_sum))
    user_support = 0.5 * ((l10_sum - l00_sum) + (l11_sum - l01_sum))
    interaction = l11_sum - l10_sum - l01_sum + l00_sum
    intact_self_effect = l11_sum - l10_sum
    intact_user_effect = l11_sum - l01_sum
    support_magnitude = abs(self_support) + abs(user_support)
    omega = (self_support - user_support) / (support_magnitude + epsilon)
    interaction_ratio = interaction / (
        abs(self_support) + abs(user_support) + abs(interaction) + epsilon
    )
    return FillerWindowResult(
        filler_id=condition.filler_id,
        l11_sum=l11_sum,
        l10_sum=l10_sum,
        l01_sum=l01_sum,
        l00_sum=l00_sum,
        self_support=self_support,
        user_support=user_support,
        interaction=interaction,
        intact_self_effect=intact_self_effect,
        intact_user_effect=intact_user_effect,
        support_magnitude=support_magnitude,
        omega=omega,
        interaction_ratio=interaction_ratio,
    )


def evaluate_window(
    metadata: WindowMetadata,
    conditions: Iterable[FactorialConditionScores],
    config: AblationConfig,
) -> AblationWindowRecord:
    """Evaluate one window and apply every exclusion before any kernel call."""

    ensemble = tuple(conditions)
    reasons: list[str] = []

    if metadata.turn_index < config.min_assistant_turn_index:
        _append_once(reasons, ExclusionReason.ASSISTANT_TURN_TOO_EARLY)
    if not metadata.natural_assistant_region:
        _append_once(reasons, ExclusionReason.UNNATURAL_ASSISTANT_REGION)
    if not metadata.exactly_aligned:
        _append_once(reasons, ExclusionReason.TOKEN_ALIGNMENT_FAILURE)
    if metadata.generation_truncated:
        _append_once(reasons, ExclusionReason.GENERATION_TRUNCATED)
    if metadata.contains_unmodeled_content:
        _append_once(reasons, ExclusionReason.UNMODELED_CONTENT)
    if len(ensemble) < config.min_fillers:
        _append_once(reasons, ExclusionReason.INSUFFICIENT_FILLERS)

    filler_ids = [condition.filler_id for condition in ensemble]
    if any(not filler_id for filler_id in filler_ids) or len(filler_ids) != len(set(filler_ids)):
        _append_once(reasons, ExclusionReason.DUPLICATE_FILLER_ID)

    structurally_valid = True
    for condition in ensemble:
        lengths = {
            len(condition.l11),
            len(condition.l10),
            len(condition.l01),
            len(condition.l00),
        }
        if lengths != {metadata.token_count}:
            _append_once(reasons, ExclusionReason.TOKEN_ALIGNMENT_FAILURE)
            structurally_valid = False
        if not _all_finite(condition):
            _append_once(reasons, ExclusionReason.MISSING_SCORE)
            structurally_valid = False

    if ensemble:
        natural = ensemble[0].l11
        if any(
            not _same_natural_scores(natural, condition.l11, config.natural_score_tolerance)
            for condition in ensemble[1:]
        ):
            _append_once(reasons, ExclusionReason.INCONSISTENT_NATURAL_SCORE)
            structurally_valid = False

    results: tuple[FillerWindowResult, ...] = ()
    if structurally_valid and ensemble:
        results = tuple(
            decompose_filler_window(condition, epsilon=config.epsilon)
            for condition in ensemble
        )

    if not results:
        return AblationWindowRecord(
            metadata=metadata,
            filler_results=(),
            self_support=None,
            user_support=None,
            interaction=None,
            intact_self_effect=None,
            intact_user_effect=None,
            support_magnitude=None,
            omega=None,
            interaction_ratio=None,
            filler_variance=None,
            eligible=False,
            exclusion_reasons=tuple(reasons),
        )

    self_support = fmean(result.self_support for result in results)
    user_support = fmean(result.user_support for result in results)
    interaction = fmean(result.interaction for result in results)
    intact_self_effect = fmean(result.intact_self_effect for result in results)
    intact_user_effect = fmean(result.intact_user_effect for result in results)
    support_magnitude = fmean(result.support_magnitude for result in results)
    omega = fmean(result.omega for result in results)
    interaction_ratio = fmean(result.interaction_ratio for result in results)
    filler_variance = variance(result.omega for result in results)

    if support_magnitude < config.min_support_magnitude:
        _append_once(reasons, ExclusionReason.LOW_SUPPORT_MAGNITUDE)
    if filler_variance > config.max_filler_variance:
        _append_once(reasons, ExclusionReason.HIGH_FILLER_SENSITIVITY)

    return AblationWindowRecord(
        metadata=metadata,
        filler_results=results,
        self_support=self_support,
        user_support=user_support,
        interaction=interaction,
        intact_self_effect=intact_self_effect,
        intact_user_effect=intact_user_effect,
        support_magnitude=support_magnitude,
        omega=omega,
        interaction_ratio=interaction_ratio,
        filler_variance=filler_variance,
        eligible=not reasons,
        exclusion_reasons=tuple(reasons),
    )
