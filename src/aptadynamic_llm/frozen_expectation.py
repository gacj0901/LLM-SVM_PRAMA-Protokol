"""Strictly causal calibration and frozen expectation for factorial ablation.

Calibration traces are emitted before a session updates the sufficient
statistics.  Confirmatory estimates are then frozen and reject temporal
overlap with the calibration partition when session ordering is supplied.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from hashlib import sha256
import json
import math
from statistics import fmean
from types import MappingProxyType
from typing import Iterable, Mapping

from aptadynamic_llm.factorial_ablation import AblationWindowRecord


DEFAULT_TURN_DEPTH_EDGES = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)


def bucket_lower_bound(value: int, edges: tuple[int, ...]) -> int:
    """Return the greatest frozen edge not exceeding ``value``."""

    if not edges or tuple(sorted(set(edges))) != edges:
        raise ValueError("bucket edges must be non-empty, unique, and increasing")
    if value < edges[0]:
        raise ValueError(f"value {value} is below the first bucket edge {edges[0]}")
    lower = edges[0]
    for edge in edges[1:]:
        if value < edge:
            break
        lower = edge
    return lower


@dataclass(frozen=True, order=True)
class ExpectationStratum:
    """Frozen context used to estimate expected factorial dependence."""

    generator_id: str
    task_family: str
    turn_depth_bucket: int
    window_position_bucket: int
    scoring_mode: str
    scoring_model_id: str

    def __post_init__(self) -> None:
        required = {
            "generator_id": self.generator_id,
            "task_family": self.task_family,
            "scoring_mode": self.scoring_mode,
            "scoring_model_id": self.scoring_model_id,
        }
        empty = [name for name, value in required.items() if not str(value).strip()]
        if empty:
            raise ValueError(f"expectation stratum has empty fields: {empty}")
        if self.scoring_mode not in {"generator-relative", "observer-relative"}:
            raise ValueError("scoring_mode must be generator-relative or observer-relative")
        if self.turn_depth_bucket < 2:
            raise ValueError("turn_depth_bucket must represent an eligible assistant turn")
        if self.window_position_bucket < 0:
            raise ValueError("window_position_bucket must be non-negative")

    def stable_id(self) -> str:
        values = (
            self.generator_id,
            self.task_family,
            str(self.turn_depth_bucket),
            str(self.window_position_bucket),
            self.scoring_mode,
            self.scoring_model_id,
        )
        return "|".join(values)


@dataclass(frozen=True)
class CalibrationObservation:
    """One eligible calibration window with a globally ordered source session."""

    session_order: int
    session_id: str
    window_index: int
    stratum: ExpectationStratum
    omega: float
    turn_index: int = 0

    def __post_init__(self) -> None:
        if self.session_order < 0:
            raise ValueError("session_order must be non-negative")
        if not self.session_id:
            raise ValueError("session_id cannot be empty")
        if self.window_index < 0:
            raise ValueError("window_index must be non-negative")
        if self.turn_index < 0:
            raise ValueError("turn_index must be non-negative")
        if not math.isfinite(self.omega):
            raise ValueError("calibration omega must be finite")


class ExpectationStatus(str, Enum):
    OBSERVED = "observed"
    INELIGIBLE = "ineligible"
    UNKNOWN_STRATUM = "unknown_stratum"
    INSUFFICIENT_CONTEXT = "insufficient_context"


@dataclass(frozen=True)
class CausalExpectationPoint:
    """Expectation available immediately before a calibration window's session."""

    observation: CalibrationObservation
    expected_omega: float | None
    status: str
    prior_session_count: int
    prior_window_count: int


def _validate_session_order(observations: tuple[CalibrationObservation, ...]) -> None:
    session_orders: dict[str, int] = {}
    order_sessions: dict[int, str] = {}
    for observation in observations:
        previous_order = session_orders.setdefault(
            observation.session_id, observation.session_order
        )
        if previous_order != observation.session_order:
            raise ValueError("one session_id cannot occur at multiple session_order values")
        previous_session = order_sessions.setdefault(
            observation.session_order, observation.session_id
        )
        if previous_session != observation.session_id:
            raise ValueError("one session_order cannot identify multiple sessions")


def causal_calibration_trace(
    observations: Iterable[CalibrationObservation],
    min_context_sessions: int,
) -> tuple[CausalExpectationPoint, ...]:
    """Compute update-after-session expectations for calibration diagnostics."""

    if min_context_sessions < 1:
        raise ValueError("min_context_sessions must be positive")
    ordered = tuple(
        sorted(
            observations,
            key=lambda item: (
                item.session_order,
                item.turn_index,
                item.window_index,
                item.stratum.stable_id(),
            ),
        )
    )
    _validate_session_order(ordered)

    sums: dict[ExpectationStratum, float] = {}
    window_counts: dict[ExpectationStratum, int] = {}
    session_ids: dict[ExpectationStratum, set[str]] = {}
    output: list[CausalExpectationPoint] = []

    by_order: dict[int, list[CalibrationObservation]] = {}
    for observation in ordered:
        by_order.setdefault(observation.session_order, []).append(observation)

    for session_order in sorted(by_order):
        session_observations = by_order[session_order]
        for observation in session_observations:
            stratum = observation.stratum
            prior_sessions = len(session_ids.get(stratum, set()))
            prior_windows = window_counts.get(stratum, 0)
            observed = prior_sessions >= min_context_sessions and prior_windows > 0
            output.append(
                CausalExpectationPoint(
                    observation=observation,
                    expected_omega=(
                        sums[stratum] / prior_windows if observed else None
                    ),
                    status=(
                        ExpectationStatus.OBSERVED.value
                        if observed
                        else ExpectationStatus.INSUFFICIENT_CONTEXT.value
                    ),
                    prior_session_count=prior_sessions,
                    prior_window_count=prior_windows,
                )
            )

        # Update only after every window in the current session was emitted.
        for observation in session_observations:
            stratum = observation.stratum
            sums[stratum] = sums.get(stratum, 0.0) + observation.omega
            window_counts[stratum] = window_counts.get(stratum, 0) + 1
            session_ids.setdefault(stratum, set()).add(observation.session_id)

    return tuple(output)


@dataclass(frozen=True)
class FrozenStatistic:
    stratum: ExpectationStratum
    mean_omega: float
    session_count: int
    window_count: int


@dataclass(frozen=True)
class ExpectedAblationWindow:
    """A window joined to a frozen expectation and ready-state decision."""

    record: AblationWindowRecord
    expected_omega: float | None
    expectation_stratum: str
    expectation_status: str
    calibration_session_count: int
    calibration_window_count: int
    estimator_id: str
    statistics_sha256: str

    @property
    def pass_to_kernel(self) -> bool:
        return bool(
            self.record.eligible
            and self.record.omega is not None
            and self.expectation_status == ExpectationStatus.OBSERVED.value
            and self.expected_omega is not None
        )

    def kernel_input(self) -> tuple[float, float] | None:
        """Return ``(omega, expected_omega)`` only for a conforming row."""

        if not self.pass_to_kernel:
            return None
        assert self.record.omega is not None
        assert self.expected_omega is not None
        return self.record.omega, self.expected_omega


@dataclass(frozen=True)
class FrozenExpectation:
    """Immutable calibration statistics for confirmatory evaluation."""

    estimator_id: str
    min_context_sessions: int
    statistics: tuple[FrozenStatistic, ...]
    calibration_max_session_order: int
    statistics_sha256: str = field(init=False)
    _by_stratum: Mapping[ExpectationStratum, FrozenStatistic] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not self.estimator_id:
            raise ValueError("estimator_id cannot be empty")
        if self.min_context_sessions < 1:
            raise ValueError("min_context_sessions must be positive")
        ordered_statistics = tuple(
            sorted(self.statistics, key=lambda item: item.stratum)
        )
        object.__setattr__(self, "statistics", ordered_statistics)
        object.__setattr__(
            self,
            "_by_stratum",
            MappingProxyType({item.stratum: item for item in ordered_statistics}),
        )
        canonical = {
            "estimator_id": self.estimator_id,
            "min_context_sessions": self.min_context_sessions,
            "calibration_max_session_order": self.calibration_max_session_order,
            "statistics": [asdict(item) for item in self.statistics],
        }
        encoded = json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        object.__setattr__(self, "statistics_sha256", sha256(encoded).hexdigest())

    @classmethod
    def fit(
        cls,
        observations: Iterable[CalibrationObservation],
        *,
        estimator_id: str,
        min_context_sessions: int,
        calibration_max_session_order: int | None = None,
    ) -> "FrozenExpectation":
        rows = tuple(observations)
        if not rows:
            raise ValueError("at least one calibration observation is required")
        _validate_session_order(rows)
        grouped: dict[ExpectationStratum, list[CalibrationObservation]] = {}
        for observation in rows:
            grouped.setdefault(observation.stratum, []).append(observation)
        statistics = tuple(
            FrozenStatistic(
                stratum=stratum,
                mean_omega=fmean(row.omega for row in members),
                session_count=len({row.session_id for row in members}),
                window_count=len(members),
            )
            for stratum, members in grouped.items()
        )
        observed_max_order = max(row.session_order for row in rows)
        if (
            calibration_max_session_order is not None
            and calibration_max_session_order < observed_max_order
        ):
            raise ValueError(
                "calibration_max_session_order precedes an eligible calibration row"
            )
        return cls(
            estimator_id=estimator_id,
            min_context_sessions=min_context_sessions,
            statistics=statistics,
            calibration_max_session_order=(
                observed_max_order
                if calibration_max_session_order is None
                else calibration_max_session_order
            ),
        )

    def attach(
        self,
        record: AblationWindowRecord,
        stratum: ExpectationStratum,
        *,
        confirmatory_session_order: int | None = None,
    ) -> ExpectedAblationWindow:
        """Attach a frozen value without updating any estimator state."""

        if (
            confirmatory_session_order is not None
            and confirmatory_session_order <= self.calibration_max_session_order
        ):
            raise ValueError(
                "confirmatory session must occur after the complete calibration partition"
            )
        statistic = self._by_stratum.get(stratum)
        if not record.eligible:
            status = ExpectationStatus.INELIGIBLE
        elif statistic is None:
            status = ExpectationStatus.UNKNOWN_STRATUM
        elif statistic.session_count < self.min_context_sessions:
            status = ExpectationStatus.INSUFFICIENT_CONTEXT
        else:
            status = ExpectationStatus.OBSERVED
        return ExpectedAblationWindow(
            record=record,
            expected_omega=(
                statistic.mean_omega
                if statistic is not None and status is ExpectationStatus.OBSERVED
                else None
            ),
            expectation_stratum=stratum.stable_id(),
            expectation_status=status.value,
            calibration_session_count=statistic.session_count if statistic else 0,
            calibration_window_count=statistic.window_count if statistic else 0,
            estimator_id=self.estimator_id,
            statistics_sha256=self.statistics_sha256,
        )
