"""Causal post-kernel Structural Conversion Differential (ODCE-v0).

ODCE preserves structural cost, obtained return and their declared channel-wise
differential.  It is deliberately not a predictor and never mutates PRAMA or
D_O v9 state.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Mapping, Sequence

from aptadynamic_llm.artifact_schema import sha256_value, validate_artifact


OPERATOR_ID = "ODCE_v0"
OPERATOR_VERSION = "ODCE-v0.1.0"
STATUS_VALUES = {"OBSERVED", "NOT_APPLICABLE", "UNAVAILABLE", "INVALID"}
COST_CHANNELS = (
    "retained_friction",
    "accumulated_debt",
    "capacity_consumption",
    "excess_persistence",
    "adverse_trend",
)
BENEFIT_CHANNELS = (
    "structural_recovery",
    "adaptive_organization_level",
    "functional_gain",
    "external_integration",
    "verified_outcome",
)
FORBIDDEN_INPUT_FIELDS = {
    "finish_reason",
    "response_time_seconds",
    "future_outcome",
    "future_label",
}
_SHA256 = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")


@dataclass(frozen=True)
class NormalizationRule:
    location: float
    scale: float
    orientation: float = 1.0
    extreme_policy: str = "preserve"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "NormalizationRule":
        rule = cls(
            location=float(value["location"]),
            scale=float(value["scale"]),
            orientation=float(value.get("orientation", 1.0)),
            extreme_policy=str(value.get("extreme_policy", "preserve")),
        )
        if not math.isfinite(rule.location):
            raise ValueError("normalization location must be finite")
        if not math.isfinite(rule.scale) or rule.scale <= 0:
            raise ValueError("normalization scale must be finite and positive")
        if rule.orientation not in {-1.0, 1.0}:
            raise ValueError("normalization orientation must be +1 or -1")
        if rule.extreme_policy != "preserve":
            raise ValueError("ODCE-v0 only supports explicit extreme_policy='preserve'")
        return rule

    def apply(self, value: float) -> float:
        return self.orientation * (float(value) - self.location) / self.scale


@dataclass(frozen=True)
class Correspondence:
    name: str
    cost_channel: str
    benefit_channel: str


@dataclass(frozen=True)
class ODCEConfig:
    window_length: int
    friction_decay: float
    differential_threshold: float
    epsilon: float
    minimum_organization_support_windows: int
    cost_normalization: Mapping[str, NormalizationRule]
    benefit_normalization: Mapping[str, NormalizationRule]
    correspondence: tuple[Correspondence, ...]
    temporal_scope: Mapping[str, Any]

    @classmethod
    def from_contract(cls, contract: Mapping[str, Any]) -> "ODCEConfig":
        if contract.get("schema") != "LLM-SVM-ODCE-contract/0.1":
            raise ValueError("unsupported ODCE contract schema")
        status = contract.get("status")
        if status not in {
            "EXPLORATORY_CAUSAL_POST_KERNEL",
            "FROZEN_PROSPECTIVE",
        }:
            raise ValueError("ODCE contract has no valid execution status")
        if contract.get("operator_id") != OPERATOR_ID:
            raise ValueError("ODCE contract operator_id mismatch")
        if contract.get("operator_version") != OPERATOR_VERSION:
            raise ValueError("ODCE contract operator_version mismatch")
        window_length = int(contract["window_length"])
        friction_decay = float(contract["friction_decay"])
        threshold = float(contract["differential_threshold"])
        epsilon = float(contract.get("epsilon", 1e-12))
        if window_length < 2:
            raise ValueError("ODCE window_length must be at least 2")
        if not 0.0 < friction_decay <= 1.0:
            raise ValueError("friction_decay must be within (0, 1]")
        if threshold < 0 or epsilon <= 0:
            raise ValueError("differential_threshold and epsilon must be nonnegative")

        normalization = contract["normalization"]
        if not isinstance(normalization, Mapping):
            raise ValueError("ODCE normalization contract must be an object")
        threshold_calibration = contract.get("differential_threshold_calibration")
        if threshold == 0.0:
            if threshold_calibration is not None:
                raise ValueError(
                    "zero differential_threshold must not claim a noise-floor calibration"
                )
        else:
            required_threshold_calibration = {
                "schema",
                "status",
                "stable_condition_id",
                "input_artifacts_sha256",
                "artifact_count",
                "session_count",
                "partitions",
                "correspondence_names",
                "estimator",
                "residual_quantile",
                "min_observations",
                "derived_differential_threshold",
                "normalization_contract_sha256",
                "normalization_modified",
                "frozen",
            }
            if (
                not isinstance(threshold_calibration, Mapping)
                or set(threshold_calibration) != required_threshold_calibration
                or threshold_calibration["schema"]
                != "LLM-SVM-ODCE-differential-noise-floor-calibration/0.1"
                or threshold_calibration["status"]
                != "EXPLORATORY_EMPIRICAL_NOISE_FLOOR"
                or threshold_calibration["normalization_modified"] is not False
                or threshold_calibration["frozen"] is not False
                or not _is_sha256(
                    threshold_calibration["input_artifacts_sha256"]
                )
                or not _is_sha256(
                    threshold_calibration["normalization_contract_sha256"]
                )
                or threshold_calibration["normalization_contract_sha256"].removeprefix(
                    "sha256:"
                )
                != sha256_value(normalization)
                or not math.isclose(
                    float(
                        threshold_calibration[
                            "derived_differential_threshold"
                        ]
                    ),
                    threshold,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
            ):
                raise ValueError(
                    "nonzero differential_threshold requires its matching empirical noise-floor calibration"
                )
        if status == "EXPLORATORY_CAUSAL_POST_KERNEL":
            if normalization.get("confirmatory_use_allowed") is not False:
                raise ValueError(
                    "exploratory ODCE normalization must forbid confirmatory use"
                )
        else:
            if normalization.get("calibration_status") != "FROZEN_DOMAIN_CALIBRATION":
                raise ValueError(
                    "FROZEN_PROSPECTIVE requires FROZEN_DOMAIN_CALIBRATION"
                )
            if normalization.get("confirmatory_use_allowed") is not True:
                raise ValueError(
                    "frozen ODCE normalization must explicitly allow confirmatory use"
                )
            for field in (
                "calibration_reference_sha256",
                "calibration_population_sha256",
            ):
                if not _is_sha256(normalization.get(field)):
                    raise ValueError(f"frozen ODCE normalization requires {field}")
            governance = contract.get("correspondence_governance")
            if (
                not isinstance(governance, Mapping)
                or governance.get("status") != "FROZEN_PROSPECTIVE"
                or not _is_sha256(
                    governance.get("rationale_reference_sha256")
                )
            ):
                raise ValueError(
                    "frozen ODCE correspondence requires prospective governance"
                )
        cost_rules = {
            name: NormalizationRule.from_mapping(normalization["cost"][name])
            for name in COST_CHANNELS
        }
        benefit_rules = {
            name: NormalizationRule.from_mapping(normalization["benefit"][name])
            for name in BENEFIT_CHANNELS
        }
        correspondence = tuple(
            Correspondence(
                name=str(row["name"]),
                cost_channel=str(row["cost_channel"]),
                benefit_channel=str(row["benefit_channel"]),
            )
            for row in contract["correspondence"]
        )
        if not correspondence or len({row.name for row in correspondence}) != len(
            correspondence
        ):
            raise ValueError("ODCE correspondence names must be nonempty and unique")
        for row in correspondence:
            if row.cost_channel not in COST_CHANNELS:
                raise ValueError(f"unknown cost channel: {row.cost_channel}")
            if row.benefit_channel not in BENEFIT_CHANNELS:
                raise ValueError(f"unknown benefit channel: {row.benefit_channel}")
            expected_name = f"{row.cost_channel}_vs_{row.benefit_channel}"
            if row.name != expected_name:
                raise ValueError(
                    "ODCE correspondence names must identify the exact cost and "
                    f"benefit channels; expected {expected_name!r}"
                )
        structural_support = contract.get("structural_support")
        expected_structural_support = {
            "observer",
            "join_identity",
            "required_channels",
            "minimum_evaluable_windows",
            "missing_identity_policy",
            "insufficient_support_policy",
        }
        if (
            not isinstance(structural_support, Mapping)
            or set(structural_support) != expected_structural_support
            or structural_support["observer"] != "D_O_v9"
            or structural_support["join_identity"]
            != ["turn_index", "window_index"]
            or structural_support["required_channels"]
            != ["transport_coherence", "variation_contraction"]
            or structural_support["missing_identity_policy"] != "UNAVAILABLE"
            or structural_support["insufficient_support_policy"] != "UNAVAILABLE"
        ):
            raise ValueError("ODCE structural_support contract is noncanonical")
        minimum_organization_support_windows = int(
            structural_support["minimum_evaluable_windows"]
        )
        if minimum_organization_support_windows < 1:
            raise ValueError(
                "minimum_evaluable_windows must be a positive integer"
            )
        temporal_scope = contract.get("temporal_scope")
        expected_scope_keys = {"cost", "benefit", "differential", "dynamics"}
        if not isinstance(temporal_scope, Mapping) or set(temporal_scope) != expected_scope_keys:
            raise ValueError("ODCE temporal_scope contract is incomplete")
        for group, channels in (("cost", COST_CHANNELS), ("benefit", BENEFIT_CHANNELS)):
            scopes = temporal_scope[group]
            if not isinstance(scopes, Mapping) or set(scopes) != set(channels):
                raise ValueError(f"ODCE temporal_scope.{group} is noncanonical")
            if not all(isinstance(value, str) and value for value in scopes.values()):
                raise ValueError(f"ODCE temporal_scope.{group} values must be named scales")
        differential_scopes = temporal_scope["differential"]
        if (
            not isinstance(differential_scopes, Mapping)
            or set(differential_scopes) != {row.name for row in correspondence}
        ):
            raise ValueError("ODCE differential temporal scopes are noncanonical")
        if temporal_scope["dynamics"] != {
            "trend": "CURRENT_VS_PREVIOUS_OBSERVED_INDEX",
            "cumulative_conversion_deficit_exposure": "SESSION_TO_DATE",
            "positive_persistence": "ROLLING_WINDOW",
        }:
            raise ValueError("ODCE dynamics temporal scopes are noncanonical")
        accumulation = contract.get("accumulation_contract")
        if not isinstance(accumulation, Mapping) or accumulation != {
            "coordinate": "cumulative_conversion_deficit_exposure",
            "update": "C_t = C_(t-1) + max(differential_t, 0)",
            "scope": "SESSION_TO_DATE",
            "irreversible": True,
            "missing_update": "NO_INCREMENT",
        }:
            raise ValueError("ODCE irreversible accumulation contract mismatch")
        return cls(
            window_length=window_length,
            friction_decay=friction_decay,
            differential_threshold=threshold,
            epsilon=epsilon,
            minimum_organization_support_windows=(
                minimum_organization_support_windows
            ),
            cost_normalization=cost_rules,
            benefit_normalization=benefit_rules,
            correspondence=correspondence,
            temporal_scope=temporal_scope,
        )


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def validate_contract_freeze(
    contract: Mapping[str, Any], freeze: Mapping[str, Any]
) -> None:
    """Verify that a prospective freeze binds the exact ODCE contract surfaces."""

    ODCEConfig.from_contract(contract)
    if contract.get("status") != "FROZEN_PROSPECTIVE":
        raise ValueError("only a FROZEN_PROSPECTIVE ODCE contract can be frozen")
    required = {
        "schema",
        "operator_id",
        "operator_version",
        "contract_canonical_sha256",
        "normalization_contract_sha256",
        "correspondence_contract_sha256",
        "calibration_reference_sha256",
        "calibration_population_sha256",
        "correspondence_rationale_reference_sha256",
        "frozen_before_confirmatory_acquisition",
    }
    if not isinstance(freeze, Mapping) or set(freeze) != required:
        raise ValueError("ODCE contract freeze manifest is noncanonical")
    if freeze["schema"] != "LLM-SVM-ODCE-contract-freeze/0.1":
        raise ValueError("unsupported ODCE contract freeze schema")
    if freeze["operator_id"] != OPERATOR_ID or freeze["operator_version"] != OPERATOR_VERSION:
        raise ValueError("ODCE contract freeze operator mismatch")
    if freeze["frozen_before_confirmatory_acquisition"] is not True:
        raise ValueError("ODCE contract must be frozen before confirmatory acquisition")
    normalization = contract["normalization"]
    governance = contract["correspondence_governance"]
    expected_hashes = {
        "contract_canonical_sha256": sha256_value(contract),
        "normalization_contract_sha256": sha256_value(normalization),
        "correspondence_contract_sha256": sha256_value(
            list(contract["correspondence"])
        ),
        "calibration_reference_sha256": normalization[
            "calibration_reference_sha256"
        ].removeprefix("sha256:"),
        "calibration_population_sha256": normalization[
            "calibration_population_sha256"
        ].removeprefix("sha256:"),
        "correspondence_rationale_reference_sha256": governance[
            "rationale_reference_sha256"
        ].removeprefix("sha256:"),
    }
    for field, expected in expected_hashes.items():
        value = freeze[field]
        if not _is_sha256(value) or value.removeprefix("sha256:") != expected:
            raise ValueError(f"ODCE contract freeze {field} mismatch")


def _reject_forbidden(rows: Sequence[Mapping[str, Any]], source: str) -> None:
    for index, row in enumerate(rows):
        leaked = sorted(FORBIDDEN_INPUT_FIELDS & row.keys())
        if leaked:
            raise ValueError(f"{source}[{index}] contains forbidden fields: {leaked}")


def _window_identity(
    row: Mapping[str, Any], *, field: str | None = None
) -> tuple[int, int]:
    value = row if field is None else row.get(field)
    if not isinstance(value, Mapping):
        label = field or "row"
        raise ValueError(f"{label} requires a temporal window identity")
    try:
        turn_index = value["turn_index"]
        window_index = value["window_index"]
    except KeyError as exc:
        label = field or "row"
        raise ValueError(f"{label} requires turn_index and window_index") from exc
    for name, index in (("turn_index", turn_index), ("window_index", window_index)):
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError(f"{field + '.' if field else ''}{name} must be a nonnegative integer")
    return turn_index, window_index


def _ordinal_by_identity(
    prama_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[tuple[int, int], int], tuple[tuple[int, int], ...]]:
    identities = tuple(_window_identity(row) for row in prama_rows)
    if len(identities) != len(set(identities)):
        raise ValueError("PRAMA temporal window identities must be unique")
    return {identity: index for index, identity in enumerate(identities)}, identities


def _number(row: Mapping[str, Any], name: str) -> float | None:
    value = row.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric or null")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _weighted_mean(values: Sequence[float], decay: float) -> float:
    weights = [decay ** (len(values) - index - 1) for index in range(len(values))]
    return sum(value * weight for value, weight in zip(values, weights)) / sum(weights)


def _observed(value: float | None) -> tuple[float | None, str]:
    return (value, "OBSERVED") if value is not None else (None, "UNAVAILABLE")


def _cost_vector(
    rows: Sequence[Mapping[str, Any]],
    *,
    initial_capacity: float | None,
    decay: float,
) -> tuple[dict[str, float | None], dict[str, str]]:
    xi = [_number(row, "xi") for row in rows]
    theta = [_number(row, "theta") for row in rows]
    trends = [_number(row, "trend") for row in rows]
    current_debt = _number(rows[-1], "accumulated_excess")
    baseline_debt = _number(rows[0], "accumulated_excess")
    current_capacity = _number(rows[-1], "capacity")

    retained = None if any(value is None for value in xi) else _weighted_mean(xi, decay)  # type: ignore[arg-type]
    debt = (
        None
        if current_debt is None or baseline_debt is None
        else max(0.0, current_debt - baseline_debt)
    )
    capacity = (
        None
        if current_capacity is None or initial_capacity is None
        else max(0.0, initial_capacity - current_capacity)
    )
    persistence = (
        None
        if any(value is None for value in xi) or any(value is None for value in theta)
        else sum(float(x) > float(limit) for x, limit in zip(xi, theta)) / len(rows)
    )
    adverse = (
        None
        if any(value is None for value in trends)
        else _weighted_mean([max(0.0, -float(value)) for value in trends], decay)
    )
    pairs = {
        "retained_friction": _observed(retained),
        "accumulated_debt": _observed(debt),
        "capacity_consumption": _observed(capacity),
        "excess_persistence": _observed(persistence),
        "adverse_trend": _observed(adverse),
    }
    return (
        {name: pair[0] for name, pair in pairs.items()},
        {name: pair[1] for name, pair in pairs.items()},
    )


def _adaptive_organization_level(
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum_evaluable_windows: int,
) -> float | None:
    if not rows:
        return None
    gains: list[float] = []
    for row in rows:
        coherence = _number(row, "transport_coherence")
        contraction = _number(row, "variation_contraction")
        if coherence is None or contraction is None:
            continue
        gains.append(coherence * (1.0 - contraction))
    if len(gains) < minimum_evaluable_windows:
        return None
    return sum(gains) / len(gains)


def _outcome_channel(
    rows: Sequence[Mapping[str, Any]], name: str
) -> tuple[float | None, str]:
    if not rows:
        return None, "UNAVAILABLE"
    for latest in reversed(rows):
        statuses = latest.get("component_status", {})
        status = (
            str(statuses.get(name, "UNAVAILABLE"))
            if isinstance(statuses, Mapping)
            else "INVALID"
        )
        if status not in STATUS_VALUES:
            raise ValueError(f"invalid outcome channel status for {name}: {status}")
        value = (
            _number(latest.get("benefit_vector", {}), name)
            if isinstance(latest.get("benefit_vector"), Mapping)
            else None
        )
        if status == "OBSERVED" and value is None:
            raise ValueError(f"observed outcome channel {name} requires a value")
        if status != "OBSERVED" and value is not None:
            raise ValueError(f"non-observed outcome channel {name} must be null")
        if status != "UNAVAILABLE":
            return value, status
    return None, "UNAVAILABLE"


def _benefit_vector(
    prama_rows: Sequence[Mapping[str, Any]],
    structural_rows: Sequence[Mapping[str, Any]],
    outcome_rows: Sequence[Mapping[str, Any]],
    *,
    structural_identity_complete: bool,
    minimum_organization_support_windows: int,
) -> tuple[dict[str, float | None], dict[str, str]]:
    capacities = [_number(row, "capacity") for row in prama_rows]
    current_capacity = _number(prama_rows[-1], "capacity")
    recovery = (
        None
        if current_capacity is None or any(value is None for value in capacities)
        else max(
            0.0,
            current_capacity - min(float(value) for value in capacities),
        )
    )
    organization = (
        _adaptive_organization_level(
            structural_rows,
            minimum_evaluable_windows=minimum_organization_support_windows,
        )
        if structural_identity_complete
        else None
    )
    functional = _outcome_channel(outcome_rows, "functional_gain")
    integration = _outcome_channel(outcome_rows, "external_integration")
    verified = _outcome_channel(outcome_rows, "verified_outcome")
    pairs = {
        "structural_recovery": _observed(recovery),
        "adaptive_organization_level": _observed(organization),
        "functional_gain": functional,
        "external_integration": integration,
        "verified_outcome": verified,
    }
    return (
        {name: pair[0] for name, pair in pairs.items()},
        {name: pair[1] for name, pair in pairs.items()},
    )


def _normalize(
    vector: Mapping[str, float | None],
    status: Mapping[str, str],
    rules: Mapping[str, NormalizationRule],
) -> dict[str, float | None]:
    return {
        name: rules[name].apply(float(value))
        if value is not None and status[name] == "OBSERVED"
        else None
        for name, value in vector.items()
    }


def compute_structural_conversion_trajectory(
    prama_rows: Sequence[Mapping[str, Any]],
    structural_rows: Sequence[Mapping[str, Any]],
    outcome_rows: Sequence[Mapping[str, Any]],
    config: ODCEConfig,
    *,
    normalization_contract: Mapping[str, Any],
    correspondence_contract: Sequence[Mapping[str, Any]],
    contract_freeze_sha256: str | None = None,
) -> list[dict[str, Any]]:
    """Compute a causal ODCE sequence for one already ordered session."""
    if not prama_rows:
        raise ValueError("ODCE requires at least one PRAMA row")
    if contract_freeze_sha256 is not None and not _is_sha256(
        contract_freeze_sha256
    ):
        raise ValueError("contract_freeze_sha256 must be a SHA-256 digest or null")
    _reject_forbidden(prama_rows, "prama_rows")
    _reject_forbidden(structural_rows, "structural_rows")
    _reject_forbidden(outcome_rows, "outcome_rows")

    ordinal, prama_identities = _ordinal_by_identity(prama_rows)
    structural_by_identity: dict[tuple[int, int], Mapping[str, Any]] = {}
    for row in structural_rows:
        identity = _window_identity(row)
        if identity not in ordinal:
            raise ValueError(f"structural observation has no PRAMA window {identity}")
        if identity in structural_by_identity:
            raise ValueError("structural observation temporal identities must be unique")
        structural_by_identity[identity] = row

    aligned_outcomes: list[tuple[int, int, Mapping[str, Any]]] = []
    outcome_identities: set[
        tuple[tuple[int, int], tuple[int, int]]
    ] = set()
    for row in outcome_rows:
        event_identity = _window_identity(row, field="event_window")
        available_identity = _window_identity(row, field="available_at_window")
        if event_identity not in ordinal:
            raise ValueError(f"domain return event has no PRAMA window {event_identity}")
        if available_identity not in ordinal:
            raise ValueError(
                f"domain return availability has no PRAMA window {available_identity}"
            )
        event_index = row.get("event_index")
        available_at_index = row.get("available_at_index")
        for name, index in (
            ("event_index", event_index),
            ("available_at_index", available_at_index),
        ):
            if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                raise ValueError(f"domain return {name} must be a nonnegative integer")
        if event_index != ordinal[event_identity]:
            raise ValueError("domain return event_index disagrees with event_window identity")
        if available_at_index != ordinal[available_identity]:
            raise ValueError(
                "domain return available_at_index disagrees with available_at_window identity"
            )
        if available_at_index < event_index:
            raise ValueError("domain return cannot be available before its event")
        outcome_identity = (event_identity, available_identity)
        if outcome_identity in outcome_identities:
            raise ValueError("domain return temporal identities must be unique")
        outcome_identities.add(outcome_identity)
        aligned_outcomes.append((available_at_index, event_index, row))
    aligned_outcomes.sort(key=lambda item: (item[0], item[1]))
    initial_capacity = _number(prama_rows[0], "capacity")
    outputs: list[dict[str, Any]] = []
    dynamics: dict[str, dict[str, float | None]] = {
        row.name: {"previous": None, "accumulation": 0.0} for row in config.correspondence
    }

    for time_index in range(len(prama_rows)):
        window_start = max(0, time_index - config.window_length + 1)
        prama_window = prama_rows[window_start : time_index + 1]
        structural_window = [
            structural_by_identity[prama_identities[index]]
            for index in range(window_start, time_index + 1)
            if prama_identities[index] in structural_by_identity
        ]
        structural_identity_complete = len(structural_window) == len(prama_window)
        causal_outcomes = [
            row
            for available_at_index, _event_index, row in aligned_outcomes
            if available_at_index <= time_index
        ]
        cost, cost_status = _cost_vector(
            prama_window, initial_capacity=initial_capacity, decay=config.friction_decay
        )
        benefit, benefit_status = _benefit_vector(
            prama_window,
            structural_window,
            causal_outcomes,
            structural_identity_complete=structural_identity_complete,
            minimum_organization_support_windows=(
                config.minimum_organization_support_windows
            ),
        )
        normalized_cost = _normalize(
            cost, cost_status, config.cost_normalization
        )
        normalized_benefit = _normalize(
            benefit, benefit_status, config.benefit_normalization
        )
        differential: dict[str, float | None] = {}
        differential_status: dict[str, str] = {}
        differential_dynamics: dict[str, dict[str, float | None]] = {}
        for mapping in config.correspondence:
            c_value = normalized_cost[mapping.cost_channel]
            b_value = normalized_benefit[mapping.benefit_channel]
            statuses = {
                cost_status[mapping.cost_channel], benefit_status[mapping.benefit_channel]
            }
            status = "OBSERVED" if statuses == {"OBSERVED"} else (
                "INVALID" if "INVALID" in statuses else (
                    "UNAVAILABLE" if "UNAVAILABLE" in statuses else "NOT_APPLICABLE"
                )
            )
            value = (
                None
                if status != "OBSERVED"
                else float(c_value) - float(b_value)  # type: ignore[arg-type]
            )
            if value is not None and abs(value) < config.epsilon:
                value = 0.0
            differential[mapping.name] = value
            differential_status[mapping.name] = status
            previous = dynamics[mapping.name]["previous"]
            trend = None if value is None or previous is None else value - float(previous)
            if value is not None:
                dynamics[mapping.name]["accumulation"] = float(
                    dynamics[mapping.name]["accumulation"] or 0.0
                ) + max(0.0, value)
                dynamics[mapping.name]["previous"] = value
            recent = [
                output["differential_vector"][mapping.name]
                for output in outputs[window_start:]
            ] + [value]
            observed_recent = [item for item in recent if item is not None]
            persistence = (
                None
                if len(observed_recent) != len(recent)
                else sum(item > config.differential_threshold for item in observed_recent)
                / len(observed_recent)
            )
            differential_dynamics[mapping.name] = {
                "trend": trend,
                "cumulative_conversion_deficit_exposure": dynamics[mapping.name][
                    "accumulation"
                ],
                "positive_persistence": persistence,
            }

        outputs.append(
            {
                "operator": OPERATOR_ID,
                "operator_version": OPERATOR_VERSION,
                "time_index": time_index,
                "window_start": window_start,
                "window_end": time_index,
                "causal": True,
                "numeric_epsilon": config.epsilon,
                "differential_threshold": config.differential_threshold,
                # Capacity consumption uses the causally observed session baseline,
                # so the reference binds the complete prefix, not only W_t.
                "kernel_state_reference_sha256": sha256_value(
                    list(prama_rows[: time_index + 1])
                ),
                "structural_observation_reference_sha256": (
                    sha256_value(structural_window) if structural_window else None
                ),
                "domain_outcome_reference_sha256": (
                    sha256_value(causal_outcomes) if causal_outcomes else None
                ),
                "cost_vector": cost,
                "benefit_vector": benefit,
                "normalized_cost_vector": normalized_cost,
                "normalized_benefit_vector": normalized_benefit,
                "differential_vector": differential,
                "component_status": {
                    "cost": cost_status,
                    "benefit": benefit_status,
                    "differential": differential_status,
                },
                "differential_dynamics": differential_dynamics,
                "temporal_scope": {
                    "cost": dict(config.temporal_scope["cost"]),
                    "benefit": dict(config.temporal_scope["benefit"]),
                    "differential": dict(config.temporal_scope["differential"]),
                    "dynamics": dict(config.temporal_scope["dynamics"]),
                },
                "normalization_contract_sha256": sha256_value(
                    normalization_contract
                ),
                "correspondence_contract_sha256": sha256_value(
                    list(correspondence_contract)
                ),
                "contract_freeze_sha256": (
                    contract_freeze_sha256.removeprefix("sha256:")
                    if contract_freeze_sha256 is not None
                    else None
                ),
                "predictive_model_used": False,
                "future_outcome_used": False,
                "causal_availability_enforced": True,
                "provider_termination_metadata_used": False,
            }
        )
    return outputs


def make_structural_conversion_differential(
    envelope: Mapping[str, Any], observation: Mapping[str, Any]
) -> dict[str, Any]:
    record = {**dict(envelope), **dict(observation)}
    validate_artifact(record, "structural_conversion_differential")
    return record
