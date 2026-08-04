"""Normative artifact boundary for the LLM structural-state contract.

The module intentionally uses only the Python standard library.  JSON Schema
files are the interchange contract; these checks provide the same critical
fail-closed invariants at runtime without adding a validator dependency.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping


CONTRACT_VERSION = "0.2.0"
ARTIFACT_VERSION = "1.0.0"


class ArtifactValidationError(ValueError):
    """Raised when an artifact cannot cross the contract boundary."""


class ChannelStatus(str, Enum):
    OBSERVED = "OBSERVED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNAVAILABLE = "UNAVAILABLE"
    INVALID = "INVALID"


ARTIFACT_TYPES = {
    "generation_observation",
    "coupling_observation",
    "external_anchor_event",
    "perturbation_response",
    "epistemic_channel",
    "prama_trajectory",
    "structural_observation",
    "structural_label",
}

COMMON_REQUIRED = {
    "contract_version",
    "artifact_type",
    "artifact_version",
    "study_id",
    "session_id",
    "producer",
    "created_at",
    "source_sha256",
    "config_sha256",
    "partition",
    "channel_status",
}

TYPE_REQUIRED = {
    "generation_observation": {
        "model_id",
        "prompt_sha256",
        "response_sha256",
        "response_time_seconds",
    },
    "coupling_observation": {
        "turn_index",
        "window_index",
        "token_start",
        "token_end",
        "self_support",
        "user_support",
        "interaction",
        "support_magnitude",
        "omega_dep",
        "expected_omega_dep",
        "self_dependence_excess",
        "filler_variance",
        "eligible",
        "expectation_status",
    },
    "external_anchor_event": {
        "anchor_id",
        "anchor_type",
        "introduced_at_window",
        "anchor_state",
        "severity",
        "externally_verifiable",
        "anchor_source_sha256",
        "uptake_detected",
        "uptake_latency_windows",
    },
    "perturbation_response": {
        "perturbation_id",
        "introduced_at_window",
        "response_horizon_windows",
        "response_class",
        "trajectory_change_magnitude",
        "recovery_detected",
    },
    "epistemic_channel": {
        "task_id",
        "pair_id",
        "condition_id",
        "task_state_sha256",
        "competence_reference",
        "competence_condition",
        "effect_vector",
        "competence_preserved",
        "channel_valid",
    },
    "prama_trajectory": {
        "turn_index",
        "window_index",
        "delta",
        "xi",
        "accumulated_excess",
        "capacity",
        "theta",
        "balance",
        "trend",
        "input_transform",
        "input_channel_status",
        "coordinate_origin",
        "kernel_identity",
        "valid",
    },
    "structural_observation": {
        "observer",
        "observer_version",
        "base_observer",
        "turn_index",
        "window_index",
        "absolute_window_index",
        "transport_status",
        "recurrence_status",
        "contraction_status",
        "mobility_status",
        "structural_state",
        "movement",
        "transport_coherence",
        "recurrence_persistence",
        "variation_contraction",
        "diagnostics",
        "alert_eligible",
        "transport_deficit",
        "cumulative_transport_deficit",
        "evidence_window_start",
        "evidence_window_end",
        "causal",
        "external_outcome_used",
        "provider_termination_metadata_used",
    },
    "structural_label": {
        "label",
        "label_version",
        "rule_id",
        "required_channels",
        "satisfied_conditions",
        "failed_conditions",
        "unavailable_conditions",
        "calibration_reference",
        "evidence_window_start",
        "evidence_window_end",
        "confidence_status",
        "claim_boundary",
        "annotation_role",
        "structural_observation_reference",
    },
}

_SHA256 = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")


def canonical_json(value: Any) -> str:
    """Return the deterministic JSON representation used for hashes."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_value(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _validate_finite(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ArtifactValidationError(f"{path}: non-finite number")
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _validate_finite(nested, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _validate_finite(nested, f"{path}[{index}]")


def _require_hash(record: Mapping[str, Any], field: str) -> None:
    value = record.get(field)
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ArtifactValidationError(f"{field} must be a SHA-256 hex digest")


def _require_unit_interval(record: Mapping[str, Any], field: str) -> None:
    value = record.get(field)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ArtifactValidationError(f"{field} must be numeric")
    if not 0.0 <= float(value) <= 1.0:
        raise ArtifactValidationError(f"{field} must be within [0, 1]")


def validate_artifact(record: Mapping[str, Any], expected_type: str | None = None) -> None:
    """Validate critical normative constraints; raise on the first violation."""

    if not isinstance(record, Mapping):
        raise ArtifactValidationError("artifact must be a JSON object")
    artifact_type = record.get("artifact_type")
    if artifact_type not in ARTIFACT_TYPES:
        raise ArtifactValidationError(f"unsupported artifact_type: {artifact_type!r}")
    if expected_type is not None and artifact_type != expected_type:
        raise ArtifactValidationError(
            f"expected artifact_type {expected_type!r}, received {artifact_type!r}"
        )
    missing = sorted((COMMON_REQUIRED | TYPE_REQUIRED[artifact_type]) - record.keys())
    if missing:
        raise ArtifactValidationError(f"missing required fields: {missing}")
    if record["contract_version"] != CONTRACT_VERSION:
        raise ArtifactValidationError(
            f"contract_version must be {CONTRACT_VERSION!r}"
        )
    if record["artifact_version"] != ARTIFACT_VERSION:
        raise ArtifactValidationError(
            f"artifact_version must be {ARTIFACT_VERSION!r}"
        )
    try:
        ChannelStatus(str(record["channel_status"]))
    except ValueError as exc:
        raise ArtifactValidationError("invalid channel_status") from exc
    if record["partition"] not in {"calibration", "confirmatory", "exploratory"}:
        raise ArtifactValidationError("invalid partition")
    if not all(
        isinstance(record[name], str) and record[name].strip()
        for name in ("study_id", "session_id", "producer")
    ):
        raise ArtifactValidationError("study_id, session_id and producer are required")
    try:
        parsed = datetime.fromisoformat(str(record["created_at"]).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ArtifactValidationError("created_at must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ArtifactValidationError("created_at must include a timezone")
    _require_hash(record, "source_sha256")
    _require_hash(record, "config_sha256")
    _validate_finite(record)

    if artifact_type == "generation_observation":
        _require_hash(record, "prompt_sha256")
        _require_hash(record, "response_sha256")
        if float(record["response_time_seconds"]) < 0:
            raise ArtifactValidationError("response_time_seconds cannot be negative")
    elif artifact_type == "coupling_observation":
        if int(record["token_end"]) <= int(record["token_start"]):
            raise ArtifactValidationError("token range must be increasing")
        observed = record["expected_omega_dep"]
        excess = record["self_dependence_excess"]
        if observed is None and excess is not None:
            raise ArtifactValidationError(
                "self_dependence_excess requires expected_omega_dep"
            )
        if observed is not None:
            expected_excess = float(record["omega_dep"]) - float(observed)
            if not math.isclose(float(excess), expected_excess, abs_tol=1e-12):
                raise ArtifactValidationError(
                    "self_dependence_excess must equal omega_dep-expected_omega_dep"
                )
    elif artifact_type == "external_anchor_event":
        _require_unit_interval(record, "severity")
        _require_hash(record, "anchor_source_sha256")
        if record["uptake_latency_windows"] is not None and int(
            record["uptake_latency_windows"]
        ) < 0:
            raise ArtifactValidationError("uptake latency cannot be negative")
        if not record["externally_verifiable"] and record["uptake_detected"]:
            raise ArtifactValidationError(
                "an unverifiable anchor cannot claim measured uptake"
            )
    elif artifact_type == "epistemic_channel":
        _require_hash(record, "task_state_sha256")
        vector = record["effect_vector"]
        expected_keys = {
            "evidence_coverage_shift",
            "verifier_relevant_omission_shift",
            "precision_shift",
            "calibration_shift",
            "response_quality_shift",
        }
        if not isinstance(vector, Mapping) or set(vector) != expected_keys:
            raise ArtifactValidationError("effect_vector has noncanonical coordinates")
    elif artifact_type == "prama_trajectory":
        identity = record["kernel_identity"]
        required_identity = {
            "package",
            "version",
            "source_tree_sha256",
            "commit",
            "kernel_api",
            "config_sha256",
            "recertification_sha256",
            "bin_scale",
        }
        if not isinstance(identity, Mapping) or not required_identity <= identity.keys():
            raise ArtifactValidationError("kernel_identity is incomplete")
        if identity["bin_scale"] != "window":
            raise ArtifactValidationError("PRAMA bin_scale must be 'window'")
        if identity["kernel_api"] != "project_v3":
            raise ArtifactValidationError("PRAMA kernel_api must be 'project_v3'")
        if record["input_transform"] != "signed_unit_affine_v1":
            raise ArtifactValidationError(
                "PRAMA input_transform must be 'signed_unit_affine_v1'"
            )
        if record["input_channel_status"] != ChannelStatus.OBSERVED.value:
            raise ArtifactValidationError(
                "PRAMA input_channel_status must be 'OBSERVED'"
            )
        if record["coordinate_origin"] != "DERIVED_KERNEL_STATE":
            raise ArtifactValidationError(
                "PRAMA coordinate_origin must be 'DERIVED_KERNEL_STATE'"
            )
        _require_hash(identity, "source_tree_sha256")
        _require_hash(identity, "config_sha256")
        _require_hash(identity, "recertification_sha256")
    elif artifact_type == "structural_observation":
        if record["observer"] != "D_O_v9" or record["observer_version"] != "D_O_v9":
            raise ArtifactValidationError("structural observer must be D_O_v9")
        if record["base_observer"] != "D_O_v6":
            raise ArtifactValidationError("D_O_v9 requires the D_O_v6 numeric base observer")
        allowed = {
            "transport_status": {
                "UNRESOLVED", "INACTIVE", "COHERENT", "PROVISIONAL", "DISRUPTED"
            },
            "recurrence_status": {
                "UNRESOLVED", "INACTIVE", "NON_RECURRENT", "RECURRENT"
            },
            "contraction_status": {
                "UNRESOLVED", "NOT_CONTRACTING", "CONTRACTING"
            },
            "mobility_status": {
                None, "VIABLE", "STAGNANT", "RECURRENT", "CRYSTALLIZING", "CRYSTALLIZED"
            },
            "structural_state": {
                "VIABLE", "STAGNANT", "RECURRENT", "CRYSTALLIZING", "CRYSTALLIZED",
                "TRANSPORT_DISRUPTED", "TRANSPORT_UNRESOLVED",
            },
        }
        for field, values in allowed.items():
            if record[field] not in values:
                raise ArtifactValidationError(f"invalid {field}")
        for field in ("turn_index", "window_index", "absolute_window_index"):
            if not isinstance(record[field], int) or isinstance(record[field], bool) or record[field] < 0:
                raise ArtifactValidationError(f"{field} must be a nonnegative integer")
        if float(record["movement"]) < 0:
            raise ArtifactValidationError("movement cannot be negative")
        for field in (
            "transport_coherence",
            "recurrence_persistence",
            "variation_contraction",
        ):
            if record[field] is not None:
                _require_unit_interval(record, field)
        for field in ("transport_deficit", "cumulative_transport_deficit"):
            if record[field] is not None and float(record[field]) < 0:
                raise ArtifactValidationError(f"{field} cannot be negative")
        if int(record["evidence_window_start"]) < 0 or int(record["evidence_window_end"]) < int(record["evidence_window_start"]):
            raise ArtifactValidationError("structural evidence window is invalid")
        if record["causal"] is not True or record["external_outcome_used"] is not False:
            raise ArtifactValidationError("D_O_v9 must be causal and outcome-blind")
        if record["provider_termination_metadata_used"] is not False:
            raise ArtifactValidationError(
                "provider termination metadata is not structural evidence"
            )
        forbidden = {"finish_reason", "response_time_seconds", "outcome", "label"}
        leaked = sorted(forbidden & record.keys())
        if leaked:
            raise ArtifactValidationError(
                f"structural observation contains forbidden fields: {leaked}"
            )
        diagnostics = record["diagnostics"]
        if (
            not isinstance(diagnostics, list)
            or not all(isinstance(value, str) and value for value in diagnostics)
            or len(diagnostics) != len(set(diagnostics))
        ):
            raise ArtifactValidationError("diagnostics must be unique nonempty strings")
    elif artifact_type == "structural_label":
        if record["annotation_role"] != "SECONDARY_INTERPRETIVE":
            raise ArtifactValidationError(
                "structural_label must be a secondary interpretive annotation"
            )
        _require_hash(record, "structural_observation_reference")


def make_envelope(
    *,
    artifact_type: str,
    study_id: str,
    session_id: str,
    producer: str,
    created_at: str,
    source_sha256: str,
    config_sha256: str,
    partition: str,
    channel_status: ChannelStatus | str,
) -> dict[str, Any]:
    """Create the common provenance envelope shared by every artifact."""

    if artifact_type not in ARTIFACT_TYPES:
        raise ArtifactValidationError(f"unsupported artifact_type: {artifact_type!r}")
    return {
        "contract_version": CONTRACT_VERSION,
        "artifact_type": artifact_type,
        "artifact_version": ARTIFACT_VERSION,
        "study_id": study_id,
        "session_id": session_id,
        "producer": producer,
        "created_at": created_at,
        "source_sha256": source_sha256.removeprefix("sha256:"),
        "config_sha256": config_sha256.removeprefix("sha256:"),
        "partition": partition,
        "channel_status": ChannelStatus(channel_status).value,
    }


def read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ArtifactValidationError(
                    f"{path}:{line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ArtifactValidationError(
                    f"{path}:{line_number}: row must be an object"
                )
            rows.append(value)
    return tuple(rows)


def write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Validate every row before atomically replacing an artifact file."""

    materialized = tuple(dict(row) for row in rows)
    for row in materialized:
        validate_artifact(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        for row in materialized:
            handle.write(canonical_json(row) + "\n")
    temporary.replace(path)
