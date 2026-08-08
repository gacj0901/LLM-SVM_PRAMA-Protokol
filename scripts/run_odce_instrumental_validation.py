#!/usr/bin/env python
"""Build deterministic causal ODCE evidence for controlled properties A--F."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from aptadynamic_llm.artifact_schema import (
    canonical_json,
    make_envelope,
    sha256_value,
    write_jsonl_atomic,
)
from aptadynamic_llm.odce_calibration import (
    calibrate_exploratory_differential_threshold,
)
from aptadynamic_llm.structural_coherence_v9 import StructuralCoherenceV9Config
from aptadynamic_llm.structural_conversion import (
    ODCEConfig,
    compute_structural_conversion_trajectory,
    make_structural_conversion_differential,
)
from aptadynamic_llm.structural_observation import (
    make_structural_observation,
    observe_structural_trajectory,
)


HASH_A = sha256_value(
    {
        "source": "deterministic controlled ODCE instrumental inputs",
        "version": 1,
    }
)
HASH_B = sha256_value(
    {
        "configuration": "controlled ODCE instrumental construction",
        "version": 1,
    }
)
KERNEL_SOURCE_TREE_SHA256 = (
    "61e1063de0b5b032cd6af09eeb3b6906614f6331954697c605603afa18f641fc"
)
KERNEL_CONFIG_SHA256 = (
    "3554d3b270c735be97adc6832865d9087970857c448b18258729c840688b31a7"
)
KERNEL_RECERTIFICATION_SHA256 = (
    "5403e4584083a5cdc59e168d66cc036870c42b93e1c4e7d99075ab8dd9756e81"
)
CREATED_AT = datetime(2026, 8, 7, tzinfo=timezone.utc).isoformat()
STUDY_ID = "odce_instrumental_validation_v1"
PRODUCER = "scripts/run_odce_instrumental_validation.py"


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(canonical_json(value) + "\n")
    temporary.replace(path)


def _envelope(artifact_type: str, session_id: str) -> dict[str, Any]:
    return make_envelope(
        artifact_type=artifact_type,
        study_id=STUDY_ID,
        session_id=session_id,
        producer=PRODUCER,
        created_at=CREATED_AT,
        source_sha256=HASH_A,
        config_sha256=HASH_B,
        partition="exploratory",
        channel_status="OBSERVED",
    )


def _prama(
    session_id: str,
    index: int,
    *,
    xi: float,
    capacity: float = 1.0,
) -> dict[str, Any]:
    payload = {
        "turn_index": 0,
        "window_index": index,
        "delta": xi,
        "xi": xi,
        "accumulated_excess": 0.0,
        "capacity": capacity,
        "theta": 0.6,
        "balance": 0.6 - xi,
        "trend": 0.0,
        "input_transform": "signed_unit_affine_v1",
        "input_channel_status": "OBSERVED",
        "coordinate_origin": "DERIVED_KERNEL_STATE",
        "kernel_identity": {
            "package": "prama-protokol",
            "version": "0.3.0",
            "source_tree_sha256": KERNEL_SOURCE_TREE_SHA256,
            "commit": "cb41d590207a09d498532b8c599e12ecab7a0dcb",
            "kernel_api": "project_v3",
            "config_sha256": KERNEL_CONFIG_SHA256,
            "recertification_sha256": KERNEL_RECERTIFICATION_SHA256,
            "bin_scale": "window",
        },
        "valid": True,
    }
    return {**_envelope("prama_trajectory", session_id), **payload}


def _structural(
    session_id: str,
    organization_levels: Sequence[float],
) -> list[dict[str, Any]]:
    numeric_windows = [
        {
            "turn_index": 0,
            "window_index": index,
            "absolute_window_index": index,
            "geometry_ready": True,
            "movement": 1.0,
            "transport_coherence": level,
            "recurrence_persistence": 0.0,
            "variation_contraction": 0.0,
        }
        for index, level in enumerate(organization_levels)
    ]
    observations = observe_structural_trajectory(
        numeric_windows, StructuralCoherenceV9Config()
    )
    return [
        make_structural_observation(
            _envelope("structural_observation", session_id), observation
        )
        for observation in observations
    ]


def _outcome(
    session_id: str,
    *,
    event_index: int,
    available_at_index: int,
    functional_gain: float | None = None,
    external_integration: float | None = None,
    verified_outcome: float | None = None,
) -> dict[str, Any]:
    values = {
        "functional_gain": functional_gain,
        "external_integration": external_integration,
        "verified_outcome": verified_outcome,
    }
    if all(value is None for value in values.values()):
        raise ValueError("a controlled outcome must observe at least one channel")
    return {
        **_envelope("domain_return_observation", session_id),
        "event_index": event_index,
        "available_at_index": available_at_index,
        "event_window": {"turn_index": 0, "window_index": event_index},
        "available_at_window": {
            "turn_index": 0,
            "window_index": available_at_index,
        },
        "benefit_vector": values,
        "component_status": {
            name: "OBSERVED" if value is not None else "UNAVAILABLE"
            for name, value in values.items()
        },
        "verifier_reference_sha256": (
            sha256_value(
                {
                    "controlled_verifier": "independent deterministic check",
                    "session_id": session_id,
                }
            )
            if verified_outcome is not None
            else None
        ),
        "retrospective_backfill": available_at_index > event_index,
        "causal_availability_declared": True,
        "provider_termination_metadata_used": False,
    }


def _materialize_odce(
    session_id: str,
    prama_rows: Sequence[Mapping[str, Any]],
    structural_rows: Sequence[Mapping[str, Any]],
    outcome_rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    observations = compute_structural_conversion_trajectory(
        prama_rows,
        structural_rows,
        outcome_rows,
        ODCEConfig.from_contract(contract),
        normalization_contract=contract["normalization"],
        correspondence_contract=contract["correspondence"],
    )
    return [
        make_structural_conversion_differential(
            _envelope("structural_conversion_differential", session_id),
            observation,
        )
        for observation in observations
    ]


def _stable_noise_rows(
    contract: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    prama_rows: list[dict[str, Any]] = []
    structural_rows: list[dict[str, Any]] = []
    terminal_odce: list[dict[str, Any]] = []
    for session_index in range(24):
        noise = (session_index - 12) * 0.0001
        session_id = f"stable-noise-{session_index:02d}"
        session_prama = [
            _prama(session_id, index, xi=0.5 + noise) for index in range(8)
        ]
        session_structural = _structural(session_id, [0.5] * 8)
        session_odce = _materialize_odce(
            session_id, session_prama, session_structural, [], contract
        )
        prama_rows.extend(session_prama)
        structural_rows.extend(session_structural)
        terminal_odce.append(session_odce[-1])
    return prama_rows, structural_rows, terminal_odce


def _joint_organization_levels(
    xi_values: Sequence[float], decay: float, window_length: int
) -> list[float]:
    levels: list[float] = []
    for index in range(len(xi_values)):
        window_start = max(0, index - window_length + 1)
        prefix = xi_values[window_start : index + 1]
        weights = [decay ** (len(prefix) - offset - 1) for offset in range(len(prefix))]
        target = sum(value * weight for value, weight in zip(prefix, weights)) / sum(weights)
        level = len(prefix) * target - sum(levels[window_start:index])
        if not 0.0 <= level <= 1.0:
            raise ValueError("joint controlled organization level left [0, 1]")
        levels.append(level)
    return levels


def _scenario_inputs(
    contract: Mapping[str, Any],
) -> tuple[
    dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    length = 44
    transition = 32
    increasing = [0.2 + (offset + 1) * (0.4 / 12) for offset in range(12)]
    xi_cost = [0.2] * transition + increasing
    xi_benefit = [0.45] * length
    benefit_levels = [0.2] * transition + increasing
    xi_joint = [0.2 + index * 0.003 for index in range(length)]
    joint_levels = _joint_organization_levels(
        xi_joint,
        float(contract["friction_decay"]),
        int(contract["window_length"]),
    )
    xi_deficit = [0.2] * transition + [
        0.4 + (offset + 1) * (0.2 / 12) for offset in range(12)
    ]
    decline = [1.0 - offset * (0.4 / 11) for offset in range(12)]
    recovery = [0.65 + offset * (0.33 / 11) for offset in range(12)]
    recovery_capacity = [1.0] * 20 + decline + recovery
    outcome_length = 68
    outcome_capacity = [1.0] + [0.6] * (outcome_length - 1)
    auxiliary_outcome_capacity = [1.0] + [0.6] * (length - 1)
    definitions = {
        "A_cost_increasing": (xi_cost, [0.2] * length, [1.0] * length, []),
        "B_benefit_increasing": (
            xi_benefit, benefit_levels, [1.0] * length, []
        ),
        "C_joint_comparable": (
            xi_joint, joint_levels, [1.0] * length, []
        ),
        "D_sustained_deficit": (
            xi_deficit, [0.2] * length, [1.0] * length, []
        ),
        "E_recovery": (
            [0.3] * length, [0.2] * length, recovery_capacity, []
        ),
        "F_causal_availability": (
            [0.2] * outcome_length,
            [0.2] * outcome_length,
            outcome_capacity,
            [
                _outcome(
                    "F_causal_availability",
                    event_index=30,
                    available_at_index=35,
                    functional_gain=0.3,
                )
            ],
        ),
        "G_external_integration": (
            [0.2] * length,
            [0.2] * length,
            auxiliary_outcome_capacity,
            [
                _outcome(
                    "G_external_integration",
                    event_index=30,
                    available_at_index=35,
                    external_integration=0.4,
                )
            ],
        ),
        "H_verified_outcome": (
            [0.2] * length,
            [0.2] * length,
            auxiliary_outcome_capacity,
            [
                _outcome(
                    "H_verified_outcome",
                    event_index=30,
                    available_at_index=35,
                    verified_outcome=1.0,
                )
            ],
        ),
    }
    scenarios = {}
    all_prama: list[dict[str, Any]] = []
    all_structural: list[dict[str, Any]] = []
    all_outcomes: list[dict[str, Any]] = []
    for session_id, (xi_values, levels, capacities, outcomes) in definitions.items():
        prama_rows = [
            _prama(
                session_id,
                index,
                xi=xi_values[index],
                capacity=capacities[index],
            )
            for index in range(len(xi_values))
        ]
        structural_rows = _structural(session_id, levels)
        scenarios[session_id] = (prama_rows, structural_rows, outcomes)
        all_prama.extend(prama_rows)
        all_structural.extend(structural_rows)
        all_outcomes.extend(outcomes)
    return scenarios, all_prama, all_structural, all_outcomes


def _nondecreasing(values: Sequence[float]) -> bool:
    return all(current >= previous for previous, current in zip(values, values[1:]))


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "run_outputs" / "odce_instrumental_validation_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    base_contract = json.loads(
        (root / "config" / "odce_v0_1_exploratory.json").read_text(
            encoding="utf-8"
        )
    )

    stable_prama, stable_structural, stable_terminal_odce = _stable_noise_rows(
        base_contract
    )
    calibrated_contract, threshold_report = (
        calibrate_exploratory_differential_threshold(
            stable_terminal_odce,
            base_contract,
            correspondence_names=[
                "retained_friction_vs_adaptive_organization_level"
            ],
            stable_condition_id="balanced_friction_organization_stable_noise_v1",
            min_observations=20,
            residual_quantile=0.95,
            allow_exploratory_input=True,
        )
    )
    threshold = float(calibrated_contract["differential_threshold"])
    if threshold <= 0.000942:
        raise ValueError("empirical noise floor did not exclude the audited micro-excursion")
    if sha256_value(base_contract["normalization"]) != sha256_value(
        calibrated_contract["normalization"]
    ):
        raise ValueError("threshold calibration modified normalization")

    scenarios, all_prama, all_structural, all_outcomes = _scenario_inputs(
        calibrated_contract
    )
    input_hash_before = sha256_value(
        {"prama": all_prama, "structural": all_structural}
    )
    trajectories = {
        session_id: _materialize_odce(
            session_id,
            prama_rows,
            structural_rows,
            outcomes,
            calibrated_contract,
        )
        for session_id, (prama_rows, structural_rows, outcomes) in scenarios.items()
    }
    input_hash_after = sha256_value(
        {"prama": all_prama, "structural": all_structural}
    )
    if input_hash_before != input_hash_after:
        raise ValueError("ODCE mutated PRAMA or D_O v9 inputs")

    organization_name = "retained_friction_vs_adaptive_organization_level"
    recovery_name = "retained_friction_vs_structural_recovery"
    function_name = "capacity_consumption_vs_functional_gain"
    a = trajectories["A_cost_increasing"]
    b = trajectories["B_benefit_increasing"]
    c = trajectories["C_joint_comparable"]
    d = trajectories["D_sustained_deficit"]
    e = trajectories["E_recovery"]
    f = trajectories["F_causal_availability"]
    g = trajectories["G_external_integration"]
    h = trajectories["H_verified_outcome"]
    observed_start = 7
    transition_index = 31
    outcome_available_index = 35

    d_persistence = [
        row["differential_dynamics"][organization_name]["positive_persistence"]
        for row in d
        if row["differential_dynamics"][organization_name][
            "positive_persistence"
        ]
        is not None
    ]
    d_exposure = [
        row["differential_dynamics"][organization_name][
            "cumulative_conversion_deficit_exposure"
        ]
        for row in d
    ]
    e_differential = [
        row["differential_vector"][recovery_name] for row in e
    ]
    e_exposure = [
        row["differential_dynamics"][recovery_name][
            "cumulative_conversion_deficit_exposure"
        ]
        for row in e
    ]
    f_function_persistence = [
        row["differential_dynamics"][function_name]["positive_persistence"]
        for row in f
        if row["differential_dynamics"][function_name][
            "positive_persistence"
        ]
        is not None
    ]
    c_organization_exposure = [
        row["differential_dynamics"][organization_name][
            "cumulative_conversion_deficit_exposure"
        ]
        for row in c
    ]
    f_organization_exposure = [
        row["differential_dynamics"][organization_name][
            "cumulative_conversion_deficit_exposure"
        ]
        for row in f
    ]
    checks = {
        "A_cost_increasing_benefit_constant": (
            a[-1]["differential_vector"][organization_name]
            > a[transition_index]["differential_vector"][organization_name]
        ),
        "B_cost_constant_benefit_increasing": (
            b[-1]["differential_vector"][organization_name]
            < b[transition_index]["differential_vector"][organization_name]
        ),
        "C_joint_comparable_stable": max(
            abs(row["differential_vector"][organization_name])
            for row in c[observed_start:]
        )
        == 0.0,
        "C_numeric_exposure_is_exact_zero": (
            c_organization_exposure[-1] == 0.0
        ),
        "D_positive_persistence_grows": (
            d_persistence[-1] > d_persistence[0]
        ),
        "D_exposure_is_nondecreasing": _nondecreasing(d_exposure),
        "D_exposure_grows": d_exposure[-1] > d_exposure[0],
        "E_structural_recovery_positive": (
            max(row["benefit_vector"]["structural_recovery"] for row in e) > 0
        ),
        "E_differential_decreases": (
            e_differential[-1] < e_differential[transition_index]
        ),
        "E_differential_can_be_negative": e_differential[-1] < 0,
        "E_exposure_is_irreversible": _nondecreasing(e_exposure),
        "F_unavailable_before_availability": all(
            row["component_status"]["benefit"]["functional_gain"]
            == "UNAVAILABLE"
            and row["benefit_vector"]["functional_gain"] is None
            for row in f[:outcome_available_index]
        ),
        "F_observed_at_availability": (
            f[outcome_available_index]["component_status"]["differential"][
                function_name
            ]
            == "OBSERVED"
        ),
        "F_functional_persistence_becomes_observed": (
            bool(f_function_persistence)
            and f_function_persistence[-1] == 1.0
        ),
        "F_numeric_organization_exposure_is_exact_zero": (
            f_organization_exposure[-1] == 0.0
        ),
        "F_outcome_reference_appears_causally": (
            all(
                row["domain_outcome_reference_sha256"] is None
                for row in f[:outcome_available_index]
            )
            and all(
                row["domain_outcome_reference_sha256"] is not None
                for row in f[outcome_available_index:]
            )
        ),
        "G_external_integration_unavailable_before_availability": all(
            row["component_status"]["benefit"]["external_integration"]
            == "UNAVAILABLE"
            and row["benefit_vector"]["external_integration"] is None
            for row in g[:outcome_available_index]
        ),
        "G_external_integration_observed_at_availability": (
            g[outcome_available_index]["component_status"]["benefit"][
                "external_integration"
            ]
            == "OBSERVED"
            and g[outcome_available_index]["benefit_vector"][
                "external_integration"
            ]
            == 0.4
        ),
        "H_verified_outcome_unavailable_before_availability": all(
            row["component_status"]["benefit"]["verified_outcome"]
            == "UNAVAILABLE"
            and row["benefit_vector"]["verified_outcome"] is None
            for row in h[:outcome_available_index]
        ),
        "H_verified_outcome_observed_at_availability": (
            h[outcome_available_index]["component_status"]["benefit"][
                "verified_outcome"
            ]
            == "OBSERVED"
            and h[outcome_available_index]["benefit_vector"][
                "verified_outcome"
            ]
            == 1.0
        ),
        "D_O_references_present": all(
            row["structural_observation_reference_sha256"] is not None
            for trajectory in trajectories.values()
            for row in trajectory
        ),
        "adaptive_organization_observed_with_support": all(
            row["component_status"]["benefit"][
                "adaptive_organization_level"
            ]
            == "OBSERVED"
            for trajectory in trajectories.values()
            for row in trajectory[observed_start:]
        ),
        "PRAMA_and_D_O_inputs_immutable": input_hash_before == input_hash_after,
        "causal_invariants": all(
            row["causal"] is True
            and row["future_outcome_used"] is False
            and row["causal_availability_enforced"] is True
            and row["provider_termination_metadata_used"] is False
            for trajectory in trajectories.values()
            for row in trajectory
        ),
        "exploratory_freeze_reference_is_null": all(
            row["contract_freeze_sha256"] is None
            for trajectory in trajectories.values()
            for row in trajectory
        ),
        "identity_normalization_raw_equals_normalized": all(
            row[f"{group}_vector"][channel]
            == row[f"normalized_{group}_vector"][channel]
            for trajectory in trajectories.values()
            for row in trajectory
            for group in ("cost", "benefit")
            for channel, status in row["component_status"][group].items()
            if status == "OBSERVED"
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"instrumental ODCE checks failed: {failed}")

    all_odce = [row for trajectory in trajectories.values() for row in trajectory]
    write_jsonl_atomic(output_dir / "stable_prama.jsonl", stable_prama)
    write_jsonl_atomic(
        output_dir / "stable_structural_observations.jsonl", stable_structural
    )
    write_jsonl_atomic(
        output_dir / "stable_terminal_odce.jsonl", stable_terminal_odce
    )
    _write_json_atomic(
        output_dir / "threshold_calibrated_contract.json", calibrated_contract
    )
    _write_json_atomic(
        output_dir / "threshold_calibration_report.json", threshold_report
    )
    write_jsonl_atomic(output_dir / "controlled_prama.jsonl", all_prama)
    write_jsonl_atomic(
        output_dir / "controlled_structural_observations.jsonl", all_structural
    )
    write_jsonl_atomic(
        output_dir / "controlled_domain_returns.jsonl", all_outcomes
    )
    write_jsonl_atomic(output_dir / "controlled_odce.jsonl", all_odce)
    report = {
        "schema": "LLM-SVM-ODCE-instrumental-validation/0.1",
        "status": "EXPLORATORY_INSTRUMENTAL_PASS",
        "operator_id": "ODCE_v0",
        "operator_version": "ODCE-v0.1.0",
        "differential_threshold": threshold,
        "normalization_modified": False,
        "normalization_validation_status": (
            "IDENTITY_NORMALIZATION_OPERATOR_LOGIC_ONLY"
        ),
        "empirical_normalizer_calibration_validated": False,
        "contract_freeze_status": "NOT_APPLICABLE_EXPLORATORY",
        "fixed_instrumental_calibration_declared": False,
        "numeric_cleanup": {
            "rule": "abs(D_t) < numeric_epsilon implies D_t = 0",
            "numeric_epsilon": float(calibrated_contract["epsilon"]),
            "differential_threshold_unchanged": threshold,
            "C_final_organization_exposure": c_organization_exposure[-1],
            "F_final_organization_exposure": f_organization_exposure[-1],
        },
        "functional_persistence_evidence": {
            "event_index": 30,
            "available_at_index": outcome_available_index,
            "trajectory_end_index": f[-1]["time_index"],
            "first_observed_at_index": next(
                row["time_index"]
                for row in f
                if row["differential_dynamics"][function_name][
                    "positive_persistence"
                ]
                is not None
            ),
            "final_positive_persistence": f_function_persistence[-1],
        },
        "outcome_channel_coverage": {
            channel: sum(
                row["component_status"]["benefit"][channel] == "OBSERVED"
                for row in all_odce
            )
            for channel in (
                "functional_gain",
                "external_integration",
                "verified_outcome",
            )
        },
        "stable_session_count": 24,
        "controlled_session_count": len(trajectories),
        "controlled_artifact_count": len(all_odce),
        "checks": checks,
        "input_immutability_sha256": input_hash_before,
        "threshold_report_sha256": sha256_value(threshold_report),
        "controlled_odce_sha256": sha256_value(all_odce),
        "claim_boundary": [
            "This battery demonstrates causal, monotonic and reproducible instrumental behavior only.",
            "Raw and normalized observed values are identical by construction; empirical normalizer calibration is not tested.",
            "It does not estimate predictive accuracy or authorize confirmatory use.",
            "contract_freeze_sha256 remains null because every artifact is exploratory and no fixed instrumental calibration is declared.",
            "PRAMA and D_O v9 are immutable inputs to the post-kernel ODCE operator.",
        ],
    }
    _write_json_atomic(output_dir / "instrumental_report.json", report)
    print(
        f"ODCE instrumental validation passed {len(checks)} checks; "
        f"differential_threshold={threshold:.12g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
