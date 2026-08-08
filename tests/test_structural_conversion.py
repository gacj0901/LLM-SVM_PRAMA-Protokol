from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import pytest

from aptadynamic_llm.artifact_schema import (
    ArtifactValidationError,
    make_envelope,
    sha256_value,
    validate_artifact,
    write_jsonl_atomic,
)
from aptadynamic_llm.structural_conversion import (
    ODCEConfig,
    compute_structural_conversion_trajectory,
    make_structural_conversion_differential,
    validate_contract_freeze,
)
from aptadynamic_llm.odce_calibration import (
    calibrate_exploratory_contract,
    calibrate_exploratory_differential_threshold,
)


HASH_A = "a" * 64
HASH_B = "b" * 64


def contract():
    root = Path(__file__).resolve().parents[1]
    return json.loads(
        (root / "config" / "odce_v0_1_exploratory.json").read_text(encoding="utf-8")
    )


def frozen_contract():
    value = contract()
    value["status"] = "FROZEN_PROSPECTIVE"
    value["normalization"]["calibration_status"] = "FROZEN_DOMAIN_CALIBRATION"
    value["normalization"]["confirmatory_use_allowed"] = True
    value["normalization"]["calibration_reference_sha256"] = HASH_A
    value["normalization"]["calibration_population_sha256"] = HASH_B
    value["correspondence_governance"] = {
        "status": "FROZEN_PROSPECTIVE",
        "rationale_reference_sha256": HASH_A,
    }
    value["differential_threshold_calibration"][
        "normalization_contract_sha256"
    ] = sha256_value(value["normalization"])
    return value


def freeze_manifest(value):
    return {
        "schema": "LLM-SVM-ODCE-contract-freeze/0.1",
        "operator_id": "ODCE_v0",
        "operator_version": "ODCE-v0.1.0",
        "contract_canonical_sha256": sha256_value(value),
        "normalization_contract_sha256": sha256_value(value["normalization"]),
        "correspondence_contract_sha256": sha256_value(value["correspondence"]),
        "calibration_reference_sha256": HASH_A,
        "calibration_population_sha256": HASH_B,
        "correspondence_rationale_reference_sha256": HASH_A,
        "frozen_before_confirmatory_acquisition": True,
    }


def prama(index, *, xi, capacity=1.0, debt=0.0, trend=0.0, theta=0.2):
    return {
        "turn_index": 0,
        "window_index": index,
        "delta": xi,
        "xi": xi,
        "accumulated_excess": debt,
        "capacity": capacity,
        "theta": theta,
        "balance": theta - xi,
        "trend": trend,
        "input_transform": "signed_unit_affine_v1",
        "input_channel_status": "OBSERVED",
        "coordinate_origin": "DERIVED_KERNEL_STATE",
        "kernel_identity": {
            "package": "prama-protokol",
            "version": "0.3.0",
            "source_tree_sha256": HASH_A,
            "commit": "abcdef1",
            "kernel_api": "project_v3",
            "config_sha256": HASH_A,
            "recertification_sha256": HASH_B,
            "bin_scale": "window",
        },
        "valid": True,
    }


def structural(index, *, coherence=0.8, contraction=0.25):
    return {
        "turn_index": 0,
        "window_index": index,
        "transport_coherence": coherence,
        "variation_contraction": contraction,
    }


def canonical_structural(index, *, session_id="session-1", coherence=0.8):
    envelope = make_envelope(
        artifact_type="structural_observation",
        study_id="study-1",
        session_id=session_id,
        producer="pytest",
        created_at=datetime.now(timezone.utc).isoformat(),
        source_sha256=HASH_A,
        config_sha256=HASH_B,
        partition="exploratory",
        channel_status="OBSERVED",
    )
    return {
        **envelope,
        "observer": "D_O_v9",
        "observer_version": "D_O_v9",
        "base_observer": "D_O_v6",
        "turn_index": 0,
        "window_index": index,
        "absolute_window_index": index,
        "transport_status": "COHERENT",
        "recurrence_status": "NON_RECURRENT",
        "contraction_status": "NOT_CONTRACTING",
        "mobility_status": "VIABLE",
        "structural_state": "VIABLE",
        "movement": 1.0,
        "transport_coherence": coherence,
        "recurrence_persistence": 0.0,
        "variation_contraction": 0.0,
        "diagnostics": [],
        "alert_eligible": index >= 16,
        "transport_deficit": 0.0,
        "cumulative_transport_deficit": 0.0,
        "evidence_window_start": 0,
        "evidence_window_end": index,
        "causal": True,
        "external_outcome_used": False,
        "provider_termination_metadata_used": False,
    }


def outcome(
    event_index,
    available_at_index,
    *,
    functional_gain=0.7,
    external_integration=None,
    verified_outcome=None,
):
    values = {
        "functional_gain": functional_gain,
        "external_integration": external_integration,
        "verified_outcome": verified_outcome,
    }
    return {
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
            HASH_B if verified_outcome is not None else None
        ),
        "retrospective_backfill": available_at_index > event_index,
        "causal_availability_declared": True,
        "provider_termination_metadata_used": False,
    }


def compute(rows, geometry, outcomes=None):
    raw = contract()
    return compute_structural_conversion_trajectory(
        rows,
        geometry,
        outcomes or [],
        ODCEConfig.from_contract(raw),
        normalization_contract=raw["normalization"],
        correspondence_contract=raw["correspondence"],
    )


def test_odce_is_causal_and_preserves_cost_benefit_and_differential():
    rows = [
        prama(0, xi=0.1),
        prama(1, xi=0.2, trend=-0.1),
        prama(2, xi=0.3, capacity=0.9, debt=0.05, trend=-0.2),
        prama(3, xi=0.15, capacity=0.92, debt=0.05, trend=0.1),
        prama(4, xi=0.15, capacity=0.92, debt=0.05),
        prama(5, xi=0.15, capacity=0.92, debt=0.05),
        prama(6, xi=0.15, capacity=0.92, debt=0.05),
        prama(7, xi=0.15, capacity=0.92, debt=0.05),
    ]
    geometry = [structural(index) for index in range(8)]
    outcomes = [outcome(2, 2, verified_outcome=1.0)]
    prefix = compute(rows[:7], geometry[:7], outcomes)
    full = compute(rows, geometry, outcomes)
    assert prefix == full[:7]
    assert full[0]["operator"] == "ODCE_v0"
    assert full[0]["operator_version"] == "ODCE-v0.1.0"
    assert full[6]["benefit_vector"]["adaptive_organization_level"] is None
    assert full[7]["benefit_vector"]["adaptive_organization_level"] == pytest.approx(0.6)
    assert full[1]["benefit_vector"]["functional_gain"] is None
    assert full[1]["component_status"]["benefit"]["functional_gain"] == "UNAVAILABLE"
    assert full[2]["benefit_vector"]["functional_gain"] == pytest.approx(0.7)
    assert full[2]["cost_vector"]["capacity_consumption"] == pytest.approx(0.1)
    function_name = "capacity_consumption_vs_functional_gain"
    assert full[2]["differential_vector"][function_name] == pytest.approx(-0.6)
    assert full[2]["differential_dynamics"][function_name]["trend"] is None
    assert full[2]["predictive_model_used"] is False
    assert full[2]["causal_availability_enforced"] is True
    assert "efficiency_vector" not in full[2]


def test_domain_return_uses_availability_not_event_time_and_persists():
    rows = [prama(index, xi=0.1) for index in range(5)]
    delayed = [outcome(1, 3, functional_gain=0.8)]
    prefix = compute(rows[:3], [], [])
    full = compute(rows, [], delayed)
    assert prefix == full[:3]
    assert all(
        row["benefit_vector"]["functional_gain"] is None for row in full[:3]
    )
    assert full[3]["benefit_vector"]["functional_gain"] == pytest.approx(0.8)
    assert full[4]["benefit_vector"]["functional_gain"] == pytest.approx(0.8)
    assert full[4]["temporal_scope"]["benefit"]["functional_gain"] == (
        "LATEST_CAUSALLY_AVAILABLE_SESSION_TO_DATE"
    )
    later_other_channel = outcome(
        4,
        4,
        functional_gain=None,
        verified_outcome=1.0,
    )
    with_later_other_channel = compute(
        rows,
        [],
        [*delayed, later_other_channel],
    )
    assert with_later_other_channel[4]["benefit_vector"][
        "functional_gain"
    ] == pytest.approx(0.8)


def test_temporal_identity_join_is_fail_closed():
    rows = [prama(0, xi=0.1), prama(1, xi=0.2)]
    mismatched = outcome(0, 1)
    mismatched["available_at_window"] = {"turn_index": 0, "window_index": 0}
    with pytest.raises(ValueError, match="available_at_index disagrees"):
        compute(rows, [], [mismatched])
    with pytest.raises(ValueError, match="no PRAMA window"):
        compute(rows, [structural(7)])
    with pytest.raises(ValueError, match="must be unique"):
        compute(rows, [], [outcome(0, 1), outcome(0, 1)])


def test_conversion_deficit_exposure_is_irreversible_session_to_date():
    rows = [prama(index, xi=0.5, capacity=1.0) for index in range(4)]
    result = compute(rows, [structural(index) for index in range(4)])
    exposure = [
        row["differential_dynamics"]["retained_friction_vs_structural_recovery"][
            "cumulative_conversion_deficit_exposure"
        ]
        for row in result
    ]
    assert exposure == sorted(exposure)
    assert exposure[0] > 0
    assert result[-1]["temporal_scope"]["dynamics"] == {
        "trend": "CURRENT_VS_PREVIOUS_OBSERVED_INDEX",
        "cumulative_conversion_deficit_exposure": "SESSION_TO_DATE",
        "positive_persistence": "ROLLING_WINDOW",
    }
    assert all(
        scope == "CURRENT_INDEX"
        for scope in result[-1]["temporal_scope"]["differential"].values()
    )


def test_controlled_cost_and_organization_monotonicity_and_joint_balance():
    name = "retained_friction_vs_adaptive_organization_level"

    def terminal(xi, organization):
        rows = [prama(index, xi=xi) for index in range(8)]
        geometry = [
            structural(index, coherence=organization, contraction=0.0)
            for index in range(8)
        ]
        return compute(rows, geometry)[-1]["differential_vector"][name]

    lower_cost = terminal(0.4, 0.3)
    higher_cost = terminal(0.5, 0.3)
    higher_benefit = terminal(0.4, 0.4)
    assert higher_cost > lower_cost
    assert higher_benefit < lower_cost
    assert terminal(0.3, 0.3) == pytest.approx(0.0)
    assert terminal(0.5, 0.5) == pytest.approx(0.0)

    numeric_residue = compute(
        [prama(index, xi=0.3) for index in range(8)],
        [
            structural(
                index,
                coherence=0.30000000000000004,
                contraction=0.0,
            )
            for index in range(8)
        ],
    )[-1]
    assert numeric_residue["numeric_epsilon"] == 1e-12
    assert numeric_residue["differential_vector"][name] == 0.0
    assert numeric_residue["differential_dynamics"][name][
        "cumulative_conversion_deficit_exposure"
    ] == 0.0


def test_real_structural_recovery_reduces_differential_without_erasing_exposure():
    capacities = [1.0, 0.9, 0.8, 0.6, 0.65, 0.7, 0.8, 0.95]
    rows = [
        prama(index, xi=0.3, capacity=capacity)
        for index, capacity in enumerate(capacities)
    ]
    result = compute(rows, [])
    name = "retained_friction_vs_structural_recovery"
    recovery = [row["benefit_vector"]["structural_recovery"] for row in result]
    differential = [row["differential_vector"][name] for row in result]
    exposure = [
        row["differential_dynamics"][name][
            "cumulative_conversion_deficit_exposure"
        ]
        for row in result
    ]
    assert recovery[3] == 0.0
    assert recovery[-1] == pytest.approx(0.35)
    assert differential[-1] < differential[3]
    assert differential[-1] < 0.0
    assert exposure == sorted(exposure)
    assert exposure[-1] == exposure[-2]


def test_empirical_threshold_separates_micro_noise_from_material_persistence():
    name = "retained_friction_vs_adaptive_organization_level"

    def trajectory(offset):
        rows = [prama(index, xi=0.5 + offset) for index in range(40)]
        geometry = [
            structural(index, coherence=0.5, contraction=0.0)
            for index in range(40)
        ]
        return compute(rows, geometry)

    micro = trajectory(0.000942)
    material = trajectory(0.02)
    assert micro[-1]["differential_threshold"] > 0.000942
    assert (
        micro[-1]["differential_dynamics"][name]["positive_persistence"]
        == 0.0
    )
    assert (
        material[-1]["differential_dynamics"][name]["positive_persistence"]
        == 1.0
    )
    assert (
        micro[-1]["differential_dynamics"][name][
            "cumulative_conversion_deficit_exposure"
        ]
        > 0.0
    )


def test_odce_does_not_turn_absence_into_zero_and_rejects_service_metadata():
    rows = [prama(0, xi=0.1)]
    rows[0]["accumulated_excess"] = None
    result = compute(rows, [])[0]
    assert result["cost_vector"]["accumulated_debt"] is None
    assert result["component_status"]["cost"]["accumulated_debt"] == "UNAVAILABLE"
    assert result["benefit_vector"]["adaptive_organization_level"] is None
    assert result["component_status"]["benefit"]["adaptive_organization_level"] == "UNAVAILABLE"
    rows[0]["finish_reason"] = "stop"
    with pytest.raises(ValueError, match="forbidden fields"):
        compute(rows, [])
    clean = [prama(0, xi=0.1)]
    with pytest.raises(ValueError, match="future_outcome"):
        compute(
            clean,
            [],
            [{**outcome(0, 0), "future_outcome": 1}],
        )


def test_odce_artifact_contract_is_fail_closed():
    observation = compute([prama(0, xi=0.1)], [structural(0)])[0]
    envelope = make_envelope(
        artifact_type="structural_conversion_differential",
        study_id="study-1",
        session_id="session-1",
        producer="pytest",
        created_at=datetime.now(timezone.utc).isoformat(),
        source_sha256=HASH_A,
        config_sha256=HASH_B,
        partition="exploratory",
        channel_status="OBSERVED",
    )
    record = make_structural_conversion_differential(envelope, observation)
    validate_artifact(record, "structural_conversion_differential")
    with pytest.raises(ArtifactValidationError, match="not a predictive model"):
        validate_artifact({**record, "predictive_model_used": True})
    with pytest.raises(ArtifactValidationError, match="efficiency_vector"):
        validate_artifact({**record, "efficiency_vector": {}})
    with pytest.raises(ArtifactValidationError, match="numeric_epsilon"):
        validate_artifact({**record, "numeric_epsilon": 0.0})
    with pytest.raises(ArtifactValidationError, match="contract_freeze_sha256"):
        validate_artifact({**record, "partition": "confirmatory"})


def test_confirmatory_contract_requires_calibration_governance_and_exact_freeze():
    status_flip = contract()
    status_flip["status"] = "FROZEN_PROSPECTIVE"
    with pytest.raises(ValueError, match="FROZEN_DOMAIN_CALIBRATION"):
        ODCEConfig.from_contract(status_flip)

    prospective = frozen_contract()
    freeze = freeze_manifest(prospective)
    ODCEConfig.from_contract(prospective)
    validate_contract_freeze(prospective, freeze)
    with pytest.raises(ValueError, match="normalization_contract_sha256 mismatch"):
        validate_contract_freeze(
            prospective,
            {**freeze, "normalization_contract_sha256": "c" * 64},
        )
    ambiguous = contract()
    ambiguous["correspondence"][0]["name"] = "cost_vs_recovery"
    ambiguous["temporal_scope"]["differential"] = {
        **ambiguous["temporal_scope"]["differential"],
        "cost_vs_recovery": "CURRENT_INDEX",
    }
    ambiguous["temporal_scope"]["differential"].pop(
        "retained_friction_vs_structural_recovery"
    )
    with pytest.raises(ValueError, match="exact cost and benefit channels"):
        ODCEConfig.from_contract(ambiguous)


def test_domain_return_artifact_contract():
    envelope = make_envelope(
        artifact_type="domain_return_observation",
        study_id="study-1",
        session_id="session-1",
        producer="pytest",
        created_at=datetime.now(timezone.utc).isoformat(),
        source_sha256=HASH_A,
        config_sha256=HASH_B,
        partition="exploratory",
        channel_status="OBSERVED",
    )
    record = {**envelope, **outcome(0, 1, verified_outcome=1.0)}
    validate_artifact(record, "domain_return_observation")
    with pytest.raises(ArtifactValidationError, match="before its event"):
        validate_artifact(
            {
                **record,
                "event_index": 1,
                "available_at_index": 0,
            }
        )


def test_canonical_odce_entry_point_emits_valid_jsonl(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "prama.jsonl"
    structural_source = tmp_path / "structural.jsonl"
    outcome_source = tmp_path / "outcomes.jsonl"
    output = tmp_path / "odce.jsonl"
    rows = []
    for index in range(10):
        envelope = make_envelope(
            artifact_type="prama_trajectory",
            study_id="study-1",
            session_id="session-1",
            producer="pytest",
            created_at=datetime.now(timezone.utc).isoformat(),
            source_sha256=HASH_A,
            config_sha256=HASH_B,
            partition="exploratory",
            channel_status="OBSERVED",
        )
        rows.append({**envelope, **prama(index, xi=0.1 + index * 0.05)})
    write_jsonl_atomic(source, rows)
    structural_rows = [canonical_structural(index) for index in range(10)]
    write_jsonl_atomic(structural_source, structural_rows)
    outcome_envelope = make_envelope(
        artifact_type="domain_return_observation",
        study_id="study-1",
        session_id="session-1",
        producer="pytest",
        created_at=datetime.now(timezone.utc).isoformat(),
        source_sha256=HASH_A,
        config_sha256=HASH_B,
        partition="exploratory",
        channel_status="OBSERVED",
    )
    write_jsonl_atomic(
        outcome_source,
        [{**outcome_envelope, **outcome(2, 8, functional_gain=0.7)}],
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "derive_structural_conversion.py"),
            "--prama",
            str(source),
            "--structural-observations",
            str(structural_source),
            "--outcomes",
            str(outcome_source),
            "--contract",
            str(root / "config" / "odce_v0_1_exploratory.json"),
            "--out",
            str(output),
            "--study-id",
            "study-1",
            "--producer",
            "pytest",
            "--partition",
            "exploratory",
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    emitted = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(emitted) == 10
    assert all(row["artifact_type"] == "structural_conversion_differential" for row in emitted)
    assert all(
        row["structural_observation_reference_sha256"] is not None
        for row in emitted
    )
    assert all(
        row["benefit_vector"]["adaptive_organization_level"] is None
        for row in emitted[:7]
    )
    assert all(
        row["component_status"]["benefit"]["adaptive_organization_level"]
        == "OBSERVED"
        for row in emitted[7:]
    )
    assert all(
        row["benefit_vector"]["functional_gain"] is None
        for row in emitted[:8]
    )
    assert emitted[8]["benefit_vector"]["functional_gain"] == pytest.approx(0.7)
    assert emitted[7]["domain_outcome_reference_sha256"] is None
    assert emitted[8]["domain_outcome_reference_sha256"] is not None

    confirmatory = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "derive_structural_conversion.py"),
            "--prama",
            str(source),
            "--contract",
            str(root / "config" / "odce_v0_1_exploratory.json"),
            "--out",
            str(tmp_path / "forbidden-confirmatory.jsonl"),
            "--study-id",
            "study-1",
            "--producer",
            "pytest",
            "--partition",
            "confirmatory",
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert confirmatory.returncode == 1
    assert "FROZEN_PROSPECTIVE" in confirmatory.stdout

    prospective = frozen_contract()
    freeze = freeze_manifest(prospective)
    contract_path = tmp_path / "odce-prospective.json"
    freeze_path = tmp_path / "odce-prospective.freeze.json"
    confirmatory_output = tmp_path / "confirmatory.jsonl"
    contract_path.write_text(json.dumps(prospective), encoding="utf-8")
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    accepted = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "derive_structural_conversion.py"),
            "--prama",
            str(source),
            "--contract",
            str(contract_path),
            "--contract-freeze",
            str(freeze_path),
            "--out",
            str(confirmatory_output),
            "--study-id",
            "study-1",
            "--producer",
            "pytest",
            "--partition",
            "confirmatory",
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert accepted.returncode == 0, accepted.stdout + accepted.stderr
    confirmatory_rows = [
        json.loads(line)
        for line in confirmatory_output.read_text(encoding="utf-8").splitlines()
    ]
    assert confirmatory_rows
    assert all(row["partition"] == "confirmatory" for row in confirmatory_rows)
    assert all(
        row["contract_freeze_sha256"] == sha256_value(freeze)
        for row in confirmatory_rows
    )


def _calibration_records(count=24, partition="calibration"):
    trajectory = [
        prama(
            index,
            xi=0.05 + index * 0.02,
            debt=index * index * 0.001,
            capacity=1.0 - index * 0.01,
            trend=(-0.01 if index % 2 else 0.005),
        )
        for index in range(count)
    ]
    observations = compute(
        trajectory,
        [
            structural(
                index,
                coherence=0.55 + index * 0.01,
                contraction=0.1 + (index % 3) * 0.05,
            )
            for index in range(count)
        ],
    )
    records = []
    for observation in observations:
        envelope = make_envelope(
            artifact_type="structural_conversion_differential",
            study_id="odce-calibration-test",
            session_id="calibration-session",
            producer="pytest",
            created_at=datetime.now(timezone.utc).isoformat(),
            source_sha256=HASH_A,
            config_sha256=HASH_B,
            partition=partition,
            channel_status="OBSERVED",
        )
        records.append(
            make_structural_conversion_differential(envelope, observation)
        )
    return records


def test_exploratory_calibration_is_partial_robust_and_never_freezes():
    calibrated, report = calibrate_exploratory_contract(
        _calibration_records(),
        contract(),
        min_observations=20,
        minimum_session_count=1,
    )
    assert calibrated["status"] == "EXPLORATORY_CAUSAL_POST_KERNEL"
    assert calibrated["normalization"]["confirmatory_use_allowed"] is False
    assert (
        calibrated["normalization"]["calibration_status"]
        == "EXPLORATORY_ROBUST_PARTIAL"
    )
    assert (
        report["channels"]["cost"]["retained_friction"]["status"]
        == "CALIBRATED_EXPLORATORY"
    )
    assert (
        report["channels"]["benefit"]["functional_gain"]["status"]
        == "INSUFFICIENT_OBSERVATIONS"
    )
    assert "benefit.functional_gain" in report["blocking_channels"]
    assert report["calibration_ready_for_freeze"] is False
    assert report["output_contract_sha256"] == sha256_value(calibrated)
    assert calibrated["differential_threshold"] == 0.0
    assert "differential_threshold_calibration" not in calibrated
    assert report["differential_threshold_recalibration_required"] is True
    assert "correspondence_governance" not in calibrated
    assert "Identity normalization" not in " ".join(calibrated["claim_boundary"])
    assert report["channels"]["cost"]["retained_friction"][
        "observed_session_count"
    ] == 1
    assert report["correspondences"][
        "retained_friction_vs_structural_recovery"
    ]["calibration_status"] == "PARTIALLY_CALIBRATED"
    assert report["correspondences"][
        "retained_friction_vs_adaptive_organization_level"
    ]["calibration_status"] == "PARTIALLY_CALIBRATED"
    assert report["correspondences"][
        "capacity_consumption_vs_functional_gain"
    ]["calibration_status"] == "PARTIALLY_CALIBRATED"
    assert calibrated["correspondence"][2]["instrumental_interpretation_allowed"] is False


def test_exploratory_calibration_requires_index_and_session_coverage():
    records = _calibration_records()
    _calibrated, report = calibrate_exploratory_contract(
        records,
        contract(),
        min_observations=20,
        minimum_observed_fraction=0.90,
        minimum_session_count=2,
        minimum_session_coverage=0.50,
    )
    friction = report["channels"]["cost"]["retained_friction"]
    assert friction["status"] == "INSUFFICIENT_SESSION_COUNT"
    assert friction["observed_fraction"] == 1.0
    assert friction["observed_session_count"] == 1
    functional = report["channels"]["benefit"]["functional_gain"]
    assert functional["status"] == "INSUFFICIENT_OBSERVATIONS"
    assert report["blocking_correspondences"]


def test_exploratory_calibration_requires_explicit_permission_and_cli(tmp_path):
    records = _calibration_records(partition="exploratory")
    with pytest.raises(ValueError, match="explicit allow_exploratory_input"):
        calibrate_exploratory_contract(records, contract())
    calibrated, _report = calibrate_exploratory_contract(
        records,
        contract(),
        allow_exploratory_input=True,
    )
    assert calibrated["normalization"]["confirmatory_use_allowed"] is False

    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "odce.jsonl"
    out_contract = tmp_path / "calibrated.json"
    out_report = tmp_path / "calibration-report.json"
    write_jsonl_atomic(source, records)
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "calibrate_odce_exploratory.py"),
            "--input",
            str(source),
            "--base-contract",
            str(root / "config" / "odce_v0_1_exploratory.json"),
            "--out-contract",
            str(out_contract),
            "--out-report",
            str(out_report),
            "--allow-exploratory-input",
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    emitted_contract = json.loads(out_contract.read_text(encoding="utf-8"))
    emitted_report = json.loads(out_report.read_text(encoding="utf-8"))
    assert emitted_report["output_contract_sha256"] == sha256_value(
        emitted_contract
    )
    assert emitted_report["status"] == "EXPLORATORY_NOT_FROZEN"


def test_empirical_noise_floor_calibration_preserves_normalization_and_cli(
    tmp_path,
):
    base = contract()
    normalization_hash = sha256_value(base["normalization"])
    records = _calibration_records()
    calibrated, report = calibrate_exploratory_differential_threshold(
        records,
        base,
        correspondence_names=[
            "retained_friction_vs_adaptive_organization_level"
        ],
        stable_condition_id="pytest-declared-stable-condition",
        min_observations=10,
        residual_quantile=0.9,
    )
    assert calibrated["differential_threshold"] > 0.0
    assert sha256_value(calibrated["normalization"]) == normalization_hash
    assert report["normalization_modified"] is False
    assert report["calibration_ready_for_freeze"] is False
    assert report["output_contract_sha256"] == sha256_value(calibrated)

    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "stable-odce.jsonl"
    out_contract = tmp_path / "threshold-contract.json"
    out_report = tmp_path / "threshold-report.json"
    write_jsonl_atomic(source, records)
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "calibrate_odce_differential_threshold.py"),
            "--input",
            str(source),
            "--base-contract",
            str(root / "config" / "odce_v0_1_exploratory.json"),
            "--correspondence",
            "retained_friction_vs_adaptive_organization_level",
            "--stable-condition-id",
            "pytest-declared-stable-condition",
            "--min-observations",
            "10",
            "--residual-quantile",
            "0.9",
            "--out-contract",
            str(out_contract),
            "--out-report",
            str(out_report),
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    cli_contract = json.loads(out_contract.read_text(encoding="utf-8"))
    cli_report = json.loads(out_report.read_text(encoding="utf-8"))
    assert cli_report["output_contract_sha256"] == sha256_value(cli_contract)
    assert sha256_value(cli_contract["normalization"]) == normalization_hash
