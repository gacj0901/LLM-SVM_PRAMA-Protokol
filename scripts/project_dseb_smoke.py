#!/usr/bin/env python3
"""Close DSEB smoke causality through PRAMA, D_O v9, outcomes, and ODCE-v0."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from aptadynamic_llm.artifact_schema import (  # noqa: E402
    ChannelStatus,
    make_envelope,
    validate_artifact,
    write_jsonl_atomic,
)
from aptadynamic_llm.operator_geometry import OperatorGeometryConfig  # noqa: E402
from aptadynamic_llm.structural_coherence_v6 import (  # noqa: E402
    StructuralCoherenceV6Config,
    observe_structural_coherence_v6,
)
from aptadynamic_llm.structural_coherence_v9 import StructuralCoherenceV9Config  # noqa: E402
from aptadynamic_llm.structural_conversion import (  # noqa: E402
    ODCEConfig,
    compute_structural_conversion_trajectory,
    make_structural_conversion_differential,
)
from aptadynamic_llm.structural_observation import (  # noqa: E402
    make_structural_observation,
    observe_structural_trajectory,
)
from benchmarks.dseb_v0.generator import DSEBGenerator  # noqa: E402
from benchmarks.dseb_v0.interaction import parse_order_response  # noqa: E402
from benchmarks.dseb_v0.preflight import canonical_sha256, file_sha256  # noqa: E402
from benchmarks.dseb_v0.protocol import load_protocol  # noqa: E402
from benchmarks.dseb_v0.verifier import verify_order  # noqa: E402
from scripts.analyze_frontend_test_battery import (  # noqa: E402
    atomic_json,
    chunks,
    normalized_tokens,
)
from scripts.project_cocc_operator_geometry import load_json  # noqa: E402
from scripts.project_cocc_prama import validate_identity  # noqa: E402
from scripts.project_cocc_prama_dynamic import project as project_dynamic  # noqa: E402
from scripts.run_dseb_smoke import RAW_SCHEMA, REPORT_SCHEMA as ACQUISITION_REPORT_SCHEMA  # noqa: E402


REPORT_SCHEMA = "LLM-SVM-DSEB-smoke-causal-projection/1"
STUDY_ID = "DSEB-v0-smoke"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected an object")
        rows.append(row)
    return rows


def _load_prama_source(path: Path | None) -> None:
    if path is None:
        return
    source = path.resolve()
    import_root = source / "src" if (source / "src").is_dir() else source
    if not import_root.is_dir():
        raise ValueError(f"PRAMA import root does not exist: {import_root}")
    sys.path.insert(0, str(import_root))
    for module_name in tuple(sys.modules):
        if module_name == "prama_protokol" or module_name.startswith("prama_protokol."):
            del sys.modules[module_name]


def _numeric_request(
    raw: Mapping[str, Any], *, observer_hash: str, source_hash: str
) -> tuple[dict[str, Any], list[list[dict[str, Any]]]]:
    request_turns: list[dict[str, Any]] = []
    token_windows: list[list[dict[str, Any]]] = []
    window_size = int(raw["window_size_tokens"])
    turns = list(raw.get("turns") or [])
    if [int(turn.get("turn_index", -1)) for turn in turns] != list(range(12)):
        raise ValueError("DSEB smoke raw must contain contiguous turns 0..11")
    for turn in turns:
        tokens = normalized_tokens({"tokens": turn.get("tokens") or []})
        request_turns.append(
            {
                "turn_index": int(turn["turn_index"]),
                "token_count": len(tokens),
                "tokens": tokens,
            }
        )
        token_windows.extend(chunks(tokens, window_size))
    request = {
        "schema": "LLM-SVM-CoCC-projector-request/1",
        "session_id": str(raw["session_id"]),
        "model_id": str(raw.get("resolved_model") or raw.get("model") or "unknown"),
        "source_session_sha256": source_hash,
        "input_channel_status": "OBSERVED",
        "observer_contract_sha256": observer_hash,
        "turns": request_turns,
    }
    serialized = json.dumps(request, sort_keys=True)
    for forbidden in (
        "user_message",
        "assistant_message",
        "finish_reason",
        "response_time_seconds",
        "phase",
        "constraint",
        "verified_outcome",
    ):
        if f'"{forbidden}"' in serialized:
            raise ValueError(f"numeric projection request leaked {forbidden}")
    return request, token_windows


def _promote_outcomes(
    *,
    raw: Mapping[str, Any],
    raw_path: Path,
    previews: Sequence[Mapping[str, Any]],
    prama_rows: Sequence[Mapping[str, Any]],
    protocol_hash: str,
) -> list[dict[str, Any]]:
    keys = [
        (int(row["turn_index"]), int(row["window_index"])) for row in prama_rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("PRAMA temporal identities are not unique")
    if len(previews) != 12:
        raise ValueError("DSEB smoke requires twelve verifier previews")
    verifier_hash = file_sha256(ROOT / "benchmarks" / "dseb_v0" / "verifier.py")
    outcomes: list[dict[str, Any]] = []
    for expected_turn, preview in enumerate(previews):
        if int(preview["benchmark_turn_index"]) != expected_turn:
            raise ValueError("verifier previews are not in contiguous turn order")
        event_window = dict(preview["event_window"])
        key = (int(event_window["turn_index"]), int(event_window["window_index"]))
        if key not in keys:
            raise ValueError(f"preview event window is absent from PRAMA: {key}")
        event_index = keys.index(key)
        if int(preview["event_index"]) != event_index:
            raise ValueError("preview event_index disagrees with canonical PRAMA order")
        integration_status = str(preview["external_integration_status"])
        benefit = {
            "functional_gain": float(preview["functional_gain"]),
            "external_integration": (
                float(preview["external_integration"])
                if preview.get("external_integration") is not None
                else None
            ),
            "verified_outcome": float(preview["verified_outcome"]),
        }
        outcome = {
            "contract_version": "0.2.0",
            "artifact_type": "domain_return_observation",
            "artifact_version": "1.0.0",
            "study_id": STUDY_ID,
            "session_id": str(raw["session_id"]),
            "producer": "scripts/project_dseb_smoke.py",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_sha256": file_sha256(raw_path),
            "config_sha256": protocol_hash,
            "partition": "exploratory",
            "channel_status": "OBSERVED",
            "benchmark_turn_index": expected_turn,
            "event_index": event_index,
            "available_at_index": event_index,
            "event_window": event_window,
            "available_at_window": event_window,
            "benefit_vector": benefit,
            "component_status": {
                "functional_gain": "OBSERVED",
                "external_integration": integration_status,
                "verified_outcome": "OBSERVED",
            },
            "verifier_reference_sha256": verifier_hash,
            "retrospective_backfill": False,
            "causal_availability_declared": True,
            "provider_termination_metadata_used": False,
        }
        validate_artifact(outcome, "domain_return_observation")
        outcomes.append(outcome)
    return outcomes


def _verify_after_structural_projection(
    *,
    raw: Mapping[str, Any],
    prama_rows: Sequence[Mapping[str, Any]],
    previews: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Execute the authoritative verifier after D_O and compare its early preview."""

    protocol_path = Path(str(raw["protocol_source"]))
    protocol_hash = file_sha256(protocol_path)
    if protocol_hash != raw["protocol_source_sha256"]:
        raise ValueError("protocol source changed after model acquisition")
    protocol = load_protocol(protocol_path)
    generated = DSEBGenerator(protocol, int(raw["seed"])).generate()
    if canonical_sha256(generated.to_dict()) != raw["generated_protocol_sha256"]:
        raise ValueError("generated DSEB trajectory changed after acquisition")
    turns = list(raw.get("turns") or [])
    if len(turns) != len(generated.turns) or len(previews) != len(turns):
        raise ValueError("raw, generated protocol, and preview turn counts differ")
    keys = [
        (int(row["turn_index"]), int(row["window_index"])) for row in prama_rows
    ]
    verified_rows: list[dict[str, Any]] = []
    for state, turn, preview in zip(
        generated.turns, turns, previews, strict=True
    ):
        turn_index = state.target.turn_index
        turn_keys = [key for key in keys if key[0] == turn_index]
        if not turn_keys:
            raise ValueError(f"turn {turn_index} has no PRAMA windows")
        event_key = turn_keys[-1]
        event_index = keys.index(event_key)
        parsed = parse_order_response(
            str(turn["assistant_message"]), generated.symbols
        )
        verification = verify_order(
            symbols=generated.symbols,
            order=parsed.order,
            persistent=state.active_constraints,
            temporary=state.temporary_constraints,
            new_constraint_ids=state.new_constraint_ids,
        )
        recomputed = {
            "benchmark_turn_index": turn_index,
            "event_window": {
                "turn_index": event_key[0],
                "window_index": event_key[1],
            },
            "event_index": event_index,
            **parsed.to_dict(),
            **verification.to_dict(),
        }
        comparison_fields = (
            "event_window",
            "event_index",
            "response_contract_valid",
            "syntax_valid",
            "functional_gain",
            "external_integration",
            "external_integration_status",
            "verified_outcome",
        )
        mismatched = [
            name
            for name in comparison_fields
            if preview.get(name) != recomputed.get(name)
        ]
        if mismatched:
            raise ValueError(
                f"turn {turn_index}: post-D_O verification differs from preview: "
                f"{mismatched}"
            )
        verified_rows.append(recomputed)
    return verified_rows


def project(args: argparse.Namespace) -> dict[str, Any]:
    acquisition_run = args.acquisition_run.resolve()
    acquisition_report = load_json(acquisition_run / "report.json")
    if acquisition_report.get("schema") != ACQUISITION_REPORT_SCHEMA:
        raise ValueError("unexpected DSEB acquisition report schema")
    if acquisition_report.get("status") != "ACQUIRED_PENDING_STRUCTURAL_PROJECTION":
        raise ValueError("DSEB acquisition is incomplete or already not in pending state")
    raw_paths = list((acquisition_run / "sessions").glob("*/raw.json"))
    if len(raw_paths) != 1:
        raise ValueError("DSEB smoke acquisition must contain exactly one raw session")
    raw_path = raw_paths[0]
    raw = load_json(raw_path)
    if raw.get("schema") != RAW_SCHEMA or raw.get("profile") != "smoke":
        raise ValueError("unexpected DSEB smoke raw artifact")
    if raw.get("contract_freeze_sha256") is not None:
        raise ValueError("exploratory DSEB raw must not declare a contract freeze")
    previews = read_jsonl(acquisition_run / "verifier_previews.jsonl")

    _load_prama_source(args.prama_source_root)
    dynamic_contract = load_json(args.dynamic_contract)
    geometry_contract = load_json(args.geometry_contract)
    coherence_contract = load_json(args.coherence_contract)
    v9_contract = load_json(args.v9_contract)
    odce_contract = load_json(args.odce_contract)
    if v9_contract.get("status") != "PRIMARY_STRUCTURAL_OBSERVER":
        raise ValueError("--v9-contract must declare PRIMARY_STRUCTURAL_OBSERVER")
    dynamic_hash = file_sha256(args.dynamic_contract)
    v9_hash = canonical_sha256(v9_contract)
    kernel_config, columns, kernel_identity = validate_identity(
        args.declaration, args.recertification
    )
    window_size = int(dynamic_contract["input"]["window_size_tokens"])
    if int(raw["window_size_tokens"]) != window_size:
        raise ValueError("acquisition and dynamic observer window sizes differ")
    source_hash = file_sha256(raw_path)
    request, token_windows = _numeric_request(
        raw, observer_hash=dynamic_hash, source_hash=source_hash
    )
    trajectory = project_dynamic(
        request,
        dynamic_contract,
        dynamic_hash,
        kernel_config,
        columns,
        kernel_identity,
    )
    for row in trajectory:
        row["study_id"] = STUDY_ID
        row["producer"] = "scripts.project_dseb_smoke/PRAMA"
        row["partition"] = "exploratory"
        validate_artifact(row, "prama_trajectory")
    if len(token_windows) != len(trajectory):
        raise ValueError("token/PRAMA window count mismatch")

    geometry_config = OperatorGeometryConfig.from_contract(geometry_contract)
    coherence_config = StructuralCoherenceV6Config.from_contract(coherence_contract)
    v9_config = StructuralCoherenceV9Config.from_contract(v9_contract)
    v6_windows = observe_structural_coherence_v6(
        token_windows, trajectory, geometry_config, coherence_config
    )
    observations = observe_structural_trajectory(v6_windows, v9_config)
    if len(observations) != len(trajectory):
        raise ValueError("D_O v9/PRAMA window count mismatch")
    created_at = datetime.now(timezone.utc).isoformat()
    structural_envelope = make_envelope(
        artifact_type="structural_observation",
        study_id=STUDY_ID,
        session_id=str(raw["session_id"]),
        producer="scripts.project_dseb_smoke/D_O_v9",
        created_at=created_at,
        source_sha256=source_hash,
        config_sha256=v9_hash,
        partition="exploratory",
        channel_status="OBSERVED",
    )
    structural_rows = [
        make_structural_observation(structural_envelope, observation)
        for observation in observations
    ]
    for prama, observation in zip(trajectory, structural_rows, strict=True):
        if (int(prama["turn_index"]), int(prama["window_index"])) != (
            int(observation["turn_index"]),
            int(observation["window_index"]),
        ):
            raise ValueError("PRAMA/D_O temporal identity mismatch")

    causal_verifications = _verify_after_structural_projection(
        raw=raw,
        prama_rows=trajectory,
        previews=previews,
    )
    protocol_hash = str(acquisition_report["protocol_source_sha256"])
    outcomes = _promote_outcomes(
        raw=raw,
        raw_path=raw_path,
        previews=causal_verifications,
        prama_rows=trajectory,
        protocol_hash=protocol_hash,
    )
    odce_config = ODCEConfig.from_contract(odce_contract)
    odce_observations = compute_structural_conversion_trajectory(
        trajectory,
        structural_rows,
        outcomes,
        odce_config,
        normalization_contract=odce_contract["normalization"],
        correspondence_contract=odce_contract["correspondence"],
        contract_freeze_sha256=None,
    )
    combined_source = sha256(
        json.dumps(
            {
                "prama": trajectory,
                "structural": structural_rows,
                "outcomes": outcomes,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    odce_hash = canonical_sha256(odce_contract)
    odce_rows = [
        make_structural_conversion_differential(
            make_envelope(
                artifact_type="structural_conversion_differential",
                study_id=STUDY_ID,
                session_id=str(raw["session_id"]),
                producer="scripts.project_dseb_smoke/ODCE_v0",
                created_at=created_at,
                source_sha256=combined_source,
                config_sha256=odce_hash,
                partition="exploratory",
                channel_status=ChannelStatus.OBSERVED,
            ),
            observation,
        )
        for observation in odce_observations
    ]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prama_path = output_dir / "prama_trajectory.jsonl"
    structural_path = output_dir / "structural_observations.jsonl"
    outcomes_path = output_dir / "domain_return_observations.jsonl"
    odce_path = output_dir / "structural_conversion_differentials.jsonl"
    write_jsonl_atomic(prama_path, trajectory)
    write_jsonl_atomic(structural_path, structural_rows)
    write_jsonl_atomic(outcomes_path, outcomes)
    write_jsonl_atomic(odce_path, odce_rows)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "COMPLETE_EXPLORATORY_CAUSAL",
        "benchmark_id": "DSEB_v0",
        "benchmark_version": "DSEB-v0",
        "profile": "smoke",
        "partition": "exploratory",
        "contract_freeze_sha256": None,
        "session_id": str(raw["session_id"]),
        "prama_observation_count": len(trajectory),
        "structural_observation_count": len(structural_rows),
        "domain_outcome_count": len(outcomes),
        "odce_observation_count": len(odce_rows),
        "causal_stage_order": [
            "terminal_window_closed",
            "O_D_observed",
            "PRAMA_projected",
            "D_O_v9_observed",
            "outcome_verified",
            "outcome_registered",
            "ODCE_executed",
        ],
        "causal_stage_order_validated": True,
        "outcomes_joined_after_structural_projection": True,
        "verifier_recomputed_after_structural_projection": True,
        "event_equals_available_at_allowed_before_first_odce_evaluation": True,
        "retrospective_backfill": False,
        "future_outcome_used": False,
        "provider_termination_metadata_used": False,
        "kernel_modified": False,
        "primary_structural_state_modified": False,
        "normalization_status": odce_contract["normalization"]["calibration_status"],
        "claim_boundary": (
            "Exploratory causal DSEB smoke projection; identity normalization is "
            "not an empirical calibration and no contract is frozen."
        ),
        "contract_hashes": {
            "dynamic_observer": dynamic_hash,
            "operator_geometry": file_sha256(args.geometry_contract),
            "structural_coherence_v6": file_sha256(args.coherence_contract),
            "D_O_v9": v9_hash,
            "ODCE_v0": odce_hash,
            "kernel": kernel_identity["config_sha256"],
        },
        "outputs": {
            "prama": str(prama_path),
            "structural_observations": str(structural_path),
            "domain_outcomes": str(outcomes_path),
            "odce": str(odce_path),
        },
    }
    atomic_json(output_dir / "report.json", report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prama-source-root", type=Path)
    parser.add_argument(
        "--dynamic-contract",
        type=Path,
        default=Path("config/cocc_dynamic_observer_contract_v1.json"),
    )
    parser.add_argument(
        "--geometry-contract",
        type=Path,
        default=Path("config/cocc_operator_geometry_observer_v1.json"),
    )
    parser.add_argument(
        "--coherence-contract",
        type=Path,
        default=Path("config/sequor_structural_coherence_observer_v6.json"),
    )
    parser.add_argument(
        "--v9-contract",
        type=Path,
        default=Path("config/sequor_structural_observer_v9.json"),
    )
    parser.add_argument(
        "--odce-contract",
        type=Path,
        default=Path("config/odce_v0_1_exploratory.json"),
    )
    parser.add_argument(
        "--declaration",
        type=Path,
        default=Path("config/window_prama_kernel_declaration.json"),
    )
    parser.add_argument(
        "--recertification",
        type=Path,
        default=Path("run_outputs/window_prama_recertification_v030_20260730.json"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        report = project(parse_args(argv))
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"DSEB smoke projection failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
