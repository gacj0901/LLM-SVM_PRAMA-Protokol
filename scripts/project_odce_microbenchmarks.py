#!/usr/bin/env python3
"""Project microbenchmark raw logprobs to canonical PRAMA and D_O v9 inputs.

The projection boundary receives numeric token observations only. Benchmark
verification and provider termination metadata are joined after PRAMA and D_O
v9 projection, and are never used as structural inputs.
"""

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
from aptadynamic_llm.structural_observation import (  # noqa: E402
    make_structural_observation,
    observe_structural_trajectory,
)
from scripts.analyze_frontend_test_battery import (  # noqa: E402
    atomic_json,
    chunks,
    normalized_tokens,
)
from scripts.project_cocc_operator_geometry import file_sha256, load_json  # noqa: E402
from scripts.project_cocc_prama import validate_identity  # noqa: E402
from scripts.project_cocc_prama_dynamic import project as project_dynamic  # noqa: E402


REPORT_SCHEMA = "LLM-SVM-odce-microbenchmark-projection/1"
RAW_SCHEMA = "LLM-SVM-odce-microbenchmark-raw/1"


def canonical_sha256(value: Any) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected an object")
        rows.append(value)
    return rows


def numeric_request(
    raw: Mapping[str, Any], turn: Mapping[str, Any], source_hash: str, observer_hash: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tokens = normalized_tokens({"tokens": turn.get("tokens") or []})
    if not tokens:
        raise ValueError(f"{raw.get('session_id')}: no numeric token observations")
    request = {
        "schema": "LLM-SVM-CoCC-projector-request/1",
        "session_id": str(raw["session_id"]),
        "model_id": str(raw.get("resolved_model") or raw.get("model") or "unknown"),
        "source_session_sha256": source_hash,
        "input_channel_status": "OBSERVED",
        "observer_contract_sha256": observer_hash,
        "turns": [
            {
                "turn_index": int(turn.get("turn_index") or 0),
                "token_count": len(tokens),
                "tokens": tokens,
            }
        ],
    }
    serialized = json.dumps(request, sort_keys=True)
    for forbidden in (
        "user_message",
        "assistant_message",
        "prompt",
        "verifier",
        "finish_reason",
        "response_time_seconds",
        "label",
    ):
        if f'"{forbidden}"' in serialized:
            raise ValueError(f"numeric projection request leaked {forbidden}")
    return request, tokens


def _load_outcomes(source_runs: Sequence[Path]) -> list[dict[str, Any]]:
    outcomes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source_run in source_runs:
        path = source_run / "domain_return_observations.jsonl"
        if not path.exists():
            raise ValueError(f"missing outcomes file: {path}")
        for row in read_jsonl(path):
            validate_artifact(row, "domain_return_observation")
            session_id = str(row["session_id"])
            if session_id in seen:
                raise ValueError(f"duplicate domain outcome for session {session_id}")
            seen.add(session_id)
            outcomes.append(row)
    return outcomes


def project(args: argparse.Namespace) -> dict[str, Any]:
    if args.prama_source_root:
        source = args.prama_source_root.resolve()
        import_root = source / "src" if (source / "src").is_dir() else source
        if not import_root.is_dir():
            raise ValueError(f"PRAMA import root does not exist: {import_root}")
        sys.path.insert(0, str(import_root))
        for module_name in tuple(sys.modules):
            if module_name == "prama_protokol" or module_name.startswith(
                "prama_protokol."
            ):
                del sys.modules[module_name]
    dynamic_contract = load_json(args.dynamic_contract)
    geometry_contract = load_json(args.geometry_contract)
    coherence_contract = load_json(args.coherence_contract)
    v9_contract = load_json(args.v9_contract)
    if v9_contract.get("status") != "PRIMARY_STRUCTURAL_OBSERVER":
        raise ValueError("--v9-contract must declare PRIMARY_STRUCTURAL_OBSERVER")
    dynamic_hash = file_sha256(args.dynamic_contract)
    geometry_hash = file_sha256(args.geometry_contract)
    coherence_hash = file_sha256(args.coherence_contract)
    v9_hash = canonical_sha256(v9_contract)
    kernel_config, columns, identity = validate_identity(
        args.declaration, args.recertification
    )
    window_size = int(dynamic_contract["input"]["window_size_tokens"])
    geometry_config = OperatorGeometryConfig.from_contract(geometry_contract)
    coherence_config = StructuralCoherenceV6Config.from_contract(coherence_contract)
    v9_config = StructuralCoherenceV9Config.from_contract(v9_contract)

    raw_paths: list[Path] = []
    for source_run in args.source_run:
        raw_paths.extend(source_run.glob("sessions/*/raw.json"))
    raw_paths = sorted(
        {path.resolve() for path in raw_paths}, key=lambda path: str(path).casefold()
    )
    if not raw_paths:
        raise ValueError("no sessions/*/raw.json files found")

    prama_rows: list[dict[str, Any]] = []
    structural_rows: list[dict[str, Any]] = []
    sessions: list[dict[str, Any]] = []
    seen_sessions: set[str] = set()
    created_at = datetime.now(timezone.utc).isoformat()
    for ordinal, raw_path in enumerate(raw_paths, 1):
        raw = load_json(raw_path)
        if raw.get("schema") != RAW_SCHEMA:
            raise ValueError(f"unexpected raw schema: {raw_path}")
        session_id = str(raw.get("session_id") or "")
        if not session_id or session_id in seen_sessions:
            raise ValueError(f"missing or duplicate session identity: {session_id!r}")
        seen_sessions.add(session_id)
        turns = list(raw.get("turns") or [])
        if len(turns) != 1 or int(turns[0].get("turn_index") or 0) != 0:
            raise ValueError(f"{session_id}: microbenchmark requires exactly turn_index 0")
        turn = turns[0]
        source_hash = file_sha256(raw_path)
        request, tokens = numeric_request(raw, turn, source_hash, dynamic_hash)
        trajectory = project_dynamic(
            request,
            dynamic_contract,
            dynamic_hash,
            kernel_config,
            columns,
            identity,
        )
        for row in trajectory:
            row["study_id"] = args.study_id
            row["session_id"] = session_id
            row["producer"] = "scripts.project_odce_microbenchmarks/PRAMA"
            row["partition"] = "exploratory"
            validate_artifact(row, "prama_trajectory")
        token_windows = chunks(tokens, window_size)
        if len(token_windows) != len(trajectory):
            raise ValueError(f"{session_id}: token/PRAMA window count mismatch")
        v6_windows = observe_structural_coherence_v6(
            token_windows, trajectory, geometry_config, coherence_config
        )
        observations = observe_structural_trajectory(v6_windows, v9_config)
        if len(observations) != len(trajectory):
            raise ValueError(f"{session_id}: D_O/PRAMA window count mismatch")
        envelope = make_envelope(
            artifact_type="structural_observation",
            study_id=args.study_id,
            session_id=session_id,
            producer="scripts.project_odce_microbenchmarks/D_O_v9",
            created_at=created_at,
            source_sha256=source_hash,
            config_sha256=v9_hash,
            partition="exploratory",
            channel_status="OBSERVED",
        )
        canonical_observations = [
            make_structural_observation(envelope, observation)
            for observation in observations
        ]
        for prama, observation in zip(trajectory, canonical_observations, strict=True):
            prama_identity = (int(prama["turn_index"]), int(prama["window_index"]))
            observation_identity = (
                int(observation["turn_index"]),
                int(observation["window_index"]),
            )
            if prama_identity != observation_identity:
                raise ValueError(f"{session_id}: temporal identity mismatch")
        prama_rows.extend(trajectory)
        structural_rows.extend(canonical_observations)
        sessions.append(
            {
                "session_id": session_id,
                "model": str(raw.get("resolved_model") or raw.get("model") or "unknown"),
                "source_raw": str(raw_path),
                "source_raw_sha256": source_hash,
                "window_count": len(trajectory),
            }
        )
        print(f"[{ordinal}/{len(raw_paths)}] projected {session_id}: {len(trajectory)} windows")

    outcomes = _load_outcomes(args.source_run)
    outcome_sessions = {str(row["session_id"]) for row in outcomes}
    if outcome_sessions != seen_sessions:
        raise ValueError(
            "outcome/session identity mismatch: "
            f"missing={sorted(seen_sessions - outcome_sessions)}, "
            f"unknown={sorted(outcome_sessions - seen_sessions)}"
        )
    prama_keys: dict[str, list[tuple[int, int]]] = {}
    for row in prama_rows:
        prama_keys.setdefault(str(row["session_id"]), []).append(
            (int(row["turn_index"]), int(row["window_index"]))
        )
    for outcome in outcomes:
        session_id = str(outcome["session_id"])
        keys = prama_keys[session_id]
        for identity_name in ("event_window", "available_at_window"):
            window_identity = outcome[identity_name]
            key = (
                int(window_identity["turn_index"]),
                int(window_identity["window_index"]),
            )
            if key not in keys:
                raise ValueError(f"{session_id}: {identity_name} is absent from PRAMA")
        if int(outcome["event_index"]) != keys.index(
            (
                int(outcome["event_window"]["turn_index"]),
                int(outcome["event_window"]["window_index"]),
            )
        ):
            raise ValueError(f"{session_id}: outcome event_index identity mismatch")
        if int(outcome["available_at_index"]) != keys.index(
            (
                int(outcome["available_at_window"]["turn_index"]),
                int(outcome["available_at_window"]["window_index"]),
            )
        ):
            raise ValueError(f"{session_id}: outcome available_at_index identity mismatch")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prama_path = args.output_dir / "prama_trajectory.jsonl"
    structural_path = args.output_dir / "structural_observations.jsonl"
    outcomes_path = args.output_dir / "domain_return_observations.jsonl"
    write_jsonl_atomic(prama_path, prama_rows)
    write_jsonl_atomic(structural_path, structural_rows)
    write_jsonl_atomic(outcomes_path, outcomes)
    report = {
        "schema": REPORT_SCHEMA,
        "generated_at": created_at,
        "status": "EXPLORATORY_OFFLINE_PROJECTION",
        "study_id": args.study_id,
        "source_runs": [str(path.resolve()) for path in args.source_run],
        "prama_source_root": (
            str(args.prama_source_root.resolve()) if args.prama_source_root else None
        ),
        "session_count": len(sessions),
        "prama_observation_count": len(prama_rows),
        "structural_observation_count": len(structural_rows),
        "domain_outcome_count": len(outcomes),
        "projection_boundary": "numeric_token_observations_only",
        "provider_termination_metadata_used": False,
        "external_evaluation_joined_after_projection": True,
        "kernel_modified": False,
        "primary_structural_state_modified": False,
        "contract_hashes": {
            "dynamic_observer": dynamic_hash,
            "operator_geometry": geometry_hash,
            "structural_coherence_v6": coherence_hash,
            "D_O_v9": v9_hash,
            "kernel": identity["config_sha256"],
        },
        "outputs": {
            "prama": str(prama_path.resolve()),
            "structural_observations": str(structural_path.resolve()),
            "outcomes": str(outcomes_path.resolve()),
        },
        "sessions": sessions,
    }
    atomic_json(args.output_dir / "report.json", report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--study-id", default="odce-microbenchmarks-v1")
    parser.add_argument(
        "--prama-source-root",
        type=Path,
        help="Explicit git checkout of the PRAMA source bound by the declaration.",
    )
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
    args = parse_args(argv)
    try:
        report = project(args)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"ODCE microbenchmark projection failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
