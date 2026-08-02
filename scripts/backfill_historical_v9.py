#!/usr/bin/env python3
"""Blind offline v6+v9 backfill for historical raw logprob sessions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for candidate in (REPO_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from aptadynamic_llm.operator_geometry import OperatorGeometryConfig  # noqa: E402
from aptadynamic_llm.structural_coherence_v6 import (  # noqa: E402
    StructuralCoherenceV6Config,
    observe_structural_coherence_v6,
)
from aptadynamic_llm.structural_coherence_v9 import (  # noqa: E402
    StructuralCoherenceV9Config,
    classify_structural_coherence_v9,
)
from scripts.analyze_frontend_test_battery import (  # noqa: E402
    atomic_json,
    chunks,
    normalized_tokens,
    prama_endpoints,
    utc_now,
)
from scripts.analyze_frontend_test_battery_v9 import (  # noqa: E402
    aggregate,
    session_endpoints,
)
from scripts.project_cocc_operator_geometry import file_sha256, load_json  # noqa: E402
from scripts.project_cocc_prama import validate_identity  # noqa: E402
from scripts.project_cocc_prama_dynamic import project as project_dynamic  # noqa: E402


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    return cleaned[:140] or "session"


def numeric_request(
    raw: Mapping[str, Any], turn: Mapping[str, Any], source_sha256: str, observer_hash: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tokens = normalized_tokens({"tokens": turn.get("tokens") or []})
    session_id = f"{raw.get('session_id', 'historical')}--turn-{int(turn.get('turn_index', 0))}"
    request = {
        "schema": "LLM-SVM-CoCC-projector-request/1",
        "session_id": session_id,
        "model_id": str(raw.get("resolved_model") or raw.get("model") or raw.get("requested_model") or "unknown"),
        "source_session_sha256": source_sha256,
        "input_channel_status": "OBSERVED",
        "observer_contract_sha256": observer_hash,
        "turns": [{"turn_index": 0, "token_count": len(tokens), "tokens": tokens}],
    }
    serialized = json.dumps(request, sort_keys=True)
    for forbidden in ("user_message", "assistant_message", "prompt", "verifier", "label"):
        if f'"{forbidden}"' in serialized:
            raise ValueError(f"numeric request leaked {forbidden}")
    return request, tokens


def compact_endpoints(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    episodes = dict(result.get("episodes") or {})
    for key, item in list(episodes.items()):
        item = dict(item)
        item.pop("episodes", None)
        episodes[key] = item
    result["episodes"] = episodes
    return result


def sha256_list(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item).casefold()):
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(file_sha256(path).encode("ascii"))
    return digest.hexdigest()


def group_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    keys = sorted({(str(row["model"]), str(row["finish_reason"])) for row in rows})
    for model, finish_reason in keys:
        members = [
            row for row in rows
            if row["model"] == model and row["finish_reason"] == finish_reason
        ]
        evaluable = [
            row for row in members
            if row["endpoints"]["session_evaluation_status"] == "EVALUABLE"
        ]
        groups.append({
            "model": model,
            "finish_reason": finish_reason,
            "n": len(members),
            "evaluable_n": len(evaluable),
            "mean_token_count": sum(int(row["token_count"]) for row in members) / len(members),
            "structural": aggregate(evaluable),
        })
    return groups


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("run_outputs/historical_v9_backfill"))
    parser.add_argument("--dynamic-contract", type=Path, default=Path("config/cocc_dynamic_observer_contract_v1.json"))
    parser.add_argument("--geometry-contract", type=Path, default=Path("config/cocc_operator_geometry_observer_v1.json"))
    parser.add_argument("--coherence-contract", type=Path, default=Path("config/sequor_structural_coherence_observer_v6.json"))
    parser.add_argument("--v9-contract", type=Path, default=Path("config/sequor_structural_coherence_observer_v9.json"))
    parser.add_argument("--declaration", type=Path, default=Path("config/window_prama_kernel_declaration.json"))
    parser.add_argument("--recertification", type=Path, default=Path("run_outputs/window_prama_recertification_v030_20260730.json"))
    parser.add_argument("--historical-kernel-label", default="PRAMA kernel v1")
    parser.add_argument("--historical-kernel-assertion-source", default="user_declared")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dynamic_contract = load_json(args.dynamic_contract)
    geometry_contract = load_json(args.geometry_contract)
    coherence_contract = load_json(args.coherence_contract)
    v9_contract = load_json(args.v9_contract)
    dynamic_hash = file_sha256(args.dynamic_contract)
    geometry_hash = file_sha256(args.geometry_contract)
    coherence_hash = file_sha256(args.coherence_contract)
    v9_hash = file_sha256(args.v9_contract)
    kernel_config, columns, identity = validate_identity(args.declaration, args.recertification)
    window_size = int(dynamic_contract["input"]["window_size_tokens"])
    geometry_config = OperatorGeometryConfig.from_contract(geometry_contract)
    coherence_config = StructuralCoherenceV6Config.from_contract(coherence_contract)
    v9_config = StructuralCoherenceV9Config.from_contract(v9_contract)

    raw_paths: list[Path] = []
    for source_run in args.source_run:
        raw_paths.extend(source_run.glob("sessions/**/raw.json"))
    raw_paths = sorted(set(path.resolve() for path in raw_paths), key=lambda item: str(item).casefold())
    if not raw_paths:
        raise SystemExit("no historical sessions/**/raw.json found")

    projection_dir = args.output_dir / "projections"
    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    total_turns = sum(len(load_json(path).get("turns") or []) for path in raw_paths)
    progress = 0
    for raw_path in raw_paths:
        raw = load_json(raw_path)
        source_sha = file_sha256(raw_path)
        source_run = raw_path.parents[2].name
        for turn in raw.get("turns") or []:
            progress += 1
            turn_index = int(turn.get("turn_index", 0))
            if not turn.get("tokens"):
                exclusions.append({
                    "source_run": source_run, "session_id": raw.get("session_id"),
                    "turn_index": turn_index, "reason": "NO_NUMERIC_TOKEN_OBSERVATIONS",
                })
                continue
            model = str(raw.get("resolved_model") or raw.get("model") or raw.get("requested_model") or "unknown")
            stem = safe_name(f"{source_run}--{raw.get('session_id')}--turn-{turn_index}")
            projection_path = projection_dir / f"{stem}.json"
            reusable = False
            if projection_path.exists() and not args.overwrite:
                existing = load_json(projection_path)
                reusable = bool(
                    existing.get("source_raw_sha256") == source_sha
                    and existing.get("v9_contract_sha256") == v9_hash
                    and existing.get("structural_coherence_v6_contract_sha256") == coherence_hash
                )
            if reusable:
                projection = existing
                print(f"[{progress}/{total_turns}] reusing {stem}", flush=True)
            else:
                print(f"[{progress}/{total_turns}] projecting {stem}", flush=True)
                request, tokens = numeric_request(raw, turn, source_sha, dynamic_hash)
                trajectory = project_dynamic(
                    request, dynamic_contract, dynamic_hash, kernel_config, columns, identity
                )
                token_windows = chunks(tokens, window_size)
                if len(token_windows) != len(trajectory):
                    raise ValueError(f"token/trajectory window mismatch: {raw_path}")
                v6_windows = observe_structural_coherence_v6(
                    token_windows, trajectory, geometry_config, coherence_config
                )
                windows = classify_structural_coherence_v9(v6_windows, v9_config)
                projection = {
                    "schema": "LLM-SVM-historical-structural-projection/9",
                    "generated_at": utc_now(),
                    "source_run": source_run,
                    "session_id": str(raw.get("session_id")),
                    "turn_index": turn_index,
                    "model": model,
                    "source_raw_sha256": source_sha,
                    "dynamic_observer_contract_sha256": dynamic_hash,
                    "operator_geometry_contract_sha256": geometry_hash,
                    "structural_coherence_v6_contract_sha256": coherence_hash,
                    "v9_contract_sha256": v9_hash,
                    "contains_prompt_or_answer": False,
                    "contains_external_evaluation": False,
                    "numeric_channels_recomputed": True,
                    "historical_generation_context": {
                        "original_kernel_label": args.historical_kernel_label,
                        "assertion_source": args.historical_kernel_assertion_source,
                        "artifact_bound_identity_available": False,
                    },
                    "reprojection_context": {
                        "interpretation": "counterfactual_reprojection_of_historical_logprobs",
                        "legacy_prama_states_reused": False,
                        "raw_numeric_token_observations_reused": True,
                        "delta_xi_recomputed": True,
                        "replay_kernel_identity": identity,
                    },
                    "kernel_modified": False,
                    "prama_trajectory": trajectory,
                    "structural_windows": windows,
                }
                atomic_json(projection_path, projection)

            windows = projection["structural_windows"]
            trajectory = projection["prama_trajectory"]
            metadata = raw.get("metadata") or {}
            endpoints = compact_endpoints(session_endpoints(windows, v9_config))
            rows.append({
                "source_run": source_run,
                "session_id": str(raw.get("session_id")),
                "turn_index": turn_index,
                "model": model,
                "provider": str(raw.get("provider") or "unknown"),
                "problem_id": str(raw.get("problem_id") or metadata.get("problem_id") or ""),
                "difficulty": metadata.get("difficulty"),
                "perturbation_type": raw.get("perturbation_type") or metadata.get("perturbation_type"),
                "finish_reason": str(turn.get("finish_reason") or "unknown"),
                "token_count": int(turn.get("token_count") or len(turn.get("tokens") or [])),
                "projection_path": str(projection_path.resolve()),
                "projection_sha256": file_sha256(projection_path),
                "prama": prama_endpoints(trajectory),
                "endpoints": endpoints,
            })

    horizons: list[dict[str, Any]] = []
    for horizon in v9_contract["fixed_causal_horizons_windows"]:
        members: list[dict[str, Any]] = []
        for row in rows:
            projection = load_json(Path(row["projection_path"]))
            windows = projection["structural_windows"]
            if len(windows) < int(horizon):
                continue
            endpoints = compact_endpoints(session_endpoints(windows[: int(horizon)], v9_config))
            members.append({**row, "endpoints": endpoints})
        warmup = int(horizon) < int(v9_contract["horizon_rules"]["minimum_inferential_horizon"])
        horizons.append({
            "horizon_windows": int(horizon),
            "approximate_token_boundary": int(horizon) * window_size,
            "analysis_status": "DESCRIPTIVE_WARMUP_ONLY" if warmup else "STRUCTURALLY_EVALUABLE_FIXED_HORIZON",
            "sessions_reaching_horizon": len(members),
            "groups": group_summary(members),
        })

    report = {
        "schema": "LLM-SVM-historical-v9-backfill/1",
        "generated_at": utc_now(),
        "status": "EXPLORATORY_OFFLINE_REPROJECTION",
        "source_runs": [str(path.resolve()) for path in args.source_run],
        "source_raw_file_count": len(raw_paths),
        "source_bundle_sha256": sha256_list(raw_paths),
        "projected_turn_count": len(rows),
        "excluded_turn_count": len(exclusions),
        "projection_boundary": "numeric_token_observations_only",
        "contains_prompt_or_answer": False,
        "external_verifier_labels_present": False,
        "finish_reason_is_not_interpreted_as_functional_correctness": True,
        "historical_generation_context": {
            "original_kernel_label": args.historical_kernel_label,
            "assertion_source": args.historical_kernel_assertion_source,
            "artifact_bound_identity_available": False,
        },
        "reprojection_context": {
            "interpretation": "counterfactual_reprojection_of_historical_logprobs",
            "legacy_prama_states_reused": False,
            "raw_numeric_token_observations_reused": True,
            "delta_xi_recomputed": True,
            "replay_kernel_identity": identity,
        },
        "interpretation_boundary": [
            "This is not v9 operating on the historical kernel v1 state trajectory.",
            "Historical token observations were replayed through the current recertified kernel before v6 plus v9 classification.",
            "Observer effects cannot be separated from kernel-version effects without a matched kernel-v1 replay.",
        ],
        "kernel_modified": False,
        "contract_hashes": {
            "dynamic": dynamic_hash, "operator_geometry": geometry_hash,
            "structural_coherence_v6": coherence_hash, "v9": v9_hash,
        },
        "groups": group_summary(rows),
        "fixed_horizons": horizons,
        "items": rows,
        "exclusions": exclusions,
    }
    report_path = args.output_dir / "report.json"
    atomic_json(report_path, report)
    print(json.dumps({
        "output": str(report_path), "projected_turns": len(rows),
        "excluded_turns": len(exclusions), "source_raw_files": len(raw_paths),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
