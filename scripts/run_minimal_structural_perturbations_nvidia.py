#!/usr/bin/env python
"""Run the complete 3-item x 4-arm minimal-perturbation study on NVIDIA.

One baseline completion is shared by the four branches of each item. Only the
numeric token observations from the perturbed response turn cross the observer
boundary. Every completed API response is saved immediately and can be reused.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for candidate in (REPO_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from aptadynamic_llm.model_payload import task_only_messages  # noqa: E402
from scripts.project_cocc_operator_geometry import file_sha256  # noqa: E402
from scripts.project_cocc_prama import project, validate_identity  # noqa: E402
from scripts.project_cocc_structural_mobility_v3 import build_artifact  # noqa: E402


NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_API_KEY_ENV = "NVIDIA_API_KEY"
DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b"
EXPECTED_ARMS = (
    "control_neutral",
    "concrete_content",
    "abstract_content",
    "minimal_structural",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if tuple(protocol.get("arms") or ()) != EXPECTED_ARMS:
        raise ValueError(f"protocol arms must be exactly {EXPECTED_ARMS}")
    items = protocol.get("diagnostic_items")
    if not isinstance(items, list) or not items:
        raise ValueError("protocol contains no diagnostic items")
    for item in items:
        perturbations = item.get("perturbations") or {}
        required = {"control_neutral", "concrete_content", "abstract_content", "minimal_structural_rule"}
        if not required.issubset(perturbations):
            raise ValueError(f"{item.get('item_id')}: incomplete perturbation mapping")
    return protocol


def perturbation_text(item: Mapping[str, Any], arm: str) -> str:
    key = "minimal_structural_rule" if arm == "minimal_structural" else arm
    return str(item["perturbations"][key])


def trials(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "item_id": str(item["item_id"]),
            "topic": str(item["topic"]),
            "arm": arm,
            "baseline_prompt": str(item["baseline_prompt"]),
            "tracked_commitment": str(item["tracked_commitment"]),
            "perturbation": perturbation_text(item, arm),
        }
        for item in protocol["diagnostic_items"]
        for arm in EXPECTED_ARMS
    ]


def entropy(logprobs: Sequence[float]) -> float:
    values = [float(value) for value in logprobs if math.isfinite(float(value))]
    if not values:
        return 0.0
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    total = sum(weights)
    return -sum((weight / total) * math.log(weight / total) for weight in weights if weight > 0.0)


def response_turn(response: Any, turn_index: int, user_message: str, response_seconds: float) -> dict[str, Any]:
    choice = response.choices[0]
    content = list(getattr(getattr(choice, "logprobs", None), "content", None) or [])
    tokens = []
    for entry in content:
        chosen = float(entry.logprob)
        candidates = sorted(
            {chosen, *[float(candidate.logprob) for candidate in list(entry.top_logprobs or [])]},
            reverse=True,
        )
        if not math.isfinite(chosen) or any(not math.isfinite(value) for value in candidates):
            raise RuntimeError("provider returned non-finite token logprobs")
        tokens.append(
            {
                "token": str(entry.token),
                "top1_logprob": chosen,
                "top_logprobs": candidates,
                "gap": candidates[0] - candidates[1] if len(candidates) > 1 else 0.0,
                "entropy": entropy(candidates),
            }
        )
    if not tokens:
        raise RuntimeError("provider returned no token logprobs")
    return {
        "turn_index": turn_index,
        "user_message": user_message,
        "assistant_message": str(choice.message.content or ""),
        "finish_reason": str(choice.finish_reason or ""),
        "token_count": len(tokens),
        "response_seconds": response_seconds,
        "tokens": tokens,
        "provider_response_id": str(getattr(response, "id", "") or ""),
        "system_fingerprint": str(getattr(response, "system_fingerprint", "") or ""),
    }


def call_model(messages: Sequence[Mapping[str, str]], args: argparse.Namespace, turn_index: int) -> tuple[dict[str, Any], str]:
    key = os.environ.get(NVIDIA_API_KEY_ENV, "").strip()
    if not key:
        raise RuntimeError(f"{NVIDIA_API_KEY_ENV} is not set in this PowerShell session")
    from openai import OpenAI

    clean_messages = task_only_messages(messages)
    last_error: Exception | None = None
    for attempt in range(1, args.max_attempts + 1):
        started = time.perf_counter()
        try:
            response = OpenAI(api_key=key, base_url=args.base_url, timeout=args.timeout).chat.completions.create(
                model=args.model,
                messages=clean_messages,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                seed=args.seed,
                logprobs=True,
                top_logprobs=args.top_logprobs,
                stream=False,
                extra_body={"chat_template_kwargs": {"enable_thinking": False, "force_nonempty_content": True}},
            )
            elapsed = time.perf_counter() - started
            return response_turn(response, turn_index, clean_messages[-1]["content"], elapsed), str(response.model or args.model)
        except Exception as exc:  # provider failures are retried and persisted responses are reused
            last_error = exc
            print(f"attempt {attempt}/{args.max_attempts} failed: {type(exc).__name__}: {exc}", flush=True)
            if attempt < args.max_attempts:
                time.sleep(args.retry_sleep)
    raise RuntimeError(f"generation failed after {args.max_attempts} attempts: {last_error}")


def numeric_request(raw: Mapping[str, Any], calibration_hash: str) -> dict[str, Any]:
    perturbation_turn = raw["turns"][-1]
    tokens = [
        {
            "top1_logprob": float(token["top1_logprob"]),
            "top_logprobs": [float(value) for value in token["top_logprobs"]],
            "gap": float(token["gap"]),
            "entropy": float(token["entropy"]),
        }
        for token in perturbation_turn["tokens"]
    ]
    request = {
        "schema": "LLM-SVM-CoCC-projector-request/1",
        "session_id": str(raw["session_id"]),
        "model_id": str(raw["model"]),
        "source_session_sha256": sha256(json.dumps(raw, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest(),
        "input_channel_status": "OBSERVED",
        "calibration_reference_sha256": calibration_hash,
        "turns": [{"turn_index": 1, "token_count": len(tokens), "tokens": tokens}],
    }
    serialized = json.dumps(request, sort_keys=True)
    for forbidden in ("arm", "topic", "prompt", "assistant_message", "tracked_commitment", "perturbation"):
        if f'"{forbidden}"' in serialized:
            raise ValueError(f"numeric request leaked {forbidden}")
    return request


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def mobility_endpoints(artifact: Mapping[str, Any], trajectory: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    windows = [window for window in artifact["windows"] if window["geometry_ready"]]
    active = [window for window in windows if float(window["path_length"]) > 0.5]
    crystallized = [window for window in windows if window["structural_mobility_state"] == "CRYSTALLIZED"]
    onset = int(crystallized[0]["window_index"]) if crystallized else None
    return {
        "n_windows": len(artifact["windows"]),
        "geometry_ready_windows": len(windows),
        "active_window_count": len(active),
        "active_window_fraction": len(active) / len(windows) if windows else 0.0,
        "maximum_path_length": max((float(window["path_length"]) for window in windows), default=0.0),
        "mean_path_length": statistics.fmean(
            [float(window["path_length"]) for window in windows]
        ) if windows else 0.0,
        "minimum_transport_efficiency": min(
            (float(window["transport_efficiency"]) for window in active), default=None
        ),
        "minimum_transport_efficiency_all_ready": min(
            (float(window["transport_efficiency"]) for window in windows), default=None
        ),
        "recurrence_persistence_area": statistics.fmean(
            [float(window["recurrence_persistence"]) for window in windows]
        ) if windows else 0.0,
        "maximum_recurrence_persistence": max(
            (float(window["recurrence_persistence"]) for window in windows), default=0.0
        ),
        "crystallized_fraction": len(crystallized) / len(windows) if windows else 0.0,
        "crystallization_onset": onset,
        "max_delta": max((float(row["delta"]) for row in trajectory if row.get("delta") is not None), default=0.0),
        "max_xi": max((float(row["xi"]) for row in trajectory if row.get("xi") is not None), default=0.0),
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metric_names = (
        "token_count",
        "response_seconds",
        "n_windows",
        "active_window_fraction",
        "maximum_path_length",
        "mean_path_length",
        "minimum_transport_efficiency_all_ready",
        "recurrence_persistence_area",
        "maximum_recurrence_persistence",
        "crystallized_fraction",
        "max_delta",
        "max_xi",
    )
    by_arm = {}
    for arm in EXPECTED_ARMS:
        members = [row for row in rows if row["arm"] == arm]
        by_arm[arm] = {
            "n": len(members),
            "finish_reasons": {
                reason: sum(row["finish_reason"] == reason for row in members)
                for reason in sorted({str(row["finish_reason"]) for row in members})
            },
            **{
                metric: {
                    "mean": statistics.fmean(
                        float(row[metric]) for row in members if row[metric] is not None
                    ),
                    "median": statistics.median(
                        float(row[metric]) for row in members if row[metric] is not None
                    ),
                }
                for metric in metric_names
            },
        }
    primary_contrasts = []
    for item_id in sorted({str(row["item_id"]) for row in rows}):
        minimal = next(row for row in rows if row["item_id"] == item_id and row["arm"] == "minimal_structural")
        abstract = next(row for row in rows if row["item_id"] == item_id and row["arm"] == "abstract_content")
        primary_contrasts.append(
            {
                "item_id": item_id,
                "recurrence_persistence_area_minimal_minus_abstract": float(minimal["recurrence_persistence_area"]) - float(abstract["recurrence_persistence_area"]),
                "maximum_recurrence_persistence_minimal_minus_abstract": float(minimal["maximum_recurrence_persistence"]) - float(abstract["maximum_recurrence_persistence"]),
                "transport_efficiency_all_ready_abstract_minus_minimal": float(abstract["minimum_transport_efficiency_all_ready"]) - float(minimal["minimum_transport_efficiency_all_ready"]),
                "active_window_fraction_minimal_minus_abstract": float(minimal["active_window_fraction"]) - float(abstract["active_window_fraction"]),
                "mean_path_length_minimal_minus_abstract": float(minimal["mean_path_length"]) - float(abstract["mean_path_length"]),
                "xi_minimal_minus_abstract": float(minimal["max_xi"]) - float(abstract["max_xi"]),
            }
        )
    return {"by_arm": by_arm, "minimal_vs_abstract_by_item": primary_contrasts}


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> Path:
    protocol = load_protocol(args.protocol)
    planned = trials(protocol)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        atomic_json(args.output_dir / "plan.json", {
            "schema": "LLM-SVM-minimal-structural-perturbation-plan/1",
            "protocol_sha256": file_sha256(args.protocol),
            "model": args.model,
            "baseline_calls": len(protocol["diagnostic_items"]),
            "perturbation_calls": len(planned),
            "total_calls": len(protocol["diagnostic_items"]) + len(planned),
            "trials": planned,
        })
        return args.output_dir / "plan.json"

    calibration_hash = file_sha256(args.calibration)
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    if calibration["model_id"] != args.model:
        raise ValueError("selected model differs from frozen calibration")
    kernel_config, columns, identity = validate_identity(args.declaration, args.recertification)
    geometry_hash = file_sha256(args.geometry_contract)
    mobility_hash = file_sha256(args.mobility_contract)
    mobility_freeze = json.loads(args.mobility_freeze.read_text(encoding="utf-8"))
    if mobility_freeze["observer_contract_sha256"] != mobility_hash:
        raise ValueError("mobility contract differs from freeze")

    baselines: dict[str, dict[str, Any]] = {}
    for item in protocol["diagnostic_items"]:
        item_id = str(item["item_id"])
        path = args.output_dir / "baselines" / item_id / "raw.json"
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            print(f"reusing baseline {item_id}: finish={raw['turn']['finish_reason']}, tokens={raw['turn']['token_count']}", flush=True)
        else:
            print(f"acquiring baseline {item_id}", flush=True)
            turn, resolved = call_model([{"role": "user", "content": str(item["baseline_prompt"])}], args, 0)
            raw = {"schema": "LLM-SVM-minimal-structural-baseline/1", "item_id": item_id, "model": resolved, "turn": turn}
            atomic_json(path, raw)
            print(f"completed baseline {item_id}: finish={turn['finish_reason']}, tokens={turn['token_count']}, seconds={turn['response_seconds']:.2f}", flush=True)
        baselines[item_id] = raw

    result_rows = []
    for index, trial in enumerate(planned, start=1):
        item_id = trial["item_id"]
        arm = trial["arm"]
        session_id = f"{item_id}--{arm}"
        session_dir = args.output_dir / "sessions" / session_id
        raw_path = session_dir / "raw.json"
        baseline = baselines[item_id]["turn"]
        if raw_path.exists():
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            turn = raw["turns"][-1]
            print(f"[{index}/{len(planned)}] reusing {session_id}: finish={turn['finish_reason']}, tokens={turn['token_count']}", flush=True)
        else:
            print(f"[{index}/{len(planned)}] acquiring {session_id}", flush=True)
            turn, resolved = call_model(
                [
                    {"role": "user", "content": trial["baseline_prompt"]},
                    {"role": "assistant", "content": baseline["assistant_message"]},
                    {"role": "user", "content": trial["perturbation"]},
                ],
                args,
                1,
            )
            raw = {
                "schema": "LLM-SVM-minimal-structural-session/1",
                "session_id": session_id,
                "item_id": item_id,
                "topic": trial["topic"],
                "arm": arm,
                "model": resolved,
                "tracked_commitment": trial["tracked_commitment"],
                "turns": [baseline, turn],
            }
            atomic_json(raw_path, raw)
            print(f"[{index}/{len(planned)}] completed {session_id}: finish={turn['finish_reason']}, tokens={turn['token_count']}, seconds={turn['response_seconds']:.2f}", flush=True)

        request = numeric_request(raw, calibration_hash)
        request_path = session_dir / "projection_request.json"
        atomic_json(request_path, request)
        trajectory = project(request, calibration, calibration_hash, kernel_config, columns, identity)
        trajectory_path = session_dir / "trajectory.jsonl"
        write_jsonl(trajectory_path, trajectory)
        artifact = build_artifact(request_path, trajectory_path, args.geometry_contract, args.mobility_contract)
        artifact_path = session_dir / "structural_mobility_v3.json"
        atomic_json(artifact_path, artifact)
        endpoint = mobility_endpoints(artifact, trajectory)
        result_rows.append(
            {
                "session_id": session_id,
                "item_id": item_id,
                "arm": arm,
                "model": raw["model"],
                "finish_reason": turn["finish_reason"],
                "token_count": turn["token_count"],
                "response_seconds": turn["response_seconds"],
                **endpoint,
            }
        )

    write_csv(args.output_dir / "session_endpoints.csv", result_rows)
    report = {
        "schema": "LLM-SVM-minimal-structural-perturbation-report/2",
        "generated_at": utc_now(),
        "study_id": protocol["study_id"],
        "protocol_path": str(args.protocol),
        "protocol_sha256": file_sha256(args.protocol),
        "model": args.model,
        "calibration_sha256": calibration_hash,
        "base_geometry_contract_sha256": geometry_hash,
        "mobility_contract_sha256": mobility_hash,
        "model_payload_boundary": "baseline_prompt, shared_baseline_answer, and arm instruction only",
        "observer_boundary": "perturbation-turn numeric token observations only",
        "llm_call_plan": {"shared_baselines": len(protocol["diagnostic_items"]), "perturbation_branches": len(planned)},
        "session_count": len(result_rows),
        "summary": summarize(result_rows),
        "session_endpoints": result_rows,
    }
    report_path = args.output_dir / "report.json"
    atomic_json(report_path, report)
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("run_outputs/minimal_structural_perturbations_nemotron_super_v1"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=NVIDIA_BASE_URL)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--top-logprobs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--calibration", type=Path, default=Path("config/cocc_nemotron3_super_calibration_v1.json"))
    parser.add_argument("--declaration", type=Path, default=Path("config/window_prama_kernel_declaration.json"))
    parser.add_argument("--recertification", type=Path, default=Path("run_outputs/window_prama_recertification_v030_20260730.json"))
    parser.add_argument("--geometry-contract", type=Path, default=Path("config/cocc_operator_geometry_observer_v1.json"))
    parser.add_argument("--mobility-contract", type=Path, default=Path("config/cocc_structural_mobility_observer_v3.json"))
    parser.add_argument("--mobility-freeze", type=Path, default=Path("config/cocc_structural_mobility_observer_v3.freeze.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        output = run(args)
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"minimal structural perturbation run failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
