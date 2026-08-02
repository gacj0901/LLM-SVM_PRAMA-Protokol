#!/usr/bin/env python3
"""Offline structural reprojection and external compliance audit for the frontend battery."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for candidate in (REPO_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from aptadynamic_llm.operator_geometry import OperatorGeometryConfig  # noqa: E402
from aptadynamic_llm.structural_coherence_v6 import (  # noqa: E402
    PRIMARY_STATES,
    StructuralCoherenceV6Config,
    observe_structural_coherence_v6,
)
from scripts.project_cocc_operator_geometry import file_sha256, load_json  # noqa: E402
from scripts.project_cocc_prama import validate_identity  # noqa: E402
from scripts.project_cocc_prama_dynamic import project as project_dynamic  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_value(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def token_entropy(logprobs: Sequence[float]) -> float:
    values = [float(value) for value in logprobs if math.isfinite(float(value))]
    if not values:
        return 0.0
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    total = sum(weights)
    return -sum((weight / total) * math.log(weight / total) for weight in weights if weight > 0.0)


def normalized_tokens(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = []
    for token in item.get("tokens") or []:
        chosen = float(token["top1_logprob"])
        candidates = sorted(
            {chosen, *[float(value) for value in token.get("top_logprobs") or []]}, reverse=True
        )
        if not math.isfinite(chosen) or any(not math.isfinite(value) for value in candidates):
            raise ValueError("non-finite token logprob")
        output.append({
            "top1_logprob": chosen,
            "top_logprobs": candidates,
            "gap": candidates[0] - candidates[1] if len(candidates) > 1 else 0.0,
            "entropy": token_entropy(candidates),
        })
    if not output:
        raise ValueError("item contains no numeric token observations")
    return output


def numeric_request(item: Mapping[str, Any], observer_hash: str) -> dict[str, Any]:
    tokens = normalized_tokens(item)
    session_id = f"frontend--{str(item['model']).replace('/', '--')}--{item['test_id']}"
    request = {
        "schema": "LLM-SVM-CoCC-projector-request/1",
        "session_id": session_id,
        "model_id": str(item["model"]),
        "source_session_sha256": sha256_value(item),
        "input_channel_status": "OBSERVED",
        "observer_contract_sha256": observer_hash,
        "turns": [{"turn_index": 0, "token_count": len(tokens), "tokens": tokens}],
    }
    serialized = json.dumps(request, sort_keys=True)
    for forbidden in ("assistant_message", "model_payload", "test_id", "protocol_completion", "failure_mode"):
        if f'"{forbidden}"' in serialized:
            raise ValueError(f"numeric request leaked {forbidden}")
    return request


def chunks(values: Sequence[Any], size: int) -> list[list[Any]]:
    return [list(values[start : start + size]) for start in range(0, len(values), size)]


def mean_or_none(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def structural_endpoints(windows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ready = [row for row in windows if row["geometry_ready"]]
    coherence = [float(row["transport_coherence"]) for row in ready if row["transport_coherence"] is not None]
    capacity = [float(row["variation_capacity"]) for row in ready if row["variation_capacity"] is not None]
    contraction = [float(row["variation_contraction"]) for row in ready if row["variation_contraction"] is not None]
    primary = [str(row["primary_state"]) for row in ready if row["primary_state"] is not None]
    statuses = [str(row["classification_status"]) for row in windows]
    return {
        "window_count": len(windows),
        "geometry_ready_windows": len(ready),
        "mean_movement": mean_or_none([float(row["movement"]) for row in ready]),
        "mean_trajectory_openness": mean_or_none([
            float(row["trajectory_openness"]) for row in ready if row["trajectory_openness"] is not None
        ]),
        "mean_inertial_echo_recurrence": mean_or_none([
            float(row["inertial_echo_recurrence"]) for row in ready
        ]),
        "mean_recurrence_persistence": mean_or_none([float(row["recurrence_persistence"]) for row in ready]),
        "mean_transport_coherence": mean_or_none(coherence),
        "terminal_transport_coherence": coherence[-1] if coherence else None,
        "mean_variation_capacity": mean_or_none(capacity),
        "maximum_variation_contraction": max(contraction) if contraction else None,
        "fraction_diagnostic_only": statuses.count("DIAGNOSTIC_ONLY") / len(statuses) if statuses else None,
        "fraction_insufficient_geometry": statuses.count("INSUFFICIENT_GEOMETRY") / len(statuses) if statuses else None,
        **{
            f"fraction_{state.lower()}": primary.count(state) / len(ready) if ready else None
            for state in PRIMARY_STATES
        },
        "fraction_imitative_echo": (
            sum(bool(row["diagnostics"]["imitative_echo"]) for row in ready) / len(ready) if ready else None
        ),
        "fraction_coherence_loss": (
            sum(bool(row["diagnostics"]["coherence_loss"]) for row in ready) / len(ready) if ready else None
        ),
        "first_recurrent_window": next(
            (int(row["absolute_window_index"]) for row in windows if row["primary_state"] == "RECURRENT"), None
        ),
        "first_crystallizing_window": next(
            (int(row["absolute_window_index"]) for row in windows if row["primary_state"] == "CRYSTALLIZING"), None
        ),
        "first_crystallized_window": next(
            (int(row["absolute_window_index"]) for row in windows if row["primary_state"] == "CRYSTALLIZED"), None
        ),
    }


def prama_endpoints(trajectory: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [row for row in trajectory if row.get("valid")]
    def values(name: str) -> list[float]:
        return [float(row[name]) for row in valid if row.get(name) is not None]
    return {
        "max_delta": max(values("delta"), default=None),
        "max_xi": max(values("xi"), default=None),
        "min_balance": min(values("balance"), default=None),
        "max_accumulated_excess": max(values("accumulated_excess"), default=None),
        "min_capacity": min(values("capacity"), default=None),
    }


def sentence_count(text: str) -> int:
    return len([part for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]) if text.strip() else 0


def compression_summary(text: str) -> str:
    match = re.search(r"(?is)(?:\*\*)?(?:existencia|respuesta final|no puede existir)", text)
    prefix = text[: match.start()] if match else text
    prefix = re.sub(r"(?im)^\s*(?:#+\s*)?\*\*?resumen:?\*\*?\s*$", "", prefix).strip()
    return prefix


def valid_orders(text: str) -> list[list[int]]:
    candidates = []
    for match in re.finditer(r"(?<!\d)([1-5])\s*(?:[,–—-]\s*|\s+)([1-5])\s*(?:[,–—-]\s*|\s+)([1-5])\s*(?:[,–—-]\s*|\s+)([1-5])\s*(?:[,–—-]\s*|\s+)([1-5])(?!\d)", text):
        order = [int(value) for value in match.groups()]
        if sorted(order) == [1, 2, 3, 4, 5]:
            candidates.append(order)
    return candidates


def constraint_checks(text: str) -> dict[str, Any]:
    orders = valid_orders(text)
    order = orders[-1] if orders else None
    if order is None:
        checks = {"two_before_five": False, "one_not_adjacent_four": False,
                  "one_between_three_five": False, "four_after_two": False}
    else:
        position = {value: index for index, value in enumerate(order)}
        checks = {
            "two_before_five": position[2] < position[5],
            "one_not_adjacent_four": abs(position[1] - position[4]) != 1,
            "one_between_three_five": abs(position[3] - position[5]) == 2,
            "four_after_two": position[4] > position[2],
        }
    return {"extracted_order": order, "checks": checks, "content_compliant": bool(order) and all(checks.values())}


def revision_checks(text: str) -> dict[str, Any]:
    normalized = text.casefold()
    conjunction = bool(re.search(r"b\s*(?:∧|&|and|y|junto\s+(?:a|con))\s*c", normalized))
    checks = {
        "identifies_b_c_conjunction": conjunction,
        "treats_a_as_nonessential": "a no" in normalized or "a es irrelevante" in normalized or "sin a" in normalized,
        "states_alternative": "alternativa" in normalized,
        "identifies_contradiction": "contradic" in normalized or "viola" in normalized or "incompatible" in normalized,
        "revises_conclusion": "revisi" in normalized or "conclusión final" in normalized or "modelo final" in normalized,
    }
    return {"checks": checks, "content_compliant": all(checks.values())}


def compression_checks(text: str) -> dict[str, Any]:
    normalized = text.casefold()
    summary = compression_summary(text)
    count = sentence_count(summary)
    checks = {
        "summary_at_most_four_sentences": 1 <= count <= 4,
        "red_implies_stable": "rojo" in normalized and "estable" in normalized,
        "stable_light_excludes_opaque": all(term in normalized for term in ("estable", "liviano", "opaco")),
        "blue_light_triangle_exception": all(term in normalized for term in ("azul", "liviano", "triang")),
        "blue_triangle_implies_opaque": all(term in normalized for term in ("azul", "triang", "opaco")),
        "opaque_excludes_red": "opaco" in normalized and "rojo" in normalized and ("ningún" in normalized or "no puede" in normalized),
        "rejects_joint_red_blue_triangle": "no puede existir" in normalized or "imposible" in normalized,
    }
    return {"summary_sentence_count": count, "checks": checks, "content_compliant": all(checks.values())}


def external_evaluation(item: Mapping[str, Any]) -> dict[str, Any]:
    text = str(item.get("assistant_message") or "")
    finish = str(item.get("finish_reason") or "")
    test_id = str(item["test_id"])
    detail = {
        "constraint": constraint_checks,
        "revision": revision_checks,
        "compression": compression_checks,
    }[test_id](text)
    response_present = bool(text.strip())
    natural_stop = finish == "stop"
    complete = bool(response_present and natural_stop and detail["content_compliant"])
    if finish == "length":
        failure = "LENGTH_EXHAUSTION"
    elif not response_present:
        failure = "EMPTY_RESPONSE"
    elif not detail["content_compliant"]:
        failure = "INSTRUCTION_OR_CONTENT_NONCOMPLIANCE"
    else:
        failure = None
    return {
        "protocol_completion": complete,
        "failure_mode": failure,
        "response_present": response_present,
        "natural_stop": natural_stop,
        **detail,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--battery-dir", type=Path, default=Path("FrontEnd/results/blind_interaction_battery_v1"))
    parser.add_argument("--dynamic-contract", type=Path, default=Path("config/cocc_dynamic_observer_contract_v1.json"))
    parser.add_argument("--geometry-contract", type=Path, default=Path("config/cocc_operator_geometry_observer_v1.json"))
    parser.add_argument("--coherence-contract", type=Path, default=Path("config/sequor_structural_coherence_observer_v6.json"))
    parser.add_argument("--declaration", type=Path, default=Path("config/window_prama_kernel_declaration.json"))
    parser.add_argument("--recertification", type=Path, default=Path("run_outputs/window_prama_recertification_v030_20260730.json"))
    args = parser.parse_args()

    dynamic_contract = load_json(args.dynamic_contract)
    geometry_contract = load_json(args.geometry_contract)
    coherence_contract = load_json(args.coherence_contract)
    dynamic_hash = file_sha256(args.dynamic_contract)
    geometry_hash = file_sha256(args.geometry_contract)
    coherence_hash = file_sha256(args.coherence_contract)
    kernel_config, columns, identity = validate_identity(args.declaration, args.recertification)
    window_size = int(dynamic_contract["input"]["window_size_tokens"])

    rows = []
    projection_dir = args.battery_dir / "structural_projection_v6"
    for item_path in sorted((args.battery_dir / "items").glob("*.json")):
        item = load_json(item_path)
        request = numeric_request(item, dynamic_hash)
        trajectory = project_dynamic(
            request, dynamic_contract, dynamic_hash, kernel_config, columns, identity
        )
        token_windows = chunks(normalized_tokens(item), window_size)
        if len(token_windows) != len(trajectory):
            raise ValueError("token/trajectory window mismatch")
        windows = observe_structural_coherence_v6(
            token_windows,
            trajectory,
            OperatorGeometryConfig.from_contract(geometry_contract),
            StructuralCoherenceV6Config.from_contract(coherence_contract),
        )
        projection = {
            "schema": "LLM-SVM-frontend-structural-projection/6",
            "generated_at": utc_now(),
            "session_id": request["session_id"],
            "model": str(item["model"]),
            "source_numeric_request_sha256": sha256_value(request),
            "source_item_sha256": sha256_value(item),
            "dynamic_observer_contract_sha256": dynamic_hash,
            "geometry_contract_sha256": geometry_hash,
            "structural_coherence_contract_sha256": coherence_hash,
            "contains_prompt_or_answer": False,
            "contains_external_evaluation": False,
            "kernel_modified": False,
            "prama_trajectory": trajectory,
            "structural_windows": windows,
        }
        projection_path = projection_dir / item_path.name
        atomic_json(projection_path, projection)
        evaluation = external_evaluation(item)
        rows.append({
            "model": str(item["model"]), "test_id": str(item["test_id"]),
            "finish_reason": str(item["finish_reason"]), "token_count": int(item["token_count"]),
            "response_time_seconds": float(item["response_time_seconds"]),
            "projection_path": str(projection_path.resolve()),
            "projection_sha256": file_sha256(projection_path),
            "external_evaluation": evaluation,
            "prama": prama_endpoints(trajectory),
            "structural": structural_endpoints(windows),
        })

    model_summary = []
    for model in sorted({row["model"] for row in rows}):
        members = [row for row in rows if row["model"] == model]
        model_summary.append({
            "model": model,
            "n": len(members),
            "protocol_completion_n": sum(row["external_evaluation"]["protocol_completion"] for row in members),
            "failure_modes": {
                mode: sum(row["external_evaluation"]["failure_mode"] == mode for row in members)
                for mode in ("LENGTH_EXHAUSTION", "EMPTY_RESPONSE", "INSTRUCTION_OR_CONTENT_NONCOMPLIANCE")
            },
            "total_tokens": sum(row["token_count"] for row in members),
            "total_response_time_seconds": sum(row["response_time_seconds"] for row in members),
        })

    report = {
        "schema": "LLM-SVM-frontend-integration-behavioral-stress-analysis/1",
        "generated_at": utc_now(),
        "status": "EXPLORATORY_OFFLINE_REPROJECTION",
        "source_battery_report_sha256": file_sha256(args.battery_dir / "report.json"),
        "projection_boundary": "numeric_token_observations_only",
        "external_evaluation_joined_after_projection": True,
        "latency_used_as_structural_observable": False,
        "kernel_modified": False,
        "model_summary": model_summary,
        "items": rows,
    }
    report_path = args.battery_dir / "structural_analysis_v6_report.json"
    atomic_json(report_path, report)
    print(json.dumps({"output": str(report_path), "items": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
