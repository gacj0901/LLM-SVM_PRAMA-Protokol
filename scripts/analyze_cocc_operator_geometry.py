#!/usr/bin/env python
"""Reproject and evaluate operator geometry on the three completed CoCC runs."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_break_the_chain_prama import auroc  # noqa: E402
from scripts.evaluate_cocc_10pair_prospective import (  # noqa: E402
    exact_one_sided_sign_p,
)
from scripts.project_cocc_operator_geometry import build_artifact  # noqa: E402


ENDPOINTS = {
    "minimum_transport_efficiency": "negative_minimum_transport_efficiency",
    "maximum_recurrence_rate": "maximum_recurrence_rate",
    "maximum_recurrent_dwell": "maximum_recurrent_dwell",
    "first_recirculation_window": "early_recirculation_score",
    "fraction_of_trajectory_recirculating": "fraction_of_trajectory_recirculating",
}


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path.cwd() / path


def average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = ((start + 1) + end) / 2.0
        for offset in range(start, end):
            result[order[offset]] = rank
        start = end
    return result


def spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    x = average_ranks(left)
    y = average_ranks(right)
    mean_x = statistics.fmean(x)
    mean_y = statistics.fmean(y)
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    denominator = math.sqrt(
        sum((a - mean_x) ** 2 for a in x) * sum((b - mean_y) ** 2 for b in y)
    )
    return numerator / denominator if denominator else 0.0


def residualize_by_length(scores: Sequence[float], lengths: Sequence[int]) -> list[float]:
    x = [math.log1p(value) for value in lengths]
    mean_x = statistics.fmean(x)
    mean_y = statistics.fmean(scores)
    variance = sum((value - mean_x) ** 2 for value in x)
    slope = (
        sum((a - mean_x) * (b - mean_y) for a, b in zip(x, scores)) / variance
        if variance
        else 0.0
    )
    intercept = mean_y - slope * mean_x
    return [score - (intercept + slope * value) for score, value in zip(scores, x)]


def endpoints(artifact: dict[str, Any], max_xi: float) -> dict[str, Any]:
    windows = artifact["windows"]
    ready = [window for window in windows if window["geometry_ready"]]
    active = [window for window in ready if window["path_length"] > 0.5]
    recirculating = [
        window
        for window in ready
        if window["structural_condition"] == "CRYSTALLIZED_RECIRCULATION"
    ]
    minimum_efficiency = (
        min(float(window["transport_efficiency"]) for window in active)
        if active
        else 1.0
    )
    first = int(recirculating[0]["window_index"]) if recirculating else None
    return {
        "n_windows": len(windows),
        "geometry_ready_windows": len(ready),
        "minimum_transport_efficiency": minimum_efficiency,
        "negative_minimum_transport_efficiency": -minimum_efficiency,
        "maximum_recurrence_rate": max(
            (float(window["recurrence_rate"]) for window in ready), default=0.0
        ),
        "maximum_recurrent_dwell": max(
            (int(window["recurrent_dwell"]) for window in ready), default=0
        ),
        "first_recirculation_window": first,
        "early_recirculation_score": 1.0 / (1.0 + first) if first is not None else 0.0,
        "fraction_of_trajectory_recirculating": (
            len(recirculating) / len(ready) if ready else 0.0
        ),
        "maximum_mode_dwell": max(
            (int(window["mode_dwell"]) for window in ready), default=0
        ),
        "mean_mode_entropy": statistics.fmean(
            [float(window["mode_entropy"]) for window in ready]
        ) if ready else 0.0,
        "conservation_fraction": (
            sum(window["mode"] == "CONSERVATION" for window in ready) / len(ready)
            if ready
            else 0.0
        ),
        "max_xi": max_xi,
    }


def paired_result(rows: list[dict[str, Any]], score_field: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["problem_id"], []).append(row)
    successes = failures = ties = informative = 0
    contrasts = []
    for problem_id, members in sorted(grouped.items(), key=lambda item: int(item[0])):
        if len(members) != 2 or members[0]["label"] == members[1]["label"]:
            continue
        informative += 1
        failed = next(member for member in members if member["label"] == 1)
        passed = next(member for member in members if member["label"] == 0)
        contrast = float(failed[score_field]) - float(passed[score_field])
        contrasts.append({"problem_id": problem_id, "fail_minus_pass": contrast})
        if contrast > 0:
            successes += 1
        elif contrast < 0:
            failures += 1
        else:
            ties += 1
    return {
        "outcome_discordant_pairs": informative,
        "successes": successes,
        "failures": failures,
        "ties": ties,
        "exact_one_sided_sign_p_exploratory": exact_one_sided_sign_p(successes, failures),
        "contrasts": contrasts,
    }


def summarize_run(
    run_dir: Path, contract_path: Path, artifact_root: Path
) -> dict[str, Any]:
    source_report_path = run_dir / "evaluation" / "report.json"
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    joined = list(csv.DictReader((run_dir / "evaluation" / "blind_join.csv").open(encoding="utf-8", newline="")))
    rows = []
    model_root = artifact_root / run_dir.name
    for joined_row in joined:
        session_id = str(joined_row["session_id"])
        request_path = run_dir / "projection" / "requests" / f"{session_id}.json"
        trajectory_path = resolve_path(joined_row["trajectory_path"])
        artifact = build_artifact(request_path, trajectory_path, contract_path)
        artifact_path = model_root / f"{session_id}.json"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        raw = json.loads((run_dir / "sessions" / session_id / "raw.json").read_text(encoding="utf-8"))
        trajectory = [json.loads(line) for line in trajectory_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        values = endpoints(
            artifact,
            max(float(window["xi"]) for window in trajectory if window.get("xi") is not None),
        )
        rows.append(
            {
                "session_id": session_id,
                "problem_id": str(raw["problem_id"]),
                "condition": str(raw["perturbation_type"]),
                "label": int(joined_row["label"]),
                **values,
            }
        )
    labels = [row["label"] for row in rows]
    metrics = {}
    for endpoint_name, score_field in ENDPOINTS.items():
        scores = [float(row[score_field]) for row in rows]
        residuals = residualize_by_length(scores, [row["n_windows"] for row in rows])
        metrics[endpoint_name] = {
            "score_direction": f"higher {score_field} = more recirculation",
            "auroc": auroc(scores, labels),
            "length_residualized_auroc": auroc(residuals, labels),
            "spearman_with_n_windows": spearman(scores, [row["n_windows"] for row in rows]),
            "spearman_with_max_xi": spearman(scores, [row["max_xi"] for row in rows]),
            "paired_inference": paired_result(rows, score_field),
        }
    by_outcome = {}
    for name, label in (("PASS", 0), ("FAIL", 1)):
        members = [row for row in rows if row["label"] == label]
        endpoint_medians = {}
        for endpoint in ENDPOINTS:
            values = [
                float(row[endpoint])
                for row in members
                if row[endpoint] is not None
            ]
            endpoint_medians[endpoint] = statistics.median(values) if values else None
        by_outcome[name] = {
            "n": len(members),
            **endpoint_medians,
            "recirculating_session_count": sum(
                row["first_recirculation_window"] is not None for row in members
            ),
            "median_n_windows": statistics.median(row["n_windows"] for row in members),
            "median_max_xi": statistics.median(row["max_xi"] for row in members),
        }
    return {
        "model": source_report["model"],
        "source_report_sha256": file_sha256(source_report_path),
        "source_design_sha256": source_report["design_sha256"],
        "session_count": len(rows),
        "outcomes": source_report["outcomes"],
        "geometry_metrics": metrics,
        "endpoint_medians_by_outcome": by_outcome,
        "session_endpoints": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=Path("config/cocc_operator_geometry_observer_v1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("run_outputs/cocc_10pair_operator_geometry_v1"))
    parser.add_argument("--run-dir", action="append", type=Path)
    args = parser.parse_args()
    run_dirs = args.run_dir or [
        Path("run_outputs/cocc_10pair_nemotron3_super_v1"),
        Path("run_outputs/cocc_10pair_mistral_medium_3_5_v1"),
        Path("run_outputs/cocc_10pair_nemotron3_ultra_v1"),
    ]
    try:
        contract_hash = file_sha256(args.contract)
        freeze = json.loads(Path("config/cocc_operator_geometry_observer_v1.freeze.json").read_text(encoding="utf-8"))
        if freeze["observer_contract_sha256"] != contract_hash:
            raise ValueError("operator-geometry contract differs from freeze record")
        models = [summarize_run(path, args.contract, args.output_dir / "artifacts") for path in run_dirs]
        report = {
            "schema": "LLM-SVM-CoCC-operator-geometry-exploratory-report/1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "inference_status": "EXPLORATORY_ONLY",
            "observer_contract_sha256": contract_hash,
            "kernel_modified": False,
            "llm_calls_executed": 0,
            "models": models,
            "hypotheses": {
                "mistral": "FAIL has more recirculation despite mostly subcritical xi",
                "super": "xi persistence confounds transport with circulation",
                "ultra": "xi advantage corresponds to geometry not explained only by length",
            },
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output = args.output_dir / "report.json"
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"operator-geometry analysis failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"output": str(output), "models": len(models)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
