#!/usr/bin/env python
"""Evaluate the frozen 10-pair CoCC study with exact paired inference."""

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
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import evaluate_break_the_chain_prama as ev


HORIZONS = (1, 2, 4, 8, 16)
PRIMARY = "max_negative_balance"
BASELINE = "max_delta"


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def exact_one_sided_sign_p(successes: int, failures: int) -> float | None:
    n = successes + failures
    if n == 0:
        return None
    return sum(math.comb(n, k) for k in range(successes, n + 1)) / (2**n)


def auc(rows: list[dict[str, Any]], field: str) -> float | None:
    labels = [int(row["label"]) for row in rows]
    if len(set(labels)) < 2:
        return None
    return ev.auroc([float(row["features"][field]) for row in rows], labels)


def metric_set(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {
        field: auc(rows, field)
        for field in (
            "max_delta",
            "max_xi",
            "max_negative_balance",
            "final_accumulated_excess",
            "capacity_loss",
            "mean_surprisal",
            "mean_entropy",
            "negative_mean_gap",
        )
    }
    result["prama_minus_delta"] = (
        result[PRIMARY] - result[BASELINE]
        if result[PRIMARY] is not None and result[BASELINE] is not None
        else None
    )
    return result


def scalar_auc(rows: list[dict[str, Any]], field: str) -> float | None:
    labels = [int(row["label"]) for row in rows]
    if len(set(labels)) < 2:
        return None
    return ev.auroc([float(row[field]) for row in rows], labels)


def load_rows(run_dir: Path) -> list[dict[str, Any]]:
    raw_by_session = {
        path.parent.name: json.loads(path.read_text(encoding="utf-8"))
        for path in (run_dir / "sessions").glob("*/raw.json")
    }
    rows = []
    with (run_dir / "evaluation" / "blind_join.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        for joined in csv.DictReader(handle):
            session_id = str(joined["session_id"])
            raw = raw_by_session[session_id]
            trajectory_path = Path(joined["trajectory_path"])
            request_path = Path(joined["projection_request_path"])
            if not trajectory_path.is_absolute():
                trajectory_path = Path.cwd() / trajectory_path
            if not request_path.is_absolute():
                request_path = Path.cwd() / request_path
            trajectory = ev._trajectory(trajectory_path)
            tokens = ev._numeric_tokens(request_path)
            turn = raw["turns"][0]
            rows.append(
                {
                    "session_id": session_id,
                    "problem_id": str(raw["problem_id"]),
                    "condition": str(raw["perturbation_type"]),
                    "label": int(joined["label"]),
                    "finish_reason": str(turn["finish_reason"]),
                    "token_count": int(turn["token_count"]),
                    "response_time_seconds": float(raw["response_time_seconds"]),
                    "trajectory": trajectory,
                    "tokens": tokens,
                    "features": ev._features(trajectory, tokens),
                }
            )
    return rows


def paired_test(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["problem_id"], []).append(row)
    table = []
    primary_success = primary_failure = primary_tie = 0
    incremental_success = incremental_failure = incremental_tie = 0
    discordant = 0
    for problem_id in sorted(grouped, key=int):
        members = grouped[problem_id]
        if len(members) != 2 or {row["condition"] for row in members} != {
            "clean_control",
            "negation_objective",
        }:
            raise ValueError(f"{problem_id}: expected one complete two-condition pair")
        informative = members[0]["label"] != members[1]["label"]
        record: dict[str, Any] = {
            "problem_id": problem_id,
            "outcome_discordant": informative,
            "sessions": [
                {
                    "session_id": row["session_id"],
                    "condition": row["condition"],
                    "outcome": "FAIL" if row["label"] else "PASS",
                    "max_delta": row["features"][BASELINE],
                    "max_xi": row["features"]["max_xi"],
                    "max_negative_balance": row["features"][PRIMARY],
                    "finish_reason": row["finish_reason"],
                    "token_count": row["token_count"],
                }
                for row in members
            ],
        }
        if informative:
            discordant += 1
            fail = next(row for row in members if row["label"] == 1)
            passed = next(row for row in members if row["label"] == 0)
            prama_contrast = fail["features"][PRIMARY] - passed["features"][PRIMARY]
            delta_contrast = fail["features"][BASELINE] - passed["features"][BASELINE]
            incremental = prama_contrast - delta_contrast
            record.update(
                {
                    "prama_fail_minus_pass": prama_contrast,
                    "delta_fail_minus_pass": delta_contrast,
                    "incremental_contrast": incremental,
                }
            )
            if prama_contrast > 0:
                primary_success += 1
            elif prama_contrast < 0:
                primary_failure += 1
            else:
                primary_tie += 1
            if incremental > 0:
                incremental_success += 1
            elif incremental < 0:
                incremental_failure += 1
            else:
                incremental_tie += 1
        table.append(record)
    primary_p = exact_one_sided_sign_p(primary_success, primary_failure)
    incremental_p = exact_one_sided_sign_p(incremental_success, incremental_failure)
    return (
        {
            "problem_pair_count": len(grouped),
            "outcome_discordant_pair_count": discordant,
            "minimum_required_outcome_discordant_pairs": 7,
            "primary": {
                "hypothesis": "max_negative_balance_FAIL > max_negative_balance_PASS within problem_id",
                "successes": primary_success,
                "failures": primary_failure,
                "ties": primary_tie,
                "effective_n": primary_success + primary_failure,
                "exact_one_sided_sign_p": primary_p,
            },
            "incremental": {
                "hypothesis": "paired PRAMA contrast > paired delta contrast",
                "successes": incremental_success,
                "failures": incremental_failure,
                "ties": incremental_tie,
                "effective_n": incremental_success + incremental_failure,
                "exact_one_sided_sign_p": incremental_p,
            },
        },
        table,
    )


def build_report(run_dir: Path, design: dict[str, Any], design_path: Path) -> dict[str, Any]:
    rows = load_rows(run_dir)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    design_digest = file_sha256(design_path)
    if len(rows) != 20:
        raise ValueError(f"expected 20 blindly joined sessions, observed {len(rows)}")
    if len({row["problem_id"] for row in rows}) != 10:
        raise ValueError("expected exactly 10 problem_id clusters")
    if manifest.get("projector_observer_sha256") != design["observer_contract_sha256"]:
        raise ValueError("run observer digest differs from frozen design")
    if manifest.get("dataset_sha256") != design["dataset_sha256"]:
        raise ValueError("run dataset digest differs from frozen design")
    if manifest.get("dataset_manifest_sha256") != design["dataset_manifest_sha256"]:
        raise ValueError("run dataset-manifest digest differs from frozen design")
    if manifest.get("confirmatory_design_sha256") != design_digest:
        raise ValueError("run manifest is not bound to this frozen design")
    model = str(manifest.get("model", ""))
    if model not in design["model_profiles"]:
        raise ValueError("run model is not declared in the frozen design")
    observed_profile = manifest.get("generation_parameter_set", {})
    for key, expected in design["model_profiles"][model].items():
        if observed_profile.get(key) != expected:
            raise ValueError(
                f"generation profile mismatch for {key}: "
                f"expected={expected!r}, observed={observed_profile.get(key)!r}"
            )
    max_tokens = int(manifest["generation_parameter_set"]["max_tokens"])
    for row in rows:
        row["resource_occupancy"] = min(1.0, row["token_count"] / max_tokens)
        row["terminal_exhaustion"] = float(row["finish_reason"] == "length")
    paired, pair_table = paired_test(rows)
    full = metric_set(rows)
    horizons = {}
    for horizon in HORIZONS:
        prefix_rows = []
        for row in rows:
            if len(row["trajectory"]) < horizon:
                continue
            prefix_rows.append(
                {
                    **row,
                    "features": ev._features(
                        row["trajectory"][:horizon], row["tokens"][: horizon * 16]
                    ),
                }
            )
        horizons[str(horizon)] = {
            "at_risk_n": len(prefix_rows),
            "metrics": metric_set(prefix_rows),
        }
    primary = paired["primary"]
    incremental = paired["incremental"]
    if paired["outcome_discordant_pair_count"] < 7:
        verdict = "inconclusive_insufficient_discordant_pairs"
    elif primary["effective_n"] < 7:
        verdict = "inconclusive_primary_score_ties"
    elif primary["exact_one_sided_sign_p"] is None or primary["exact_one_sided_sign_p"] >= 0.01:
        verdict = "prospective_not_confirmed"
    elif (
        incremental["effective_n"] >= 7
        and incremental["exact_one_sided_sign_p"] is not None
        and incremental["exact_one_sided_sign_p"] < 0.01
        and full["prama_minus_delta"] is not None
        and full["prama_minus_delta"] >= 0.05
    ):
        verdict = "prospective_incremental_prama_confirmed"
    else:
        verdict = "prospective_association_confirmed_incremental_not_confirmed"
    finish_audit: dict[str, dict[str, int]] = {}
    for row in rows:
        outcome = "FAIL" if row["label"] else "PASS"
        finish_audit.setdefault(outcome, {}).setdefault(row["finish_reason"], 0)
        finish_audit[outcome][row["finish_reason"]] += 1
    latency_by_outcome = {}
    for outcome, label in (("PASS", 0), ("FAIL", 1)):
        values = [
            row["response_time_seconds"] for row in rows if row["label"] == label
        ]
        latency_by_outcome[outcome] = {
            "n": len(values),
            "median_seconds": statistics.median(values) if values else None,
            "mean_seconds": statistics.fmean(values) if values else None,
        }
    return {
        "schema": "LLM-SVM-CoCC-10pair-prospective-evaluation/1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "design_id": design["design_id"],
        "design_sha256": design_digest,
        "run_manifest_sha256": file_sha256(manifest_path),
        "model": manifest["model"],
        "independent_unit": "problem_id",
        "session_count": len(rows),
        "problem_pair_count": 10,
        "outcomes": {
            "pass_n": sum(row["label"] == 0 for row in rows),
            "fail_n": sum(row["label"] == 1 for row in rows),
        },
        "full_trajectory_metrics": full,
        "paired_exact_inference": paired,
        "absolute_window_horizons": horizons,
        "finish_reason_by_outcome": finish_audit,
        "response_latency_by_outcome": latency_by_outcome,
        "operational_channel": {
            "resource_occupancy_auroc": scalar_auc(rows, "resource_occupancy"),
            "terminal_exhaustion_auroc": scalar_auc(rows, "terminal_exhaustion"),
            "kept_separate_from_structural_state": True,
        },
        "activation_audit": ev._activation_audit(rows, 1e-12),
        "pair_table": pair_table,
        "verdict": verdict,
        "decision_rule": design["decision_rule"],
        "claim_boundary": (
            "Prospective paired evidence for external PASS/FAIL discrimination on "
            "these ten prespecified problems; not a population-wide benchmark claim."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--design", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    try:
        design = json.loads(args.design.read_text(encoding="utf-8"))
        report = build_report(args.run_dir, design, args.design)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"10-pair prospective evaluation failed: {exc}")
        return 1
    print(json.dumps({"output": str(args.out), "verdict": report["verdict"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
