#!/usr/bin/env python3
"""Select a CoCC generation budget using calibration clean controls only.

The selector is deliberately unable to consume verifier outputs or holdout rows.
Candidate runs must form a contiguous prefix of the preregistered budgets.  The
smallest candidate whose observed ``finish_reason=length`` rate is at or below
the frozen threshold is selected.  If no observed candidate passes, the output
requests the next preregistered budget or blocks design freeze.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "LLM-SVM-CoCC-generation-budget-selection/1"
FORBIDDEN_OUTCOME_FIELDS = {
    "label",
    "outcome",
    "passed",
    "verification",
    "verification_result",
    "verifier_result",
}


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = _load_json(path)
    if protocol.get("schema") != "LLM-SVM-CoCC-next-confirmatory-protocol-draft/1":
        raise ValueError("unexpected protocol schema")
    plan = protocol.get("generation_budget_calibration") or {}
    if plan.get("partition") != "calibration":
        raise ValueError("budget selection must use partition=calibration")
    if plan.get("required_perturbation_type") != "clean_control":
        raise ValueError("budget selection must use clean_control only")
    if plan.get("outcome_labels_allowed") is not False:
        raise ValueError("outcome labels must be prohibited during budget selection")
    budgets = [int(value) for value in plan.get("candidate_max_tokens") or []]
    if not budgets or budgets != sorted(set(budgets)) or budgets[0] <= 0:
        raise ValueError("candidate_max_tokens must be unique increasing positives")
    threshold = float(plan.get("maximum_observed_length_rate"))
    if not 0.0 <= threshold < 1.0:
        raise ValueError("maximum_observed_length_rate must lie in [0, 1)")
    if int(plan.get("minimum_clean_control_sessions") or 0) <= 0:
        raise ValueError("minimum_clean_control_sessions must be positive")
    if plan.get("selection_rule") != "smallest_candidate_meeting_rate_threshold":
        raise ValueError("unexpected budget selection rule")
    return protocol


def _load_calibration_items(dataset_path: Path) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    with dataset_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"dataset line {line_number}: expected object")
            if row.get("split") != "calibration":
                continue
            if row.get("perturbation_type") != "clean_control":
                raise ValueError(
                    f"dataset line {line_number}: calibration row is not clean_control"
                )
            item_id = str(row.get("item_id") or "")
            if not item_id or item_id in selected:
                raise ValueError(f"dataset line {line_number}: invalid duplicate item_id")
            selected[item_id] = row
    if not selected:
        raise ValueError("dataset contains no calibration clean controls")
    return selected


def parse_candidate_run(value: str) -> tuple[int, Path]:
    budget_text, separator, directory = value.partition("=")
    if not separator or not directory:
        raise argparse.ArgumentTypeError("candidate run must be MAX_TOKENS=PATH")
    try:
        budget = int(budget_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("candidate max tokens must be an integer") from exc
    if budget <= 0:
        raise argparse.ArgumentTypeError("candidate max tokens must be positive")
    return budget, Path(directory)


def _raw_paths(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("sessions/*/raw.json"))


def _audit_candidate(
    budget: int,
    run_dir: Path,
    allowed_items: dict[str, dict[str, Any]],
    minimum_sessions: int,
) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    manifest = _load_json(manifest_path)
    observed_budget = int(
        (manifest.get("generation_parameter_set") or {}).get("max_tokens") or 0
    )
    if observed_budget != budget:
        raise ValueError(
            f"{run_dir}: manifest max_tokens={observed_budget}, expected {budget}"
        )
    paths = _raw_paths(run_dir)
    if len(paths) < minimum_sessions:
        raise ValueError(
            f"{run_dir}: found {len(paths)} raw sessions; require {minimum_sessions}"
        )
    records: list[dict[str, Any]] = []
    seen_items: set[str] = set()
    for path in paths:
        raw = _load_json(path)
        forbidden = sorted(FORBIDDEN_OUTCOME_FIELDS.intersection(raw))
        if forbidden:
            raise ValueError(f"{path}: forbidden outcome fields present: {forbidden}")
        item_id = str(raw.get("item_id") or "")
        if item_id not in allowed_items:
            raise ValueError(f"{path}: item is not in the calibration clean-control set")
        if item_id in seen_items:
            raise ValueError(f"{path}: duplicate item_id {item_id}")
        seen_items.add(item_id)
        if raw.get("perturbation_type") != "clean_control":
            raise ValueError(f"{path}: perturbation_type must be clean_control")
        turns = raw.get("turns") or []
        if len(turns) != 1 or not isinstance(turns[0], dict):
            raise ValueError(f"{path}: expected exactly one generated turn")
        turn = turns[0]
        forbidden_turn = sorted(FORBIDDEN_OUTCOME_FIELDS.intersection(turn))
        if forbidden_turn:
            raise ValueError(
                f"{path}: forbidden turn outcome fields present: {forbidden_turn}"
            )
        reason = str(turn.get("finish_reason") or "").lower()
        if reason not in {"stop", "length"}:
            raise ValueError(f"{path}: unsupported finish_reason {reason!r}")
        token_count = int(turn.get("token_count") or len(turn.get("tokens") or []))
        if token_count < 0 or token_count > budget:
            raise ValueError(f"{path}: token_count outside candidate budget")
        records.append(
            {
                "session_id": str(raw.get("session_id") or ""),
                "item_id": item_id,
                "finish_reason": reason,
                "token_count": token_count,
            }
        )
    if len(seen_items) != len(allowed_items):
        missing = sorted(set(allowed_items) - seen_items)
        raise ValueError(
            f"{run_dir}: candidate did not acquire the complete frozen calibration set; "
            f"missing {len(missing)} items"
        )
    length_count = sum(row["finish_reason"] == "length" for row in records)
    return {
        "max_tokens": budget,
        "run_directory": str(run_dir),
        "run_manifest_sha256": file_sha256(manifest_path),
        "session_count": len(records),
        "session_ids_sha256": canonical_sha256(
            sorted(row["session_id"] for row in records)
        ),
        "item_ids_sha256": canonical_sha256(sorted(seen_items)),
        "finish_reason_counts": {
            "length": length_count,
            "stop": len(records) - length_count,
        },
        "observed_length_rate": length_count / len(records),
        "token_count_summary": {
            "minimum": min(row["token_count"] for row in records),
            "maximum": max(row["token_count"] for row in records),
            "mean": sum(row["token_count"] for row in records) / len(records),
        },
        "provider": manifest.get("provider"),
        "provider_endpoint": manifest.get("provider_endpoint"),
        "model": manifest.get("model"),
        "resolved_models": (manifest.get("provider_response_identity") or {}).get(
            "resolved_models"
        ),
        "system_fingerprints": (
            manifest.get("provider_response_identity") or {}
        ).get("system_fingerprints"),
    }


def select_budget(
    protocol: dict[str, Any],
    dataset_path: Path,
    candidate_runs: Iterable[tuple[int, Path]],
) -> dict[str, Any]:
    plan = protocol["generation_budget_calibration"]
    budgets = [int(value) for value in plan["candidate_max_tokens"]]
    supplied = list(candidate_runs)
    supplied_budgets = [budget for budget, _ in supplied]
    if supplied_budgets != budgets[: len(supplied_budgets)]:
        raise ValueError(
            "candidate runs must be a contiguous prefix of candidate_max_tokens"
        )
    if not supplied:
        raise ValueError("at least one candidate run is required")
    allowed_items = _load_calibration_items(dataset_path)
    minimum_sessions = int(plan["minimum_clean_control_sessions"])
    if len(allowed_items) < minimum_sessions:
        raise ValueError(
            f"dataset has {len(allowed_items)} calibration controls; "
            f"require {minimum_sessions}"
        )
    audits = [
        _audit_candidate(budget, run_dir, allowed_items, minimum_sessions)
        for budget, run_dir in supplied
    ]
    identities = {
        (
            row["provider"],
            row["provider_endpoint"],
            row["model"],
            json.dumps(row["resolved_models"], sort_keys=True),
            json.dumps(row["system_fingerprints"], sort_keys=True),
        )
        for row in audits
    }
    if len(identities) != 1:
        raise ValueError("candidate runs differ in provider/model response identity")
    item_hashes = {row["item_ids_sha256"] for row in audits}
    if len(item_hashes) != 1:
        raise ValueError("candidate runs used different calibration items")
    threshold = float(plan["maximum_observed_length_rate"])
    selected = next(
        (row for row in audits if row["observed_length_rate"] <= threshold), None
    )
    if selected is not None:
        status = "SELECTED"
        next_budget = None
    elif len(audits) < len(budgets):
        status = "MORE_CALIBRATION_REQUIRED"
        next_budget = budgets[len(audits)]
    else:
        status = "BLOCKED_NO_BUDGET_MEETS_CAP_RATE"
        next_budget = None
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "interpretation": "CALIBRATION_ONLY_GENERATION_CAP_SELECTION",
        "protocol_id": protocol["protocol_id"],
        "protocol_status": protocol["status"],
        "dataset": str(dataset_path),
        "dataset_sha256": file_sha256(dataset_path),
        "calibration_item_count": len(allowed_items),
        "calibration_item_ids_sha256": canonical_sha256(sorted(allowed_items)),
        "outcome_labels_consumed": False,
        "holdout_rows_consumed": False,
        "selection_rule": plan["selection_rule"],
        "maximum_observed_length_rate": threshold,
        "candidate_audits": audits,
        "selected_max_tokens": (
            selected["max_tokens"] if selected is not None else None
        ),
        "next_candidate_max_tokens": next_budget,
        "design_freeze_allowed": status == "SELECTED",
        "caveat": (
            "The observed calibration cap rate is a design-selection diagnostic, "
            "not evidence about PASS/FAIL or PRAMA discrimination."
        ),
    }


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument(
        "--candidate-run",
        required=True,
        action="append",
        type=parse_candidate_run,
        metavar="MAX_TOKENS=PATH",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        protocol = _load_protocol(args.protocol)
        report = select_budget(protocol, args.dataset, args.candidate_run)
        _atomic_write(args.output, report)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "status": report["status"],
                    "selected_max_tokens": report["selected_max_tokens"],
                    "next_candidate_max_tokens": report["next_candidate_max_tokens"],
                    "sha256": file_sha256(args.output),
                },
                ensure_ascii=False,
            )
        )
        return 0 if report["status"] == "SELECTED" else 2
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"CoCC generation-budget selection failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
