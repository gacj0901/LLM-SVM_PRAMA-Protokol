#!/usr/bin/env python
"""Project raw LLM sessions with the certified PRAMA Protokol kernel.

This is the E-P1 boundary between the LLM Observation Interface and the
domain-blind kernel.  It stages all projections in memory, enforces the global
C3 gate, and only then writes evaluator-compatible ``raw.json`` files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import prama_protokol
from prama_protokol import KernelConfig, compliance, project

from aptadynamic_llm.ingest import load_sessions
from aptadynamic_llm.omega import omega_sessions
from aptadynamic_llm.ep1_config import (
    BASE_SEED,
    K,
    KERNEL_G_SMOOTH,
    KERNEL_TAU,
    MIN_SESSIONS,
    MIN_VALID_TOKENS,
    TEST_FRACTION,
    WINDOW_SIZE,
)

CFG = KernelConfig(tau_memory=KERNEL_TAU, g_smooth=KERNEL_G_SMOOTH)
MIN_TOKENS = MIN_VALID_TOKENS
SPLIT_SEED = BASE_SEED

LABEL_FIELDS = (
    "session_id",
    "label",
    "event_token",
    "event_turn",
    "event_type",
    "split",
    "verifier_name",
    "finish_reason",
    "source_session_id",
    "prompt_id",
)


class StudyInvalid(RuntimeError):
    """The Observation Interface failed a preregistered study gate."""


@dataclass
class StagedSession:
    generation_id: str
    source_session_id: str
    prompt_id: str
    model: str
    finish_reason: str
    omega: np.ndarray
    gamma: Any
    token_rows: list[dict[str, float]]


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _token_payloads(group: Any, limit: int) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for _, row in group.sort_values("pos").iloc[:limit].iterrows():
        rows.append(
            {
                "top1_logprob": -_finite_float(row.get("surprisal")),
                "entropy": _finite_float(row.get("entropy")),
                "gap": _finite_float(row.get("gap")),
            }
        )
    return rows


def stage_sessions(
    sessions_dir: Path,
    cfg: KernelConfig = CFG,
    k: int = K,
    min_tokens: int = MIN_TOKENS,
    min_sessions: int = MIN_SESSIONS,
) -> tuple[list[StagedSession], list[dict[str, str]], dict[str, Any]]:
    df = load_sessions(str(sessions_dir))
    if df.empty:
        raise ValueError(f"no verifiable token sessions found under {sessions_dir}")
    models = {str(value) for value in df["model"].dropna().unique()}
    if len(models) != 1 or "" in models:
        raise ValueError(
            f"E-P1 requires exactly one non-empty model identity; found {sorted(models)!r}"
        )

    group_col = "generation_id" if "generation_id" in df.columns else "session_id"
    groups = {str(gid): group for gid, group in df.groupby(group_col, sort=False)}
    staged: list[StagedSession] = []
    excluded: list[dict[str, str]] = []
    all_delta: list[np.ndarray] = []
    all_omega: list[np.ndarray] = []

    for gid_value, finish_reason, omega, expected in omega_sessions(df, min_sessions=min_sessions):
        gid = str(gid_value)
        group = groups[gid]
        omega_k = np.asarray(omega[:k], dtype=float)
        expected_k = np.asarray(expected[:k], dtype=float)
        if len(omega_k) < min_tokens:
            excluded.append({"session_id": gid, "reason": "fewer_than_min_observed_tokens"})
            continue
        gamma = project(omega_k, expected_k, cfg)
        valid = gamma["valid"].to_numpy(dtype=bool)
        if int(valid.sum()) < min_tokens:
            excluded.append({"session_id": gid, "reason": "fewer_than_min_valid_tokens"})
            continue
        all_delta.append(gamma["delta"].to_numpy(dtype=float)[valid])
        all_omega.append(omega_k[valid])
        staged.append(
            StagedSession(
                generation_id=gid,
                source_session_id=str(group["session_id"].iloc[0]),
                prompt_id=str(group["prompt_id"].iloc[0] or group["session_id"].iloc[0]),
                model=str(group["model"].iloc[0]),
                finish_reason=str(finish_reason or ""),
                omega=omega_k,
                gamma=gamma,
                token_rows=_token_payloads(group, len(omega_k)),
            )
        )

    if not staged:
        raise ValueError("no evaluable sessions remain after warm-up and valid-token exclusions")

    c3 = compliance.check_degeneration(np.concatenate(all_delta), np.concatenate(all_omega))
    if not c3["passed"]:
        raise StudyInvalid(
            "C3 failed: delta degenerated into activity; E-P1 is invalid and the diagnosis belongs to O_D. "
            + str(c3.get("detail", ""))
        )
    stats = {
        "input_tokens": int(len(df)),
        "input_sessions": int(df["session_id"].nunique()),
        "input_generations": int(df[group_col].nunique()),
        "evaluable_generations": len(staged),
        "excluded_generations": len(excluded),
        "C3": c3,
    }
    return staged, excluded, stats


def _last_valid(values: np.ndarray, valid: np.ndarray) -> float:
    return float(values[valid][-1])


def build_turns(session: StagedSession, window_size: int = WINDOW_SIZE) -> list[dict[str, Any]]:
    gamma = session.gamma
    valid = gamma["valid"].to_numpy(dtype=bool)
    latent = gamma["latent_collapse"].to_numpy(dtype=bool)
    arrays = {name: gamma[name].to_numpy() for name in ("delta", "xi", "M", "G", "theta", "stratum")}
    turns: list[dict[str, Any]] = []
    for start in range(0, len(session.omega), window_size):
        stop = min(start + window_size, len(session.omega))
        window_valid = valid[start:stop]
        if not window_valid.any():
            continue
        window_latent = latent[start:stop]
        summary = {
            "latent_occupancy": float(window_latent[window_valid].mean()),
            "delta": float(arrays["delta"][start:stop][window_valid].mean()),
            "xi": _last_valid(arrays["xi"][start:stop], window_valid),
            "M": _last_valid(arrays["M"][start:stop], window_valid),
            "G": _last_valid(arrays["G"][start:stop], window_valid),
            "theta": _last_valid(arrays["theta"][start:stop], window_valid),
            "stratum": int(_last_valid(arrays["stratum"][start:stop], window_valid)),
        }
        summary["neg_M"] = -summary["M"]
        turns.append(
            {
                "turn_index": len(turns),
                "token_count": stop - start,
                "valid_token_count": int(window_valid.sum()),
                "source_token_start": start,
                "source_token_end": stop,
                "tokens": session.token_rows[start:stop],
                "summary": summary,
            }
        )
    if not turns:
        raise ValueError(f"{session.generation_id}: projection produced no valid windows")
    return turns


def _read_external_labels(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"session_id", "label"}.issubset(reader.fieldnames):
            raise ValueError("external labels file requires session_id,label columns")
        rows = [row for row in reader if row.get("session_id")]
    ids = [row["session_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("external labels file contains duplicate session_id values")
    return {row["session_id"]: row for row in rows}


def _binary_label(value: Any) -> int:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "fail", "failed", "positive", "pathological"}:
        return 1
    if normalized in {"0", "false", "pass", "passed", "negative", "healthy"}:
        return 0
    raise ValueError(f"unsupported binary label: {value!r}")


def _assign_stratified_splits(rows: list[dict[str, Any]], seed: int, test_fraction: float) -> None:
    explicit = [str(row.get("split") or "").strip().lower() for row in rows]
    if all(value in {"train", "calibration", "calib", "test"} for value in explicit):
        for row, value in zip(rows, explicit):
            row["split"] = "train" if value in {"train", "calibration", "calib"} else "test"
        group_splits: dict[str, set[str]] = {}
        for row in rows:
            group_splits.setdefault(str(row["prompt_id"]), set()).add(str(row["split"]))
        leaked = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
        if leaked:
            raise ValueError(f"prompt groups cross explicit train/test splits: {leaked!r}")
    else:
        rng = np.random.default_rng(seed)
        groups: dict[str, list[int]] = {}
        for index, row in enumerate(rows):
            group = str(row.get("prompt_id") or "").strip()
            if not group:
                raise ValueError("grouped split requires a non-empty prompt_id for every session")
            groups.setdefault(group, []).append(index)
        if len(groups) < 2:
            raise ValueError("grouped split requires at least two distinct prompt_id groups")
        totals = {label: sum(int(row["label"]) == label for row in rows) for label in (0, 1)}
        targets = {
            label: min(totals[label] - 1, max(1, int(round(totals[label] * test_fraction))))
            for label in (0, 1)
        }
        group_names = np.array(sorted(groups), dtype=object)
        rng.shuffle(group_names)
        n_test_groups = min(len(groups) - 1, max(1, int(round(len(groups) * test_fraction))))
        selected: list[str] = []
        counts = {0: 0, 1: 0}
        remaining = list(group_names)
        for _ in range(n_test_groups):
            def candidate_score(group: str) -> tuple[int, int]:
                additions = {
                    label: sum(int(rows[index]["label"]) == label for index in groups[group])
                    for label in (0, 1)
                }
                distance = sum(abs((counts[label] + additions[label]) - targets[label]) for label in (0, 1))
                overshoot = sum(max(0, counts[label] + additions[label] - targets[label]) for label in (0, 1))
                return distance, overshoot
            chosen = min(remaining, key=candidate_score)
            selected.append(chosen)
            for index in groups[chosen]:
                counts[int(rows[index]["label"])] += 1
            remaining.remove(chosen)
        test_groups = set(selected)
        for group, indices in groups.items():
            split = "test" if group in test_groups else "train"
            for index in indices:
                rows[index]["split"] = split

    for split in ("train", "test"):
        labels = {int(row["label"]) for row in rows if row["split"] == split}
        if labels != {0, 1}:
            raise ValueError(f"split {split!r} must contain both label classes")


def build_labels(
    staged: list[StagedSession],
    turns_by_id: dict[str, list[dict[str, Any]]],
    labels_file: Path | None,
    seed: int,
    test_fraction: float,
) -> list[dict[str, Any]]:
    external = _read_external_labels(labels_file)
    source_counts: dict[str, int] = {}
    for session in staged:
        source_counts[session.source_session_id] = source_counts.get(session.source_session_id, 0) + 1

    rows: list[dict[str, Any]] = []
    for session in staged:
        supplied = external.get(session.generation_id)
        if supplied is None and source_counts[session.source_session_id] == 1:
            supplied = external.get(session.source_session_id)
        if labels_file is not None and supplied is None:
            raise ValueError(f"missing external label for evaluable generation {session.generation_id}")
        if supplied:
            label = _binary_label(supplied["label"])
        else:
            normalized_reason = session.finish_reason.strip().lower()
            if normalized_reason not in {"length", "stop"}:
                raise ValueError(
                    f"{session.generation_id}: finish_reason must be 'length' or 'stop', got {session.finish_reason!r}"
                )
            label = int(normalized_reason == "length")
        turns = turns_by_id[session.generation_id]
        represented = sum(int(turn["token_count"]) for turn in turns)
        rows.append(
            {
                "session_id": session.generation_id,
                "label": label,
                "event_token": represented,
                "event_turn": len(turns) - 1,
                "event_type": (supplied or {}).get("event_type") or "final_outcome",
                "split": (supplied or {}).get("split", ""),
                "verifier_name": (supplied or {}).get("verifier_name") or ("external" if supplied else "finish_reason"),
                "finish_reason": session.finish_reason,
                "source_session_id": session.source_session_id,
                "prompt_id": session.prompt_id,
            }
        )
    _assign_stratified_splits(rows, seed, test_fraction)
    return rows


def _source_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.json")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def write_outputs(
    out: Path,
    sessions_dir: Path,
    staged: list[StagedSession],
    excluded: list[dict[str, str]],
    stats: dict[str, Any],
    labels: list[dict[str, Any]],
    turns_by_id: dict[str, list[dict[str, Any]]],
    cfg: KernelConfig,
    settings: dict[str, Any],
) -> None:
    if out.exists():
        raise FileExistsError(f"output directory already exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ep1-", dir=out.parent) as temporary:
        root = Path(temporary) / out.name
        sessions_root = root / "sessions"
        sessions_root.mkdir(parents=True)
        for session in staged:
            safe_id = session.generation_id.replace("/", "_").replace("\\", "_").replace(":", "_")
            session_dir = sessions_root / safe_id
            session_dir.mkdir()
            raw = {
                "session_id": session.generation_id,
                "source_session_id": session.source_session_id,
                "prompt_id": session.prompt_id,
                "model": session.model,
                "finish_reason": session.finish_reason,
                "final_outcome_proxy": True,
                "kernel_package": "prama-protokol",
                "kernel_version": getattr(prama_protokol, "__version__", "unknown"),
                "turns": turns_by_id[session.generation_id],
            }
            (session_dir / "raw.json").write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")

        with (root / "labels.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=LABEL_FIELDS)
            writer.writeheader()
            writer.writerows(labels)

        manifest = {
            "schema": "LLM-SVM-E-P1/1",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_sessions_dir": str(sessions_dir),
            "source_sha256": _source_hash(sessions_dir),
            "kernel": {
                "package": "prama-protokol",
                "version": getattr(prama_protokol, "__version__", "unknown"),
                "config": asdict(cfg),
            },
            "settings": settings,
            "models": sorted({session.model for session in staged}),
            "statistics": stats,
            "excluded": excluded,
            "label_counts": {
                "0": sum(int(row["label"]) == 0 for row in labels),
                "1": sum(int(row["label"]) == 1 for row in labels),
            },
            "split_counts": {
                "train": sum(row["split"] == "train" for row in labels),
                "test": sum(row["split"] == "test" for row in labels),
            },
        }
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        shutil.move(str(root), str(out))


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.K <= 0 or args.min_tokens <= 0 or args.window_size <= 0 or args.min_sessions < 0:
        raise ValueError("K, min-tokens and window-size must be positive; min-sessions cannot be negative")
    if not 0.0 < args.test_fraction < 1.0:
        raise ValueError("test-fraction must be between 0 and 1")
    cfg = KernelConfig(tau_memory=args.tau_memory, g_smooth=args.g_smooth)
    staged, excluded, stats = stage_sessions(
        args.sessions_dir, cfg=cfg, k=args.K, min_tokens=args.min_tokens, min_sessions=args.min_sessions
    )
    turns_by_id = {session.generation_id: build_turns(session, args.window_size) for session in staged}
    labels = build_labels(staged, turns_by_id, args.labels_file, args.seed, args.test_fraction)
    settings = {
        "K": args.K,
        "min_tokens": args.min_tokens,
        "window_size": args.window_size,
        "min_sessions": args.min_sessions,
        "split_seed": args.seed,
        "test_fraction": args.test_fraction,
        "primary_score": "latent_occupancy",
        "outcome_mode": "external" if args.labels_file else "finish_reason",
    }
    write_outputs(
        args.out, args.sessions_dir, staged, excluded, stats, labels, turns_by_id, cfg, settings
    )
    return {"output": str(args.out), "statistics": stats, "settings": settings}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions-dir", required=True, type=Path)
    parser.add_argument("--labels-file", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--K", type=int, default=K)
    parser.add_argument("--min-tokens", type=int, default=MIN_TOKENS)
    parser.add_argument("--min-sessions", type=int, default=MIN_SESSIONS)
    parser.add_argument("--tau-memory", type=float, default=CFG.tau_memory)
    parser.add_argument("--g-smooth", type=int, default=CFG.g_smooth)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--test-fraction", type=float, default=TEST_FRACTION)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(args)
    except StudyInvalid as exc:
        print(f"STUDY INVALID — O_D interface failure: {exc}")
        return 2
    except (FileExistsError, ValueError, OSError) as exc:
        print(f"scoring failed: {exc}")
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
