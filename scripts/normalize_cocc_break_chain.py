#!/usr/bin/env python
"""Normalize official Chain-of-Code Collapse pickle files for isolated evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import tempfile
from typing import Any

CANONICAL_BENCHMARK = "chain_of_code_collapse"
BENCHMARK_ALIAS = "break_the_chain_code_generation"
REQUIRED_COLUMNS = {
    "question_content",
    "question_id",
    "starter_code",
    "public_test_cases",
    "private_test_cases",
    "metadata",
}


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def discover_pickles(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    direct = sorted(root.glob("modified_problems_*.pkl"))
    if direct:
        return direct
    preferred = root / "Chain-of-Code-Collapse_fresh" / "data_modified"
    if preferred.is_dir():
        direct = sorted(preferred.glob("modified_problems_*.pkl"))
        if direct:
            return direct
    return sorted(root.rglob("modified_problems_*.pkl"))


def split_for(problem_id: str, seed: str, calibration_fraction: float) -> str:
    digest = sha256(f"{seed}:{problem_id}".encode("utf-8")).digest()
    unit = int.from_bytes(digest[:8], "big") / float(2**64)
    return "calibration" if unit < calibration_fraction else "test"


def natural_task_prompt(problem: str, starter_code: str) -> str:
    prefix = (
        "You are an expert Python programmer. Solve the following programming problem.\n\n"
        f"{problem.strip()}\n\n"
    )
    if starter_code.strip():
        prefix += (
            "Use the following starter code and return the complete implementation:\n\n"
            f"```python\n{starter_code.rstrip()}\n```\n\n"
        )
    else:
        prefix += (
            "Write a complete program that reads from standard input and writes to "
            "standard output.\n\n"
        )
    return prefix + "Return only the final Python solution inside one fenced code block."


def normalize_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool, list, dict)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def load_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "pandas is required to read the official CoCC pickle files"
        ) from exc
    frame = pd.read_pickle(path)
    columns = set(str(value) for value in frame.columns)
    missing = REQUIRED_COLUMNS - columns
    modified = sorted(column for column in columns if column.endswith("_modified"))
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    if len(modified) != 1:
        raise ValueError(f"{path.name}: expected exactly one *_modified column")
    prompt_column = modified[0]
    perturbation = prompt_column.removesuffix("_modified")
    result = []
    for row_index, raw in frame.iterrows():
        row = raw.to_dict()
        problem_id = str(normalize_scalar(row["question_id"]))
        problem = str(normalize_scalar(row[prompt_column]) or "")
        starter = str(normalize_scalar(row.get("starter_code")) or "")
        if not problem.strip():
            continue
        result.append(
            {
                "benchmark_name": CANONICAL_BENCHMARK,
                "benchmark_alias": BENCHMARK_ALIAS,
                "problem_id": problem_id,
                "item_id": f"{problem_id}:{perturbation}:{row_index}",
                "perturbation_type": perturbation,
                "perturbed_prompt": natural_task_prompt(problem, starter),
                "clean_prompt": normalize_scalar(row.get("question_content")),
                "starter_code": starter,
                "public_test_cases": normalize_scalar(row.get("public_test_cases")),
                "private_test_cases": normalize_scalar(row.get("private_test_cases")),
                "metadata": normalize_scalar(row.get("metadata")),
                "question_title": normalize_scalar(row.get("question_title")),
                "difficulty": normalize_scalar(row.get("difficulty")),
                "verifier_ref": f"cocc:{problem_id}:{perturbation}",
            }
        )
    return result


def build(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    files = discover_pickles(args.source)
    if not files:
        raise ValueError("no official modified_problems_*.pkl files found")
    selected = set(args.perturbation or [])
    rows: list[dict[str, Any]] = []
    sources = []
    for path in files:
        perturbation = path.stem.removeprefix("modified_problems_")
        if selected and perturbation not in selected:
            continue
        loaded = load_rows(path)
        official_row_count = len(loaded)
        if args.include_clean_control:
            clean_rows = []
            for attacked in loaded:
                clean = dict(attacked)
                clean["perturbation_type"] = "clean_control"
                clean["item_id"] = f"{clean['problem_id']}:clean_control"
                clean["perturbed_prompt"] = natural_task_prompt(
                    str(clean["clean_prompt"] or ""), str(clean["starter_code"] or "")
                )
                clean["verifier_ref"] = f"cocc:{clean['problem_id']}:clean_control"
                clean_rows.append(clean)
            loaded.extend(clean_rows)
        source_hash = file_sha256(path)
        for row in loaded:
            row["split"] = split_for(
                row["problem_id"], args.seed, args.calibration_fraction
            )
            row["source_file_sha256"] = source_hash
        rows.extend(loaded)
        sources.append(
            {
                "name": path.name,
                "sha256": source_hash,
                "row_count": official_row_count,
                "perturbation_type": perturbation,
            }
        )
    found = {row["perturbation_type"] for row in rows}
    if selected and not selected <= found:
        raise ValueError(f"requested perturbations not found: {sorted(selected - found)}")
    if not rows:
        raise ValueError("selection contains no rows")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if (
            args.clean_calibration_only
            and row["split"] == "calibration"
            and row["perturbation_type"] != "clean_control"
        ):
            continue
        grouped.setdefault((row["split"], row["perturbation_type"]), []).append(row)
    selected_rows: list[dict[str, Any]] = []
    for split in ("calibration", "test"):
        limit = args.limit_calibration if split == "calibration" else args.limit_test
        for perturbation in sorted(found):
            members = sorted(
                grouped.get((split, perturbation), []),
                key=lambda row: (row["problem_id"], row["item_id"]),
            )
            selected_rows.extend(members[:limit] if limit else members)
    split_map: dict[str, str] = {}
    for row in selected_rows:
        previous = split_map.setdefault(row["problem_id"], row["split"])
        if previous != row["split"]:
            raise RuntimeError("problem-level split leakage detected")
    counts: dict[str, int] = {}
    for row in selected_rows:
        key = f"{row['split']}:{row['perturbation_type']}"
        counts[key] = counts.get(key, 0) + 1
    if not any(row["split"] == "calibration" for row in selected_rows):
        raise ValueError("selection contains no calibration rows")
    if not any(row["split"] == "test" for row in selected_rows):
        raise ValueError("selection contains no test rows")
    manifest = {
        "schema": "LLM-SVM-CoCC-normalization/1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(args.source.resolve()),
        "source_files": sources,
        "seed": args.seed,
        "calibration_fraction": args.calibration_fraction,
        "row_count": len(selected_rows),
        "counts": counts,
        "problem_level_split": True,
        "model_prompt_boundary": "natural_code_generation_task_only",
    }
    return selected_rows, manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--perturbation", action="append")
    parser.add_argument("--include-clean-control", action="store_true")
    parser.add_argument(
        "--clean-calibration-only",
        action="store_true",
        help="Keep attacked rows only in test; requires --include-clean-control.",
    )
    parser.add_argument("--seed", default="cocc-prama-v1")
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--limit-calibration", type=int, default=0)
    parser.add_argument("--limit-test", type=int, default=0)
    args = parser.parse_args(argv)
    if not 0.0 < args.calibration_fraction < 1.0:
        parser.error("--calibration-fraction must lie strictly between 0 and 1")
    if args.limit_calibration < 0 or args.limit_test < 0:
        parser.error("limits must be nonnegative")
    if args.clean_calibration_only and not args.include_clean_control:
        parser.error("--clean-calibration-only requires --include-clean-control")
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        rows, manifest = build(args)
        atomic_jsonl(args.output, rows)
        manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
        manifest["output_sha256"] = file_sha256(args.output)
        atomic_json(manifest_path, manifest)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"CoCC normalization failed: {exc}")
        return 1
    print(f"wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
