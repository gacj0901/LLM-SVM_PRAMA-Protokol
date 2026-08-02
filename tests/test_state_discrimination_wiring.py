"""Synthetic wiring test only; this is not empirical evidence for E-P1."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.score_sessions_prama import _assign_stratified_splits

from aptadynamic_llm.evaluation.state_discrimination_metrics import auroc
from scripts.evaluate_state_discrimination import load_examples, load_labels
from scripts.score_sessions_prama import main as score_main, stage_sessions
from scripts.validate_state_discrimination_inputs import validate_inputs


def test_synthetic_structural_corpus_wires_kernel_scorer_validator_and_evaluator(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "structural"
    scored = tmp_path / "ep1"
    subprocess.run(
        [sys.executable, str(root / "examples" / "make_synthetic.py"), "structural", str(source)],
        check=True,
        cwd=root,
        capture_output=True,
        text=True,
    )

    assert score_main(["--sessions-dir", str(source), "--out", str(scored)]) == 0
    manifest = json.loads((scored / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["statistics"]["C3"]["r_delta_omega"] == pytest.approx(
        -0.09194754007715042, abs=1e-12
    )
    raw_paths = sorted((scored / "sessions").rglob("raw.json"))
    labels_path = scored / "labels.csv"
    report = validate_inputs(labels_path, raw_paths)
    assert report["valid"], report["errors"]
    assert report["primary_score_range"] > 0.0

    labels = load_labels(labels_path)
    examples = load_examples(raw_paths, labels)
    test = [row for row in examples if row["split"] == "test"]
    latent_auc = auroc([row["scores"]["latent_occupancy"] for row in test], [row["label"] for row in test])
    # This is a wiring invariant under the repository's pinned dependency, not an
    # empirical performance gate. The historical 0.939 result used an earlier
    # kernel/configuration and is not silently inherited by the current lock.
    assert latent_auc == pytest.approx(0.6236842105263158, abs=1e-12)
    assert latent_auc > 0.5

    first = json.loads(raw_paths[0].read_text(encoding="utf-8"))
    summary = first["turns"][-1]["summary"]
    assert {"latent_occupancy", "delta", "xi", "M", "G", "theta", "stratum"}.issubset(summary)
    assert "boundary_pressure" not in summary


def test_scorer_rejects_mixed_model_corpus(tmp_path):
    for index, model in enumerate(("model-a", "model-b")):
        payload = {
            "session_id": f"s{index}",
            "model": model,
            "turns": [
                {
                    "finish_reason": "stop",
                    "tokens": [{"top1_logprob": -0.5} for _ in range(40)],
                }
            ],
        }
        (tmp_path / f"s{index}.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one non-empty model"):
        stage_sessions(tmp_path, min_sessions=0)


def test_grouped_split_never_leaks_prompt_ids():
    rows = [
        {"session_id": f"s{index}", "prompt_id": f"p{index // 2}", "label": index % 2, "split": ""}
        for index in range(16)
    ]
    _assign_stratified_splits(rows, seed=1337, test_fraction=0.25)
    by_prompt: dict[str, set[str]] = {}
    for row in rows:
        by_prompt.setdefault(row["prompt_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in by_prompt.values())
    assert {row["label"] for row in rows if row["split"] == "train"} == {0, 1}
    assert {row["label"] for row in rows if row["split"] == "test"} == {0, 1}


def test_explicit_split_rejects_prompt_leakage():
    rows = [
        {"session_id": "a", "prompt_id": "shared", "label": 0, "split": "train"},
        {"session_id": "b", "prompt_id": "shared", "label": 1, "split": "test"},
    ]
    with pytest.raises(ValueError, match="prompt groups cross"):
        _assign_stratified_splits(rows, seed=1337, test_fraction=0.25)
