"""Tests for the sole authoritative E-P1 verdict path."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import scripts.analyze_ep1 as analysis


def test_script_entrypoint_resolves_repository_imports():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "scripts" / "analyze_ep1.py"), "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--replication-freeze" in completed.stdout


def test_missing_collection_manifest_emits_interface_failure(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    out = tmp_path / "analysis"
    assert analysis.main(["--sessions-dir", str(source), "--out", str(out)]) == 2
    verdict = json.loads((out / "ep1_verdict.json").read_text(encoding="utf-8"))
    assert verdict["verdict"] == "interface_failure"
    assert verdict["analysis_executed"] is False


def test_power_gate_stops_before_score_evaluation(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "confirmatory"
    out = tmp_path / "analysis"
    subprocess.run(
        [sys.executable, str(root / "examples" / "make_synthetic.py"), "structural", str(source)],
        check=True,
        cwd=root,
        capture_output=True,
        text=True,
    )
    for path in source.glob("s*.json"):
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["model"] = "hermes3:8b"
        path.write_text(json.dumps(raw), encoding="utf-8")
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "mode": "confirmatory",
                "model": "hermes3:8b",
                "model_digest": "4f6b83f30b62-test",
                "ollama_version": "0.30.11",
                "temperature": 0.7,
                "top_p": 0.9,
                "top_logprobs": 5,
                "seed": 1337,
                "seed_per_index": True,
                "num_predict": 512,
                "prompt_suite_sha256": "fixture-hash",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(analysis, "FROZEN_NUM_PREDICT", 512)
    monkeypatch.setattr(analysis, "FROZEN_PROMPT_SUITE_SHA256", "fixture-hash")

    assert analysis.main(["--sessions-dir", str(source), "--out", str(out)]) == 3
    verdict = json.loads((out / "ep1_verdict.json").read_text(encoding="utf-8"))
    assert verdict["verdict"] == "underpowered"
    assert verdict["analysis_executed"] is False
    assert verdict["gates"]["power"]["test_positives"] < 15
    assert verdict["gates"]["power"]["scores_computed_at_gate"] is False
    assert verdict["gates"]["C3"]["status"] == "not_run"
    assert not (out / "scored").exists()
    assert not (out / "evaluation").exists()

    # Lowering only the test fixture's power gate exercises the complete verdict path.
    complete_out = tmp_path / "complete_analysis"
    monkeypatch.setattr(analysis, "MIN_TEST_POSITIVES", 1)
    assert analysis.main(["--sessions-dir", str(source), "--out", str(complete_out)]) == 0
    complete = json.loads((complete_out / "ep1_verdict.json").read_text(encoding="utf-8"))
    assert complete["verdict"] in {"positive", "honest_null"}
    assert complete["analysis_executed"] is True
    assert complete["gates"]["C3"]["passed"] is True
    assert complete["baseline_fields"] == [
        "mean_surprisal",
        "mean_entropy",
        "negative_mean_top1_gap",
        "markovian_surprisal_tau64",
    ]
    assert set(complete["gates"]) == {
        "C3",
        "power",
        "permutation",
        "auroc_vs_all_four_baselines",
        "tpr_at_fpr_vs_all_four_baselines",
    }
