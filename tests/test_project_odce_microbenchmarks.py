from argparse import Namespace
import json
from pathlib import Path

from aptadynamic_llm.artifact_schema import validate_artifact
from scripts import project_odce_microbenchmarks as projection
from scripts.run_odce_microbenchmarks import make_domain_outcome


ROOT = Path(__file__).parents[1]


def _token(index):
    return {
        "token": str(index),
        "top1_logprob": -0.2,
        "top_logprobs": [-0.2, -1.2],
        "gap": 1.0,
        "entropy": 0.5,
    }


def test_numeric_request_preserves_session_identity_and_excludes_metadata():
    raw = {"session_id": "s1", "model": "hermes3:8b"}
    turn = {
        "turn_index": 0,
        "finish_reason": "stop",
        "assistant_message": "secret",
        "tokens": [_token(0), _token(1)],
    }
    request, tokens = projection.numeric_request(raw, turn, "a" * 64, "b" * 64)
    encoded = json.dumps(request)
    assert request["session_id"] == "s1"
    assert len(tokens) == 2
    assert "finish_reason" not in encoded
    assert "assistant_message" not in encoded


def test_projection_writes_joinable_prama_do_v9_and_outcome(tmp_path, monkeypatch):
    source_run = tmp_path / "source"
    raw_dir = source_run / "sessions" / "s1"
    raw_dir.mkdir(parents=True)
    tokens = [_token(index) for index in range(17)]
    raw = {
        "schema": projection.RAW_SCHEMA,
        "session_id": "s1",
        "provider": "ollama",
        "model": "hermes3:8b",
        "resolved_model": "hermes3:8b",
        "turns": [
            {
                "turn_index": 0,
                "assistant_message": "ok",
                "finish_reason": "stop",
                "token_count": len(tokens),
                "tokens": tokens,
            }
        ],
    }
    raw_path = raw_dir / "raw.json"
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    outcome = make_domain_outcome(
        study_id="test-study",
        raw_path=raw_path,
        raw=raw,
        verification={"functional_gain": 1.0, "verified_outcome": 1.0},
        suite_sha256="a" * 64,
        verifier_sha256="b" * 64,
        window_size=16,
    )
    (source_run / "domain_return_observations.jsonl").write_text(
        json.dumps(outcome) + "\n", encoding="utf-8"
    )

    identity = {
        "package": "prama-protokol",
        "version": "0.3.0",
        "source_tree_sha256": "c" * 64,
        "commit": "d" * 40,
        "kernel_api": "project_v3",
        "config_sha256": "e" * 64,
        "recertification_sha256": "f" * 64,
        "bin_scale": "window",
    }
    monkeypatch.setattr(
        projection,
        "validate_identity",
        lambda *_args: ({}, {}, identity),
    )

    def fake_project(request, *_args):
        rows = []
        for index, count in enumerate((16, 1)):
            rows.append(
                {
                    "contract_version": "0.2.0",
                    "artifact_type": "prama_trajectory",
                    "artifact_version": "1.0.0",
                    "study_id": "old",
                    "session_id": request["session_id"],
                    "producer": "fake",
                    "created_at": "2026-08-08T00:00:00+00:00",
                    "source_sha256": "1" * 64,
                    "config_sha256": "2" * 64,
                    "partition": "exploratory",
                    "channel_status": "OBSERVED",
                    "turn_index": 0,
                    "window_index": index,
                    "n_tokens_in_window": count,
                    "delta": 0.1,
                    "xi": 0.2,
                    "accumulated_excess": 0.0,
                    "capacity": 1.0,
                    "theta": 0.5,
                    "balance": 0.5,
                    "trend": 0.0,
                    "input_transform": "signed_unit_affine_v1",
                    "input_channel_status": "OBSERVED",
                    "coordinate_origin": "DERIVED_KERNEL_STATE",
                    "kernel_identity": identity,
                    "valid": True,
                }
            )
        return rows

    monkeypatch.setattr(projection, "project_dynamic", fake_project)
    monkeypatch.setattr(
        projection,
        "observe_structural_coherence_v6",
        lambda token_windows, trajectory, *_args: [
            {
                "turn_index": 0,
                "window_index": index,
                "absolute_window_index": index,
                "geometry_ready": False,
                "movement": 0.0,
                "transport_coherence": None,
                "recurrence_persistence": 0.0,
                "variation_contraction": None,
            }
            for index in range(len(trajectory))
        ],
    )
    monkeypatch.setattr(
        projection.OperatorGeometryConfig,
        "from_contract",
        lambda _contract: object(),
    )
    monkeypatch.setattr(
        projection.StructuralCoherenceV6Config,
        "from_contract",
        lambda _contract: object(),
    )
    output_dir = tmp_path / "output"
    args = Namespace(
        source_run=[source_run],
        output_dir=output_dir,
        study_id="test-study",
        prama_source_root=None,
        dynamic_contract=ROOT / "config" / "cocc_dynamic_observer_contract_v1.json",
        geometry_contract=ROOT / "config" / "cocc_operator_geometry_observer_v1.json",
        coherence_contract=ROOT / "config" / "sequor_structural_coherence_observer_v6.json",
        v9_contract=ROOT / "config" / "sequor_structural_observer_v9.json",
        declaration=ROOT / "config" / "window_prama_kernel_declaration.json",
        recertification=ROOT / "run_outputs" / "window_prama_recertification_v030_20260730.json",
    )
    report = projection.project(args)
    assert report["session_count"] == 1
    assert report["prama_observation_count"] == 2
    assert report["structural_observation_count"] == 2
    assert report["contract_hashes"]["kernel"] == "e" * 64
    prama = projection.read_jsonl(output_dir / "prama_trajectory.jsonl")
    structural = projection.read_jsonl(output_dir / "structural_observations.jsonl")
    outcomes = projection.read_jsonl(output_dir / "domain_return_observations.jsonl")
    assert {row["session_id"] for row in prama + structural + outcomes} == {"s1"}
    assert [(row["turn_index"], row["window_index"]) for row in prama] == [
        (0, 0),
        (0, 1),
    ]
    for row in prama:
        validate_artifact(row, "prama_trajectory")
    for row in structural:
        validate_artifact(row, "structural_observation")
