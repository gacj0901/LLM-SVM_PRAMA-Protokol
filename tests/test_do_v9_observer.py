from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import pytest

from aptadynamic_llm.artifact_schema import (
    ArtifactValidationError,
    make_envelope,
    validate_artifact,
)
from aptadynamic_llm.structural_coherence_v9 import StructuralCoherenceV9Config
from aptadynamic_llm.structural_observation import (
    make_structural_observation,
    observe_structural_trajectory,
)


HASH_A = "a" * 64
HASH_B = "b" * 64


def config():
    return StructuralCoherenceV9Config(
        activity_path_length_threshold=0.5,
        recurrence_threshold=0.3,
        coherence_threshold=0.5,
        variation_contraction_threshold=0.25,
        tau_windows=4,
        hysteresis_grace_tau=0.25,
        minimum_session_transport_windows=2,
    )


def window(index, *, coherence=0.9, recurrence=0.1, contraction=0.1, **extra):
    return {
        "turn_index": 0,
        "window_index": index,
        "absolute_window_index": index,
        "geometry_ready": True,
        "movement": 1.0,
        "transport_coherence": coherence,
        "recurrence_persistence": recurrence,
        "variation_contraction": contraction,
        **extra,
    }


def test_do_v9_is_causal_and_resolves_recurrence_contraction_and_mobility():
    windows = [window(index) for index in range(6)]
    windows += [
        window(6, recurrence=0.8),
        window(7, recurrence=0.8, contraction=0.8),
        window(8, recurrence=0.8, contraction=0.8),
        window(9, recurrence=0.8, contraction=0.8),
        window(10, recurrence=0.8, contraction=0.8),
    ]
    prefix = observe_structural_trajectory(windows[:9], config())
    full = observe_structural_trajectory(windows, config())
    assert prefix == full[:9]
    assert full[6]["recurrence_status"] == "RECURRENT"
    assert full[6]["structural_state"] == "RECURRENT"
    assert full[7]["contraction_status"] == "CONTRACTING"
    assert full[7]["structural_state"] == "CRYSTALLIZING"
    assert full[10]["structural_state"] == "CRYSTALLIZED"


def test_do_v9_ignores_and_contract_rejects_provider_termination_metadata():
    base = [window(index, finish_reason="stop") for index in range(8)]
    changed = [window(index, finish_reason="length") for index in range(8)]
    assert observe_structural_trajectory(base, config()) == observe_structural_trajectory(
        changed, config()
    )

    observation = observe_structural_trajectory(base, config())[-1]
    envelope = make_envelope(
        artifact_type="structural_observation",
        study_id="study-1",
        session_id="session-1",
        producer="pytest",
        created_at=datetime.now(timezone.utc).isoformat(),
        source_sha256=HASH_A,
        config_sha256=HASH_B,
        partition="exploratory",
        channel_status="OBSERVED",
    )
    record = make_structural_observation(envelope, observation)
    validate_artifact(record, "structural_observation")
    with pytest.raises(ArtifactValidationError, match="forbidden fields"):
        validate_artifact({**record, "finish_reason": "stop"})


def test_canonical_do_v9_entry_point_emits_structural_observations(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "v6_sessions.jsonl"
    output = tmp_path / "structural_observation.jsonl"
    source.write_text(
        json.dumps(
            {
                "session_id": "session-1",
                "windows": [window(index) for index in range(8)],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "observe_structural_trajectory.py"),
            "--input",
            str(source),
            "--contract",
            str(root / "config" / "sequor_structural_observer_v9.json"),
            "--out",
            str(output),
            "--study-id",
            "study-1",
            "--producer",
            "pytest",
            "--partition",
            "exploratory",
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    rows = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 8
    assert all(row["artifact_type"] == "structural_observation" for row in rows)
    assert all(row["provider_termination_metadata_used"] is False for row in rows)
