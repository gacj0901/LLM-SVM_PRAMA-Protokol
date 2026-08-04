from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from aptadynamic_llm.window_prama import validate_window_kernel_declaration
from aptadynamic_llm.repository_artifact import (
    matches_frozen_sha256,
    repository_artifact_sha256_candidates,
)


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_kernel_lock_binds_dependency_declaration_and_recertification():
    lock = json.loads((ROOT / "config/kernel.lock.json").read_text(encoding="utf-8"))
    declaration_path = ROOT / lock["declaration"]["path"]
    recertification_path = ROOT / lock["recertification"]["path"]
    declaration = json.loads(declaration_path.read_text(encoding="utf-8"))
    recertification = json.loads(recertification_path.read_text(encoding="utf-8"))

    assert digest(declaration_path) == lock["declaration"]["sha256"]
    assert digest(recertification_path) == lock["recertification"]["sha256"]
    assert recertification["status"] == lock["recertification"]["required_status"]
    assert declaration["kernel_identity"]["commit"] == lock["git_commit"]
    assert declaration["kernel_identity"]["source_tree_sha256"] == lock["source_tree_sha256"]
    assert declaration["kernel_identity"]["config_sha256"] == lock["config_sha256"]
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert lock["dependency_spec"] in pyproject


def test_retained_data_and_results_manifests_respect_directory_boundary():
    retained = json.loads(
        (ROOT / "data/retained_manifest_v1.json").read_text(encoding="utf-8")
    )
    results = json.loads(
        (ROOT / "run_outputs/reproducible_results_manifest_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert retained["files"] and results["artifacts"]
    assert all(item["path"].startswith("data/") for item in retained["files"])
    assert all(
        item["path"].startswith("run_outputs/") for item in results["artifacts"]
    )
    assert all(len(item["sha256"]) == 64 for item in retained["files"])
    assert all(len(item["sha256"]) == 64 for item in results["artifacts"])
    assert all(
        matches_frozen_sha256(ROOT / item["path"], item["sha256"])
        for item in retained["files"]
    )
    assert all(
        matches_frozen_sha256(ROOT / item["path"], item["sha256"])
        for item in results["artifacts"]
    )


def test_corrected_historical_join_has_complete_provenance_chain():
    root = ROOT / "run_outputs/historical_v9_backfill_cocc_462"
    amendment_v1_path = root / "provenance_amendment.json"
    amendment_v2 = json.loads(
        (root / "provenance_amendment_v2.json").read_text(encoding="utf-8")
    )
    corrected_report_path = root / "verified_overlap_120_verifier_v2/report.json"
    transition_path = root / "verifier_transition_v1_to_v2.json"
    review_path = root / "verifier_manual_review_v1.json"
    corrected_report = json.loads(corrected_report_path.read_text(encoding="utf-8"))

    assert matches_frozen_sha256(
        amendment_v1_path, amendment_v2["supersedes_amendment_sha256"]
    )
    assert matches_frozen_sha256(
        corrected_report_path,
        amendment_v2["bound_artifacts"]["corrected_verifier_v2_report_sha256"],
    )
    assert matches_frozen_sha256(
        transition_path,
        amendment_v2["bound_artifacts"]["verifier_transition_audit_sha256"],
    )
    assert matches_frozen_sha256(
        review_path,
        amendment_v2["bound_artifacts"]["verifier_manual_review_sha256"],
    )
    assert corrected_report["provenance_context_source_sha256"] == digest(
        amendment_v1_path
    )
    assert corrected_report["historical_generation_context"] is not None
    assert corrected_report["reprojection_context"] is not None
    assert corrected_report["interpretation_boundary"]


def test_repository_artifact_hash_accepts_only_line_endings_or_lfs_oid(tmp_path):
    crlf = tmp_path / "artifact.json"
    crlf.write_bytes(b'{\r\n  "value": 1\r\n}\r\n')
    lf_digest = sha256(b'{\n  "value": 1\n}\n').hexdigest()
    assert matches_frozen_sha256(crlf, lf_digest)

    oid = "a" * 64
    pointer = tmp_path / "artifact.jsonl"
    pointer.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{oid}\nsize 123\n",
        encoding="ascii",
    )
    assert repository_artifact_sha256_candidates(pointer)["git_lfs_object_oid"] == oid
    assert matches_frozen_sha256(pointer, oid)
    assert not matches_frozen_sha256(pointer, "b" * 64)


def test_mismatched_installed_kernel_is_rejected_before_projection():
    lock = json.loads((ROOT / "config/kernel.lock.json").read_text(encoding="utf-8"))
    declaration = json.loads(
        (ROOT / lock["declaration"]["path"]).read_text(encoding="utf-8")
    )
    with pytest.raises(
        ValueError, match="installed PRAMA source tree differs from frozen declaration"
    ):
        validate_window_kernel_declaration(
            declaration,
            actual_version=lock["version"],
            actual_source_tree_sha256="0" * 64,
            actual_commit=lock["git_commit"],
            recertification_sha256=lock["recertification"]["sha256"],
        )
