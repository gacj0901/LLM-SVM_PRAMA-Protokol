from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path


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
