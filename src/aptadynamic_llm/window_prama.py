"""Fail-closed identity declaration for window-scale PRAMA projection."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import subprocess
from typing import Any, Mapping

from aptadynamic_llm.artifact_schema import sha256_value


SIGNED_UNIT_AFFINE_V1 = {
    "name": "signed_unit_affine_v1",
    "source_min": -1.0,
    "source_max": 1.0,
    "target_min": 0.0,
    "target_max": 1.0,
}


def signed_unit_affine_v1(value: float) -> float:
    """Map a signed coupling coordinate into PRAMA's nonnegative input."""

    numeric = float(value)
    if not -1.0 <= numeric <= 1.0:
        raise ValueError("signed coupling input must lie in [-1, 1]")
    return (numeric + 1.0) / 2.0


def source_tree_sha256(package_root: str | Path) -> str:
    """Hash the importable Python source tree using stable relative paths."""

    root = Path(package_root).resolve()
    if not root.is_dir():
        raise ValueError(f"package source root does not exist: {root}")
    files = sorted(
        (
            path
            for path in root.rglob("*.py")
            if "__pycache__" not in path.parts
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not files:
        raise ValueError(f"package source root contains no Python files: {root}")
    digest = sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def discover_git_commit(package_root: str | Path) -> str | None:
    """Return the containing repository HEAD when Git metadata is available."""

    root = Path(package_root).resolve()
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip().lower()
    if len(value) == 40 and all(char in "0123456789abcdef" for char in value):
        return value
    return None


@dataclass(frozen=True)
class WindowKernelIdentity:
    package: str
    version: str
    source_tree_sha256: str
    commit: str | None
    kernel_api: str
    config_sha256: str
    recertification_sha256: str
    bin_scale: str

    def __post_init__(self) -> None:
        if self.package != "prama-protokol":
            raise ValueError("kernel package must be prama-protokol")
        if self.bin_scale != "window":
            raise ValueError("kernel bin_scale must be window")
        if not self.version:
            raise ValueError("kernel version must be pinned")
        if self.commit is not None and len(self.commit) < 7:
            raise ValueError("kernel commit must be null or a pinned revision")
        if self.kernel_api != "project_v3":
            raise ValueError("kernel_api must be project_v3")
        for name in (
            "source_tree_sha256",
            "config_sha256",
            "recertification_sha256",
        ):
            value = getattr(self, name).removeprefix("sha256:")
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"{name} must be a SHA-256 digest")

    def to_dict(self) -> dict[str, str | None]:
        return {
            "package": self.package,
            "version": self.version,
            "source_tree_sha256": self.source_tree_sha256.removeprefix("sha256:"),
            "commit": self.commit,
            "kernel_api": self.kernel_api,
            "config_sha256": self.config_sha256.removeprefix("sha256:"),
            "recertification_sha256": self.recertification_sha256.removeprefix(
                "sha256:"
            ),
            "bin_scale": self.bin_scale,
        }


def validate_window_kernel_declaration(
    declaration: Mapping[str, Any],
    *,
    actual_version: str,
    actual_source_tree_sha256: str,
    actual_commit: str | None,
    recertification_sha256: str,
) -> tuple[
    WindowKernelIdentity,
    dict[str, Any],
    dict[str, Any],
    dict[str, str],
]:
    """Verify identity, config and recertification before any kernel call."""

    identity = WindowKernelIdentity(**declaration["kernel_identity"])
    config = dict(declaration["kernel_config"])
    input_transform = dict(declaration["input_transform"])
    column_map = dict(declaration["column_map"])
    if identity.version != actual_version:
        raise ValueError("installed PRAMA identity differs from frozen declaration")
    if (
        identity.source_tree_sha256.removeprefix("sha256:")
        != actual_source_tree_sha256.removeprefix("sha256:")
    ):
        raise ValueError("installed PRAMA source tree differs from frozen declaration")
    if identity.commit is not None and identity.commit != actual_commit:
        raise ValueError("installed PRAMA commit differs from frozen declaration")
    if input_transform != SIGNED_UNIT_AFFINE_V1:
        raise ValueError("input_transform must be signed_unit_affine_v1")
    projection_config = {
        "kernel_config": config,
        "input_transform": input_transform,
    }
    if (
        identity.config_sha256.removeprefix("sha256:")
        != sha256_value(projection_config)
    ):
        raise ValueError("projection configuration does not match config_sha256")
    if (
        identity.recertification_sha256.removeprefix("sha256:")
        != recertification_sha256.removeprefix("sha256:")
    ):
        raise ValueError("window-scale recertification hash mismatch")
    required = {
        "delta",
        "xi",
        "accumulated_excess",
        "capacity",
        "theta",
        "balance",
        "trend",
        "valid",
    }
    if not required <= column_map.keys():
        raise ValueError(f"column_map is missing {sorted(required - column_map.keys())}")
    return identity, config, input_transform, column_map
