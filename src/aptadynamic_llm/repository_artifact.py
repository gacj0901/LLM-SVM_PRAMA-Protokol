"""Portable verification of repository-bound artifact SHA-256 identities."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import re


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LFS_OID = re.compile(
    rb"(?:^|\n)oid sha256:([0-9a-f]{64})(?:\n|$)"
)
_LFS_HEADER = b"version https://git-lfs.github.com/spec/v1"


def repository_artifact_sha256_candidates(path: Path) -> dict[str, str]:
    """Return identities allowed after Git/LFS materialization.

    Frozen historical records contain raw Windows-worktree hashes. GitHub Actions
    checks out ordinary text with LF and may leave Git LFS objects as pointer files.
    This function permits only those transport-level representations: exact bytes,
    the exact LFS object OID, or UTF-8 text differing solely by CRLF versus LF.
    """
    raw = path.read_bytes()
    candidates = {"raw_bytes": sha256(raw).hexdigest()}
    if raw.startswith(_LFS_HEADER):
        match = _LFS_OID.search(raw.replace(b"\r\n", b"\n"))
        if match:
            candidates["git_lfs_object_oid"] = match.group(1).decode("ascii")
        return candidates

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return candidates
    if "\x00" in text:
        return candidates
    lf_text = text.replace("\r\n", "\n")
    candidates["utf8_lf"] = sha256(lf_text.encode("utf-8")).hexdigest()
    candidates["utf8_crlf"] = sha256(
        lf_text.replace("\n", "\r\n").encode("utf-8")
    ).hexdigest()
    return candidates


def matches_frozen_sha256(path: Path, expected: str) -> bool:
    expected = str(expected).lower()
    if not _SHA256.fullmatch(expected):
        raise ValueError("expected SHA-256 must be 64 lowercase hexadecimal characters")
    return expected in repository_artifact_sha256_candidates(path).values()
