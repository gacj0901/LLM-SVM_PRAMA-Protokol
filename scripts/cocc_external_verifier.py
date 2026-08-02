#!/usr/bin/env python
"""Verify one generated answer against CoCC tests in an isolated child process."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import pickle
import re
import subprocess
import sys
from typing import Any
import zlib

FENCE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
MISSING_CALLABLE_MESSAGE = re.compile(
    r"^callable '[A-Za-z_][A-Za-z0-9_]*' not found$"
)


def audit_worker_exception(result: dict[str, Any]) -> dict[str, Any]:
    """Fail closed when worker exception class and message do not support its label."""
    exception_type = result.get("exception_type")
    exception_message = result.get("exception_message")
    failure_kind = result.get("failure_kind")
    if exception_type is None and exception_message is None:
        if failure_kind == "missing_callable":
            raise ValueError("missing_callable lacks exception audit evidence")
        return result
    if not isinstance(exception_type, str) or not isinstance(exception_message, str):
        raise ValueError("worker exception audit requires string type and message")
    message_matches = bool(MISSING_CALLABLE_MESSAGE.fullmatch(exception_message))
    supported = exception_type == "AttributeError" and message_matches
    if (failure_kind == "missing_callable") != supported:
        raise ValueError(
            "worker missing_callable classification disagrees with exception message"
        )
    result["exception_audit"] = {
        "class_and_message_consistent": True,
        "missing_callable_message_match": message_matches,
    }
    return result


def load_dataset(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            result[str(row["item_id"])] = row
    if not result:
        raise ValueError("normalized dataset is empty")
    return result


def decoded_cases(value: Any, private: bool = False) -> list[dict[str, Any]]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        if private:
            try:
                decoded = pickle.loads(zlib.decompress(base64.b64decode(value)))
                value = decoded.decode("utf-8") if isinstance(decoded, bytes) else decoded
            except Exception:
                pass
        decoded = json.loads(value) if isinstance(value, str) else value
        return decoded if isinstance(decoded, list) else [decoded]
    raise ValueError("unsupported test-case encoding")


def metadata_func_name(value: Any) -> str:
    if isinstance(value, str):
        value = json.loads(value)
    return str((value or {}).get("func_name") or "")


def extract_source(answer: str) -> str:
    blocks = [block.strip() for block in FENCE.findall(answer) if block.strip()]
    return max(blocks, key=len) if blocks else answer.strip()


def verify(
    request: dict[str, Any],
    rows: dict[str, dict[str, Any]],
    worker: Path,
    timeout: int,
) -> dict[str, Any]:
    item_id = str(request["item_id"])
    if item_id not in rows:
        raise ValueError(f"item_id not found in verifier dataset: {item_id}")
    row = rows[item_id]
    tests = decoded_cases(row.get("public_test_cases"))
    tests.extend(decoded_cases(row.get("private_test_cases"), private=True))
    if not tests:
        raise ValueError("problem has no decodable tests")
    payload = {
        "source": extract_source(str(request.get("observed_answer") or "")),
        "tests": tests,
        "func_name": metadata_func_name(row.get("metadata")),
    }
    try:
        completed = subprocess.run(
            [sys.executable, "-I", str(worker)],
            input=json.dumps(payload, ensure_ascii=False),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "status": "failed",
            "scope": "public_and_private_tests",
            "failure_kind": "timeout",
            "tests_passed": 0,
            "tests_total": len(tests),
        }
    if completed.returncode != 0 or not completed.stdout.strip():
        return {
            "passed": False,
            "status": "failed",
            "scope": "public_and_private_tests",
            "failure_kind": "worker_error",
            "tests_passed": 0,
            "tests_total": len(tests),
        }
    result = audit_worker_exception(json.loads(completed.stdout))
    result["scope"] = "public_and_private_tests"
    result["tests_total"] = len(tests)
    result["verifier_is_external_to_model"] = True
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument(
        "--worker",
        type=Path,
        default=Path(__file__).with_name("_cocc_verify_worker.py"),
    )
    parser.add_argument("--timeout", type=int, default=20)
    args = parser.parse_args(argv)
    try:
        result = verify(
            json.load(sys.stdin),
            load_dataset(args.dataset),
            args.worker.resolve(),
            args.timeout,
        )
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
        print(f"CoCC verification failed: {exc}", file=sys.stderr)
        return 1
    sys.stdout.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
