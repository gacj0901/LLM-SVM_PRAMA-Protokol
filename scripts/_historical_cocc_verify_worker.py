#!/usr/bin/env python3
"""CoCC worker variant that isolates incidental stdout from functional candidates."""

from __future__ import annotations

import contextlib
import ast
import io
import json
import math
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

FORBIDDEN_IMPORT_ROOTS = {
    "asyncio", "ctypes", "http", "multiprocessing", "os", "pathlib", "shutil",
    "signal", "socket", "subprocess", "threading", "urllib", "winreg",
}
FORBIDDEN_CALLS = {"__import__", "breakpoint", "compile", "eval", "exec", "exit", "open", "quit"}
MISSING_CALLABLE_MESSAGE = re.compile(
    r"^callable '[A-Za-z_][A-Za-z0-9_]*' not found$"
)


def exception_result(exc: BaseException, tests_total: int):
    """Return class and message so missing-callable audits are evidence based."""
    exception_type = type(exc).__name__
    exception_message = str(exc)[:1000]
    missing_callable = bool(
        exception_type == "AttributeError"
        and MISSING_CALLABLE_MESSAGE.fullmatch(exception_message)
    )
    return {
        "passed": False,
        "status": "failed",
        "tests_passed": 0,
        "tests_total": int(tests_total),
        "failure_kind": "missing_callable" if missing_callable else exception_type,
        "exception_type": exception_type,
        "exception_message": exception_message,
    }


def safety_gate(source):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = {alias.name.split(".", 1)[0] for alias in node.names}
            if roots & FORBIDDEN_IMPORT_ROOTS:
                raise ValueError("forbidden import")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".", 1)[0] in FORBIDDEN_IMPORT_ROOTS:
                raise ValueError("forbidden import")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
                raise ValueError("forbidden call")
            if isinstance(node.func, ast.Attribute) and node.func.attr in {
                "popen", "remove", "rename", "replace", "rmdir", "system", "unlink",
            }:
                raise ValueError("forbidden attribute call")


def parse_arguments(raw):
    return [json.loads(line) for line in str(raw).splitlines() if line.strip()]


def parse_expected(raw):
    try:
        return json.loads(str(raw))
    except json.JSONDecodeError:
        return str(raw).strip()


def comparable(value):
    if isinstance(value, tuple):
        return [comparable(item) for item in value]
    if isinstance(value, list):
        return [comparable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): comparable(member) for key, member in value.items()}
    return value


def equal(actual, expected):
    actual, expected = comparable(actual), comparable(expected)
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(float(actual), float(expected), rel_tol=1e-6, abs_tol=1e-6)
    return actual == expected


def functional_case(namespace, case, func_name):
    solution = namespace.get("Solution")
    target = getattr(solution(), func_name) if isinstance(solution, type) else namespace.get(func_name)
    if not callable(target):
        raise AttributeError(f"callable {func_name!r} not found")
    return equal(target(*parse_arguments(case["input"])), parse_expected(case["output"]))


def stdin_case(source, namespace, case):
    stdin = io.StringIO(str(case["input"]))
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
        previous = sys.stdin
        sys.stdin = stdin
        try:
            namespace["__name__"] = "__main__"
            exec(compile(source, "<candidate>", "exec"), namespace, namespace)
        finally:
            sys.stdin = previous
    actual = "\n".join(line.rstrip() for line in stdout.getvalue().strip().splitlines())
    expected = "\n".join(line.rstrip() for line in str(case["output"]).strip().splitlines())
    return actual == expected


def quiet_functional_case(namespace, case, func_name):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return functional_case(namespace, case, func_name)


def evaluate(payload):
    source = str(payload["source"])
    tests = list(payload.get("tests") or [])
    func_name = str(payload.get("func_name") or "")
    safety_gate(source)
    namespace = {
        "__name__": "__candidate__",
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "Set": Set,
        "Tuple": Tuple,
    }
    if func_name:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec(compile(source, "<candidate>", "exec"), namespace, namespace)
    passed = 0
    for case in tests:
        test_type = str(case.get("testtype") or ("functional" if func_name else "stdin"))
        okay = (
            quiet_functional_case(namespace, case, func_name)
            if test_type == "functional"
            else stdin_case(source, namespace, case)
        )
        if not okay:
            return {
                "passed": False, "status": "failed", "tests_passed": passed,
                "tests_total": len(tests), "failure_kind": "wrong_answer",
            }
        passed += 1
    return {
        "passed": True, "status": "passed", "tests_passed": passed,
        "tests_total": len(tests), "failure_kind": None,
    }


def main() -> int:
    try:
        result = evaluate(json.load(sys.stdin))
    except BaseException as exc:
        result = exception_result(exc, 0)
    sys.stdout.write(json.dumps(result, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
