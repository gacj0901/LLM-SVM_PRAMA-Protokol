#!/usr/bin/env python
"""Numerically recertify PRAMA v0.3 for window bins and freeze a declaration."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Callable

from aptadynamic_llm.artifact_schema import sha256_value
from aptadynamic_llm.window_prama import (
    SIGNED_UNIT_AFFINE_V1,
    discover_git_commit,
    signed_unit_affine_v1,
    source_tree_sha256,
)


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        delete=False,
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def _test(name: str, operation: Callable[[], tuple[bool, dict[str, Any]]]):
    try:
        passed, metrics = operation()
        return {"name": name, "passed": bool(passed), "metrics": metrics}
    except Exception as exc:  # the ledger must record a fail-closed test result
        return {
            "name": name,
            "passed": False,
            "metrics": {"error": f"{type(exc).__name__}: {exc}"},
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--declaration-out", required=True, type=Path)
    parser.add_argument("--tau", type=float, default=16.0)
    parser.add_argument("--theta-scale", type=float, default=0.2)
    parser.add_argument("--kappa-v3", type=float, default=0.001)
    parser.add_argument("--g-smooth", type=int, default=8)
    args = parser.parse_args(argv)

    try:
        import numpy as np
        import prama_protokol
        from prama_protokol import (
            KernelConfigV3,
            KernelV3,
            V3ProjectionError,
            project_v3,
        )

        package_root = Path(prama_protokol.__file__).resolve().parent
        source_hash = source_tree_sha256(package_root)
        commit = discover_git_commit(package_root)
        config_object = KernelConfigV3(
            h=1.0,
            tau=args.tau,
            theta_scale=args.theta_scale,
            lambda_0=1.0,
            lambda_min=0.1,
            lambda_max=1.0,
            kappa_v3=args.kappa_v3,
            g_smooth=args.g_smooth,
            delta_ref=1.0,
        )
        config = asdict(config_object)
        projection_config = {
            "kernel_config": config,
            "input_transform": SIGNED_UNIT_AFFINE_V1,
        }
        config_hash = sha256_value(projection_config)
        expected_columns = {
            "delta",
            "delta_tilde",
            "e",
            "xi",
            "A",
            "lambda",
            "theta",
            "M",
            "G",
            "u_lambda",
            "sigma_op",
            "valid",
            "input_index",
            "state_index",
        }
        pulse = np.concatenate(
            (np.zeros(8), np.ones(24), np.zeros(40))
        ).astype(np.float64)
        null = np.zeros_like(pulse)

        def schema_test():
            trajectory = project_v3(pulse, null, config_object)
            columns = set(trajectory.as_dict())
            return columns == expected_columns, {
                "actual_columns": sorted(columns),
                "expected_columns": sorted(expected_columns),
                "row_count": len(trajectory),
            }

        def input_transform_test():
            source = (-1.0, 0.0, 1.0)
            transformed = tuple(signed_unit_affine_v1(value) for value in source)
            rejected = False
            try:
                signed_unit_affine_v1(1.000001)
            except ValueError:
                rejected = True
            passed = transformed == (0.0, 0.5, 1.0) and rejected
            return passed, {
                "source": source,
                "transformed": transformed,
                "out_of_range_rejected": rejected,
            }

        def determinism_test():
            first = project_v3(pulse, null, config_object).as_dict()
            second = project_v3(pulse, null, config_object).as_dict()
            exact = all(np.array_equal(first[key], second[key]) for key in first)
            return exact, {"exact_array_equality": exact}

        def causal_prefix_test():
            full = project_v3(pulse, null, config_object).as_dict()
            prefix_length = 40
            prefix = project_v3(
                pulse[:prefix_length], null[:prefix_length], config_object
            ).as_dict()
            exact = all(
                np.array_equal(prefix[key], full[key][:prefix_length])
                for key in prefix
            )
            return exact, {
                "prefix_length": prefix_length,
                "exact_array_equality": exact,
            }

        def quiescent_test():
            zero = np.zeros(48, dtype=np.float64)
            trajectory = project_v3(zero, zero, config_object)
            passed = (
                np.all(trajectory.delta == 0.0)
                and np.all(trajectory.xi == 0.0)
                and np.all(trajectory.A == 0.0)
                and np.all(trajectory.lambda_ == config_object.lambda_0)
                and np.all(trajectory.valid)
            )
            return bool(passed), {
                "max_abs_delta": float(np.max(np.abs(trajectory.delta))),
                "max_abs_xi": float(np.max(np.abs(trajectory.xi))),
                "max_abs_A": float(np.max(np.abs(trajectory.A))),
            }

        def bounded_pulse_test():
            trajectory = project_v3(pulse, null, config_object)
            numeric_columns = (
                trajectory.delta,
                trajectory.xi,
                trajectory.A,
                trajectory.lambda_,
                trajectory.theta,
                trajectory.M,
                trajectory.G,
            )
            finite = all(np.isfinite(column).all() for column in numeric_columns)
            crossed = bool(np.any(trajectory.e > 0.0))
            accumulated = float(trajectory.A[-1]) > 0.0
            bounded = bool(
                np.all(trajectory.lambda_ >= config_object.lambda_min)
                and np.all(trajectory.lambda_ <= config_object.lambda_max)
            )
            pulse_end = 8 + 24 - 1
            recovered = bool(
                trajectory.xi[-1] < trajectory.xi[pulse_end]
                and trajectory.M[-1] > 0.0
            )
            monotone_A = bool(np.all(np.diff(trajectory.A) >= -1e-15))
            passed = finite and crossed and accumulated and bounded and recovered and monotone_A
            return passed, {
                "finite": finite,
                "threshold_crossing_observed": crossed,
                "max_xi": float(np.max(trajectory.xi)),
                "final_xi": float(trajectory.xi[-1]),
                "final_A": float(trajectory.A[-1]),
                "minimum_lambda": float(np.min(trajectory.lambda_)),
                "final_margin": float(trajectory.M[-1]),
                "post_pulse_recovery_observed": recovered,
                "accumulated_excess_monotone": monotone_A,
            }

        def streaming_equivalence_test():
            batch = project_v3(pulse, null, config_object).rows()
            kernel = KernelV3(config_object)
            streamed = [
                row.as_dict()
                for omega, expected in zip(pulse, null)
                if (row := kernel.step(float(omega), float(expected))) is not None
            ]
            exact = batch == streamed
            audit = kernel.numeric_audit
            residual = max(
                abs(audit.lambda_step_residual),
                abs(audit.lambda_ledger_residual),
            )
            return exact and residual <= 1e-12, {
                "exact_row_equality": exact,
                "maximum_lambda_ledger_residual": residual,
                "resummation_count": audit.resummation_count,
            }

        def fail_closed_test():
            kernel = KernelV3(config_object)
            kernel.step(0.0, 0.0)
            before = kernel.numeric_audit
            raised = False
            try:
                kernel.step(0.1, math.nan)
            except V3ProjectionError:
                raised = True
            after = kernel.numeric_audit
            unchanged = before == after
            return raised and unchanged, {
                "internal_missing_rejected": raised,
                "state_unchanged_after_rejection": unchanged,
            }

        tests = [
            _test("frozen_gamma_v3_schema", schema_test),
            _test("signed_unit_affine_input_transform", input_transform_test),
            _test("deterministic_batch_projection", determinism_test),
            _test("causal_prefix_invariance", causal_prefix_test),
            _test("quiescent_zero_response", quiescent_test),
            _test("bounded_pulse_crossing_and_recovery", bounded_pulse_test),
            _test("streaming_batch_and_numeric_ledger", streaming_equivalence_test),
            _test("fail_closed_internal_missing", fail_closed_test),
        ]
        passed = all(test["passed"] for test in tests)
        recertification = {
            "artifact_type": "window_prama_recertification",
            "recertification_version": "0.1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "PASS" if passed else "FAIL",
            "scope": "numeric compatibility at bin_scale=window; not construct validation",
            "kernel_identity": {
                "package": "prama-protokol",
                "version": str(getattr(prama_protokol, "__version__", "")),
                "source_tree_sha256": source_hash,
                "commit": commit,
                "kernel_api": "project_v3",
                "bin_scale": "window",
            },
            "kernel_config": config,
            "input_transform": SIGNED_UNIT_AFFINE_V1,
            "config_sha256": config_hash,
            "tests": tests,
        }
        _write_json_atomic(args.out, recertification)
        recertification_hash = sha256(args.out.read_bytes()).hexdigest()
        declaration = {
            "declaration_version": "0.1.0",
            "kernel_identity": {
                **recertification["kernel_identity"],
                "config_sha256": config_hash,
                "recertification_sha256": recertification_hash,
            },
            "kernel_config": config,
            "input_transform": SIGNED_UNIT_AFFINE_V1,
            "column_map": {
                "delta": "delta",
                "xi": "xi",
                "accumulated_excess": "A",
                "capacity": "lambda",
                "theta": "theta",
                "balance": "M",
                "trend": "G",
                "valid": "valid",
            },
        }
        _write_json_atomic(args.declaration_out, declaration)
    except (ImportError, OSError, TypeError, ValueError) as exc:
        print(f"window PRAMA recertification failed: {exc}")
        return 1

    print(
        f"{recertification['status']}: {len(tests)} numerical checks; "
        f"recertification={args.out}; declaration={args.declaration_out}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
