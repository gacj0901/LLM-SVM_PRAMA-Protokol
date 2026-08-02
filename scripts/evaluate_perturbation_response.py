#!/usr/bin/env python
"""Build perturbation-response artifacts from frozen pre/post measurements."""

from __future__ import annotations

import argparse
from dataclasses import asdict

from aptadynamic_llm.artifact_schema import write_jsonl_atomic
from aptadynamic_llm.perturbation_response import (
    PerturbationConfig,
    evaluate_perturbation_response,
)
from _structural_artifact_cli import add_common_arguments, envelope_for, read_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--response-horizon-windows", type=int, required=True)
    parser.add_argument("--minimum-uptake-gain", type=float, required=True)
    parser.add_argument("--maximum-self-dependence-increase", type=float, required=True)
    parser.add_argument("--minimum-integrated-uptake", type=float, required=True)
    parser.add_argument("--counter-reactive-margin", type=float, required=True)
    args = parser.parse_args(argv)
    try:
        config = PerturbationConfig(
            args.response_horizon_windows,
            args.minimum_uptake_gain,
            args.maximum_self_dependence_increase,
            args.minimum_integrated_uptake,
            args.counter_reactive_margin,
        )
        outputs = []
        for row in read_rows(args.input):
            fields = dict(row)
            session_id = str(fields.pop("session_id"))
            outputs.append(
                evaluate_perturbation_response(
                    envelope=envelope_for(
                        args,
                        artifact_type="perturbation_response",
                        session_id=session_id,
                        config=asdict(config),
                    ),
                    config=config,
                    **fields,
                )
            )
        write_jsonl_atomic(args.out, outputs)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(f"perturbation evaluation failed: {exc}")
        return 1
    print(f"wrote {len(outputs)} perturbation artifacts to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
