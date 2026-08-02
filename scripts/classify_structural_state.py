#!/usr/bin/env python
"""Apply the deterministic structural-label precedence rules."""

from __future__ import annotations

import argparse

from aptadynamic_llm.artifact_schema import write_jsonl_atomic
from aptadynamic_llm.structural_labels import (
    StructuralLabelInput,
    classify_structural_state,
)
from _structural_artifact_cli import add_common_arguments, envelope_for, read_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--calibration-reference", required=True)
    args = parser.parse_args(argv)
    try:
        outputs = []
        for row in read_rows(args.input):
            session_id = str(row["session_id"])
            inputs = StructuralLabelInput(**row["inputs"])
            outputs.append(
                classify_structural_state(
                    envelope=envelope_for(
                        args,
                        artifact_type="structural_label",
                        session_id=session_id,
                        config={
                            "label_version": "structural-labels/1.0.0",
                            "calibration_reference": args.calibration_reference,
                        },
                    ),
                    inputs=inputs,
                    calibration_reference=args.calibration_reference,
                )
            )
        write_jsonl_atomic(args.out, outputs)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(f"structural classification failed: {exc}")
        return 1
    print(f"wrote {len(outputs)} structural labels to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
