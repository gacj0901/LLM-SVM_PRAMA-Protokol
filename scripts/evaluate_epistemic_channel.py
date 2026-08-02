#!/usr/bin/env python
"""Build paired epistemic-channel vectors with backend-only condition IDs."""

from __future__ import annotations

import argparse
from dataclasses import asdict

from aptadynamic_llm.artifact_schema import write_jsonl_atomic
from aptadynamic_llm.epistemic_channel import (
    EpistemicChannelConfig,
    evaluate_epistemic_pair,
)
from _structural_artifact_cli import add_common_arguments, envelope_for, read_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--competence-tolerance", type=float, required=True)
    parser.add_argument("--minimum-coordinate-effect", type=float, required=True)
    args = parser.parse_args(argv)
    try:
        config = EpistemicChannelConfig(
            args.competence_tolerance, args.minimum_coordinate_effect
        )
        outputs = []
        for row in read_rows(args.input):
            fields = dict(row)
            session_id = str(fields.pop("session_id"))
            outputs.append(
                evaluate_epistemic_pair(
                    envelope=envelope_for(
                        args,
                        artifact_type="epistemic_channel",
                        session_id=session_id,
                        config=asdict(config),
                    ),
                    config=config,
                    **fields,
                )
            )
        write_jsonl_atomic(args.out, outputs)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(f"epistemic-channel evaluation failed: {exc}")
        return 1
    print(f"wrote {len(outputs)} epistemic-channel artifacts to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
