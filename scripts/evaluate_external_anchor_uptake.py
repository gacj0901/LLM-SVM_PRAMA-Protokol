#!/usr/bin/env python
"""Build external-anchor artifacts from verifier-scored response windows."""

from __future__ import annotations

import argparse
from dataclasses import asdict

from aptadynamic_llm.artifact_schema import write_jsonl_atomic
from aptadynamic_llm.external_anchor import AnchorUptakeConfig, evaluate_anchor_uptake
from _structural_artifact_cli import add_common_arguments, envelope_for, read_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--uptake-threshold", type=float, required=True)
    parser.add_argument("--response-horizon-windows", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        config = AnchorUptakeConfig(
            args.uptake_threshold, args.response_horizon_windows
        )
        outputs = []
        for row in read_rows(args.input):
            outputs.append(
                evaluate_anchor_uptake(
                    envelope=envelope_for(
                        args,
                        artifact_type="external_anchor_event",
                        session_id=str(row["session_id"]),
                        config=asdict(config),
                    ),
                    anchor_id=str(row["anchor_id"]),
                    anchor_type=str(row["anchor_type"]),
                    introduced_at_window=int(row["introduced_at_window"]),
                    anchor_state=str(row["anchor_state"]),
                    severity=float(row["severity"]),
                    externally_verifiable=bool(row["externally_verifiable"]),
                    anchor_source_sha256=str(row["anchor_source_sha256"]),
                    source_is_evaluated_trajectory=bool(
                        row.get("source_is_evaluated_trajectory", False)
                    ),
                    response_windows=row["response_windows"],
                    config=config,
                )
            )
        write_jsonl_atomic(args.out, outputs)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(f"external-anchor evaluation failed: {exc}")
        return 1
    print(f"wrote {len(outputs)} external-anchor artifacts to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
