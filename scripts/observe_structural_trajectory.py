#!/usr/bin/env python
"""Emit canonical D_O v9 observations from causal D_O v6 numeric windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aptadynamic_llm.artifact_schema import write_jsonl_atomic
from aptadynamic_llm.structural_coherence_v9 import StructuralCoherenceV9Config
from aptadynamic_llm.structural_observation import (
    make_structural_observation,
    observe_structural_trajectory,
)
from _structural_artifact_cli import add_common_arguments, envelope_for, read_rows


FORBIDDEN_INPUT_FIELDS = {
    "finish_reason",
    "response_time_seconds",
    "outcome",
    "label",
    "passed",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("config/sequor_structural_observer_v9.json"),
    )
    args = parser.parse_args(argv)
    try:
        contract = json.loads(args.contract.read_text(encoding="utf-8"))
        if contract.get("status") != "PRIMARY_STRUCTURAL_OBSERVER":
            raise ValueError(
                "canonical observation requires the prospective primary-observer declaration"
            )
        config = StructuralCoherenceV9Config.from_contract(contract)
        outputs = []
        seen_sessions: set[str] = set()
        for row in read_rows(args.input):
            leaked = sorted(FORBIDDEN_INPUT_FIELDS & row.keys())
            if leaked:
                raise ValueError(
                    f"structural observer input contains service/outcome fields: {leaked}"
                )
            session_id = str(row["session_id"])
            if not session_id or session_id in seen_sessions:
                raise ValueError("session_id must be nonempty and unique")
            seen_sessions.add(session_id)
            windows = row.get("structural_windows", row.get("windows"))
            if not isinstance(windows, list) or not windows:
                raise ValueError(
                    f"{session_id}: expected a nonempty windows/structural_windows array"
                )
            if any(FORBIDDEN_INPUT_FIELDS & window.keys() for window in windows):
                raise ValueError(
                    f"{session_id}: a numeric window contains service/outcome metadata"
                )
            observations = observe_structural_trajectory(windows, config)
            for observation in observations:
                outputs.append(
                    make_structural_observation(
                        envelope_for(
                            args,
                            artifact_type="structural_observation",
                            session_id=session_id,
                            config=contract,
                        ),
                        observation,
                    )
                )
        write_jsonl_atomic(args.out, outputs)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"D_O v9 structural observation failed: {exc}")
        return 1
    print(f"wrote {len(outputs)} D_O v9 structural observations to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
