#!/usr/bin/env python
"""Build the cross-model summary from completed minimal-structural v4 reports."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.project_cocc_operator_geometry import file_sha256  # noqa: E402
from scripts.reproject_minimal_structural_mobility_v4 import atomic_json  # noqa: E402


DEFAULT_REPORTS = (
    Path("run_outputs/minimal_structural_nemotron_super_dynamic_v4/report.json"),
    Path("run_outputs/minimal_structural_mistral_medium_3_5_dynamic_v4/report.json"),
    Path("run_outputs/minimal_structural_nemotron_ultra_dynamic_v4/report.json"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", type=Path, default=list(DEFAULT_REPORTS))
    parser.add_argument("--out", type=Path, default=Path("run_outputs/minimal_structural_three_model_v4_summary.json"))
    args = parser.parse_args()
    reports = []
    observer_hashes = set()
    mobility_hashes = set()
    for path in args.reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        observer_hashes.add(report["dynamic_observer_contract_sha256"])
        mobility_hashes.add(report["mobility_contract_sha256"])
        reports.append({
            "model": report["model"],
            "report": str(path),
            "report_sha256": file_sha256(path),
            "minimal_full": report["summary_full"]["minimal_structural"],
            "minimal_fixed_h16": report["summary_fixed_horizon"]["minimal_structural"],
            "minimal_vs_abstract_full": report["minimal_vs_abstract_full"],
            "minimal_vs_abstract_h16": report["minimal_vs_abstract_fixed_horizon"],
        })
    if len(observer_hashes) != 1 or len(mobility_hashes) != 1:
        raise SystemExit("reports do not share one observer and one mobility contract")
    summary = {
        "schema": "LLM-SVM-minimal-structural-three-model-v4-summary/2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "observer_contract_sha256": observer_hashes.pop(),
        "mobility_contract_sha256": mobility_hashes.pop(),
        "short_trajectory_policy": "structural metrics are null when geometry_ready_windows == 0",
        "reports": reports,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out, summary)
    print(json.dumps({"output": str(args.out), "sha256": file_sha256(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
