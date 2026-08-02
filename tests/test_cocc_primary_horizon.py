from scripts.evaluate_break_the_chain_prama import _primary_horizon_test_rows


def _trajectory(count):
    rows = []
    for index in range(count):
        delta = 0.1 if index < 8 else 0.9
        xi = 0.05 if index < 8 else 0.8
        rows.append(
            {
                "delta": delta,
                "xi": xi,
                "balance": 0.2 - xi,
                "accumulated_excess": 0.0,
                "capacity": 1.0,
            }
        )
    return rows


def _row(problem_id, condition, windows):
    return {
        "problem_id": problem_id,
        "perturbation_type": condition,
        "trajectory": _trajectory(windows),
        "tokens": [
            {"top1_logprob": -0.1, "entropy": 0.2, "gap": 1.0}
            for _ in range(windows * 16)
        ],
        "features": {},
    }


def test_primary_horizon_uses_prefix_and_excludes_incomplete_problem_cluster():
    design = {
        "temporal_censoring": {
            "primary_absolute_window_horizon": 8,
            "window_size_tokens": 16,
        },
        "independent_unit": {"expected_sessions_per_cluster": 2},
    }
    rows = [
        _row("p1", "clean_control", 10),
        _row("p1", "negation_objective", 10),
        _row("p2", "clean_control", 7),
        _row("p2", "negation_objective", 10),
    ]

    retained, audit = _primary_horizon_test_rows(rows, design)

    assert len(retained) == 2
    assert {row["problem_id"] for row in retained} == {"p1"}
    assert all(row["features"]["max_delta"] == 0.1 for row in retained)
    assert audit["at_risk_problem_cluster_count"] == 1
    assert audit["excluded_problem_ids"] == ["p2"]
