import pytest

from scripts.audit_cocc_length_confound import auroc, residualized_auc, stratified_auc


def _rows():
    return [
        {"session_id": "a", "label": 0, "n_windows": 2, "finish_reason": "stop", "scores": {"s": 0.1}},
        {"session_id": "b", "label": 1, "n_windows": 2, "finish_reason": "stop", "scores": {"s": 0.4}},
        {"session_id": "c", "label": 0, "n_windows": 4, "finish_reason": "length", "scores": {"s": 0.8}},
        {"session_id": "d", "label": 1, "n_windows": 4, "finish_reason": "length", "scores": {"s": 0.9}},
    ]


def test_auroc_and_exact_length_stratification():
    rows = _rows()
    assert auroc([0.1, 0.4, 0.8, 0.9], [0, 1, 0, 1]) == pytest.approx(0.75)
    result = stratified_auc(rows, "s", "n_windows")
    assert result["auroc"] == 1.0
    assert result["informative_pair_count"] == 2
    assert result["informative_session_count"] == 4


def test_residualization_is_outcome_blind_and_finite():
    result = residualized_auc(_rows(), "s")
    assert result["method"].startswith("outcome_blind_ols")
    assert result["design_rank"] >= 2
    assert 0.0 <= result["residual_auroc"] <= 1.0
