"""Pure metrics for the LLM-SVM PRAMA state-discrimination study.

This module intentionally contains pure evaluation helpers only. It does not
modify the PRAMA Protokol kernel, regime classification, thresholds, or semantic
verification. Scores are evaluated causally: callers must pass only values
available at or before the evaluated token/window.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

MARKOVIAN_BASELINE_TAU = 64.0

PRIMARY_PRAMA_SCORE = "latent_occupancy"
REQUIRED_LABEL_FIELDS = {"session_id", "label", "event_token", "event_turn", "event_type"}

PRAMA_SCORE_FIELDS = (
    "latent_occupancy",
    "delta",
    "xi",
    "neg_M",
)

BASELINE_SCORE_FIELDS = (
    "mean_surprisal",
    "mean_entropy",
    "negative_mean_top1_gap",
    "markovian_surprisal_tau64",
)

@dataclass(frozen=True)
class Confusion:
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def fpr(self) -> float:
        denom = self.fp + self.tn
        return self.fp / denom if denom else 0.0

    @property
    def tpr(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _as_binary_labels(labels: Sequence[Any]) -> list[int]:
    out: list[int] = []
    for label in labels:
        if isinstance(label, str):
            normalized = label.strip().lower()
            out.append(1 if normalized in {"1", "true", "fail", "failed", "pathological", "positive"} else 0)
        else:
            out.append(1 if int(label) else 0)
    return out


def _validate_scores_labels(scores: Sequence[float], labels: Sequence[Any]) -> tuple[list[float], list[int]]:
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    if not scores:
        raise ValueError("scores and labels must not be empty")
    clean_scores = [float(score) for score in scores]
    clean_labels = _as_binary_labels(labels)
    if not any(clean_labels):
        raise ValueError("at least one positive label is required; all labels were negative")
    if not any(label == 0 for label in clean_labels):
        raise ValueError("at least one negative label is required; all labels were positive")
    return clean_scores, clean_labels


def auroc(scores: Sequence[float], labels: Sequence[Any]) -> float:
    """Compute AUROC using average ranks, with ties handled exactly."""

    clean_scores, clean_labels = _validate_scores_labels(scores, labels)
    ranked = sorted(enumerate(clean_scores), key=lambda item: item[1])
    ranks = [0.0] * len(clean_scores)
    i = 0
    while i < len(ranked):
        j = i + 1
        while j < len(ranked) and ranked[j][1] == ranked[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[ranked[k][0]] = avg_rank
        i = j

    n_pos = sum(clean_labels)
    n_neg = len(clean_labels) - n_pos
    pos_rank_sum = sum(rank for rank, label in zip(ranks, clean_labels) if label == 1)
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def permutation_pvalue(
    scores: Sequence[float],
    labels: Sequence[Any],
    permutations: int = 1000,
    seed: int = 0,
) -> float:
    """One-sided AUROC label-permutation p-value with plus-one correction."""

    if permutations <= 0:
        raise ValueError("permutations must be positive")
    clean_scores, clean_labels = _validate_scores_labels(scores, labels)
    observed = auroc(clean_scores, clean_labels)
    rng = random.Random(seed)
    exceedances = 0
    shuffled = list(clean_labels)
    for _ in range(permutations):
        rng.shuffle(shuffled)
        exceedances += auroc(clean_scores, shuffled) >= observed
    return (exceedances + 1) / (permutations + 1)


def confusion_at_threshold(scores: Sequence[float], labels: Sequence[Any], threshold: float) -> dict[str, int]:
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    clean_labels = _as_binary_labels(labels)
    tp = fp = tn = fn = 0
    for score, label in zip(scores, clean_labels):
        predicted = float(score) >= threshold
        if predicted and label == 1:
            tp += 1
        elif predicted and label == 0:
            fp += 1
        elif not predicted and label == 0:
            tn += 1
        else:
            fn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def precision_recall_at_threshold(scores: Sequence[float], labels: Sequence[Any], threshold: float) -> dict[str, float]:
    counts = confusion_at_threshold(scores, labels, threshold)
    precision_den = counts["tp"] + counts["fp"]
    recall_den = counts["tp"] + counts["fn"]
    return {
        "precision": counts["tp"] / precision_den if precision_den else 0.0,
        "recall": counts["tp"] / recall_den if recall_den else 0.0,
    }


def roc_points(scores: Sequence[float], labels: Sequence[Any]) -> list[dict[str, float]]:
    clean_scores, clean_labels = _validate_scores_labels(scores, labels)
    thresholds = sorted(set(clean_scores), reverse=True)
    points = [{"threshold": float("inf"), "fpr": 0.0, "tpr": 0.0}]
    for threshold in thresholds:
        counts = confusion_at_threshold(clean_scores, clean_labels, threshold)
        conf = Confusion(**counts)
        points.append({"threshold": threshold, "fpr": conf.fpr, "tpr": conf.tpr})
    points.append({"threshold": float("-inf"), "fpr": 1.0, "tpr": 1.0})
    return points


def threshold_at_fpr(scores: Sequence[float], labels: Sequence[Any], target_fpr: float) -> float:
    """Return the most permissive threshold whose empirical FPR is <= target_fpr."""

    if not 0.0 <= target_fpr <= 1.0:
        raise ValueError("target_fpr must be between 0 and 1")
    points = roc_points(scores, labels)
    valid = [point for point in points if point["fpr"] <= target_fpr]
    if not valid:
        return float("inf")
    return min(valid, key=lambda point: point["threshold"])["threshold"]


def first_alarm_index(scores: Sequence[float], threshold: float) -> int | None:
    for index, score in enumerate(scores):
        if float(score) >= threshold:
            return index
    return None


def lead_time_tokens(alarm_token: int | None, event_token: int | None) -> int | None:
    if alarm_token is None or event_token is None:
        return None
    lead = int(event_token) - int(alarm_token)
    return lead if lead >= 0 else None


def lead_time_turns(alarm_turn: int | None, event_turn: int | None) -> int | None:
    if alarm_turn is None or event_turn is None:
        return None
    lead = int(event_turn) - int(alarm_turn)
    return lead if lead >= 0 else None


def _median(values: Iterable[float]) -> float | None:
    clean = sorted(float(value) for value in values)
    if not clean:
        return None
    mid = len(clean) // 2
    if len(clean) % 2:
        return clean[mid]
    return (clean[mid - 1] + clean[mid]) / 2.0


# Baseline helpers. They consume only provided prefixes/windows; callers are
# responsible for passing causal data.
def mean_entropy(tokens: Sequence[dict[str, Any]]) -> float:
    values = [safe_float(token.get("entropy")) for token in tokens]
    return sum(values) / len(values) if values else 0.0


def mean_logprob(tokens: Sequence[dict[str, Any]]) -> float:
    values = [safe_float(token.get("top1_logprob")) for token in tokens]
    return sum(values) / len(values) if values else 0.0


def mean_surprisal(tokens: Sequence[dict[str, Any]]) -> float:
    return -mean_logprob(tokens)


def top1_gap(tokens: Sequence[dict[str, Any]]) -> float:
    values = [safe_float(token.get("gap")) for token in tokens]
    return sum(values) / len(values) if values else 0.0


def negative_mean_top1_gap(tokens: Sequence[dict[str, Any]]) -> float:
    """Sign-inverted gap so every study score has the direction greater = worse."""

    return -top1_gap(tokens)


def markovian_surprisal_intensity(
    tokens: Sequence[dict[str, Any]], tau: float = MARKOVIAN_BASELINE_TAU
) -> float:
    """Raw-surprisal EMA with the kernel's declared memory scale.

    I(0)=0 and I(t)=a*I(t-1)+(1-a)*omega(t), a=exp(-1/tau), for t>=1.
    This is the preregistered Markovian rival to the Protokol's structural
    accumulator at the same tau.
    """

    if tau <= 0:
        raise ValueError("tau must be positive")
    surprises = [-safe_float(token.get("top1_logprob")) for token in tokens]
    intensity = 0.0
    a = math.exp(-1.0 / tau)
    for value in surprises[1:]:
        intensity = a * intensity + (1.0 - a) * value
    return intensity


# Certified-kernel score helpers.  All fields come from
# ``prama_protokol.project`` summaries; no legacy aptadynamik score is valid.
def latent_occupancy(row: dict[str, Any]) -> float:
    return safe_float(row.get("latent_occupancy"))


def delta(row: dict[str, Any]) -> float:
    return safe_float(row.get("delta"))


def xi(row: dict[str, Any]) -> float:
    return safe_float(row.get("xi"))


def neg_M(row: dict[str, Any]) -> float:
    return -safe_float(row.get("M"))


def baseline_score_from_tokens(field: str, tokens: Sequence[dict[str, Any]]) -> float:
    helpers = {
        "mean_surprisal": mean_surprisal,
        "mean_entropy": mean_entropy,
        "negative_mean_top1_gap": negative_mean_top1_gap,
        "markovian_surprisal_tau64": markovian_surprisal_intensity,
    }
    if field not in helpers:
        raise ValueError(f"unknown baseline score field: {field}")
    return helpers[field](tokens)


def prama_score_from_row(field: str, row: dict[str, Any]) -> float:
    helpers = {
        "latent_occupancy": latent_occupancy,
        "delta": delta,
        "xi": xi,
        "neg_M": neg_M,
    }
    if field not in helpers:
        raise ValueError(f"unknown PRAMA score field: {field}")
    return helpers[field](row)


def matched_fpr_comparison(
    prama_scores: Sequence[float],
    baseline_scores: Sequence[float],
    labels: Sequence[Any],
    token_positions: Sequence[int],
    event_tokens: Sequence[int],
    target_fpr: float = 0.10,
) -> dict[str, Any]:
    """Compare PRAMA and a baseline at matched FPR.

    Inputs are per-example causal scores and token positions. Lead time is
    computed only for true-positive examples with an alarm before the event.
    """

    if not (len(prama_scores) == len(baseline_scores) == len(labels) == len(token_positions) == len(event_tokens)):
        raise ValueError("all inputs must have the same length")
    clean_labels = _as_binary_labels(labels)
    prama_threshold = threshold_at_fpr(prama_scores, clean_labels, target_fpr)
    baseline_threshold = threshold_at_fpr(baseline_scores, clean_labels, target_fpr)

    prama_leads: list[int] = []
    baseline_leads: list[int] = []
    for prama_score, baseline_score, label, token_position, event_token in zip(
        prama_scores, baseline_scores, clean_labels, token_positions, event_tokens
    ):
        if label != 1:
            continue
        if float(prama_score) >= prama_threshold:
            lead = lead_time_tokens(int(token_position), int(event_token))
            if lead is not None:
                prama_leads.append(lead)
        if float(baseline_score) >= baseline_threshold:
            lead = lead_time_tokens(int(token_position), int(event_token))
            if lead is not None:
                baseline_leads.append(lead)

    prama_lead = _median(prama_leads)
    baseline_lead = _median(baseline_leads)
    prama_auc = auroc(prama_scores, clean_labels)
    baseline_auc = auroc(baseline_scores, clean_labels)
    return {
        "target_fpr": target_fpr,
        "prama_threshold": prama_threshold,
        "baseline_threshold": baseline_threshold,
        "prama_auroc": prama_auc,
        "baseline_auroc": baseline_auc,
        "auroc_delta": prama_auc - baseline_auc,
        "prama_confusion": confusion_at_threshold(prama_scores, clean_labels, prama_threshold),
        "baseline_confusion": confusion_at_threshold(baseline_scores, clean_labels, baseline_threshold),
        "prama_median_lead_tokens": prama_lead,
        "baseline_median_lead_tokens": baseline_lead,
        "lead_token_delta": None if prama_lead is None or baseline_lead is None else prama_lead - baseline_lead,
    }
