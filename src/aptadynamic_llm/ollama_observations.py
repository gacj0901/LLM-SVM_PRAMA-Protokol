"""Provider-neutral conversion of Ollama logprobs into numeric observations."""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from typing import Any

from aptadynamic_llm.model_payload import encode_task_only_payload


def request_json(
    base_url: str,
    endpoint: str,
    payload: dict[str, Any] | None = None,
    timeout: int = 600,
) -> dict[str, Any]:
    """Send a task-only JSON request to an Ollama-compatible endpoint."""
    url = base_url.rstrip("/") + endpoint
    data = None if payload is None else encode_task_only_payload(payload)
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama request failed at {url}: {exc}") from exc


def _valid_logprob(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _candidate_logprobs(item: dict[str, Any], limit: int) -> list[float]:
    values = [
        value
        for candidate in item.get("top_logprobs") or []
        if (value := _valid_logprob(candidate.get("logprob"))) is not None
    ]
    chosen = _valid_logprob(item.get("logprob"))
    if chosen is not None and chosen not in values:
        values.append(chosen)
    return sorted(values, reverse=True)[:limit]


def _normalized_entropy(logprobs: list[float]) -> float:
    if len(logprobs) < 2:
        return 0.0
    maximum = max(logprobs)
    weights = [math.exp(value - maximum) for value in logprobs]
    total = sum(weights)
    probabilities = [weight / total for weight in weights]
    entropy = -sum(
        probability * math.log(probability + 1e-15)
        for probability in probabilities
    )
    return max(0.0, min(1.0, entropy / math.log(len(probabilities))))


def response_tokens(
    response: dict[str, Any], max_top_logprobs: int = 5
) -> list[dict[str, Any]]:
    """Convert an Ollama response's logprobs to the repository token schema."""
    if max_top_logprobs < 2:
        raise ValueError("max_top_logprobs must be at least 2")
    tokens: list[dict[str, Any]] = []
    for item in response.get("logprobs") or []:
        chosen = _valid_logprob(item.get("logprob"))
        if chosen is None:
            continue
        candidates = _candidate_logprobs(item, max_top_logprobs)
        tokens.append(
            {
                "token": str(item.get("token") or ""),
                "bytes": item.get("bytes") or [],
                "top1_logprob": chosen,
                "top_logprobs": candidates,
                "gap": (
                    float(candidates[0] - candidates[1])
                    if len(candidates) >= 2
                    else 0.0
                ),
                "entropy": _normalized_entropy(candidates),
            }
        )
    if not tokens:
        raise RuntimeError("Ollama returned no usable token logprobs")
    return tokens
