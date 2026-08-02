"""Exact teacher-forced scoring through the llama.cpp HTTP server.

Ollama's generation API returns logprobs only for generated tokens.  The
bundled llama.cpp server can instead receive exact prompt token IDs.  This
adapter forces each recorded target token as the next generated token while
requesting *pre-sampling* probabilities.  llama.cpp then returns the original
model logprob for that token, including when it lies outside the requested
top-N alternatives.

The preflight verifies that:

* the server is healthy and loaded from the declared weights hash;
* a natural next-token logprob is unchanged when that token is forced;
* the response identifies the exact forced target token.
"""

from __future__ import annotations

import json
import math
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from aptadynamic_llm.factorial_scoring import RenderedCondition, ScorerIdentity


class LlamaServerError(RuntimeError):
    """The local scorer server violated the teacher-forcing contract."""


class LlamaServerTeacherForcedScorer:
    """Full-vocabulary scorer backed by a loaded llama.cpp server."""

    def __init__(
        self,
        *,
        base_url: str,
        identity: ScorerIdentity,
        force_bias: float = 100.0,
        timeout_seconds: float = 120.0,
        raw_logprob_tolerance: float = 1e-4,
    ) -> None:
        if not base_url.startswith(("http://127.0.0.1", "http://localhost")):
            raise ValueError("the factorial scorer must use an explicitly local server")
        if force_bias <= 0 or not math.isfinite(force_bias):
            raise ValueError("force_bias must be positive and finite")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if raw_logprob_tolerance < 0:
            raise ValueError("raw_logprob_tolerance must be non-negative")
        self.base_url = base_url.rstrip("/")
        self._identity = identity
        self.force_bias = float(force_bias)
        self.timeout_seconds = float(timeout_seconds)
        self.raw_logprob_tolerance = float(raw_logprob_tolerance)

    @property
    def identity(self) -> ScorerIdentity:
        return self._identity

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                value = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LlamaServerError(f"{path}: scorer request failed: {exc}") from exc
        if not isinstance(value, dict):
            raise LlamaServerError(f"{path}: scorer returned a non-object response")
        return value

    @staticmethod
    def _completion_payload(
        prefix_token_ids: Sequence[int],
        *,
        forced_token_id: int | None,
        force_bias: float,
    ) -> dict[str, Any]:
        if not prefix_token_ids:
            raise ValueError("a scoring prefix must contain at least one token")
        payload: dict[str, Any] = {
            "prompt": [int(token_id) for token_id in prefix_token_ids],
            "n_predict": 1,
            "n_probs": 1,
            "temperature": 0.0,
            "top_k": 0,
            "top_p": 1.0,
            "min_p": 0.0,
            "repeat_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "stream": False,
            "cache_prompt": True,
            "post_sampling_probs": False,
        }
        if forced_token_id is not None:
            payload["logit_bias"] = [[int(forced_token_id), force_bias]]
        return payload

    @staticmethod
    def _probability_item(response: dict[str, Any]) -> dict[str, Any]:
        probabilities = response.get("completion_probabilities")
        if isinstance(probabilities, dict):
            item = probabilities
        elif isinstance(probabilities, list) and probabilities:
            item = probabilities[0]
        else:
            raise LlamaServerError("completion response omitted token probabilities")
        if not isinstance(item, dict):
            raise LlamaServerError("completion probability item is not an object")
        return item

    def _score_next(
        self,
        prefix_token_ids: Sequence[int],
        forced_token_id: int | None,
    ) -> tuple[int, float]:
        response = self._request_json(
            "POST",
            "/completion",
            self._completion_payload(
                prefix_token_ids,
                forced_token_id=forced_token_id,
                force_bias=self.force_bias,
            ),
        )
        item = self._probability_item(response)
        try:
            returned_id = int(item["id"])
            logprob = float(item["logprob"])
        except (KeyError, TypeError, ValueError) as exc:
            raise LlamaServerError(
                "completion probability omitted a valid token id or logprob"
            ) from exc
        if not math.isfinite(logprob):
            raise LlamaServerError("completion returned a non-finite logprob")
        if forced_token_id is not None and returned_id != forced_token_id:
            raise LlamaServerError(
                f"forced token {forced_token_id} was not selected; got {returned_id}. "
                "Increase force_bias or reject the scoring window."
            )
        return returned_id, logprob

    def score(self, rendered: RenderedCondition) -> tuple[float, ...]:
        """Score each recorded target token with an exact growing prefix."""

        rendered.validate_basic()
        values = []
        for offset, target_token_id in enumerate(rendered.target_token_ids):
            prefix_stop = rendered.target_start + offset
            prefix = rendered.token_ids[:prefix_stop]
            _, logprob = self._score_next(prefix, target_token_id)
            values.append(logprob)
        return tuple(values)

    def preflight(self, probe_prefix_token_ids: Sequence[int]) -> dict[str, Any]:
        """Verify server identity and raw-logprob preservation."""

        health = self._request_json("GET", "/health")
        if health.get("status") != "ok":
            raise LlamaServerError(f"scorer health is not ok: {health!r}")
        props = self._request_json("GET", "/props")
        model_path = str(props.get("model_path") or "")
        expected_hash = self.identity.weights_sha256.lower()
        if expected_hash not in model_path.lower():
            raise LlamaServerError(
                "loaded scorer model does not match the declared weights hash"
            )

        natural_token_id, natural_logprob = self._score_next(
            probe_prefix_token_ids, forced_token_id=None
        )
        forced_token_id, forced_logprob = self._score_next(
            probe_prefix_token_ids, forced_token_id=natural_token_id
        )
        delta = abs(natural_logprob - forced_logprob)
        if forced_token_id != natural_token_id or delta > self.raw_logprob_tolerance:
            raise LlamaServerError(
                "forcing changed the reported raw logprob; the scorer contract is invalid"
            )
        return {
            "status": "ok",
            "model_path": model_path,
            "build_info": str(props.get("build_info") or ""),
            "weights_sha256": expected_hash,
            "natural_probe_token_id": natural_token_id,
            "natural_logprob": natural_logprob,
            "forced_logprob": forced_logprob,
            "absolute_logprob_delta": delta,
            "raw_logprobs_preserved": True,
        }
