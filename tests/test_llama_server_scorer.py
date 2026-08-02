import pytest

from aptadynamic_llm.factorial_scoring import RenderedCondition, ScorerIdentity
from aptadynamic_llm.hermes3_factorial_config import (
    GGUF_MODEL_SHA256,
    OLLAMA_MANIFEST_SHA256,
    generator_relative_scorer_identity,
)
from aptadynamic_llm.llama_server_scorer import (
    LlamaServerError,
    LlamaServerTeacherForcedScorer,
)


def _identity():
    return ScorerIdentity(
        scoring_mode="generator-relative",
        model_id="model@manifest",
        weights_sha256="abc123",
        tokenizer_id="tokenizer@model",
        tokenizer_hash="abc123",
        chat_template_id="template@hash",
    )


def _rendered():
    return RenderedCondition(
        condition="L11",
        token_ids=[128000, 10, 11, 20, 21],
        target_start=3,
        target_token_ids=[20, 21],
        current_turn_prefix_start=1,
        role_sequence=["user", "assistant"],
        message_token_counts=[1, 2],
        chat_template_id="template@hash",
        template_control_hash="control-hash",
    )


class StubScorer(LlamaServerTeacherForcedScorer):
    def __init__(self):
        super().__init__(
            base_url="http://127.0.0.1:11435",
            identity=_identity(),
        )
        self.requests = []

    def _request_json(self, method, path, payload=None):
        self.requests.append((method, path, payload))
        if path == "/health":
            return {"status": "ok"}
        if path == "/props":
            return {
                "model_path": "C:/models/sha256-abc123",
                "build_info": "b1-test",
            }
        forced = (payload or {}).get("logit_bias")
        token_id = int(forced[0][0]) if forced else 42
        logprob = -1.50001 if forced and token_id == 42 else (
            -1.5 if not forced else -token_id / 10.0
        )
        return {
            "completion_probabilities": [
                {"id": token_id, "logprob": logprob, "top_logprobs": []}
            ]
        }


def test_teacher_forcing_uses_exact_growing_token_prefix():
    scorer = StubScorer()
    scores = scorer.score(_rendered())

    assert scores == pytest.approx((-2.0, -2.1))
    completion_requests = [
        payload for method, path, payload in scorer.requests if path == "/completion"
    ]
    assert completion_requests[0]["prompt"] == [128000, 10, 11]
    assert completion_requests[0]["logit_bias"] == [[20, 100.0]]
    assert completion_requests[0]["post_sampling_probs"] is False
    assert completion_requests[1]["prompt"] == [128000, 10, 11, 20]
    assert completion_requests[1]["logit_bias"] == [[21, 100.0]]


def test_preflight_verifies_weights_and_raw_logprob_preservation():
    scorer = StubScorer()
    report = scorer.preflight([128000, 10, 11])

    assert report["status"] == "ok"
    assert report["raw_logprobs_preserved"]
    assert report["absolute_logprob_delta"] == pytest.approx(1e-5)


def test_preflight_rejects_a_different_loaded_model():
    scorer = StubScorer()

    def wrong_props(method, path, payload=None):
        if path == "/health":
            return {"status": "ok"}
        if path == "/props":
            return {"model_path": "C:/models/sha256-different"}
        raise AssertionError("completion must not run after identity failure")

    scorer._request_json = wrong_props
    with pytest.raises(LlamaServerError, match="weights hash"):
        scorer.preflight([128000, 10])


def test_forced_token_mismatch_is_rejected():
    scorer = StubScorer()

    def mismatching_response(method, path, payload=None):
        return {
            "completion_probabilities": [
                {"id": 999, "logprob": -2.0, "top_logprobs": []}
            ]
        }

    scorer._request_json = mismatching_response
    with pytest.raises(LlamaServerError, match="was not selected"):
        scorer.score(_rendered())


def test_hermes_identity_uses_full_manifest_and_exact_gguf_hashes():
    identity = generator_relative_scorer_identity()

    assert OLLAMA_MANIFEST_SHA256.startswith("4f6b83f30b62")
    assert len(OLLAMA_MANIFEST_SHA256) == 64
    assert identity.weights_sha256 == GGUF_MODEL_SHA256
    assert identity.scoring_mode == "generator-relative"
    assert "hermes3:8b" in identity.model_id
