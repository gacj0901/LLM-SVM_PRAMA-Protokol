from __future__ import annotations

import pytest

from FrontEnd import server


def test_nvidia_models_are_explicitly_allowlisted() -> None:
    assert set(server.MODEL_PROFILES) == {
        "nvidia/nemotron-3-super-120b-a12b",
        "mistralai/mistral-medium-3.5-128b",
        "nvidia/nemotron-3-ultra-550b-a55b",
    }


def test_model_payload_contains_only_roles_and_content() -> None:
    turns = [
        {
            "user_message": "Primera pregunta",
            "assistant_message": "Primera respuesta",
            "tokens": [{"token": "ignored"}],
            "windows": [{"rigidity": 0.8}],
            "label": "must-not-cross",
        }
    ]
    messages = server.provider_messages(turns, "Segunda pregunta")
    assert messages == [
        {"role": "user", "content": "Primera pregunta"},
        {"role": "assistant", "content": "Primera respuesta"},
        {"role": "user", "content": "Segunda pregunta"},
    ]
    assert all(set(message) == {"role", "content"} for message in messages)


@pytest.mark.parametrize("prompt", ["Habla de PRAMA", "Describe esta interfaz", "Lee el regime_label"])
def test_internal_terms_are_rejected_before_provider_call(prompt: str) -> None:
    with pytest.raises(ValueError, match="términos internos"):
        server.validate_prompt(prompt)


def test_response_time_and_observation_metrics_are_session_side_only() -> None:
    tokens = [
        {
            "token": "x",
            "top1_logprob": -0.1,
            "top_logprobs": [-0.1, -1.1],
            "gap": 1.0,
            "entropy": 0.4,
        }
        for _ in range(20)
    ]
    windows = server.build_windows(tokens)
    session = server.Session(
        session_id="test",
        provider="nvidia_nim",
        model="nvidia/nemotron-3-super-120b-a12b",
        created_at=server.utc_now(),
        turns=[
            {
                "token_count": len(tokens),
                "response_time_seconds": 1.25,
                "windows": windows,
                "resolved_model": "nvidia/nemotron-3-super-120b-a12b",
            }
        ],
    )
    summary = server.session_summary(session)
    assert summary["response_time_seconds"] == 1.25
    assert summary["window_count"] == 2
    assert 0.0 <= summary["avg_entropy_norm"] <= 1.0
    assert 0.0 <= summary["avg_gap_norm"] <= 1.0
