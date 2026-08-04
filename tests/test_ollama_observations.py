import math

from aptadynamic_llm.ollama_observations import response_tokens


def test_response_tokens_preserves_chosen_logprob_and_derives_observations():
    response = {
        "logprobs": [
            {
                "token": "hello",
                "logprob": -0.2,
                "bytes": [104],
                "top_logprobs": [
                    {"token": "hello", "logprob": -0.2},
                    {"token": "world", "logprob": -1.2},
                ],
            }
        ]
    }
    tokens = response_tokens(response)
    assert tokens[0]["top1_logprob"] == -0.2
    assert tokens[0]["gap"] == 1.0
    assert 0.0 < tokens[0]["entropy"] < 1.0
    assert all(math.isfinite(value) for value in tokens[0]["top_logprobs"])
