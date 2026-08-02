"""Frozen model identity for the Hermes 3 factorial pilot."""

from __future__ import annotations

from aptadynamic_llm.factorial_scoring import ScorerIdentity


OLLAMA_MODEL_NAME = "hermes3:8b"
OLLAMA_MIN_VERSION = (0, 30, 11)

# Full local Ollama manifest identity. ``ollama list`` displays its first
# twelve hexadecimal characters: 4f6b83f30b62.
OLLAMA_MANIFEST_SHA256 = (
    "4f6b83f30b62bc3d0cf9be09266db222805ee815c8fd7d8b38f863f655be78b7"
)

# Q4_0 GGUF layer loaded by both Ollama and the bundled llama.cpp scorer.
GGUF_MODEL_SHA256 = (
    "c8985d236593f7a17da2a3da49588aa951a9b1e57ce97753364fbf59e63af84a"
)
GGUF_QUANTIZATION = "Q4_0"
GGUF_ARCHITECTURE = "llama"
GGUF_PARAMETER_SIZE = "8.0B"
GGUF_CONTEXT_LENGTH = 131072

# Ollama Modelfile template layer. Rendering must use this template, even
# though llama-server exposes a different embedded Jinja template.
OLLAMA_TEMPLATE_SHA256 = (
    "c7ec478a7939bd2449e43dde848f31c31bc68b75990d4085ced5ac30f299b24a"
)

LLAMA_SERVER_BUILD = "b1-cb295bf59"


def generator_relative_scorer_identity() -> ScorerIdentity:
    """Identity for exact scoring with the same quantized generator weights."""

    return ScorerIdentity(
        scoring_mode="generator-relative",
        model_id=(
            f"ollama:{OLLAMA_MODEL_NAME}@sha256:{OLLAMA_MANIFEST_SHA256}"
        ),
        weights_sha256=GGUF_MODEL_SHA256,
        tokenizer_id=(
            f"gguf-embedded-tokenizer@sha256:{GGUF_MODEL_SHA256}"
        ),
        # The tokenizer is embedded in, and therefore transitively identified
        # by, the byte-exact GGUF blob.
        tokenizer_hash=GGUF_MODEL_SHA256,
        chat_template_id=f"ollama-template@sha256:{OLLAMA_TEMPLATE_SHA256}",
    )
