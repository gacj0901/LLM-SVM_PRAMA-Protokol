#!/usr/bin/env python
"""Verify the local Hermes 3 generator-relative factorial scoring stack."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any
from urllib.request import Request, urlopen

from aptadynamic_llm.hermes3_factorial_config import (
    GGUF_ARCHITECTURE,
    GGUF_MODEL_SHA256,
    GGUF_QUANTIZATION,
    LLAMA_SERVER_BUILD,
    OLLAMA_MANIFEST_SHA256,
    OLLAMA_MIN_VERSION,
    OLLAMA_MODEL_NAME,
    OLLAMA_TEMPLATE_SHA256,
    generator_relative_scorer_identity,
)
from aptadynamic_llm.llama_server_scorer import (
    LlamaServerError,
    LlamaServerTeacherForcedScorer,
)


def _request_json(
    base_url: str,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
        headers["Content-Type"] = "application/json"
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers=headers,
        method=method,
    )
    with urlopen(request, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _version_tuple(value: str) -> tuple[int, int, int]:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", value)
    if not match:
        raise ValueError(f"unsupported Ollama version string: {value!r}")
    return tuple(int(part) for part in match.groups())


def _tokenize_probe(llama_url: str) -> list[int]:
    response = _request_json(
        llama_url,
        "POST",
        "/tokenize",
        {
            "content": "Factorial scorer preflight",
            "add_special": True,
            "parse_special": True,
        },
    )
    tokens = response.get("tokens")
    if not isinstance(tokens, list) or not tokens:
        raise ValueError("llama-server tokenizer returned no probe tokens")
    return [int(token_id) for token_id in tokens]


def run(args: argparse.Namespace) -> dict[str, Any]:
    version_response = _request_json(args.ollama_url, "GET", "/api/version")
    ollama_version = str(version_response.get("version") or "")
    if _version_tuple(ollama_version) < OLLAMA_MIN_VERSION:
        raise ValueError(
            f"Ollama {ollama_version} is below minimum "
            f"{'.'.join(map(str, OLLAMA_MIN_VERSION))}"
        )

    show = _request_json(
        args.ollama_url,
        "POST",
        "/api/show",
        {"model": OLLAMA_MODEL_NAME, "verbose": True},
    )
    details = show.get("details") or {}
    if details.get("family") != GGUF_ARCHITECTURE:
        raise ValueError(f"unexpected model family: {details.get('family')!r}")
    if details.get("quantization_level") != GGUF_QUANTIZATION:
        raise ValueError(
            f"unexpected quantization: {details.get('quantization_level')!r}"
        )
    template_hash = sha256(str(show.get("template") or "").encode("utf-8")).hexdigest()
    if template_hash != OLLAMA_TEMPLATE_SHA256:
        raise ValueError("Ollama chat template hash does not match the frozen template")

    manifest_hash = sha256(args.manifest.read_bytes()).hexdigest()
    if manifest_hash != OLLAMA_MANIFEST_SHA256:
        raise ValueError("local Ollama manifest hash does not match hermes3:8b")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    layer_digests = {
        str(layer.get("digest") or "") for layer in manifest.get("layers") or []
    }
    if f"sha256:{GGUF_MODEL_SHA256}" not in layer_digests:
        raise ValueError("Hermes manifest does not reference the frozen GGUF blob")

    scorer = LlamaServerTeacherForcedScorer(
        base_url=args.llama_url,
        identity=generator_relative_scorer_identity(),
        force_bias=args.force_bias,
        timeout_seconds=args.timeout,
    )
    scorer_report = scorer.preflight(_tokenize_probe(args.llama_url))
    report = {
        "status": "ok",
        "ollama": {
            "model": OLLAMA_MODEL_NAME,
            "version": ollama_version,
            "minimum_version": ".".join(map(str, OLLAMA_MIN_VERSION)),
            "manifest_sha256": manifest_hash,
            "gguf_sha256": GGUF_MODEL_SHA256,
            "quantization": details.get("quantization_level"),
            "architecture": details.get("family"),
            "parameter_size": details.get("parameter_size"),
            "template_sha256": template_hash,
        },
        "scorer": scorer_report,
        "expected_llama_server_build": LLAMA_SERVER_BUILD,
        "exact_build_match": scorer_report["build_info"] == LLAMA_SERVER_BUILD,
        "scoring_mode": "generator-relative",
    }
    if not report["exact_build_match"]:
        raise ValueError(
            "llama-server build differs from the frozen pilot build: "
            f"{scorer_report['build_info']!r}"
        )
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_manifest = (
        Path.home()
        / ".ollama"
        / "models"
        / "manifests"
        / "registry.ollama.ai"
        / "library"
        / "hermes3"
        / "8b"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--llama-url", default="http://127.0.0.1:11435")
    parser.add_argument("--manifest", type=Path, default=default_manifest)
    parser.add_argument("--force-bias", type=float, default=100.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run(args)
    except (LlamaServerError, OSError, TypeError, ValueError) as exc:
        print(f"Hermes factorial preflight failed: {exc}")
        return 1
    encoded = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8", newline="\n")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
