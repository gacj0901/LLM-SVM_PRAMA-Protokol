#!/usr/bin/env python3
"""Run a resumable blind interaction battery across approved NVIDIA models."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

BASE_URL = "https://integrate.api.nvidia.com/v1"
KEY_ENV = "NVIDIA_API_KEY"
TOP_LOGPROBS = 5

MODEL_PROFILES: dict[str, dict[str, Any]] = {
    "nvidia/nemotron-3-super-120b-a12b": {
        "temperature": 1.0, "top_p": 0.95, "max_tokens": 4096,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False, "force_nonempty_content": True}},
    },
    "mistralai/mistral-medium-3.5-128b": {
        "temperature": 0.7, "top_p": 1.0, "max_tokens": 16384,
        "extra_body": {"reasoning_effort": "high"},
    },
    "nvidia/nemotron-3-ultra-550b-a55b": {
        "temperature": 1.0, "top_p": 0.95, "max_tokens": 4096,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False, "force_nonempty_content": True}},
    },
}

TESTS = {
    "constraint": (
        "Analiza el siguiente problema sin omitir ninguna restricción. Una caja contiene cinco fichas "
        "numeradas del 1 al 5. Debes ordenar las cinco fichas de modo que: (1) la 2 quede antes que la 5; "
        "(2) la 1 no quede junto a la 4; (3) exactamente una ficha quede entre la 3 y la 5; y (4) la 4 "
        "quede después de la 2. Propón un orden válido, comprueba cada restricción por separado y, si "
        "detectas un error en tu propuesta, corrígelo antes de concluir."
    ),
    "revision": (
        "Un sistema tiene tres causas posibles A, B y C. Se observa que al retirar A el efecto persiste, "
        "al retirar B desaparece, y al restaurar B reaparece sólo cuando C está presente. Explica la "
        "estructura causal mínima compatible con esas observaciones. Después formula una alternativa "
        "plausible, identifica qué observación la contradice y revisa tu conclusión."
    ),
    "compression": (
        "Resume el siguiente conjunto de reglas en no más de cuatro frases sin perder ninguna condición: "
        "todo miembro rojo es estable; ningún miembro estable y liviano puede ser opaco; los miembros "
        "azules son livianos salvo cuando son triangulares; todo triángulo azul es opaco; y ningún objeto "
        "opaco puede ser rojo. Después indica si puede existir un miembro simultáneamente rojo, azul y "
        "triangular, justificando la respuesta sólo con las reglas dadas."
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def acquire(model: str, test_id: str, prompt: str, timeout: float) -> dict[str, Any]:
    from openai import OpenAI

    profile = MODEL_PROFILES[model]
    started = perf_counter()
    stream = OpenAI(
        api_key=os.environ[KEY_ENV].strip(), base_url=BASE_URL, timeout=timeout, max_retries=1,
    ).chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=float(profile["temperature"]), top_p=float(profile["top_p"]),
        max_tokens=int(profile["max_tokens"]), seed=1337,
        logprobs=True, top_logprobs=TOP_LOGPROBS,
        extra_body=dict(profile["extra_body"]),
        stream=True,
    )
    response_parts: list[str] = []
    tokens: list[dict[str, Any]] = []
    finish_reason = ""
    resolved_model = model
    for chunk in stream:
        resolved_model = str(getattr(chunk, "model", "") or resolved_model)
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        piece = getattr(choice.delta, "content", None)
        if piece:
            response_parts.append(str(piece))
        for entry in list(getattr(getattr(choice, "logprobs", None), "content", None) or []):
            tokens.append(
                {
                    "token": str(entry.token),
                    "top1_logprob": float(entry.logprob),
                    "top_logprobs": [float(item.logprob) for item in list(entry.top_logprobs or [])],
                }
            )
        if choice.finish_reason:
            finish_reason = str(choice.finish_reason)
    elapsed = perf_counter() - started
    return {
        "schema": "LLM-SVM-frontend-test-battery-item/1",
        "generated_at": utc_now(),
        "provider": "nvidia_nim", "provider_endpoint": BASE_URL,
        "model": model, "resolved_model": resolved_model,
        "test_id": test_id,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "model_payload": {"messages": [{"role": "user", "content": prompt}]},
        "assistant_message": "".join(response_parts),
        "finish_reason": finish_reason,
        "response_time_seconds": elapsed,
        "token_count": len(tokens), "tokens": tokens,
        "generation_parameter_set": {
            "temperature": profile["temperature"], "top_p": profile["top_p"],
            "max_tokens": profile["max_tokens"], "seed": 1337,
            "top_logprobs": TOP_LOGPROBS, "extra_body": profile["extra_body"],
        },
        "response_clock": "immediately_before_provider_request_to_complete_response",
        "model_payload_boundary": "single_role_and_content_prompt_only",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("FrontEnd/results/blind_interaction_battery_v1"))
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()
    if not os.environ.get(KEY_ENV, "").strip():
        parser.error("NVIDIA_API_KEY is not set in this PowerShell session")

    jobs = [(model, test_id, prompt) for model in MODEL_PROFILES for test_id, prompt in TESTS.items()]
    items: list[dict[str, Any]] = []
    for index, (model, test_id, prompt) in enumerate(jobs, 1):
        path = args.output_dir / "items" / f"{model.replace('/', '--')}--{test_id}.json"
        if path.is_file():
            item = json.loads(path.read_text(encoding="utf-8"))
            print(f"[{index}/{len(jobs)}] reusing {model} / {test_id}", flush=True)
        else:
            print(f"[{index}/{len(jobs)}] acquiring {model} / {test_id}", flush=True)
            item = acquire(model, test_id, prompt, args.timeout)
            write_json(path, item)
            print(
                f"[{index}/{len(jobs)}] completed {model} / {test_id}: "
                f"finish={item['finish_reason']}, tokens={item['token_count']}, "
                f"seconds={item['response_time_seconds']:.2f}", flush=True,
            )
        items.append(item)

    durations = [float(item["response_time_seconds"]) for item in items]
    report = {
        "schema": "LLM-SVM-frontend-test-battery/1", "generated_at": utc_now(),
        "provider": "nvidia_nim", "models": list(MODEL_PROFILES), "tests": list(TESTS),
        "job_count": len(items), "completed_count": len(items),
        "response_time_seconds": {"total": sum(durations), "mean": sum(durations) / len(durations)},
        "items": [
            {key: item[key] for key in (
                "model", "resolved_model", "test_id", "prompt_sha256", "finish_reason",
                "response_time_seconds", "token_count",
            )}
            for item in items
        ],
    }
    report_path = args.output_dir / "report.json"
    write_json(report_path, report)
    print(json.dumps({"output": str(report_path)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
