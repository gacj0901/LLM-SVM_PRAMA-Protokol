#!/usr/bin/env python3
"""Serve the PRAMA monitor and proxy approved NVIDIA models safely.

The browser never receives the NVIDIA API key.  The model boundary contains
only the selected model id, generation parameters, and ordinary user/assistant
messages.  Monitor state, interface names, experimental conditions, labels,
and PRAMA coordinates are never included in the provider request.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
import mimetypes
import os
from pathlib import Path
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import RLock
from time import perf_counter
from typing import Any, Iterable
from urllib.parse import parse_qs, unquote, urlparse
from uuid import uuid4


FRONTEND_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = FRONTEND_ROOT / "results"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_API_KEY_ENV = "NVIDIA_API_KEY"
WINDOW_SIZE = 16
TOP_LOGPROBS = 5

MODEL_PROFILES: dict[str, dict[str, Any]] = {
    "nvidia/nemotron-3-super-120b-a12b": {
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 4096,
        "extra_body": {
            "chat_template_kwargs": {
                "enable_thinking": False,
                "force_nonempty_content": True,
            }
        },
    },
    "mistralai/mistral-medium-3.5-128b": {
        "temperature": 0.7,
        "top_p": 1.0,
        "max_tokens": 16384,
        "extra_body": {"reasoning_effort": "high"},
    },
    "nvidia/nemotron-3-ultra-550b-a55b": {
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 4096,
        "extra_body": {
            "chat_template_kwargs": {
                "enable_thinking": False,
                "force_nonempty_content": True,
            }
        },
    },
}

# These words are allowed in ordinary assistant answers, but not in a new user
# prompt sent through this deliberately blind observation interface.
FORBIDDEN_PROMPT_MARKERS = (
    "prama",
    "interfaz",
    "interface",
    "monitor",
    "regime_label",
    "trajectory_assessment",
    "condition_id",
    "pass/fail",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_session_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def entropy(logprobs: Iterable[float]) -> float:
    values = [float(value) for value in logprobs if math.isfinite(float(value))]
    if not values:
        return 0.0
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    total = sum(weights)
    return -sum((weight / total) * math.log(weight / total) for weight in weights if weight)


def token_row(entry: Any) -> dict[str, Any]:
    chosen = float(entry.logprob)
    candidates = [float(value.logprob) for value in list(entry.top_logprobs or [])]
    if chosen not in candidates:
        candidates.append(chosen)
    candidates = sorted((value for value in candidates if math.isfinite(value)), reverse=True)
    return {
        "token": str(entry.token),
        "top1_logprob": chosen,
        "top_logprobs": candidates,
        "gap": candidates[0] - candidates[1] if len(candidates) > 1 else 0.0,
        "entropy": entropy(candidates),
    }


def summarize_window(tokens: list[dict[str, Any]], index: int) -> dict[str, Any]:
    entropy_norms: list[float] = []
    gaps: list[float] = []
    for token in tokens:
        candidates = token.get("top_logprobs") or []
        scale = math.log(max(2, len(candidates)))
        entropy_norms.append(min(1.0, max(0.0, float(token["entropy"]) / scale)))
        gaps.append(max(0.0, float(token["gap"])))
    entropy_norm = sum(entropy_norms) / len(entropy_norms)
    gap_norm = sum(1.0 - math.exp(-value) for value in gaps) / len(gaps)
    rigidity = (1.0 - entropy_norm) * gap_norm
    uncertainty = entropy_norm * (1.0 - gap_norm)
    return {
        "window_index": index,
        "entropy_raw": sum(float(token["entropy"]) for token in tokens) / len(tokens),
        "entropy_norm": entropy_norm,
        "gap_norm": gap_norm,
        "rigidity": rigidity,
        "uncertainty": uncertainty,
        "entropy_range": max(entropy_norms) - min(entropy_norms),
        "n_tokens_in_window": len(tokens),
        "channel_status": "OBSERVED",
    }


def build_windows(tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        summarize_window(tokens[start : start + WINDOW_SIZE], start // WINDOW_SIZE)
        for start in range(0, len(tokens), WINDOW_SIZE)
        if tokens[start : start + WINDOW_SIZE]
    ]


def validate_prompt(prompt: Any) -> str:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("El mensaje no puede estar vacío.")
    text = prompt.strip()
    if len(text) > 20_000:
        raise ValueError("El mensaje supera el límite de 20 000 caracteres.")
    normalized = text.casefold()
    leaked = [marker for marker in FORBIDDEN_PROMPT_MARKERS if marker in normalized]
    if leaked:
        raise ValueError(
            "El mensaje contiene términos internos que esta conexión ciega no envía al modelo: "
            + ", ".join(leaked)
        )
    return text


def provider_messages(turns: list[dict[str, Any]], prompt: str) -> list[dict[str, str]]:
    """Create a strict role/content-only view of recent conversation history."""

    messages: list[dict[str, str]] = []
    for turn in turns[-6:]:
        messages.append({"role": "user", "content": str(turn["user_message"])})
        messages.append({"role": "assistant", "content": str(turn["assistant_message"])})
    messages.append({"role": "user", "content": prompt})
    if any(set(message) != {"role", "content"} for message in messages):
        raise RuntimeError("La frontera del modelo contiene campos no permitidos.")
    if any(message["role"] not in {"user", "assistant"} for message in messages):
        raise RuntimeError("La frontera del modelo contiene un rol no permitido.")
    return messages


@dataclass
class Session:
    session_id: str
    provider: str
    model: str
    created_at: str
    status: str = "active"
    stopped_at: str | None = None
    turns: list[dict[str, Any]] = field(default_factory=list)


SESSIONS: dict[str, Session] = {}
SESSIONS_LOCK = RLock()


def session_summary(session: Session) -> dict[str, Any]:
    windows = [window for turn in session.turns for window in turn.get("windows", [])]
    last_turn = session.turns[-1] if session.turns else {}

    def average(name: str) -> float:
        return sum(float(window.get(name, 0.0)) for window in windows) / len(windows) if windows else 0.0

    return {
        "session_id": session.session_id,
        "status": session.status,
        "provider": session.provider,
        "model": session.model,
        "requested_model": session.model,
        "resolved_model": last_turn.get("resolved_model", session.model),
        "turn_count": len(session.turns),
        "token_count": sum(int(turn.get("token_count", 0)) for turn in session.turns),
        "window_count": len(windows),
        "response_time_seconds": last_turn.get("response_time_seconds"),
        "avg_entropy_norm": average("entropy_norm"),
        "avg_gap_norm": average("gap_norm"),
        "avg_rigidity": average("rigidity"),
        "avg_uncertainty": average("uncertainty"),
        "max_entropy_range": max((float(window.get("entropy_range", 0.0)) for window in windows), default=0.0),
    }


class MonitorHandler(BaseHTTPRequestHandler):
    server_version = "PRAMAFrontend/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}", flush=True)

    def _json_body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            decoded = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("El cuerpo JSON no es válido.") from exc
        if not isinstance(decoded, dict):
            raise ValueError("El cuerpo debe ser un objeto JSON.")
        return decoded

    def _send_json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _send_error_json(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"error": message}, status)

    def _ndjson(self, value: Any) -> None:
        encoded = (json.dumps(value, ensure_ascii=False) + "\n").encode("utf-8")
        self.wfile.write(encoded)
        self.wfile.flush()

    def _session(self, session_id: Any) -> Session:
        with SESSIONS_LOCK:
            session = SESSIONS.get(str(session_id or ""))
        if session is None:
            raise ValueError("La sesión no existe.")
        return session

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self._send_json(
                {
                    "status": "ready",
                    "provider": "nvidia_nim",
                    "api_key_present": bool(os.environ.get(NVIDIA_API_KEY_ENV, "").strip()),
                    "models": list(MODEL_PROFILES),
                }
            )
            return
        if parsed.path == "/download":
            requested = unquote(parse_qs(parsed.query).get("path", [""])[0])
            try:
                path = Path(requested).resolve()
                path.relative_to(RESULTS_ROOT.resolve())
                self._serve_file(path, attachment=True)
            except (OSError, ValueError):
                self._send_error_json(HTTPStatus.NOT_FOUND, "Archivo no disponible.")
            return
        relative = "prama_monitor.html" if parsed.path in {"", "/"} else parsed.path.lstrip("/")
        try:
            path = (FRONTEND_ROOT / relative).resolve()
            path.relative_to(FRONTEND_ROOT)
        except ValueError:
            self._send_error_json(HTTPStatus.NOT_FOUND, "Ruta no disponible.")
            return
        self._serve_file(path)

    def _serve_file(self, path: Path, attachment: bool = False) -> None:
        if not path.is_file():
            self._send_error_json(HTTPStatus.NOT_FOUND, "Archivo no encontrado.")
            return
        content = path.read_bytes()
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{mime}; charset=utf-8" if mime.startswith("text/") else mime)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        if attachment:
            self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self) -> None:  # noqa: N802
        try:
            if self.path == "/session/start":
                self._start_session()
            elif self.path == "/chat":
                self._chat()
            elif self.path == "/session/stop":
                self._stop_session()
            elif self.path == "/session/report":
                self._report_session()
            else:
                self._send_error_json(HTTPStatus.NOT_FOUND, "Ruta no disponible.")
        except ValueError as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:  # keep provider details server-side
            print(f"server error: {type(exc).__name__}: {exc}", flush=True)
            self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, "Error interno del servidor.")

    def _start_session(self) -> None:
        body = self._json_body()
        provider = str(body.get("provider") or "")
        model = str(body.get("model") or "")
        if provider != "nvidia_nim":
            raise ValueError("Proveedor no permitido.")
        if model not in MODEL_PROFILES:
            raise ValueError("Modelo no permitido.")
        if not os.environ.get(NVIDIA_API_KEY_ENV, "").strip():
            self._send_error_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "NVIDIA_API_KEY no está activa en la consola que ejecutó este servidor.",
            )
            return
        session = Session(
            session_id=uuid4().hex,
            provider=provider,
            model=model,
            created_at=utc_now(),
        )
        with SESSIONS_LOCK:
            SESSIONS[session.session_id] = session
        self._send_json(session_summary(session), HTTPStatus.CREATED)

    def _chat(self) -> None:
        body = self._json_body()
        session = self._session(body.get("session_id"))
        if session.status != "active":
            raise ValueError("La sesión ya está cerrada.")
        prompt = validate_prompt(body.get("user_message"))
        profile = MODEL_PROFILES[session.model]
        messages = provider_messages(session.turns, prompt)

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Falta instalar el paquete openai.") from exc

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        started = perf_counter()
        response_parts: list[str] = []
        tokens: list[dict[str, Any]] = []
        finish_reason = ""
        resolved_model = session.model
        try:
            stream = OpenAI(
                api_key=os.environ[NVIDIA_API_KEY_ENV].strip(),
                base_url=NVIDIA_BASE_URL,
                timeout=600.0,
                max_retries=1,
            ).chat.completions.create(
                model=session.model,
                messages=messages,
                temperature=float(profile["temperature"]),
                top_p=float(profile["top_p"]),
                max_tokens=int(profile["max_tokens"]),
                seed=1337,
                logprobs=True,
                top_logprobs=TOP_LOGPROBS,
                stream=True,
                extra_body=dict(profile["extra_body"]),
            )
            for chunk in stream:
                resolved_model = str(getattr(chunk, "model", "") or resolved_model)
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                piece = getattr(choice.delta, "content", None)
                if piece:
                    text = str(piece)
                    response_parts.append(text)
                    self._ndjson({"type": "chunk", "text": text})
                for entry in list(getattr(getattr(choice, "logprobs", None), "content", None) or []):
                    tokens.append(token_row(entry))
                if choice.finish_reason:
                    finish_reason = str(choice.finish_reason)
        except Exception as exc:
            print(f"NVIDIA request failed: {type(exc).__name__}: {exc}", flush=True)
            self._ndjson({"type": "error", "message": f"La solicitud a NVIDIA falló: {type(exc).__name__}."})
            return

        elapsed = perf_counter() - started
        assistant_message = "".join(response_parts)
        windows = build_windows(tokens)
        turn = {
            "turn_index": len(session.turns),
            "timestamp": utc_now(),
            "user_message": prompt,
            "assistant_message": assistant_message,
            "finish_reason": finish_reason,
            "response_time_seconds": elapsed,
            "token_count": len(tokens),
            "tokens": tokens,
            "windows": windows,
            "resolved_model": resolved_model,
        }
        with SESSIONS_LOCK:
            session.turns.append(turn)
        self._ndjson({"type": "turn_summary", "session": session_summary(session)})

    def _stop_session(self) -> None:
        body = self._json_body()
        session = self._session(body.get("session_id"))
        session.status = "closed"
        session.stopped_at = utc_now()
        self._send_json(session_summary(session))

    def _report_session(self) -> None:
        body = self._json_body()
        session = self._session(body.get("session_id"))
        output = RESULTS_ROOT / safe_session_name(session.session_id)
        output.mkdir(parents=True, exist_ok=True)
        conversation = [
            {"user": turn["user_message"], "assistant": turn["assistant_message"]}
            for turn in session.turns
        ]
        markdown: list[str] = [f"# Session {session.session_id}", ""]
        for turn in session.turns:
            markdown.extend(
                [
                    f"## Turn {int(turn['turn_index']) + 1}",
                    "",
                    "### User",
                    "",
                    str(turn["user_message"]),
                    "",
                    "### Assistant",
                    "",
                    str(turn["assistant_message"]),
                    "",
                    f"Response time: {float(turn['response_time_seconds']):.3f} s",
                    "",
                ]
            )
        files = {
            "conversation_md": output / "conversation.md",
            "conversation_json": output / "conversation.json",
            "raw": output / "raw.json",
            "metadata": output / "metadata.json",
        }
        files["conversation_md"].write_text("\n".join(markdown), encoding="utf-8")
        files["conversation_json"].write_text(json.dumps(conversation, indent=2, ensure_ascii=False), encoding="utf-8")
        files["raw"].write_text(
            json.dumps(
                {
                    "session_id": session.session_id,
                    "provider": session.provider,
                    "model": session.model,
                    "created_at": session.created_at,
                    "stopped_at": session.stopped_at,
                    "status": session.status,
                    "turns": session.turns,
                    "summary": session_summary(session),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        files["metadata"].write_text(
            json.dumps(
                {
                    "schema": "LLM-SVM-frontend-session/1",
                    "provider_endpoint": NVIDIA_BASE_URL,
                    "api_key_environment_variable": NVIDIA_API_KEY_ENV,
                    "model_payload_boundary": "role_and_content_conversation_only",
                    "response_clock": "server_request_start_to_provider_stream_completion",
                    "summary": session_summary(session),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        self._send_json(
            {
                "output_folder": output.name,
                "files": {key: str(path.resolve()) for key, path in files.items()},
            }
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    server = ThreadingHTTPServer((args.host, args.port), MonitorHandler)
    print(f"PRAMA frontend: http://{args.host}:{args.port}/", flush=True)
    print(
        f"NVIDIA key: {'ready' if os.environ.get(NVIDIA_API_KEY_ENV, '').strip() else 'NOT SET'}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
