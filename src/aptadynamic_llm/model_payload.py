"""Fail-closed separation between monitor state and the evaluated LLM."""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping


class ModelPayloadLeakageError(ValueError):
    """Internal state or interface metadata reached the model boundary."""


FORBIDDEN_KEYS = {
    "artifact_type",
    "channel_status",
    "condition_id",
    "interface",
    "label",
    "label_version",
    "monitor",
    "observation_validity",
    "prama",
    "regime",
    "structural_state",
    "trajectory_assessment",
}

FORBIDDEN_KEY_MARKERS = (
    "artifact",
    "calibration",
    "condition",
    "interface",
    "label",
    "monitor",
    "prama",
    "regime",
    "structural",
    "threshold",
    "trajectory",
)

FORBIDDEN_TEXT_MARKERS = (
    "prama",
    "condition_id",
    "monitor.",
    "interface",
    "interfaz",
    "structural state",
    "trajectory_assessment",
    "regime_label",
    "interface metadata",
)


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            yield str(key).casefold()
            yield from _walk_keys(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _walk_keys(nested)


def assert_no_monitor_metadata(value: Any) -> None:
    """Reject nested internal keys before serialization."""

    keys = set(_walk_keys(value))
    leaked = sorted(
        key
        for key in keys
        if key in FORBIDDEN_KEYS
        or any(marker in key for marker in FORBIDDEN_KEY_MARKERS)
    )
    if leaked:
        raise ModelPayloadLeakageError(
            f"model payload contains forbidden internal keys: {leaked}"
        )


def task_only_prompt(prompt: str) -> str:
    """Validate a plain task prompt with no hidden system/interface channel."""

    if not isinstance(prompt, str) or not prompt.strip():
        raise ModelPayloadLeakageError("prompt must be a non-empty string")
    normalized = prompt.casefold()
    markers = [marker for marker in FORBIDDEN_TEXT_MARKERS if marker in normalized]
    if markers:
        raise ModelPayloadLeakageError(
            f"task text contains forbidden monitor/interface markers: {markers}"
        )
    return prompt


def task_only_messages(messages: Iterable[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Copy only role/content pairs; reject system, tool and metadata channels."""

    output: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if set(message) != {"role", "content"}:
            raise ModelPayloadLeakageError(
                f"message {index} must contain exactly role and content"
            )
        role = str(message["role"])
        if role not in {"user", "assistant"}:
            raise ModelPayloadLeakageError(
                f"message {index} uses forbidden role {role!r}"
            )
        content = task_only_prompt(str(message["content"]))
        output.append({"role": role, "content": content})
    if not output or output[-1]["role"] != "user":
        raise ModelPayloadLeakageError("the final task-only message must be user")
    assert_no_monitor_metadata(output)
    return output


def encode_task_only_payload(payload: Mapping[str, Any]) -> bytes:
    """Serialize an already allow-listed provider payload."""

    allowed = {"model", "prompt", "messages", "stream", "logprobs", "top_logprobs", "options"}
    unknown = set(payload) - allowed
    if unknown:
        raise ModelPayloadLeakageError(f"provider payload has unknown keys: {sorted(unknown)}")
    assert_no_monitor_metadata(payload)
    if "prompt" in payload:
        task_only_prompt(str(payload["prompt"]))
    if "messages" in payload:
        task_only_messages(payload["messages"])
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")
