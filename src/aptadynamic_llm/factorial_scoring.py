"""Contracts for rendering and teacher-forced factorial scoring.

The current Ollama top-k collector cannot satisfy this contract.  An adapter
must return the log-likelihood of every recorded target token under an exact
rendering of all four contexts.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Iterable, Protocol, Sequence, runtime_checkable

from aptadynamic_llm.factorial_ablation import FactorialConditionScores


FACTORIAL_CONDITIONS = ("L11", "L10", "L01", "L00")


def _int_tuple(values: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


@dataclass(frozen=True)
class ScorerIdentity:
    scoring_mode: str
    model_id: str
    weights_sha256: str
    tokenizer_id: str
    tokenizer_hash: str
    chat_template_id: str
    log_base: str = "e"

    def __post_init__(self) -> None:
        if self.scoring_mode not in {"generator-relative", "observer-relative"}:
            raise ValueError("invalid scoring_mode")
        for name in (
            "model_id",
            "weights_sha256",
            "tokenizer_id",
            "tokenizer_hash",
            "chat_template_id",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} cannot be empty")
        if self.log_base != "e":
            raise ValueError("this implementation contract requires natural logarithms")

    def stable_id(self) -> str:
        encoded = json.dumps(
            self.__dict__,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return sha256(encoded).hexdigest()


@dataclass(frozen=True)
class RenderedCondition:
    """A fully rendered condition with an explicit recorded-token target span."""

    condition: str
    token_ids: tuple[int, ...]
    target_start: int
    target_token_ids: tuple[int, ...]
    current_turn_prefix_start: int
    role_sequence: tuple[str, ...]
    message_token_counts: tuple[int, ...]
    chat_template_id: str
    template_control_hash: str
    user_filler_hash: str | None = None
    assistant_filler_hash: str | None = None

    def __init__(
        self,
        *,
        condition: str,
        token_ids: Sequence[int],
        target_start: int,
        target_token_ids: Sequence[int],
        current_turn_prefix_start: int,
        role_sequence: Sequence[str],
        message_token_counts: Sequence[int],
        chat_template_id: str,
        template_control_hash: str,
        user_filler_hash: str | None = None,
        assistant_filler_hash: str | None = None,
    ) -> None:
        object.__setattr__(self, "condition", condition)
        object.__setattr__(self, "token_ids", _int_tuple(token_ids))
        object.__setattr__(self, "target_start", int(target_start))
        object.__setattr__(self, "target_token_ids", _int_tuple(target_token_ids))
        object.__setattr__(
            self, "current_turn_prefix_start", int(current_turn_prefix_start)
        )
        object.__setattr__(self, "role_sequence", tuple(str(role) for role in role_sequence))
        object.__setattr__(
            self, "message_token_counts", _int_tuple(message_token_counts)
        )
        object.__setattr__(self, "chat_template_id", str(chat_template_id))
        object.__setattr__(self, "template_control_hash", str(template_control_hash))
        object.__setattr__(self, "user_filler_hash", user_filler_hash)
        object.__setattr__(self, "assistant_filler_hash", assistant_filler_hash)

    def __post_init__(self) -> None:
        # A custom __init__ is used to normalize sequences, so validate here
        # explicitly through validate_basic() at the ensemble boundary.
        pass

    def validate_basic(self) -> None:
        if self.condition not in FACTORIAL_CONDITIONS:
            raise ValueError(f"unknown factorial condition: {self.condition}")
        if not self.target_token_ids:
            raise ValueError("target_token_ids cannot be empty")
        if self.target_start < 0:
            raise ValueError("target_start must be non-negative")
        if not 0 <= self.current_turn_prefix_start <= self.target_start:
            raise ValueError("invalid current-turn prefix span")
        stop = self.target_start + len(self.target_token_ids)
        if stop > len(self.token_ids):
            raise ValueError("target span falls outside rendered token_ids")
        if self.token_ids[self.target_start:stop] != self.target_token_ids:
            raise ValueError("rendered target span does not equal target_token_ids")
        if len(self.role_sequence) != len(self.message_token_counts):
            raise ValueError("role and message-token-count arrays must align")
        if any(count < 0 for count in self.message_token_counts):
            raise ValueError("message token counts must be non-negative")
        if not self.chat_template_id or not self.template_control_hash:
            raise ValueError("chat-template identity and control hash are mandatory")

    def deterministic_hash(self) -> str:
        payload = {
            "condition": self.condition,
            "token_ids": self.token_ids,
            "target_start": self.target_start,
            "target_token_ids": self.target_token_ids,
            "current_turn_prefix_start": self.current_turn_prefix_start,
            "role_sequence": self.role_sequence,
            "message_token_counts": self.message_token_counts,
            "chat_template_id": self.chat_template_id,
            "template_control_hash": self.template_control_hash,
            "user_filler_hash": self.user_filler_hash,
            "assistant_filler_hash": self.assistant_filler_hash,
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return sha256(encoded).hexdigest()


@dataclass(frozen=True)
class FactorialRenderedContexts:
    """Four coupled renderings for one filler realization."""

    filler_id: str
    l11: RenderedCondition
    l10: RenderedCondition
    l01: RenderedCondition
    l00: RenderedCondition

    def conditions(self) -> tuple[RenderedCondition, ...]:
        return self.l11, self.l10, self.l01, self.l00


def validate_rendered_contexts(contexts: FactorialRenderedContexts) -> None:
    """Enforce positional, template, prefix, target, and filler coupling."""

    if not contexts.filler_id:
        raise ValueError("filler_id cannot be empty")
    conditions = contexts.conditions()
    expected_labels = FACTORIAL_CONDITIONS
    for rendered, expected in zip(conditions, expected_labels):
        rendered.validate_basic()
        if rendered.condition != expected:
            raise ValueError(
                f"condition slot expected {expected}, got {rendered.condition}"
            )

    reference = contexts.l11
    for rendered in conditions[1:]:
        if len(rendered.token_ids) != len(reference.token_ids):
            raise ValueError("all conditions must preserve absolute rendered length")
        if rendered.target_start != reference.target_start:
            raise ValueError("all conditions must preserve the exact target position")
        if rendered.target_token_ids != reference.target_token_ids:
            raise ValueError("all conditions must score identical recorded target tokens")
        if rendered.current_turn_prefix_start != reference.current_turn_prefix_start:
            raise ValueError("current-turn prefix start differs across conditions")
        if rendered.role_sequence != reference.role_sequence:
            raise ValueError("role sequence differs across conditions")
        if rendered.message_token_counts != reference.message_token_counts:
            raise ValueError("rendered per-message token counts differ across conditions")
        if rendered.chat_template_id != reference.chat_template_id:
            raise ValueError("chat template differs across conditions")
        if rendered.template_control_hash != reference.template_control_hash:
            raise ValueError("chat-template control tokens differ across conditions")
        reference_prefix = reference.token_ids[
            reference.current_turn_prefix_start:reference.target_start
        ]
        candidate_prefix = rendered.token_ids[
            rendered.current_turn_prefix_start:rendered.target_start
        ]
        if candidate_prefix != reference_prefix:
            raise ValueError("current assistant-turn prefix is not preserved exactly")

    if contexts.l01.user_filler_hash is None or contexts.l00.user_filler_hash is None:
        raise ValueError("neutralized user-history conditions require a filler hash")
    if contexts.l01.user_filler_hash != contexts.l00.user_filler_hash:
        raise ValueError("user filler is not coupled between L01 and L00")
    if (
        contexts.l10.assistant_filler_hash is None
        or contexts.l00.assistant_filler_hash is None
    ):
        raise ValueError("neutralized self-history conditions require a filler hash")
    if contexts.l10.assistant_filler_hash != contexts.l00.assistant_filler_hash:
        raise ValueError("assistant filler is not coupled between L10 and L00")
    if reference.user_filler_hash is not None or reference.assistant_filler_hash is not None:
        raise ValueError("L11 must contain no neutralization filler")
    if contexts.l10.user_filler_hash is not None:
        raise ValueError("L10 must preserve natural user history")
    if contexts.l01.assistant_filler_hash is not None:
        raise ValueError("L01 must preserve natural assistant history")


@runtime_checkable
class TeacherForcedScorer(Protocol):
    """Adapter contract for a full-logit local or remote scoring stack."""

    @property
    def identity(self) -> ScorerIdentity:
        ...

    def score(self, rendered: RenderedCondition) -> Sequence[float]:
        """Return one natural-log likelihood per recorded target token."""


@runtime_checkable
class FactorialRenderer(Protocol):
    """Tokenizer-aware renderer/neutralizer implemented by a scoring backend."""

    def render(self, filler_id: str) -> FactorialRenderedContexts:
        ...


def _validated_scores(
    scorer: TeacherForcedScorer,
    rendered: RenderedCondition,
) -> tuple[float, ...]:
    scores = tuple(float(value) for value in scorer.score(rendered))
    if len(scores) != len(rendered.target_token_ids):
        raise ValueError(
            f"{rendered.condition}: scorer returned {len(scores)} values for "
            f"{len(rendered.target_token_ids)} target tokens"
        )
    if not all(math.isfinite(value) for value in scores):
        raise ValueError(f"{rendered.condition}: scorer returned a non-finite value")
    return scores


def score_rendered_ensemble(
    scorer: TeacherForcedScorer,
    ensemble: Iterable[FactorialRenderedContexts],
) -> tuple[FactorialConditionScores, ...]:
    """Validate and score a coupled filler ensemble.

    ``L11`` is scored once because its rendering must be byte-for-byte
    equivalent across fillers.  This prevents filler-dependent numerical noise
    in the natural reference cell.
    """

    contexts = tuple(ensemble)
    if len(contexts) < 3:
        raise ValueError("at least three rendered filler realizations are required")
    filler_ids = [item.filler_id for item in contexts]
    if len(filler_ids) != len(set(filler_ids)):
        raise ValueError("filler_id values must be unique")
    for item in contexts:
        validate_rendered_contexts(item)

    natural_hash = contexts[0].l11.deterministic_hash()
    if any(item.l11.deterministic_hash() != natural_hash for item in contexts[1:]):
        raise ValueError("L11 rendering differs across the filler ensemble")

    l11 = _validated_scores(scorer, contexts[0].l11)
    output = []
    for item in contexts:
        output.append(
            FactorialConditionScores(
                filler_id=item.filler_id,
                l11=l11,
                l10=_validated_scores(scorer, item.l10),
                l01=_validated_scores(scorer, item.l01),
                l00=_validated_scores(scorer, item.l00),
            )
        )
    return tuple(output)
