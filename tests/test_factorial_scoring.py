from collections import Counter

import pytest

from aptadynamic_llm.factorial_scoring import (
    FactorialRenderedContexts,
    RenderedCondition,
    ScorerIdentity,
    score_rendered_ensemble,
    validate_rendered_contexts,
)


def _rendered(
    condition,
    *,
    history_marker,
    user_filler_hash=None,
    assistant_filler_hash=None,
):
    # Positions 6..7 are the preserved current-turn prefix. Positions 8..9
    # are the recorded target tokens. Only prior-history positions vary.
    token_ids = [history_marker, 2, 3, 4, 5, 6, 70, 71, 80, 81]
    return RenderedCondition(
        condition=condition,
        token_ids=token_ids,
        target_start=8,
        target_token_ids=[80, 81],
        current_turn_prefix_start=6,
        role_sequence=["system", "user", "assistant", "user", "assistant"],
        message_token_counts=[1, 2, 2, 2, 2],
        chat_template_id="template@1",
        template_control_hash="control-sha256",
        user_filler_hash=user_filler_hash,
        assistant_filler_hash=assistant_filler_hash,
    )


def _contexts(filler_id, marker_offset=0):
    return FactorialRenderedContexts(
        filler_id=filler_id,
        l11=_rendered("L11", history_marker=1),
        l10=_rendered(
            "L10",
            history_marker=10 + marker_offset,
            assistant_filler_hash=f"assistant-{filler_id}",
        ),
        l01=_rendered(
            "L01",
            history_marker=20 + marker_offset,
            user_filler_hash=f"user-{filler_id}",
        ),
        l00=_rendered(
            "L00",
            history_marker=30 + marker_offset,
            user_filler_hash=f"user-{filler_id}",
            assistant_filler_hash=f"assistant-{filler_id}",
        ),
    )


class FakeScorer:
    identity = ScorerIdentity(
        scoring_mode="observer-relative",
        model_id="fake-model@1",
        weights_sha256="weights",
        tokenizer_id="fake-tokenizer@1",
        tokenizer_hash="tokenizer",
        chat_template_id="template@1",
    )

    def __init__(self):
        self.calls = Counter()

    def score(self, rendered):
        self.calls[rendered.condition] += 1
        values = {
            "L11": (-1.0, -1.0),
            "L10": (-2.0, -2.0),
            "L01": (-3.0, -3.0),
            "L00": (-4.0, -4.0),
        }
        return values[rendered.condition]


def test_valid_rendering_enforces_all_structural_invariants():
    validate_rendered_contexts(_contexts("filler-a"))


def test_user_filler_must_be_coupled_between_l01_and_l00():
    contexts = _contexts("filler-a")
    broken_l00 = _rendered(
        "L00",
        history_marker=30,
        user_filler_hash="different-user-filler",
        assistant_filler_hash="assistant-filler-a",
    )
    broken = FactorialRenderedContexts(
        filler_id=contexts.filler_id,
        l11=contexts.l11,
        l10=contexts.l10,
        l01=contexts.l01,
        l00=broken_l00,
    )

    with pytest.raises(ValueError, match="user filler is not coupled"):
        validate_rendered_contexts(broken)


def test_current_turn_prefix_must_be_identical():
    contexts = _contexts("filler-a")
    altered = RenderedCondition(
        condition="L10",
        token_ids=[10, 2, 3, 4, 5, 6, 99, 71, 80, 81],
        target_start=8,
        target_token_ids=[80, 81],
        current_turn_prefix_start=6,
        role_sequence=contexts.l10.role_sequence,
        message_token_counts=contexts.l10.message_token_counts,
        chat_template_id=contexts.l10.chat_template_id,
        template_control_hash=contexts.l10.template_control_hash,
        assistant_filler_hash=contexts.l10.assistant_filler_hash,
    )
    broken = FactorialRenderedContexts(
        filler_id=contexts.filler_id,
        l11=contexts.l11,
        l10=altered,
        l01=contexts.l01,
        l00=contexts.l00,
    )

    with pytest.raises(ValueError, match="prefix is not preserved"):
        validate_rendered_contexts(broken)


def test_ensemble_scores_natural_reference_once():
    scorer = FakeScorer()
    scored = score_rendered_ensemble(
        scorer,
        [
            _contexts("filler-a", 0),
            _contexts("filler-b", 1),
            _contexts("filler-c", 2),
        ],
    )

    assert len(scored) == 3
    assert all(item.l11 == (-1.0, -1.0) for item in scored)
    assert scorer.calls == Counter({"L10": 3, "L01": 3, "L00": 3, "L11": 1})


def test_natural_rendering_must_be_identical_across_fillers():
    first = _contexts("filler-a", 0)
    second = _contexts("filler-b", 1)
    changed_natural = _rendered("L11", history_marker=999)
    second = FactorialRenderedContexts(
        filler_id=second.filler_id,
        l11=changed_natural,
        l10=second.l10,
        l01=second.l01,
        l00=second.l00,
    )

    with pytest.raises(ValueError, match="L11 rendering differs"):
        score_rendered_ensemble(
            FakeScorer(),
            [first, second, _contexts("filler-c", 2)],
        )


def test_scorer_must_return_one_value_per_target_token():
    class ShortScorer(FakeScorer):
        def score(self, rendered):
            return [-1.0]

    with pytest.raises(ValueError, match="2 target tokens"):
        score_rendered_ensemble(
            ShortScorer(),
            [_contexts("a", 0), _contexts("b", 1), _contexts("c", 2)],
        )
