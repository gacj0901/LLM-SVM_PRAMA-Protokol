"""LLM ingest: raw JSON sessions to canonical token rows.

Canonical rows preserve the source ``session_id`` and add ``generation_id`` as
the unit analyzed by the observation interface. A one-turn raw file keeps
``generation_id == session_id``; multi-turn files get one generation id per
turn so independent completions are never interleaved by position.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_sessions(root: str) -> pd.DataFrame:
    rows = []
    files = sorted(Path(root).rglob("*.json"))
    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        turns = d.get("turns")
        if not turns:
            continue
        sid = d.get("session_id", f.stem)
        model = d.get("model", "")
        prompt_id = d.get("prompt_id", "")
        for turn_index, turn in enumerate(turns):
            toks = turn.get("tokens") or []
            fr = turn.get("finish_reason", "")
            generation_id = sid if len(turns) == 1 else f"{sid}:turn{turn_index:03d}"
            for i, t in enumerate(toks):
                rows.append({
                    "session_id": sid,
                    "generation_id": generation_id,
                    "turn_index": turn_index,
                    "model": model,
                    "prompt_id": prompt_id,
                    "pos": i,
                    "surprisal": -t["top1_logprob"],
                    "entropy": t.get("entropy", 0.0),
                    "gap": t.get("gap", 0.0),
                    "finish_reason": fr,
                })
    return pd.DataFrame(rows)
