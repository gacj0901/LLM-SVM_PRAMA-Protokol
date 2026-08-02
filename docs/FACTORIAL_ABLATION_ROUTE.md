# Parallel factorial-ablation route

Status: implemented pre-kernel vertical slice. Experimental; not part of E-P1.

This route realizes the arithmetic, validity, rendering contract, calibration,
and artifact boundary of `OD_ABLATION_FACTORIAL_SPEC_v0.2.md` without replacing
`aptadynamic_llm.omega`.

It deliberately stops before `prama_protokol.project`. The factorial interface
emits one observation per token window, whereas E-P1 declares one kernel bin per
token. A window-scale kernel configuration and numeric recertification must be
frozen before this route may mutate kernel state.

## Implemented components

- `aptadynamic_llm.factorial_scoring`
  - full-logit teacher-forced scorer protocol;
  - exact rendered-length, role, template, target-position and current-prefix
    invariants;
  - coupled user and assistant fillers across the factorial cells;
  - one shared natural `L11` reference across fillers.
- `aptadynamic_llm.factorial_ablation`
  - ratio-of-sums factorial decomposition;
  - per-filler raw condition records;
  - mean of per-filler normalized coordinates;
  - sample filler variance (`ddof=1`);
  - support, alignment and provenance exclusions before the kernel boundary;
  - positive intact-context self-effect gate for self-dominance candidates.
- `aptadynamic_llm.frozen_expectation`
  - update-after-session calibration trace;
  - frozen confirmatory estimator;
  - distinct-session warm-up;
  - temporal-overlap rejection;
  - window-position-aware strata and deterministic statistics hash.
- `aptadynamic_llm.factorial_pipeline`
  - calibration/confirmatory partition enforcement;
  - complete observation records;
  - kernel-ready rows only when `eligible=true` and expectation is observed.
- `aptadynamic_llm.llama_server_scorer`
  - exact teacher forcing against the same Ollama GGUF token IDs;
  - forced-target selection with pre-sampling logprob recovery;
  - loaded-weight verification and raw-logprob preservation preflight.
- `aptadynamic_llm.hermes3_factorial_config`
  - full Ollama manifest, GGUF, template and llama-server build identities.
- `scripts/evaluate_factorial_ablation.py`
  - deterministic JSONL artifact generation from precomputed condition scores.
- `scripts/preflight_factorial_hermes3.py`
  - live generator/scorer identity and numerical-contract verification.

## Frozen implementation decisions

1. `turn_index` is zero-based. Index `1` is the second assistant turn and is the
   first eligible default.
2. The primary ensemble value is
   `mean_j(omega_j)`, not a ratio reconstructed from mean supports.
3. `support_magnitude` is `mean_j(|S_self,j| + |S_user,j|)`.
4. `filler_variance` is the unbiased sample variance over `omega_j`.
5. `L11` must be identical across the filler ensemble and is scored once.
6. A component-level self-dominance candidate requires both an omega threshold
   and positive `mean_j(L11-L10)`. It is never labeled as echo by this code.
7. Calibration statistics update after all windows of a session, preventing
   intra-session leakage.
8. Every calibration session must precede every confirmatory session.

## Precomputed-score input

The evaluator consumes one JSON object per window:

```json
{
  "session_id": "session-0001",
  "session_order": 12,
  "partition": "calibration",
  "turn_index": 1,
  "window_index": 0,
  "token_start": 0,
  "token_end": 2,
  "expectation_stratum": {
    "generator_id": "generator@version",
    "task_family": "structured_analysis",
    "turn_depth_bucket": 2,
    "window_position_bucket": 0,
    "scoring_mode": "observer-relative",
    "scoring_model_id": "observer@weights-hash"
  },
  "conditions": [
    {
      "filler_id": "neutral-grammar-01",
      "L11": [-0.7, -1.2],
      "L10": [-1.1, -1.5],
      "L01": [-0.8, -1.3],
      "L00": [-1.4, -1.8]
    },
    {
      "filler_id": "unrelated-session-01",
      "L11": [-0.7, -1.2],
      "L10": [-1.0, -1.6],
      "L01": [-0.9, -1.4],
      "L00": [-1.5, -1.9]
    },
    {
      "filler_id": "lexical-permutation-01",
      "L11": [-0.7, -1.2],
      "L10": [-1.2, -1.4],
      "L01": [-0.85, -1.35],
      "L00": [-1.45, -1.85]
    }
  ]
}
```

Every confirmatory row uses the same schema with
`"partition": "confirmatory"` and a `session_order` greater than the complete
calibration partition.

Run:

```bash
python scripts/evaluate_factorial_ablation.py \
  --input factorial_scores.jsonl \
  --out outputs/factorial_od \
  --estimator-id factorial-frozen-v1 \
  --min-context-sessions 5 \
  --min-support-magnitude 0.1 \
  --max-filler-variance 0.05
```

Outputs:

- `observations.jsonl`: every eligible and excluded calibration/confirmatory
  window with per-filler scores and expectation status;
- `kernel_inputs.jsonl`: only conforming confirmatory scalar pairs;
- `manifest.json`: source hash, frozen parameters, estimator hash, counts and
  the explicit `kernel_not_invoked` status.

## Frozen Hermes 3 pilot identity

- Ollama model: `hermes3:8b`
- displayed model ID: `4f6b83f30b62`
- full manifest SHA-256:
  `4f6b83f30b62bc3d0cf9be09266db222805ee815c8fd7d8b38f863f655be78b7`
- GGUF SHA-256:
  `c8985d236593f7a17da2a3da49588aa951a9b1e57ce97753364fbf59e63af84a`
- quantization: `Q4_0`
- Ollama template SHA-256:
  `c7ec478a7939bd2449e43dde848f31c31bc68b75990d4085ced5ac30f299b24a`
- minimum Ollama version: `0.30.11`
- locally observed Ollama client/server version during preflight: `0.32.1`
- bundled llama-server build: `b1-cb295bf59`

The scorer is generator-relative because it loads the byte-exact quantized
GGUF used by Ollama. It receives pre-rendered token IDs, so the experiment must
still render messages with the Ollama Modelfile template rather than
llama-server's different embedded Jinja template.

With the bundled llama-server running on port 11435, verify the stack:

```bash
python scripts/preflight_factorial_hermes3.py
```

The preflight rejects a different model blob, template, manifest, architecture,
quantization or llama-server build. It also proves that forcing a token does
not alter the raw logprob returned by the server.

## Remaining work before real-model use

1. Implement tokenizer-aware filler generation and retain every rendered
   context hash.
2. Implement the Ollama-template renderer that produces the exact token-ID
   contexts accepted by the scorer.
3. Freeze the filler ensemble and validity thresholds in
   a new preregistration.
4. Pin the PRAMA release, source-tree digest, optional Git commit, API entry
   point, and numeric-recertification hash.
5. Select and recertify kernel parameters with `bin = window`.
6. Execute synthetic identifiability and filler-robustness stages before any
   confirmatory partition is scored.
