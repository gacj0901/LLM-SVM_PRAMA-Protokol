# Break-The-Chain / PRAMA external-degradation evaluation

## Purpose

This study tests whether a PRAMA trajectory discriminates externally verified
failure on the normalized Chain-of-Code Collapse benchmark. It does not use
dataset labels, `finish_reason`, PRAMA labels, or monitor annotations as ground
truth.

The implementation is:

- `scripts/normalize_cocc_break_chain.py` for official CoCC pickle
  normalization and problem-level splitting;
- `scripts/run_break_the_chain_prama_eval.py` for acquisition and channel
  isolation;
- `scripts/cocc_external_verifier.py` plus `_cocc_verify_worker.py` for
  public/private test execution in a child process;
- `scripts/calibrate_cocc_projector.py` for a frozen, numeric-only
  position-wise surprisal expectation;
- `scripts/project_cocc_prama.py` for the pinned and recertified PRAMA
  projection;
- `scripts/evaluate_break_the_chain_prama.py` for the blind statistical join.

## Channel boundary

The run has four sequential phases:

1. Acquisition sends only the perturbed natural-language task and generation
   parameters to the evaluated LLM.
2. External verification receives the answer and benchmark verifier mapping,
   but no PRAMA coordinate.
3. Projection receives only numeric token observations and a frozen
   calibration digest. It receives no prompt, answer, condition, verifier
   mapping, correctness field, response time, or source label.
4. The blind join occurs only after verifier and trajectory files exist.

`response_time_seconds` is measured with a monotonic clock around the successful
provider request. Failed-attempt sleep time is not included. Dry runs emit zero
with `channel_status=NOT_APPLICABLE`.

The model prompt never includes the benchmark condition, correctness label,
PRAMA name, interface name, monitoring state, entropy, logprob, trajectory, or
timing measurement. These fields exist only outside the provider payload.

## Build a normalized official dataset

The normalizer reads `modified_problems_*.pkl`, records each source SHA-256 and
uses a deterministic problem-level split. All perturbations of one problem stay
on the same side of the calibration/test boundary.

```powershell
python scripts/normalize_cocc_break_chain.py `
  --source PATH_TO_COCC_ROOT `
  --output data/cocc_clean_negation_confirmatory_v2.jsonl `
  --manifest data/cocc_clean_negation_confirmatory_v2.manifest.json `
  --perturbation negation_objective `
  --include-clean-control `
  --clean-calibration-only `
  --seed 1337 `
  --limit-calibration 20 `
  --limit-test 120
```

`--clean-calibration-only` makes the causal order explicit: calibration receives
only clean controls; attacked variants remain in the untouched test partition.
The frozen v2 artifact contains 20 clean calibration sessions and 120 paired
problem groups in holdout (120 clean controls plus 120 negation variants). Its
SHA-256 is
`f22083c38c8399cbbfeb72c98276568f277081bcec38b44b18d971ac617b3433`.
Every frozen holdout session is acquired and verified; PASS/FAIL outcomes cannot
trigger early stopping, replacement, or enrichment.
The evaluator recomputes the dataset hash and the deterministic session IDs,
split assignments, and problem-group hashes. The blind join must contain the
complete frozen membership exactly; a partial, duplicated, old, or foreign run
fails closed before any statistic is calculated.

The v2 binding also freezes the normalization-manifest hash, train/test ID-list
hashes, immutable Ollama model blob, Ollama version, observation-interface
schema and source hashes, generation parameters, PRAMA kernel identity and
kernel parameters. The acquisition runner resolves the local Ollama blob before
the first request and fails if it differs. Its run manifest attests all these
values; the evaluator requires and validates that manifest for a confirmatory
design.

## Freeze calibration before test

First acquire only the calibration prefix with `--queue-only`, then freeze the
numeric projection requests:

```powershell
python scripts/calibrate_cocc_projector.py `
  --requests-dir RUN/projection/requests `
  --output config/cocc_model_calibration.json `
  --window-size 16 `
  --min-sessions 20
```

The calibration artifact contains no prompt, answer, correctness label, token
text, perturbation name, or timing. Its file SHA-256 is passed unchanged to the
complete run and must be echoed by every trajectory row.

## Mandatory projector condition

A per-session PRAMA projection is valid only when its projector is started with
an expectation/calibration object frozen before the evaluated test partition.
The runner does not estimate an expectation from the session being scored.

Every returned trajectory row must contain:

```text
session_id
input_channel_status = OBSERVED
coordinate_origin = DERIVED_KERNEL_STATE
calibration_reference_sha256
kernel_identity
```

The calibration digest returned in every row must match
`--projector-calibration-sha256`. A missing or mismatched digest fails closed.
This prevents a projector from silently fitting its baseline on the evaluated
answer.

## Preflight

The bundled fixture performs no model call:

```powershell
python scripts/run_break_the_chain_prama_eval.py `
  --dataset tests/fixtures/cocc_smoke.jsonl `
  --output-dir run_outputs/cocc_preflight `
  --provider ollama `
  --model hermes3:8b `
  --dry-run `
  --queue-only
```

The generated `projection/requests/*.json` files are the exact numeric-only
payloads accepted by the projector boundary.

## Complete run

```powershell
python scripts/run_break_the_chain_prama_eval.py `
  --dataset PATH_TO_NORMALIZED_COCC.jsonl `
  --dataset-manifest PATH_TO_NORMALIZATION_MANIFEST.json `
  --output-dir run_outputs/cocc_confirmatory `
  --provider ollama `
  --model hermes3:8b `
  --expected-model-blob-sha256 c8985d236593f7a17da2a3da49588aa951a9b1e57ce97753364fbf59e63af84a `
  --confirmatory-design-sha256 cf6d2baba84ab71c8ae4f58671f2cad3d3fa3741f7e50140008137d532154de1 `
  --verifier-command python PATH_TO_VERIFIER.py `
  --projector-command python PATH_TO_PROJECTOR.py `
  --projector-calibration-sha256 FROZEN_CALIBRATION_SHA256
```

The verifier and projector commands read one JSON object from stdin and write
JSON to stdout. They are invoked without `shell=True`.

The bundled verifier applies an AST deny-list, runs each answer under
`python -I` in a child process and enforces a parent timeout. This limits
accidental damage but is not a hardened security sandbox. Answers from
untrusted remote providers should be executed inside a disposable VM or
container with no secrets, no network and a read-only filesystem.

Commands that themselves require options should use the JSON-array forms:

```text
--verifier-command-json '["python","scripts/cocc_external_verifier.py","--dataset","data/cocc.jsonl"]'
--projector-command-json '["python","scripts/project_cocc_prama.py", ...]'
```

An interrupted run can use `--resume`. Resume fails closed unless every saved
raw acquisition and request is present and byte-equivalent after deterministic
reconstruction; it does not call the model again.

After the run:

```powershell
python scripts/evaluate_break_the_chain_prama.py `
  --blind-join run_outputs/cocc_confirmatory/evaluation/blind_join.csv `
  --out run_outputs/cocc_confirmatory/evaluation/report.json `
  --design config/cocc_confirmatory_design_v3.json `
  --dataset data/cocc_clean_negation_confirmatory_v2.jsonl `
  --dataset-manifest data/cocc_clean_negation_confirmatory_v2.manifest.json `
  --run-manifest run_outputs/cocc_confirmatory/manifest.json `
  --session-horizon-csv run_outputs/cocc_confirmatory/evaluation/session_horizon_table.csv `
  --primary-score max_negative_balance
```

The canonical v3 design SHA-256 is
`cf6d2baba84ab71c8ae4f58671f2cad3d3fa3741f7e50140008137d532154de1`.
It is recorded, together with the raw file hash and canonicalization rule, in
`config/cocc_confirmatory_design_v3.freeze.json`. The acquisition runner writes
that canonical SHA into its run manifest and the evaluator rejects a mismatch.

## Frozen evaluation direction

Higher scores always mean worse state:

- primary: `max_negative_balance`;
- PRAMA alternatives: `max_xi`, `final_accumulated_excess`,
  `capacity_loss`;
- instantaneous/logprob baselines: `max_delta`, `mean_surprisal`,
  `mean_entropy`, `negative_mean_gap`.

A positive confirmatory verdict treats `problem_id`, not session, as the
independent unit. Every holdout cluster must contain exactly one
`clean_control` and one `negation_objective` session. The primary association
test reassigns each complete ordered two-outcome vector among problem clusters;
it never permutes the two sessions as independent observations.

The formal incremental statistic is:

```text
AUROC(max_negative_balance) - AUROC(max_delta)
```

It is tested with a paired, one-sided bootstrap that resamples whole
`problem_id` clusters. The frozen null is difference `<= 0`. Confirmation
requires all of the following:

- primary cluster-permutation `p < 0.01`;
- incremental cluster-bootstrap `p < 0.01`;
- one-sided 99% percentile lower bound greater than zero;
- observed AUROC gain of at least `0.05`;
- primary AUROC greater than every other frozen baseline.

The `0.05` gain is the minimum effect of scientific interest: a smaller
improvement is not considered material even if statistically significant.
Any other adequately supported completed result is reported as `honest_null`.
The validity gate requires all 120 frozen problem clusters, at least 20
externally verified FAIL and 20 PASS sessions, at least 20 problem clusters
containing a FAIL, and at least 20 containing a PASS. If any quota is missed,
the result is
`inconclusive_insufficient_class_support`; the threshold is not relaxed and the
holdout is not topped up after seeing outcomes.

The report always compares `max_delta` vs `max_xi` vs
`max_negative_balance`. Since `balance = theta - xi` with fixed theta, the
last two can be order-equivalent; that equivalence is reported instead of
being treated as independent evidence. A dynamic coordinate must strictly
exceed `max_delta` for the original confirmatory rule to pass.

Anticipation is assessed without changing the final external label. The
absolute window horizons 1, 2, 4, 8 and 16 use only sessions still at risk at
that horizon. They are prefixes retained from the beginning: `h=1` means only
the first window and `h=4` means the first four windows; they are not windows
removed from the end. Relative prefixes 25%, 50%, 75% and 100% are reported as a
length-normalized sensitivity analysis, not as the primary real-time estimate.
The evaluator writes a session-level CSV containing label, perturbation,
difficulty, trajectory length, every coordinate at every absolute horizon and
the full-trajectory score. These temporal tables are explicitly exploratory
and descriptive in v3; they do not silently fall back to session-level
inferential p-values.

For the v3 confirmatory design, cluster outcome-profile assignments are
enumerated exactly when their multinomial count is at most 100,000. Otherwise
the frozen procedure uses 100,000 Monte Carlo draws, seed `2026073101`,
Python 3.12 MT19937 shuffle semantics, and a plus-one correction. Invalid
cluster geometry fails closed; there is no session-level fallback.

The paired cluster bootstrap uses 100,000 valid draws, seed `2026073102`,
Python 3.12 MT19937 `randrange` semantics and a plus-one correction.
Single-class draws are discarded and redrawn, up to 1,000,000 attempts; failure
to obtain all valid draws aborts inference. CLI `--seed` and `--permutations`
remain only for legacy v2 evaluations and cannot alter the frozen v3 tests.
The 7/1 retrospective pilot therefore still reports exactly
`p=1/8=0.125`, without Monte Carlo jitter.

Before interpreting `final_accumulated_excess` or `capacity_loss`, the evaluator
audits whether `xi > theta`, accumulated excess became positive, or capacity
fell from its initial value. Kernel parameters remain frozen until this audit
is complete.

This is evidence of external outcome discrimination, not proof that a PRAMA
coordinate is semantically identical to degradation.

## Hermes 3 8B exploratory pilot (2026-07-31)

The local pilot used the official file
`modified_problems_negation_objective.pkl` with source SHA-256
`0e0ab6d641eb07fc030683e99637eb5818aa841a600ff6dcb9d6051143193d8b`.
Four clean sessions froze calibration; the untouched test contained four clean
controls and four negation variants.

Results:

- 12 acquired, 12 externally verified, 12 projected, 12 blindly joined;
- test labels: 7 FAIL and 1 PASS;
- primary `max_negative_balance`: AUROC 1.0, exact permutation p = 0.125;
- instantaneous `max_delta`: AUROC 1.0;
- original verdict: `honest_null`;
- v2 retrospective validity verdict:
  `inconclusive_insufficient_class_support`.

The p-value cannot satisfy the preregistered 0.01 threshold with this class
count, and the primary score did not exceed the strongest instantaneous
baseline. The pilot therefore validates the machinery and isolation boundary,
not the scientific prediction. A powered study needs substantially more clean
cases that Hermes can solve, attacked variants of those same problems, and
group-aware inference at the problem level.

The v2 activation audit also found zero threshold crossings, zero sessions with
positive accumulated excess, and zero sessions with capacity degradation
(`max accumulated_excess = 0`, `min capacity = 1`). Their AUROC 0.5 therefore
reflects mechanisms that were never activated, not a demonstrated failure of
those mechanisms. No kernel change is justified by this pilot.

Across the 12 complete-run calls, measured response time was 13.62–40.70 s
(median 20.32 s, mean 23.15 s). These numbers cover only the successful provider
call, not verification, projection, retry sleeps, or orchestration.
