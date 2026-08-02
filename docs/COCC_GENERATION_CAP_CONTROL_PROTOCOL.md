# CoCC generation-cap control protocol

The Nemotron 3 Super 512-token study is preserved as **“Nemotron 3 Super / 512-token confirmatory run — honest null with generation-cap confound.”** Its frozen verdict is not changed by later descriptive analyses.

The next model candidate is NVIDIA NIM `mistralai/mistral-medium-3.5-128b`. Its model-specific runner is separate from the frozen Nemotron runner. The generation profile follows NVIDIA's published API example: `temperature=0.7`, `top_p=1` and `reasoning_effort=high`; token-logprob compatibility remains a mandatory preflight gate.

## Prospective sequence

1. Bind the next provider and model identity.
2. Freeze the candidate token budgets and the acceptable clean-control cap rate.
3. Run only the frozen `calibration` clean controls at increasing candidate budgets.
4. Select the smallest budget with `finish_reason=length` rate at or below 5%. No correctness label or holdout row may enter this selection.
5. Bind the budget-selection artifact and choose one operational absolute-window horizon from 1, 2, 4, 8 or 16.
6. Freeze the complete confirmatory design.
7. Acquire the whole holdout with one budget and no replacements.
8. Before interpreting PRAMA or any accumulated score, report `finish_reason × outcome`, `finish_reason × perturbation_type`, and token/window length by outcome.

If more than 5% of clean holdout controls end at the generation cap, the truncation-independent early-warning claim is `inconclusive_generation_cap_violation`. The run remains reportable and is never repaired by replacing sessions.

Relative-prefix analyses remain exploratory because a percentage prefix contains different numbers of state updates in trajectories of different lengths. The primary early-warning estimand uses a single preregistered absolute horizon and an explicitly reported at-risk set.

Runaway generation is a separate viability estimand. It may be scientifically legitimate, but evidence for it cannot substitute for evidence that PRAMA anticipates externally verified code failure independently of truncation.

## Budget selector

First run the model-specific endpoint check in the same PowerShell session that contains `NVIDIA_API_KEY`:

```powershell
py -3.12 scripts\preflight_nvidia_mistral_medium.py --execute
```

Only after that check resolves the exact model and reports `token_logprobs_supported: true` may the calibration-budget plan be frozen.

The Mistral preflight passed. Plan v1 exposed a manifest/runner compatibility error before output-directory creation or any provider call. The transparent pre-acquisition amendment is frozen as `config/cocc_mistral_medium_3_5_budget_calibration_plan_v2.json`; no scientific rule changed. The first candidate acquisition uses the physically isolated calibration-only dataset:

```powershell
$env:PYTHONUTF8 = "1"
py -3.12 scripts\run_break_the_chain_prama_eval_nvidia_mistral.py `
  --dataset data\cocc_clean_calibration_only_v2.jsonl `
  --dataset-manifest data\cocc_clean_calibration_only_v2.runner_manifest.json `
  --output-dir run_outputs\cocc_mistral_medium_3_5_budget_512 `
  --n 20 `
  --max-tokens 512 `
  --queue-only `
  --calibration-only
```

`--queue-only` performs model acquisition and records finish reasons/logprobs, but does not execute the verifier or PRAMA projector. This prevents correctness outcomes from entering budget selection.

During the 512-token acquisition, repeated provider stalls reached the original 600-second timeout. After 3/20 complete sessions, the run was stopped without deleting or replacing completed artifacts. Operational amendment v3 reduces only the client timeout to 120 seconds and requires resume:

The first v3 resume attempt exposed an implementation defect before making a new provider call: `--resume` required an already complete acquisition. Amendment v4 corrects only that operational behavior. Existing `raw.json` files are validated and reused byte-for-byte; only missing sessions are acquired. The runner now prints `reusing`, `acquiring`, retry failures and `completed` status per session.

```powershell
$env:PYTHONUTF8 = "1"
py -3.12 scripts\run_break_the_chain_prama_eval_nvidia_mistral.py `
  --dataset data\cocc_clean_calibration_only_v2.jsonl `
  --dataset-manifest data\cocc_clean_calibration_only_v2.runner_manifest.json `
  --output-dir run_outputs\cocc_mistral_medium_3_5_budget_512 `
  --n 20 `
  --max-tokens 512 `
  --timeout 120 `
  --queue-only `
  --calibration-only `
  --resume
```

The model payload, generation profile, candidate budget, stopping rule and scientific estimand are unchanged. The three completed sessions are reused byte-for-byte.

The completed 512-token candidate produced `finish_reason=length` in 20/20 calibration controls and is therefore rejected against the frozen 5% maximum. Before starting 1024, amendment v5 disables the OpenAI SDK's two implicit retries (`max_retries=0`). The three explicit runner attempts and 120-second timeout remain unchanged. This makes request counts auditable and ensures each successful `response_time_seconds` value corresponds to one provider request rather than a bundle of hidden SDK retries.

The 1024-token candidate also produced `finish_reason=length` in 20/20 controls. No 2048- or 4096-token acquisition was started. User-directed amendment v6 replaces that unacquired staircase with one 16384-token calibration candidate. NVIDIA's model-specific API reference permits `max_tokens` from 1 through 32768, and the official Build example for `reasoning_effort=high` uses 16384. This value is treated as a ceiling, not a requested response length. The single high-ceiling run uses a 1800-second client timeout, one explicit attempt, no SDK retries, and safe manual `--resume` for missing sessions.

```powershell
py -3.12 scripts\select_cocc_generation_budget.py `
  --protocol config\cocc_next_confirmatory_protocol.draft.json `
  --dataset data\cocc_clean_negation_confirmatory_v2.jsonl `
  --candidate-run 512=run_outputs\next_model_budget_512 `
  --candidate-run 1024=run_outputs\next_model_budget_1024 `
  --output config\next_model_generation_budget_selection.json
```

Candidate runs must be supplied in increasing preregistered order. If none supplied so far passes, the selector reports `MORE_CALIBRATION_REQUIRED` and names the next budget. A confirmatory design cannot be frozen until the selector reports `SELECTED`.
