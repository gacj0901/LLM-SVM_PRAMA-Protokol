# NVIDIA Nemotron 3 Super runner

This route leaves `scripts/run_break_the_chain_prama_eval.py` unchanged because
that file is already bound by the frozen Hermes design. NVIDIA acquisition uses
the separate runner:

```text
scripts/run_break_the_chain_prama_eval_nvidia.py
```

Frozen provider identity:

```text
endpoint  https://integrate.api.nvidia.com/v1
model     nvidia/nemotron-3-super-120b-a12b
key env   NVIDIA_API_KEY
```

The key is read only at request time. It is never accepted as a command-line
argument and is not written to raw sessions, projection requests or manifests.

## PowerShell activation

```powershell
$env:NVIDIA_API_KEY = Read-Host "Pega tu NVIDIA API key" -MaskInput
```

Do not put the key in the repository or in a design JSON.

## Required endpoint preflight

First perform the local-only check, which consumes no credits:

```powershell
py -3.12 scripts/preflight_nvidia_nemotron.py
```

Then explicitly authorize one tiny remote request:

```powershell
py -3.12 scripts/preflight_nvidia_nemotron.py --execute
```

The remote preflight requests eight output tokens with reasoning disabled and
requires native token `logprobs` plus `top_logprobs`. A normal text response
without token log probabilities fails closed because PRAMA cannot project it.
The preflight does not print the key or the generated text.

## Local dry run

```powershell
py -3.12 scripts/run_break_the_chain_prama_eval_nvidia.py `
  --dataset data/cocc_clean_negation_confirmatory_v2.jsonl `
  --dataset-manifest data/cocc_clean_negation_confirmatory_v2.manifest.json `
  --output-dir run_outputs/nemotron3_super_dry_run `
  --n 4 `
  --dry-run `
  --queue-only
```

The model page recommends `temperature=1.0` and `top_p=0.95`; this runner
rejects drift from those values. Thinking is disabled by default. Enabling it
requires both `--enable-thinking` and an explicit `--reasoning-budget` no larger
than `--max-tokens`.

Do not use the Hermes v3 design SHA for an NVIDIA run. After the endpoint
preflight succeeds, create and freeze a separate Nemotron design containing the
NVIDIA model, endpoint, runner SHA, thinking mode, token budget and observed
provider identity before confirmatory acquisition.

## Frozen calibration before the confirmatory design

Nemotron must not reuse the Hermes surprisal expectation. The calibration plan
is frozen in:

```text
config/cocc_nemotron3_super_calibration_plan_v1.json
config/cocc_nemotron3_super_calibration_plan_v1.freeze.json
```

It binds exactly the 20 `calibration`/`clean_control` rows at the beginning of
the normalized dataset. Acquire those rows without verification or projection:

```powershell
py -3.12 scripts/run_break_the_chain_prama_eval_nvidia.py `
  --dataset data/cocc_clean_negation_confirmatory_v2.jsonl `
  --dataset-manifest data/cocc_clean_negation_confirmatory_v2.manifest.json `
  --output-dir run_outputs/cocc_nemotron3_super_calibration_v1 `
  --n 20 `
  --queue-only
```

Then freeze the numeric-only calibration artifact:

```powershell
py -3.12 scripts/calibrate_cocc_projector.py `
  --requests-dir run_outputs/cocc_nemotron3_super_calibration_v1/projection/requests `
  --output config/cocc_nemotron3_super_calibration_v1.json `
  --window-size 16 `
  --min-sessions 20
```

Only after that artifact exists is its SHA inserted into the NVIDIA
confirmatory design. The final design SHA must be frozen before acquiring any
of the 240 holdout sessions.

## Frozen confirmatory design

The Nemotron calibration and final design are now frozen as:

```text
calibration SHA-256  b9ea07856fd5a8097cb37ea8b582d4c9bb404563022facca1e943c95a60bea4f
design SHA-256       707755b216001ad203502afb2474ec6e2c33f4b1076b4aa25a5722c245c2e8a5
```

The NVIDIA endpoint did not expose an immutable weight digest or a non-null
`system_fingerprint`. The design therefore freezes the endpoint, exact model
alias, resolved-model response, request parameters, client version and local
software hashes, while explicitly declining to claim immutable-weight
reproducibility.

The complete confirmatory acquisition command is intentionally documented but
must not be launched until the operator has checked available NVIDIA credits:

```powershell
py -3.12 scripts/run_break_the_chain_prama_eval_nvidia.py `
  --dataset data/cocc_clean_negation_confirmatory_v2.jsonl `
  --dataset-manifest data/cocc_clean_negation_confirmatory_v2.manifest.json `
  --output-dir run_outputs/cocc_nemotron3_super_confirmatory_v4 `
  --confirmatory-design-sha256 707755b216001ad203502afb2474ec6e2c33f4b1076b4aa25a5722c245c2e8a5 `
  --projector-calibration-sha256 b9ea07856fd5a8097cb37ea8b582d4c9bb404563022facca1e943c95a60bea4f `
  --verifier-command-json '["py","-3.12","scripts/cocc_external_verifier.py","--dataset","data/cocc_clean_negation_confirmatory_v2.jsonl"]' `
  --projector-command-json '["py","-3.12","scripts/project_cocc_prama.py","--calibration","config/cocc_nemotron3_super_calibration_v1.json","--declaration","config/window_prama_kernel_declaration.json","--recertification","run_outputs/window_prama_recertification_v030_20260730.json"]'
```

This command acquires all 260 frozen dataset rows: 20 calibration rows retained
for exact dataset membership plus the untouched 240-session holdout. Formal
inference uses only the holdout partition.

After acquisition, evaluate with:

```powershell
py -3.12 scripts/evaluate_break_the_chain_prama.py `
  --blind-join run_outputs/cocc_nemotron3_super_confirmatory_v4/evaluation/blind_join.csv `
  --out run_outputs/cocc_nemotron3_super_confirmatory_v4/evaluation/report.json `
  --design config/cocc_confirmatory_design_v4_nvidia_nemotron3_super.json `
  --dataset data/cocc_clean_negation_confirmatory_v2.jsonl `
  --dataset-manifest data/cocc_clean_negation_confirmatory_v2.manifest.json `
  --run-manifest run_outputs/cocc_nemotron3_super_confirmatory_v4/manifest.json `
  --session-horizon-csv run_outputs/cocc_nemotron3_super_confirmatory_v4/evaluation/session_horizon_table.csv `
  --primary-score max_negative_balance
```

Official references:

- https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b?nim=hosted
- https://docs.api.nvidia.com/nim/reference/llm-apis
