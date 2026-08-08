# DSEB-v0 offline preflight

This implementation is the deterministic, model-free entry stage of DSEB-v0. It validates
the benchmark terrain before any LLM is called. It does not execute or modify `O_D`, PRAMA,
`D_O` v9, or ODCE-v0.

## Scope

The preflight checks:

- a 36-turn contiguous protocol generated from a recorded seed;
- satisfiability of every `C_t ∪ P_t` using a current-state CSP oracle;
- separate accounting for ordinary `R_t`, checkpoint `K_t`, and
  `retired_constraint_count`;
- the recovery path `H_21..H_25 = 0,1,2,3,4`;
- turn-local counterfactual scope;
- functional gain, external-integration status, and verified outcome;
- canonical `(turn_index, window_index)` identities and global ordinals;
- the required causal stage order
  `close → O_D → PRAMA → D_O v9 → verifier → outcome registration → ODCE`.

The stage order is simulated and validated offline. Output artifacts explicitly say
`pipeline_execution_mode=SIMULATED_OFFLINE`; they do not claim that a structural observer
was executed.

## Run

```powershell
py -3.12 scripts\run_dseb_offline_preflight.py `
  --seed 7 `
  --output-dir run_outputs\dseb_v0\offline_preflight_seed0007
```

Use `--verbose` only when the complete list of checks should also be printed. The complete
report is always written to `report.json`.

## Artifacts

```text
benchmark_protocol.json
benchmark_turns.jsonl
verifier_outcomes.jsonl
report.json
```

The reference protocol is
`benchmarks/dseb_v0/configs/dseb_v0.json`. It remains exploratory and is not a frozen
calibration population. The witness seed is a technical preflight seed, not evidence that
one seed is statistically sufficient.
