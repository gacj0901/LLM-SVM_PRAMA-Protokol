# Aptadynamic LLM-SVM_PRAMA-Protokol
LLM Structural Viability Monitor


**LLM domain implementation of the PRAMA Protokol (exploratory draft): structural
viability of generation sessions from token-level observables and preregistered
final-state discrimination of operational degradation in generative trajectories**

G.A.C.J. — ORCID: [0009-0009-5649-1359](https://orcid.org/0009-0009-5649-1359)
Part of the **AptadynamiK** program.
Normative specification: [AS-1](https://github.com/gacj0901/aptadynamic-cybernetics) ·
Engine: [`prama-protokol`](https://github.com/gacj0901/prama-protokol)

> **Status:** E-P1 is the preregistered confirmatory route. The structural
> Observation Interface through D_O v9 is a separate exploratory program and does
> not amend, extend, or retrospectively validate E-P1. See
> [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Reproducibility boundary

The repository separates three classes of material:

| Class | Location | Rule |
|---|---|---|
| implementation | `src/`, `scripts/`, `config/`, `schemas/`, `FrontEnd/`, `tests/` | executable code, contracts and tests |
| retained inputs | `data/` | immutable datasets listed in `data/retained_manifest_v1.json`; large data use Git LFS |
| reproducible results | `run_outputs/` | only artifacts listed in `run_outputs/reproducible_results_manifest_v1.json` carry a reproducibility claim |

Unlisted run output is retained as exploratory evidence or working material, not as
a canonical result. Local scratch data belongs under the ignored `_local/` or
`_scratch/` directories.

The window kernel is pinned twice: the Python dependency is fixed to Git commit
`cb41d590207a09d498532b8c599e12ecab7a0dcb`, and
[`config/kernel.lock.json`](config/kernel.lock.json) fixes source-tree SHA-256
`61e1063de0b5b032cd6af09eeb3b6906614f6331954697c605603afa18f641fc`,
configuration SHA and recertification SHA. Projection fails closed when the installed
kernel differs from the declaration.

## Structural artifact contract v0.2

The implementable multi-channel contract is
[`docs/LLM_STRUCTURAL_STATE_ARTIFACT_CONTRACT_v0.2.md`](docs/LLM_STRUCTURAL_STATE_ARTIFACT_CONTRACT_v0.2.md).
Its JSON Schemas live in [`schemas/`](schemas/) and its runtime boundary in
`aptadynamic_llm.artifact_schema`.

The evaluated model receives task content only. `aptadynamic_llm.model_payload`
fails closed if provider payloads contain monitor labels, experimental condition
IDs, PRAMA/structural state, interface metadata, system roles, or tool roles.
Experimental metadata and response timing are attached only after the provider
response has completed.

## The question

Does a generation session sustain itself, or merely appear to? Event = token; the
observable is surprisal. The observation interface builds a strictly causal expectation
per position bucket over previous sessions of the same model; the **unmodified certified
kernel** projects the stream onto Γ and flags **latent collapse** — the session still
generates while its structural margin is being consumed.

First target outcome: `finish_reason = "length"` vs `"stop"` — aptadynamically, the
failure of the **resolution** phase: the session that cannot conclude. Because this
outcome is known only at the final token, E-P1 is post-hoc state discrimination now;
event-localized early warning is declared future work, not a confirmatory claim.

## Architecture (AS-1 P7 — enforced by dependency and identity checks)

```
sessions (raw json)                       aptadynamic_llm.ingest
  → token stream (surprisal, pos, ...)    aptadynamic_llm.omega   ← the ONLY
  → (ω, ω̂) strictly causal                                          domain part
  → Γ = (Δ, Ξ, λ, Θ, M, G), latent        prama_protokol.project  ← certified kernel,
  → certified window summaries            scripts/score_sessions_prama
  → D6 gates + sole verdict                 scripts/analyze_ep1.py
```

There is deliberately **no independent PRAMA kernel implementation in this
repository**. Projection adapters import `prama-protokol` and verify its version,
commit, source-tree SHA, configuration SHA and recertification artifact before use.
The complete component and evidence boundaries are documented in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Experimental programs

| Program | Status | Authoritative claim path |
|---|---|---|
| E-P1 | preregistered confirmatory study | `scripts/analyze_ep1.py` under `PREREGISTRATION_P1.md` |
| CoCC/NVIDIA experiments | protocol-specific; confirmatory only where a frozen design says so | frozen CoCC design, runner and external verifier |
| D_O v9 structural replay | exploratory, descriptive and offline | v6 numeric observer followed by the v9 layered state classifier |

D_O v9 studies transport coherence, recurrence, contraction and mobility states. It
must not be reported as an E-P1 endpoint. Historical v9 backfills reuse numeric token
observations; where Delta and Xi are recomputed with the current kernel they are
counterfactual reprojections, not re-analyses of stored kernel-v1 state.

## Validated on synthetic data (2026-07)

Two scenarios generated by `examples/make_synthetic.py`:

| Scenario | Latent occupancy | Trivial baseline (mean surprisal) |
|---|---|---|
| **volume** (runaway = rising mean) | AUC 0.61 | **AUC 0.93 — baseline wins, as it should** |
| **structural** (runaway = mean-preserving oscillation: loops/repetition) | **AUC 0.939, ratio 4.45, perm p < 0.001** | AUC 0.357 — the volume monitor is blind |

The table is a historical July 2026 result from the kernel/configuration named in that
section; it is not inherited by the current window-kernel lock. The current synthetic
test is deliberately a wiring invariant, not a performance or confirmatory threshold.

**Bin-scale finding (feeds AS-1 v1.1):** the grid-validated kernel configuration
(tau_memory = 336) is mechanically inapplicable at K = 256 token windows — memory longer
than the trajectory means the accumulator never reaches regime. A declared sweep shows a
robust plateau at tau ∈ [16, 64] (AUC 0.90–0.98 across g_smooth ∈ {8, 16, 24}); the study
configuration (64/16) sits on that plateau and is declared in
[`PREREGISTRATION_P1.md`](PREREGISTRATION_P1.md) (D3), with the validated configuration
retained as documented sensitivity.

The division of labor mirrors the electrical-grid finding: when degeneration is
structural rather than volumetric, the accumulator sees what activity monitoring cannot.
Compliance: the interface's Δ **passes the C3 degeneration statistic**
(r_Δω = −0.09194754 with the seed-7 structural corpus and `prama-protokol==0.1.0`)
via the Engine's compliance module, which the study script runs and enforces before
reporting any result.

## Synthetic wiring check

```bash
pip install -e .                                   # installs the local project plus the Git dependency
python examples/make_synthetic.py structural data_s
python scripts/latent_llm_test.py data_s
```

This is a wiring test, not confirmatory evidence.

## E-P1 pipeline (Hermes 3 8B through Ollama)

```bash
# Uncensored redesign pilot: 40 distinct, bounded structured-analysis prompts.
python scripts/collect_ollama.py --pilot --model hermes3:8b --n 40 \
  --num-predict 2048 --seed-per-index \
  --prompts examples/prompts_ep1_v2.jsonl --out data_pilot_v2

# After freezing the pilot-derived num_predict cap in PREREGISTRATION_P1.md:
python scripts/collect_ollama.py --confirmatory --model hermes3:8b --n 400 \
  --num-predict <frozen-cap> --seed-per-index \
  --prompts prompts_ep1.jsonl --out data_confirmatory

python scripts/analyze_ep1.py \
  --sessions-dir data_confirmatory --out outputs/ep1
```

`analyze_ep1.py` is the only authoritative verdict path. It checks the frozen collection
identity, runs C3, applies the power gate before inspecting scores, validates inputs, and
evaluates permutation + AUROC + TPR at the train-calibrated FPR against exactly four
baselines. It prints `positive`, `honest_null`, `interface_failure`, or the pre-verdict
state `underpowered`. Component scripts cannot establish the confirmatory claim.

The confirmatory channel is only `latent_occupancy`; `delta`, `xi`, and `neg_M` remain
separate exploratory/diagnostic channels.

## Conceptual foundation of the domain

- *The Ontological Condition of Artificial Intelligence* — DOI:
  [10.5281/zenodo.20666529](https://doi.org/10.5281/zenodo.20666529)
- *Testable Predictions of the Cyberprotobiont Theory (ℬₒ)* — DOI:
  [10.5281/zenodo.19508233](https://doi.org/10.5281/zenodo.19508233)
- [ORDSPOC](https://github.com/gacj0901/prama-protokol-ordspoc) — trajectory-level risk
  as executable argument.

## License

AGPL-3.0. Commercial licensing, industrial collaborations and academic research
partnerships may be available separately.
