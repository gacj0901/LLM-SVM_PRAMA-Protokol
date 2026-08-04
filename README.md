# Aptadynamic LLM-SVM_PRAMA-Protokol
LLM Structural Viability Monitor


**LLM-domain implementation of the PRAMA Protokol: structural observability and
viability analysis of generation trajectories from token-level numeric observables**

G.A.C.J. — ORCID: [0009-0009-5649-1359](https://orcid.org/0009-0009-5649-1359)
Part of the **AptadynamiK** program.
Normative specification: [AS-1](https://github.com/gacj0901/aptadynamic-cybernetics) ·
Engine: [`prama-protokol`](https://github.com/gacj0901/prama-protokol)

> **Status:** CoCC/NVIDIA protocols and the D_O v9 structural observer are separate
> experimental programs over the same locked kernel boundary. Each claim is limited
> to its frozen design and bound evidence. See
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

Does a generation trajectory sustain coherent structural motion under perturbation, or
merely continue emitting tokens? Event = token; the observable channel is numeric
logprob geometry. The observation interface builds strictly causal windows, the
**unmodified certified kernel** projects perturbation and accumulated state, and
structural observers study transport, recurrence, contraction and mobility without
feeding those labels back to the model.

Provider termination metadata such as `finish_reason` is retained only as acquisition
metadata. It is not a structural-viability coordinate.

## Architecture (AS-1 P7 — enforced by dependency and identity checks)

```
sessions (raw json)                       aptadynamic_llm.ingest
  → token stream (surprisal, pos, ...)    aptadynamic_llm.omega   ← the ONLY
  → (ω, ω̂) strictly causal                                          domain part
  → Γ = (Δ, Ξ, λ, Θ, M, G), latent        prama_protokol.project  ← certified kernel,
  → structural observer coordinates        D_O v9 / protocol-specific analysis
  → external outcome joined after projection
```

There is deliberately **no independent PRAMA kernel implementation in this
repository**. Projection adapters import `prama-protokol` and verify its version,
commit, source-tree SHA, configuration SHA and recertification artifact before use.
The complete component and evidence boundaries are documented in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Experimental programs

| Program | Status | Authoritative claim path |
|---|---|---|
| CoCC/NVIDIA experiments | protocol-specific; confirmatory only where a frozen design says so | frozen CoCC design, runner and external verifier |
| D_O v9 structural replay | exploratory, descriptive and offline | v6 numeric observer followed by the v9 layered state classifier |

D_O v9 studies transport coherence, recurrence, contraction and mobility states. It
is exploratory unless a prospective protocol explicitly freezes an endpoint. Historical
v9 backfills reuse numeric token observations; where Delta and Xi are recomputed with
the current kernel they are counterfactual reprojections, not re-analyses of stored
kernel-v1 state.

The corrected historical CoCC outcome join is bound to its provenance by
`run_outputs/historical_v9_backfill_cocc_462/provenance_amendment_v2.json`. Its paired
v1-to-v2 transition audit and stratified manual verifier review are retained beside the
report; the superseded join is preserved explicitly and must not be used as current
outcome truth.

The CoCC verifier executes generated code and is therefore restricted to controlled
research use. See [`docs/VERIFIER_SECURITY.md`](docs/VERIFIER_SECURITY.md) for the
current protections and the container boundary required for general use.

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
configuration (64/16) sits on that plateau and is retained as documented sensitivity.

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
