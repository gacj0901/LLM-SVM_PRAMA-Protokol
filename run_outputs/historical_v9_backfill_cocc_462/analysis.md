# Historical CoCC v9 backfill — exploratory audit

## Scope and provenance

- 462 historical responses were projected offline: 232 DeepSeek and 230 GPT-4.1.
- No source result was modified and no model/API call was made.
- Projection used numeric token observations only.
- The 462 projections contain no prompts, answers, or external outcomes.
- 120 problem IDs overlapped the current normalized verifier dataset, producing 240 externally verified responses.
- Outcomes were joined only after blind v9 projection.

### Kernel-version boundary

- The historical responses and their original PRAMA analyses belong to the **kernel v1** era. This provenance is user-declared because the inspected raw artifacts do not bind a complete legacy kernel identity.
- The offline backfill did **not** reuse the v1 PRAMA state trajectories. It reused the saved numeric token observations, recomputed Delta and Xi with the currently recertified `prama-protokol 0.3.0 / project_v3` kernel, and then applied the v6 structural observer plus the v9 state layer.
- Consequently, this audit is a **counterfactual reprojection of historical logprobs through the current kernel**, not “v9 running on kernel v1.” Its results cannot isolate observer-version effects from kernel-version effects.
- A matched kernel comparison requires replaying the same token observations twice: once through the exactly reconstructed v1 configuration and once through the current kernel, holding the v9 observer fixed.

## Historical inventory result

The 230 problem IDs shared by both historical model runs had the following termination pattern:

- both stopped naturally: 193;
- DeepSeek reached the length limit and GPT-4.1 stopped: 21;
- GPT-4.1 reached the length limit and DeepSeek stopped: 16;
- both reached the length limit: 0.

Length exhaustion was therefore model-specific in this corpus, not merely problem-specific.

## Descriptive length audit

For DeepSeek, length-exhausted trajectories had more full-trajectory structural degradation than natural stops:

- post-stabilization disrupted fraction: 0.484 vs 0.323;
- recurrence persistence: 0.338 vs 0.209;
- crystallizing fraction: 0.302 vs 0.131;
- maximum disrupted dwell: 12.62 vs 4.81;
- terminal transport coherence: 0.356 vs 0.436.

For GPT-4.1 the direction was mostly reversed:

- post-stabilization disrupted fraction: 0.300 vs 0.492;
- recurrence persistence: 0.223 vs 0.288;
- crystallizing fraction: 0.187 vs 0.256;
- terminal transport coherence: 0.529 vs 0.228.

At H64, length-exhausted trajectories did not have higher disrupted occupancy in either model. The DeepSeek full-trajectory separation therefore emerges with later exposure and is not a universal early-warning result.

## External outcome audit

The current verifier dataset covered 120 paired problems (240 responses). Class support was extremely imbalanced:

- DeepSeek: 2 PASS, 118 FAIL;
- GPT-4.1: 2 PASS, 118 FAIL.

Full-trajectory AUROC, with FAIL as the positive class:

| Endpoint | DeepSeek | GPT-4.1 |
|---|---:|---:|
| max delta | 0.487 | 0.428 |
| max xi | 0.475 | 0.318 |
| negative terminal transport coherence | 0.931 | 0.758 |
| crystallizing fraction | 0.650 | 0.610 |
| recurrence persistence | 0.654 | 0.449 |
| post-stabilization disrupted fraction | 0.531 | 0.386 |
| mean transport deficit | 0.837 | 0.474 |

At H32, the channels directionally above chance in both models were:

| Endpoint | DeepSeek | GPT-4.1 |
|---|---:|---:|
| disrupted fraction | 0.810 | 0.647 |
| recurrence persistence | 0.905 | 0.674 |
| crystallizing fraction | 0.637 | 0.606 |

However, H32 contained only one DeepSeek PASS and two GPT-4.1 PASS responses. H64 contained one PASS per model, and H128 contained none. These AUROCs are descriptive only and cannot support a confirmatory claim.

## Interpretation boundary

The backfill provides evidence that v9 processes historical providers differently and that mobility channels are not reducible to delta or xi. The most consistent full-trajectory coordinate was loss of terminal transport coherence; the most consistent H32 coordinates were disruption, recurrence, and crystallizing occupancy.

It does not validate prediction of functional failure because the historical negation-objective corpus contains insufficient PASS support. A future blind outcome join must provide adequate cases in both classes without changing the frozen v9 observer.
