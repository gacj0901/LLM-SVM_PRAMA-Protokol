# ODCE-v0 — Structural Conversion Differential

## Role

ODCE-v0 is an executable causal layer after the PRAMA kernel and D_O v9:

```text
numeric observations -> PRAMA Gamma -> D_O v9 Q -> ODCE -> optional inference
```

It does not modify the kernel or observer, emit a viability verdict, train a
predictive model, or control the evaluated system. It preserves three objects at
every causal index: structural cost, causally obtained return, and their declared
channel-wise differential.

## Exploratory development construction

For the rolling window `W_t` declared in
`config/odce_v0_1_exploratory.json`, the implementation constructs:

```text
retained_friction    = exponentially weighted mean(xi in W_t)
accumulated_debt     = max(A_t - A_window_start, 0)
capacity_consumption = max(lambda_session_start - lambda_t, 0)
excess_persistence   = mean(xi > theta in W_t)
adverse_trend        = exponentially weighted mean(max(-G, 0) in W_t)
```

and:

```text
structural_recovery = max(lambda_t - min(lambda_tau for tau in W_t), 0)
adaptive_organization_level = mean(transport_coherence * (1 - variation_contraction))
functional_gain     = latest causally available domain measurement in W_t
external_integration = latest causally available domain measurement in W_t
verified_outcome     = latest causally available independent measurement in W_t
```

`adaptive_organization_level` is unavailable unless D_O v9 covers every PRAMA
identity in the causal window and at least eight D_O observations contain both
`transport_coherence` and `variation_contraction`. The aggregation uses only those
evaluable D_O observations; missing identities fail closed instead of being joined
by list position. External return is unavailable until its `available_at_index` is
reached. `event_index` records when the measured event occurred; it never authorizes
earlier use. Both indices must agree with their canonical
`(turn_index, window_index)` identities, and the join fails closed on a mismatch.
A final outcome is never copied backward into a prefix.

## Normalization and correspondence

The operator reads all normalization rules and cost/return correspondences from its
contract. ODCE-v0 supports an explicit `preserve` extreme policy; it does not clip
values silently. The checked-in development contract uses identity normalization,
so it is suitable for exercising the architecture but not for confirmatory claims.
Changing only `status` cannot promote it: `FROZEN_PROSPECTIVE` additionally
requires a frozen domain calibration, calibration-population provenance, explicit
prospective correspondence governance, and a separate freeze manifest whose hashes
bind the complete contract, normalization, and correspondence surfaces.

The default differentials are:

```text
retained_friction_vs_structural_recovery = retained_friction - structural_recovery
retained_friction_vs_adaptive_organization_level = retained_friction - adaptive_organization_level
capacity_consumption_vs_functional_gain = capacity_consumption - functional_gain
```

All underlying coordinates remain in the artifact. ODCE-v0.1.0 does not define or
emit an `efficiency_vector`.

## Exploratory calibration

`scripts/calibrate_odce_exploratory.py` fits each raw cost and benefit coordinate
independently with a robust median location and
`max(1.4826 * MAD, IQR / 1.349)` scale. A fitted rule additionally requires the
declared minimum number and fraction of observed indices, minimum contributing
session count, minimum session coverage, and a nondegenerate scale. Every other
coordinate retains its base rule and is listed in `blocking_channels`; absence is
never treated as zero.

Calibration artifacts are accepted directly. Exploratory artifacts require the
explicit `--allow-exploratory-input` acknowledgement. Confirmatory artifacts are
always rejected as calibration input. The emitted contract preserves:

```json
{
  "status": "EXPLORATORY_CAUSAL_POST_KERNEL",
  "normalization": {
    "calibration_status": "EXPLORATORY_ROBUST_PARTIAL",
    "confirmatory_use_allowed": false
  }
}
```

The accompanying report records channel coverage, estimator diagnostics, retained
fallback rules and blockers. Index-level weighting is explicitly exploratory; a
future freeze requires domain-representative data, declared session/cluster
weighting, outcome coverage and reviewed correspondences.

Every emitted correspondence also records `calibration_status` as `CALIBRATED`,
`PARTIALLY_CALIBRATED`, or `UNCALIBRATED`. Its
`instrumental_interpretation_allowed` flag is true only when both channel rules are
calibrated. ODCE may still compute the algebraic differential for exploratory
diagnostics, but a partial or uncalibrated result is not a metrically comparable
instrumental reading.

Changing normalization invalidates the empirical `differential_threshold`. The
normalization calibrator therefore clears the old threshold calibration and marks
threshold recalibration as required; it never silently carries a noise floor across
new coordinate scales.

## Empirical material-differential threshold

`positive_persistence` does not use zero as an implicit significance boundary.
The checked-in exploratory threshold is derived from explicitly declared stable
ODCE rows by `scripts/calibrate_odce_differential_threshold.py`. For each selected
correspondence the estimator measures:

```text
center_j = median(D_j)
noise_floor_j = abs(center_j) + Q_q(abs(D_j - center_j))
delta_D = max_j(noise_floor_j)
```

The current exploratory controlled baseline uses 24 stable sessions and `q=0.95`,
yielding `delta_D = 0.0011849999999999803`. The procedure modifies only
`differential_threshold`; every normalization rule and its hash remain unchanged.
The calibration is exploratory and not frozen.

Each newly derived ODCE artifact records the exact `differential_threshold` used;
legacy v0.1.0 artifacts without that additive field remain readable. Thus the three
dynamics remain distinct:

```text
D_t = normalized_cost_t - normalized_benefit_t
A_t = A_(t-1) + max(D_t, 0)
P_t = mean(1[D_tau > delta_D] for tau in W_t)
```

`delta_D` affects only `P_t`. It does not alter `D_t` or irreversible exposure
`A_t`.

Before either dynamic is updated, ODCE applies only IEEE-754 cleanup:

```text
if abs(D_t) < numeric_epsilon: D_t = 0
numeric_epsilon = 1e-12
```

This is not a semantic threshold. It removes arithmetic cancellation residue from
the represented differential itself; `differential_threshold` remains exclusively
the materiality boundary used by `P_t`.

## Temporal scales and irreversible accumulation

Every emitted artifact declares `temporal_scope`. Friction, debt, persistence,
adverse trend, structural recovery, and adaptive organization are rolling-window
coordinates. Capacity consumption is session-to-date. Domain returns are the latest
causally available observation session-to-date. Differentials are current-index
comparisons of their declared coordinates. Within dynamics, trend compares the
current value with the previous observed index, persistence uses the rolling window,
and cumulative exposure is session-to-date.

For each differential `D_t`, cumulative conversion-deficit exposure is:

```text
C_t = C_(t-1) + max(D_t, 0)
```

The accumulation is irreversible. A negative or unavailable differential adds zero
and can never reduce prior exposure.

## Missing channels

The common envelope retains its scalar `channel_status`. Per-coordinate states live
under `component_status` and can be `OBSERVED`, `NOT_APPLICABLE`, `UNAVAILABLE`, or
`INVALID`. A non-observed coordinate is always `null`; absence is never numeric zero.

## External return input

The optional outcome JSONL contains one or more causal measurements:

```json
{
  "session_id": "session-1",
  "artifact_type": "domain_return_observation",
  "event_index": 41,
  "available_at_index": 63,
  "event_window": {"turn_index": 2, "window_index": 9},
  "available_at_window": {"turn_index": 3, "window_index": 15},
  "benefit_vector": {
    "functional_gain": 0.8,
    "external_integration": null,
    "verified_outcome": 1.0
  },
  "component_status": {
    "functional_gain": "OBSERVED",
    "external_integration": "NOT_APPLICABLE",
    "verified_outcome": "OBSERVED"
  },
  "verifier_reference_sha256": "<sha256>",
  "retrospective_backfill": true,
  "causal_availability_declared": true,
  "provider_termination_metadata_used": false
}
```

The full record also carries the common provenance envelope. The command validates
the domain-return artifact, rejects index/identity disagreement, out-of-range
identities, future-outcome fields, response latency, and `finish_reason`.

## Execution

```bash
python scripts/derive_structural_conversion.py \
  --prama prama_trajectory.jsonl \
  --structural-observations structural_observation.jsonl \
  --outcomes causal_domain_return.jsonl \
  --contract config/odce_v0_1_exploratory.json \
  --out structural_conversion_differential.jsonl \
  --study-id study-1 \
  --producer local \
  --partition exploratory
```

The writer validates every record before atomic replacement. The exploratory
contract fails closed if asked to emit `--partition confirmatory`.

After deriving raw identity-normalized exploratory ODCE artifacts, calibrate them
with:

```bash
python scripts/calibrate_odce_exploratory.py \
  --input raw_identity_odce.jsonl \
  --base-contract config/odce_v0_1_exploratory.json \
  --out-contract odce_v0_1_exploratory_calibrated.json \
  --out-report odce_v0_1_exploratory_calibration_report.json \
  --allow-exploratory-input
```

Confirmatory execution, once a legitimate calibrated contract exists, also requires:

```bash
  --contract prospective_odce_contract.json \
  --contract-freeze prospective_odce_contract.freeze.json \
  --partition confirmatory
```

The checked-in `odce_v0_1_exploratory.json` is deliberately not such a contract and
has no freeze manifest. Every confirmatory differential records the canonical hash
of the supplied freeze; exploratory artifacts require that reference to be `null`.

## Instrumental validation

The deterministic battery can be regenerated with:

```bash
python scripts/run_odce_instrumental_validation.py
```

It materializes canonical PRAMA, D_O v9 structural observations, delayed
functional, external-integration and independently verified outcomes, stable
noise-floor rows, calibrated-threshold provenance and eight controlled
trajectories. The functional trajectory extends beyond 32 post-availability
windows so its persistence becomes observed without changing missing-window logic.
Its report checks increasing cost, increasing
organization, matched joint growth, sustained deficit, real capacity recovery,
causal outcome availability, irreversible exposure, input immutability and all
causal metadata invariants. This is an instrumental validation, not a prediction or
failure-accuracy experiment.

All observed raw and normalized values are deliberately identical in this battery.
It validates ODCE operator algebra under identity normalization, not empirical
normalizer calibration. Because every artifact is exploratory,
`contract_freeze_sha256` remains `null`; the report explicitly forbids interpreting
the result as a fixed instrumental calibration.

## Inferential boundary

A later predictor may consume causal histories of Gamma, Q, and ODCE. It is not
justified merely because an ODCE differential correlates with an outcome. A
prospective protocol must freeze the outcome horizon, clusters, calibration,
missing-data rules, model class, baselines, ablations, minimum effect of interest,
and inference procedure. It must show incremental value beyond Delta, Xi, capacity,
trend, D_O v9 coordinates, and domain-specific conventional baselines.
