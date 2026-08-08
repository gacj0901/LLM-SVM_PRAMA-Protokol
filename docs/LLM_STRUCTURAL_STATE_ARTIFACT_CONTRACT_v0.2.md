# LLM Structural State Artifact Contract

Version: **0.2.0**  
Artifact version: **1.0.0**  
Status: implementable normative contract; scientific interpretations remain
exploratory until the validation stages below are passed.

## 1. Scope and claim boundary

This contract defines auditable artifacts for:

1. teacher-forced contextual support of a recorded continuation;
2. uptake of independently verifiable external anchors;
3. response to relevant perturbation;
4. paired epistemic-channel modulation;
5. historical accumulation through a pinned, window-recertified PRAMA kernel;
6. causal post-kernel structural observation through D_O v9;
7. causal post-observer structural conversion through ODCE-v0;
8. optional deterministic multichannel annotations.

The factorial channel does **not** directly observe model internals and does not
establish a general causal property of free generation. It estimates the relative
support of a fixed recorded target under controlled context ablations.

No artifact may claim consciousness, deception, ego, agency, a complete viability
regime, or ground truth about an interpretive annotation.

## 2. Normative language

`MUST`, `MUST NOT`, `SHOULD`, and `MAY` are normative. A conforming producer MUST:

- emit contract version `0.2.0`;
- emit artifact version `1.0.0`;
- use the canonical fields in this document;
- preserve all exclusions and unavailable channels;
- reject non-finite JSON numbers;
- freeze calibration before confirmatory exposure;
- separate `monitor.*` annotations from `outcome.*` labels;
- keep all monitor, condition, interface, calibration, and label metadata outside
  the evaluated model payload.

## 3. Immutable source artifact

`generation_observation` is the source-of-record for one provider response. It MUST
contain hashes of prompt and response, exact model identity when available, provider
termination metadata, token count, and:

```json
{
  "response_time_seconds": 1.4281
}
```

`finish_reason` and `response_time_seconds` are service-execution metadata only.
Neither is a structural-viability coordinate and neither may enter D_O v9. The latter
is elapsed monotonic wall time from immediately before the provider request is sent
until its complete response is received.

Raw task and response text MAY be retained in access-controlled source files, but
the interoperable artifact SHOULD use hashes to avoid accidental prompt leakage.

## 4. Common provenance envelope

Every JSONL record MUST contain:

```json
{
  "contract_version": "0.2.0",
  "artifact_type": "coupling_observation",
  "artifact_version": "1.0.0",
  "study_id": "study-1",
  "session_id": "session-1",
  "producer": "package-or-script@version",
  "created_at": "2026-07-28T10:00:00+00:00",
  "source_sha256": "64-lowercase-hex",
  "config_sha256": "64-lowercase-hex",
  "partition": "calibration",
  "channel_status": "OBSERVED"
}
```

Allowed partitions are `calibration`, `confirmatory`, and `exploratory`.

Allowed channel states are:

- `OBSERVED`: required measurements passed their validity gates;
- `NOT_APPLICABLE`: the protocol declares the channel unnecessary for this case;
- `UNAVAILABLE`: the channel was relevant but could not be measured;
- `INVALID`: measurements exist but failed a validity gate.

Absence is never encoded as a negative observation.

## 5. Canonical artifacts

The normative schemas are in `schemas/`.

| Artifact type | File convention | Purpose |
|---|---|---|
| `generation_observation` | `generation_observation.jsonl` | immutable provider execution |
| `coupling_observation` | `coupling_observation.jsonl` | teacher-forced context support |
| `external_anchor_event` | `external_anchor_events.jsonl` | independent constraint and uptake |
| `perturbation_response` | `perturbation_response.jsonl` | preregistered pre/post response |
| `epistemic_channel` | `epistemic_channel.jsonl` | matched-pair report modulation |
| `prama_trajectory` | `prama_trajectory.jsonl` | recertified historical accumulation |
| `structural_observation` | `structural_observation.jsonl` | primary D_O v9 post-kernel observation |
| `structural_conversion_differential` | `structural_conversion_differential.jsonl` | ODCE-v0 cost/return/differential object |
| `structural_label` | `structural_labels.jsonl` | optional secondary interpretive annotation |

### 5.1 Coupling

Canonical names are:

```text
omega_dep
expected_omega_dep
self_dependence_excess
```

Legacy `omega` and `expected_omega` MAY be emitted for backward compatibility but
are not canonical.

For token log-likelihood sums \(L_{ab}\):

\[
S_{\rm self}=\frac{(L_{01}-L_{00})+(L_{11}-L_{10})}{2}
\]

\[
S_{\rm user}=\frac{(L_{10}-L_{00})+(L_{11}-L_{01})}{2}
\]

\[
I=L_{11}-L_{10}-L_{01}+L_{00}
\]

\[
m=|S_{\rm self}|+|S_{\rm user}|
\]

\[
\omega_{\rm dep}=
\frac{S_{\rm self}-S_{\rm user}}{m+\epsilon}
\in[-1,1]
\]

The frozen expectation is \(\widehat\omega_{\rm dep}\), estimated only from prior
calibration sessions in the declared stratum. The anomaly coordinate is:

\[
D_{\rm self}=\omega_{\rm dep}-\widehat\omega_{\rm dep}\in[-2,2].
\]

When no expectation exists, both `expected_omega_dep` and
`self_dependence_excess` MUST be `null`.

Ensemble aggregation is the arithmetic mean of each filler-level normalized
coordinate. `filler_variance` is the unbiased sample variance (`ddof=1`).
Support magnitude, alignment, natural-score identity, minimum filler count, and
filler-variance exclusions occur before any kernel call.

### 5.2 External anchors

An anchor MUST:

- originate outside the evaluated trajectory;
- have a SHA-256 provenance hash;
- be independently auditable;
- precede the response window used to score uptake;
- not depend on the monitor annotation being evaluated.

Uptake requires both a preregistered `uptake_score` threshold and an independent
verifier pass. Similar wording alone is insufficient. Latency is:

\[
L_{\rm uptake}=k_{\rm first\ verified\ uptake}-k_{\rm introduced}.
\]

### 5.3 Perturbation response

The response horizon and numeric gates MUST be frozen. The implementation uses:

\[
\Delta D=D_{\rm post}-D_{\rm pre}, \qquad
\Delta U=U_{\rm post}-U_{\rm pre}
\]

\[
\text{trajectory\_change\_magnitude}=
\sqrt{(\Delta D)^2+(\Delta U)^2}.
\]

Classes are:

- `adaptive_integration`;
- `partial_integration`;
- `persistent_nonintegration`;
- `counterfactual_rejection`;
- `indeterminate`.

The class definitions are executable in
`aptadynamic_llm.perturbation_response.PerturbationConfig`.

### 5.4 Epistemic channel

Only matched task states are comparable. `task_state_sha256` MUST be identical
between pair members. `condition_id` is backend-only and MUST be attached after
the model response.

The canonical result is a vector:

```text
evidence_coverage_shift
verifier_relevant_omission_shift
precision_shift
calibration_shift
response_quality_shift
```

There is no canonical scalar \(\mathcal B_o\). A scalar reduction requires a
separate preregistration and construct validation. The paired channel is valid
only when task state matches and competence remains within the frozen tolerance.

### 5.5 PRAMA trajectory

PRAMA receives ordered pairs
`(omega_dep, expected_omega_dep)` at `bin_scale="window"`. Because these
coupling coordinates are signed while PRAMA v0.3 requires a nonnegative
expectation, both coordinates MUST first use the frozen affine map:

\[
\omega_{\rm kernel}=(\omega_{\rm dep}+1)/2,\qquad
\widehat\omega_{\rm kernel}=(\widehat\omega_{\rm dep}+1)/2.
\]

The declaration identifies this map as `signed_unit_affine_v1`; values outside
`[-1,1]` fail closed. The `config_sha256` covers both the kernel parameters and
this input transform.

A projection MUST NOT run unless all are pinned and verified:

```text
package
version
source_tree_sha256
commit (nullable provenance)
kernel_api = project_v3
config_sha256
recertification_sha256
bin_scale = window
input_transform = signed_unit_affine_v1
column_map
```

The recertification file and the complete importable Python source tree are
hashed at runtime. The source-tree digest is the authoritative executable
identity, including editable or dirty installations. A containing Git commit is
recorded and checked when available, but it is not a substitute for the source
digest and MAY be `null` when Git metadata is unavailable. The v0.3
`project_v3` entry point is mandatory because it exposes the frozen trajectory
coordinates, including accumulated excess `A`. Token-scale parameters MUST NOT
be reused silently. Historical coordinates describe persistence; they do not
identify semantic cause.

For `prama_trajectory`, the envelope value `channel_status="OBSERVED"` means
that the accepted coupling input channel was observed and valid. It MUST NOT be
read as if every trajectory coordinate were directly observed. Each output row
therefore also declares:

```text
input_channel_status = OBSERVED
coordinate_origin = DERIVED_KERNEL_STATE
```

`delta` is computed from the observed and frozen-expected coupling pair after
the declared affine transform. `xi`, `accumulated_excess`, `capacity`, `theta`,
`balance`, and `trend` are derived kernel states.

### 5.6 D_O v9 structural observation

D_O v9 is the primary structural observer of PRAMA-projected generative
trajectories. It is post-kernel and therefore is not the Observation Interface:

```text
O_D != D_O_v9
```

The canonical boundary is:

```text
numeric Observation Interface
  -> recertified PRAMA trajectory
numeric token windows + PRAMA trajectory
  -> D_O_v6 numeric structural channels
  -> structural_observation (D_O_v9)
  -> optional structural_label integration
```

A conforming `structural_observation` MUST declare transport, recurrence,
contraction and coherent-mobility status; the resolved structural state; its causal
evidence-window bounds; observer identity; and the numeric coordinates supporting
the classification. It MUST declare:

```json
{
  "artifact_type": "structural_observation",
  "observer": "D_O_v9",
  "observer_version": "D_O_v9",
  "transport_status": "COHERENT",
  "recurrence_status": "RECURRENT",
  "contraction_status": "NOT_CONTRACTING",
  "mobility_status": "RECURRENT",
  "structural_state": "RECURRENT",
  "evidence_window_start": 0,
  "evidence_window_end": 63,
  "causal": true,
  "external_outcome_used": false,
  "provider_termination_metadata_used": false
}
```

The observer MUST read only causal numeric windows. It MUST NOT read
`finish_reason`, response latency, verifier outcomes, semantic labels, prompts or
answers. Historical replay is one application of D_O v9, not its architectural
definition. The retained historical replay declaration and its hashes remain
separate from the prospective primary-observer declaration.

### 5.7 ODCE-v0 structural conversion differential

ODCE-v0 is a causal post-observer construction, not part of the PRAMA kernel and
not a predictive model:

```text
prama_trajectory + structural_observation + causally available domain return
  -> structural_conversion_differential
```

Every record MUST preserve separately:

- the raw and normalized structural cost vectors;
- the raw and normalized obtained-return vectors;
- the declared channel-wise differential;
- per-channel status and differential dynamics;
- the empirical `differential_threshold` used for material-positive persistence;
- hashes of the normalization and correspondence contracts;
- hashes of the exact causal upstream windows used.

The canonical cost coordinates are `retained_friction`, `accumulated_debt`,
`capacity_consumption`, `excess_persistence`, and `adverse_trend`. The canonical
return coordinates are `structural_recovery`, `adaptive_organization_level`,
`functional_gain`, `external_integration`, and `verified_outcome`.
`structural_recovery` is recovery from the minimum capacity already observed in
the causal rolling window. `adaptive_organization_level` requires complete D_O v9
identity coverage and the contractually declared minimum evaluable support.

Correspondence names MUST state both exact coordinates. The exploratory contract
therefore uses `retained_friction_vs_structural_recovery`,
`retained_friction_vs_adaptive_organization_level`, and
`capacity_consumption_vs_functional_gain`; ambiguous aggregate names are rejected.

Because the common envelope already uses `channel_status` for the overall artifact,
the nested per-coordinate states use `component_status`. An absent coordinate MUST
be `null` and have status `NOT_APPLICABLE`, `UNAVAILABLE`, or `INVALID`; absence MUST
NOT be represented by numeric zero.

ODCE MUST reject provider termination metadata and future outcomes. It MUST declare
`causal=true`, `predictive_model_used=false`, `future_outcome_used=false`, and
`provider_termination_metadata_used=false`. A functional or verified result observed
for event window `e` with `available_at_index=t` MAY affect ODCE at `t` and later
causal windows, but never a prefix that precedes availability. Both indices MUST be
bound to canonical `(turn_index, window_index)` identities; joins by an unverified
integer position are forbidden.

Every coordinate MUST declare its temporal scale. Rolling-window costs and returns,
session-to-date capacity consumption, latest-causally-available domain returns,
current-index differentials, previous-observed-index trend, rolling persistence,
and session-to-date exposure are distinct scopes.
`cumulative_conversion_deficit_exposure` MUST use
`C_t = C_(t-1) + max(differential_t, 0)`; it is irreversible and a missing
differential produces no increment.
`positive_persistence` MUST use `1[D_t > differential_threshold]`, where the
threshold is bound to an empirical stable-condition noise-floor calibration. The
threshold changes persistence only; it MUST NOT alter the instantaneous
differential or irreversible exposure.
Implementations MAY canonicalize floating-point cancellation residue with the
declared positive `numeric_epsilon`: `abs(D_t) < numeric_epsilon` becomes exact
zero before dynamics are updated. This arithmetic cleanup is distinct from the
material `differential_threshold` and MUST NOT be used to redefine accumulation.

The initial `odce_v0_1_exploratory.json` contract uses identity normalization solely
to exercise the architecture. It is not a frozen domain calibration and MUST NOT
produce a confirmatory artifact. Confirmatory use requires a prospectively frozen
normalization population, estimators, channel correspondence, horizons, missing-data
rules and effect criterion.

A confirmatory invocation MUST provide a separate
`LLM-SVM-ODCE-contract-freeze/0.1` manifest. The manifest MUST bind canonical hashes
of the complete ODCE contract, normalization surface, and correspondence surface,
plus the calibration reference, calibration population, and correspondence
rationale. Merely changing the contract `status` is nonconforming. Every
confirmatory differential MUST carry `contract_freeze_sha256`; this field MUST be
`null` for calibration and exploratory output.

## 6. Model-payload isolation

The evaluated LLM may receive only:

- task text;
- prior user/assistant task messages when the protocol requires history;
- provider generation parameters needed to produce the response.

It MUST NOT receive:

```text
condition_id
monitor labels
outcome labels
PRAMA state or terminology
structural states
calibration values
thresholds
interface identity or metadata
artifact fields
timing results
```

System and tool roles are rejected by the default task-only boundary. The runtime
implementation is `aptadynamic_llm.model_payload`; provider serialization occurs
only after this check. Experimental condition assignment is stored in backend
artifacts after the response.

## 7. Optional deterministic annotation contract

When the optional multichannel integrator is invoked, exactly one secondary
`monitor.*` annotation is emitted. It MUST bind the SHA-256 of its upstream
`structural_observation` and declare
`annotation_role="SECONDARY_INTERPRETIVE"`. Precedence is:

1. `monitor.CRYSTALLIZATION_CANDIDATE`;
2. `monitor.RECURSIVE_IMITATIVE_ITERATION`;
3. `monitor.UNRESOLVED_FRICTION`;
4. `monitor.VIABLE_INTERACTION`;
5. `monitor.INDETERMINATE`.

The name `CRYSTALLIZATION_STATE` is not used because validation is incomplete.

### 7.1 Crystallization candidate

Requires all:

- persistent `SELF_DOMINANT_CANDIDATE`;
- observed `ANCHOR_NOT_INTEGRATED`;
- observed `RIGID` or `COUNTER_REACTIVE` perturbation response;
- observed `DEGRADING_PARTIAL` or `CRITICAL_PARTIAL` history;
- continued surface operation.

### 7.2 Recursive imitative iteration

Requires all:

- persistent `SELF_DOMINANT_CANDIDATE`;
- observed weak or absent external uptake;
- continued operation.

History and epistemic channels are optional for this annotation.

### 7.3 Unresolved friction

Requires all:

- observed relevant external friction;
- observed partial or rigid response;
- observed non-recovering history.

### 7.4 Viable interaction

Requires all:

- observed functional coupling;
- integrated/no relevant anchor, or anchor channel `NOT_APPLICABLE`;
- adaptive/partial response, or perturbation channel `NOT_APPLICABLE`;
- observed stable/recovering history.

### 7.5 Indeterminate

Mandatory when no rule is satisfied or any channel required by the selected rule
is `UNAVAILABLE` or `INVALID`. Contradictory evidence is not reconciled by
inventing a confidence probability.

Each label records rule ID, required channels, satisfied/failed/unavailable
conditions, calibration reference, evidence window, and categorical confidence.

## 8. Namespace separation

Primary structural observations use `structural_observation`. Secondary conversion
objects use `structural_conversion_differential`. Interpretive annotations use
`monitor.*`. Independent outcomes use `outcome.*`.
The observer and label engine never read outcomes. Empirical validation joins the
namespaces only after structural artifacts are frozen.

## 9. Threshold artifacts

Every frozen threshold set MUST record:

```text
value
calibration_partition
calibration_sha256
estimator
target_operating_point
created_at
study_id
```

No threshold may change after confirmatory outcome exposure. Fixed raw rules such
as `omega_dep > 0.75 => label` are nonconformant.

## 10. Executable entry points

```text
scripts/evaluate_factorial_ablation.py
scripts/evaluate_external_anchor_uptake.py
scripts/evaluate_perturbation_response.py
scripts/evaluate_epistemic_channel.py
scripts/project_window_prama.py
scripts/observe_structural_trajectory.py
scripts/derive_structural_conversion.py
scripts/classify_structural_state.py  # optional, secondary
scripts/validate_structural_artifacts.py
scripts/validate_structural_labels.py
```

All artifact writers validate before atomic replacement. A failed validation
returns nonzero and leaves no partially accepted artifact.

The canonical post-kernel sequence is:

```bash
python scripts/project_window_prama.py ...
python scripts/observe_structural_trajectory.py \
  --input do_v6_sessions.jsonl \
  --contract config/sequor_structural_observer_v9.json \
  --out structural_observation.jsonl ...
python scripts/derive_structural_conversion.py \
  --prama prama_trajectory.jsonl \
  --structural-observations structural_observation.jsonl \
  --out structural_conversion_differential.jsonl \
  --study-id study-1 --producer local --partition exploratory
python scripts/classify_structural_state.py \
  --structural-observation-reference <sha256> ...  # optional
```

## 11. Validation hierarchy

1. Synthetic identifiability of user/self/interaction/none partitions.
2. Filler and scorer robustness.
3. Controlled externally verified perturbations.
4. Matched paired epistemic-channel study.
5. Window-scale PRAMA recertification and comparison with EMA, CUSUM, and
   change-point baselines.
6. Prospective comparison of frozen `monitor.*` annotations with independent
   `outcome.*` events.
7. Prospective incremental-value test of Gamma plus ODCE against kernel coordinates,
   D_O v9 channels and domain baselines, including cost/benefit ablations.

Negative results MUST be retained. If PRAMA adds no prospective value beyond
simple baselines, the PRAMA layer is not justified for this domain.

## 12. Conformance

A build is mechanically conformant when:

- every emitted JSONL row passes `validate_structural_artifacts.py`;
- payload-isolation tests pass;
- canonical and legacy coupling names agree;
- D_O v9 observations are causal and reject provider termination metadata;
- ODCE records preserve cost, return and differential, reject future outcomes and
  provider termination metadata, and never turn missing channels into zero;
- label classification is deterministic;
- each structural label binds an upstream structural observation and declares its
  secondary interpretive role;
- monitor/outcome namespaces do not mix;
- PRAMA projection refuses an unpinned or non-recertified kernel.

Mechanical conformance is not scientific validation.
