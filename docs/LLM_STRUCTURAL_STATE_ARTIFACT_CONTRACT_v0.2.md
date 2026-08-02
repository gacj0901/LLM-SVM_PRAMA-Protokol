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
6. deterministic monitor annotations.

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
contain hashes of prompt and response, exact model identity when available, finish
reason, token count, and:

```json
{
  "response_time_seconds": 1.4281
}
```

`response_time_seconds` is elapsed monotonic wall time from immediately before the
provider request is sent until its complete response is received. It is execution
metadata only. It MUST NOT be used as structural evidence unless a separate study
preregisters that use.

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
| `structural_label` | `structural_labels.jsonl` | deterministic monitor annotation |

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

## 7. Deterministic annotation contract

Exactly one primary monitor annotation is emitted. Precedence is:

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

Monitor annotations use `monitor.*`. Independent outcomes use `outcome.*`.
The label engine never reads outcomes. Empirical validation joins the two
namespaces only after monitor artifacts are frozen.

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
scripts/classify_structural_state.py
scripts/validate_structural_artifacts.py
scripts/validate_structural_labels.py
```

All artifact writers validate before atomic replacement. A failed validation
returns nonzero and leaves no partially accepted artifact.

## 11. Validation hierarchy

1. Synthetic identifiability of user/self/interaction/none partitions.
2. Filler and scorer robustness.
3. Controlled externally verified perturbations.
4. Matched paired epistemic-channel study.
5. Window-scale PRAMA recertification and comparison with EMA, CUSUM, and
   change-point baselines.
6. Prospective comparison of frozen `monitor.*` annotations with independent
   `outcome.*` events.

Negative results MUST be retained. If PRAMA adds no prospective value beyond
simple baselines, the PRAMA layer is not justified for this domain.

## 12. Conformance

A build is mechanically conformant when:

- every emitted JSONL row passes `validate_structural_artifacts.py`;
- payload-isolation tests pass;
- canonical and legacy coupling names agree;
- label classification is deterministic;
- monitor/outcome namespaces do not mix;
- PRAMA projection refuses an unpinned or non-recertified kernel.

Mechanical conformance is not scientific validation.
