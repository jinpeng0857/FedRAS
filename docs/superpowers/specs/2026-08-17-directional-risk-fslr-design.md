# Directional Risk-Calibrated FSLR Design

## 1. Objective

Improve the existing Fisher-guided Sparse Lookahead Restoration (FSLR) without
copying ASR or replacing FSLR with a reset method. Continual accuracy is the
primary optimization target, while Domain-wise accuracy, post-stream Clean
accuracy, and Revisit accuracy are non-regression constraints.

The practical target is a Continual accuracy above 63% on the designated
validation checkpoint. The paper-level claim requires paired improvements over
the locked FedDR baseline on independent training checkpoints; crossing 63% in
one run is not sufficient evidence by itself.

## 2. Problem in the Current Controller

The current dynamic FSLR score is

\[
s_{t,j}=\widehat F_j\widehat{|\theta_{t,j}-\theta_{0,j}|},
\]

where \(F_j\) is source Fisher importance and \(\theta_0\) is the fixed
federated Lookahead anchor. This score identifies important coordinates with
large accumulated drift, but it treats all drift as equally harmful.

In test-time adaptation, some motion away from the anchor is useful target-domain
adaptation, while some motion is persistent error accumulation. Restoring both
types creates a stability-plasticity conflict: aggressive restoration can hurt
Domain-wise and Revisit accuracy, while conservative restoration can leave
Continual error accumulation uncontrolled.

The existing controller also uses discontinuous drift tiers and, in the DLRS +
FSLR path, uses reliability both as a hard authorization gate and as a budget
multiplier. This can suppress restoration on difficult batches and makes the
restoration amount sensitive to threshold crossings.

## 3. Scope and Invariants

The new mechanism is named **Directional Risk-Calibrated FSLR (DRC-FSLR)**.
It changes the FSLR controller, not the restoration identity.

The following invariants are mandatory:

1. Only trainable `BatchNorm2d` affine coordinates are eligible.
2. The fixed `lookahead_anchor` remains the sole restoration target.
3. Offline source Fisher remains the parameter-importance signal.
4. Absolute anchor drift remains a primary selection signal.
5. Restoration remains sparse and soft:

   \[
   \theta_{t,j}\leftarrow
   (1-\lambda)\theta_{t,j}+\lambda\theta_{0,j}.
   \]

6. Selected optimizer-state coordinates are attenuated using the existing
   policy.
7. Domain-wise reset semantics, pre-update evaluation, corruption order,
   checkpoint, and statistics artifact remain unchanged.
8. Legacy modes remain reproducible. New controls default to disabled unless
   the new controller mode is explicitly selected.

The design does not add prediction concentration, a moving knowledge anchor,
layer-depth priors, a source-model forward pass, or a hard reset branch.

## 4. Component A: Directional Risk Accumulation

### 4.1 Observed update

For each adaptation batch, cache the trainable BN-affine values immediately
before `optimizer.step()` and observe them immediately afterward:

\[
u_{t,j}=\theta^{+}_{t,j}-\theta^{-}_{t,j},
\qquad
\delta^{-}_{t,j}=\theta^{-}_{t,j}-\theta_{0,j}.
\]

Using the actual parameter displacement makes the signal valid for both SGD and
Adam, including momentum and adaptive optimizer effects. It requires no extra
forward or backward pass.

If optimization is skipped because the loss or gradient is non-finite, the
snapshot is cleared and the directional-risk state only decays; no artificial
zero update is counted as evidence.

### 4.2 Bounded outward-motion evidence

Define the increase in anchor distance:

\[
e_{t,j}=\max\left(0,
|\delta^{-}_{t,j}+u_{t,j}|-|\delta^{-}_{t,j}|
\right).
\]

Normalize it into a bounded coordinate-wise observation:

\[
o_{t,j}=\frac{e_{t,j}}
{|\delta^{-}_{t,j}|+|u_{t,j}|+\epsilon_j},
\qquad o_{t,j}\in[0,1].
\]

The scale floor is

\[
\epsilon_j=\epsilon\max(1,|\theta_{0,j}|),
\]

so zero or near-zero anchor coordinates remain numerically stable.

Maintain an exponential risk trace:

\[
q_{t,j}=\mu_q q_{t-1,j}+(1-\mu_q)o_{t,j}.
\]

The trace is updated every successful optimizer step and is attenuated on a
restored coordinate by the same factor as the parameter residual:

\[
q_{t,j}\leftarrow(1-\lambda)q_{t,j},
\qquad j\in\mathcal K_t.
\]

This prevents already-corrected risk from remaining permanently high.

### 4.3 Direction-aware ranking

At a scheduled restoration event, rank positive-drift coordinates by

\[
s^{\mathrm{DRC}}_{t,j}=
\widehat F_j\widehat{|\delta_{t,j}|}
(\alpha_q\widehat F_j\widehat{|\delta_{t,j}|}q_{t,j}).
\]

Equivalently,

\[
s^{\mathrm{DRC}}_{t,j}=
\widehat F_j\widehat{|\delta_{t,j}|}(1+\alpha_q q_{t,j}).
\]

Thus Fisher x drift remains the base score. Directional risk can reprioritize
only coordinates that are already important and actually drifted; it cannot
make a zero-Fisher or zero-drift coordinate eligible.

Setting \(\alpha_q=0\) exactly recovers the current Fisher x drift ranking.

## 5. Component B: Continuous Hysteretic Budget

### 5.1 Global directional risk

Aggregate coordinate risk using the current Fisher-drift mass:

\[
w_{t,j}=\widehat F_j\widehat{|\delta_{t,j}|},
\qquad
R_t=\frac{\sum_j w_{t,j}q_{t,j}}
{\sum_j w_{t,j}+\epsilon}.
\]

Because \(q_{t,j}\in[0,1]\), \(R_t\in[0,1]\). The signal measures the share
of currently important drift that is also being persistently pushed outward.

### 5.2 Hysteresis state

Maintain a controller state \(H_t\in\{0,1\}\):

\[
H_t=1 \quad\text{if}\quad R_t\ge\tau_{\mathrm{on}},
\]

\[
H_t=0 \quad\text{if}\quad R_t\le\tau_{\mathrm{off}},
\]

and otherwise keep the previous state, with
\(0\le\tau_{\mathrm{off}}<\tau_{\mathrm{on}}\le1\).

Hysteresis prevents restoration from repeatedly switching on and off around one
threshold. It does not create immediate off-schedule restoration; the existing
restoration interval remains unchanged.

### 5.3 Continuous drift budget

Replace the stepwise drift tiers in the new mode with a bounded piecewise-linear
base ratio \(\rho_D(D_t)\), interpolated through these fixed knots:

| Global drift \(D_t\) | Base ratio |
|---:|---:|
| 0.00 | 0.00 |
| 0.10 | 0.00 |
| 0.20 | 0.02 |
| 0.35 | 0.05 |
| 0.50 | 0.10 |

Values below 0.10 map to zero and values above 0.50 are capped at 0.10.

Directional risk adds a bounded urgency term only while hysteresis is active:

\[
\rho_t=\operatorname{clip}\left(
\rho_D(D_t)+H_t\gamma_R
\frac{[R_t-\tau_{\mathrm{off}}]_+}
{1-\tau_{\mathrm{off}}},
0,0.10
\right).
\]

The maximum restoration ratio therefore remains 10%. Reliability is retained
for diagnostics and DLRS sample weighting but is removed as an FSLR hard gate
and budget multiplier in the new controller mode.

Setting \(\gamma_R=0\) disables risk-budget modulation. Legacy
`drift_tiered` behavior remains available as a separate mode.

## 6. Runtime Data Flow

For every test batch:

1. Generate the pre-update prediction used for evaluation.
2. Compute the existing DLRS adaptation loss and gradients.
3. Capture the current BN-affine coordinates before `optimizer.step()`.
4. Apply the optimizer update.
5. Observe the actual BN-affine displacement and update \(q_t\).
6. At the existing restoration interval:
   - compute global drift \(D_t\);
   - compute directional risk \(R_t\);
   - update hysteresis state \(H_t\);
   - compute continuous budget \(\rho_t\);
   - rank coordinates by \(s^{\mathrm{DRC}}_{t,j}\);
   - restore top-K positive-score coordinates toward `lookahead_anchor`;
   - attenuate matching optimizer state and risk trace.
7. Record diagnostics without altering frozen post-stream evaluation.

## 7. Interfaces and Configuration

`PeriodicSparseRestorer` receives one new explicit controller mode rather than
silently changing `drift_tiered`:

- `budget_mode="directional_risk"`
- `directional_risk_weight` for \(\alpha_q\)
- `directional_risk_ema_decay` for \(\mu_q\)
- `directional_risk_on_threshold` for \(\tau_{\mathrm{on}}\)
- `directional_risk_off_threshold` for \(\tau_{\mathrm{off}}\)
- `directional_risk_budget_gain` for \(\gamma_R\)

New lifecycle methods encapsulate snapshots inside the restorer:

- `capture_pre_update()` before a valid optimizer step;
- `observe_post_update(updated: bool)` after the attempted step.

Calling `observe_post_update(updated=False)` clears the snapshot and decays the
risk trace. Calling it without a matching snapshot is an error in the new mode
and a no-op in legacy modes.

All new numerical arguments are finite and range-checked. The default CLI mode
remains the existing locked FSLR behavior. The new mode is opt-in, so old JSONs
and commands remain reproducible.

## 8. Diagnostics

The result JSON must add:

- current and mean global directional risk;
- hysteresis active-event count and state transitions;
- mean/max coordinate risk trace;
- outward-moving coordinate count per observation;
- base drift ratio, risk increment, and final ratio per event;
- directionally boosted selected-coordinate count;
- selected outward-risk mass ratio;
- per-layer drift energy, risk mass, selection count, and restoration-induced
  drift reduction;
- number of successful, skipped, and non-finite update observations.

Existing restoration, drift, selection-Jaccard, unique-coverage, and optimizer
state diagnostics remain intact.

## 9. Validation Strategy

### 9.1 Static and unit validation

Tests must verify:

1. Outward updates increase risk; inward updates do not.
2. Risk remains finite and in [0, 1] for zero anchors and zero updates.
3. Restoring a coordinate attenuates its risk trace.
4. `directional_risk_weight=0` preserves Fisher x drift ordering.
5. Hysteresis enters and exits only at its two thresholds.
6. The continuous budget is monotonic in drift and active risk and never
   exceeds 10%.
7. Reliability does not gate or scale restoration in the new mode.
8. Legacy fixed and drift-tiered tests remain unchanged and pass.
9. SGD and Adam actual-displacement observations are both accepted.
10. Frozen evaluation does not mutate risk, parameters, optimizer state, or
    counters.

### 9.2 Controlled experimental sequence

Use one designated validation training checkpoint for mechanism selection and
keep checkpoint, `lookahead_anchor`, source statistics, corruption order,
severity, optimizer, learning rate, batch size, and DLRS configuration fixed.

Run in this order:

1. **Diagnostic-only:** collect directional-risk traces with
   \(\alpha_q=0\) and \(\gamma_R=0\).
2. **A only:** direction-aware ranking with the locked legacy budget.
3. **B only:** continuous hysteretic budget with \(\alpha_q=0\).
4. **A+B:** full DRC-FSLR.
5. **Confirmation:** freeze the selected configuration and evaluate it on the
   two remaining independent training checkpoints.

Thresholds are selected from the diagnostic validation trace, then frozen:

- \(\tau_{\mathrm{on}}\) is chosen from the 75th percentile of scheduled-event
  \(R_t\);
- \(\tau_{\mathrm{off}}\) is chosen from the 50th percentile;
- if the percentiles are equal, use the nearest distinct lower observed value
  for \(\tau_{\mathrm{off}}\); if none exists, hysteresis is disabled and the
  design is rejected for that trace.

The limited candidate grids are:

- \(\mu_q\in\{0.90,0.99\}\);
- \(\alpha_q\in\{0.5,1.0\}\);
- \(\gamma_R\in\{0.02,0.05\}\).

No other FSLR or DLRS parameter changes are allowed during this selection.

## 10. Acceptance Criteria

The candidate is retained only if all four metrics improve in the paired mean
over the locked FedDR baseline across three independent training checkpoints:

- Domain-wise mean delta > 0;
- Continual mean delta > 0 and validation Continual accuracy > 63%;
- post-stream Clean mean delta > 0;
- Revisit mean delta > 0.

Continual is the primary metric. In addition:

- at least two of three checkpoints must have a positive Continual delta;
- final-cycle Continual accuracy and cycle degradation must not worsen in the
  paired mean;
- improvements must be reported with per-checkpoint values, mean, standard
  deviation, and paired deltas;
- a single-checkpoint crossing of 63% is a candidate result, not a paper claim.

If no candidate satisfies all four constraints, the mechanism is not merged
into the paper method. The failure is reported by component and metric rather
than selecting a configuration solely for crossing 63%.

## 11. Paper Positioning

The central claim is not that all anchor drift is harmful. The paper should
state:

> FSLR distinguishes persistent outward parameter motion from generic target
> adaptation. It restores coordinates only when source importance, accumulated
> anchor drift, and directional error-accumulation risk coincide, while a
> continuous hysteretic controller balances restoration urgency and adaptation
> plasticity.

The novelty is the parameter-space feedback mechanism: source Fisher answers
which coordinates encode important source knowledge, absolute drift answers how
far they have moved, and actual update direction answers whether they continue
to move away. This differs from prediction-concentration reset triggers,
moving-anchor recovery, and fixed layer-depth priors.

Claims about accuracy, superiority, or robustness are withheld until the full
three-checkpoint confirmation is complete.
