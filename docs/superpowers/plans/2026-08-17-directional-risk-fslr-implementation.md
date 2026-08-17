# Directional Risk-Calibrated FSLR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Implement the approved Directional Risk-Calibrated FSLR controller on the remote FEDEATA checkout, preserving legacy FSLR behavior while adding direction-aware ranking and continuous hysteretic budgets.

**Architecture:** `PeriodicSparseRestorer` owns update snapshots, bounded directional-risk state, ranking, hysteresis, budget computation, restoration, and diagnostics. `DLRSAdapter` only brackets the existing optimizer step with `capture_pre_update()` and `observe_post_update(updated=...)`. The new controller does not use reliability as a restoration gate or budget multiplier.

**Tech Stack:** Python 3.11, PyTorch, `unittest`, argparse, remote `vllm` conda environment.

## Global Constraints

- Keep `lookahead_anchor` as the sole restoration target.
- Keep trainable `BatchNorm2d` affine scope, Fisher importance, absolute anchor drift, sparse selection, soft interpolation, optimizer-state attenuation, and pre-update evaluation.
- Keep legacy `fixed` and `drift_tiered` modes behaviorally unchanged.
- Do not add ASR concentration, a moving knowledge anchor, layer-depth priors, or an extra source-model forward pass.
- New controller maximum restoration ratio is 10%.
- Do not delete, reset, clean, or overwrite unrelated remote dirty/untracked files.
- Validate focused tests and syntax before any long experiment.
- A single checkpoint crossing 63% is candidate evidence only; final claims require three independent training checkpoints and all four metrics.

---

### Task 1: Upload design and implementation plan

**Files:**
- Create remote: `docs/superpowers/specs/2026-08-17-directional-risk-fslr-design.md`
- Create remote: `docs/superpowers/plans/2026-08-17-directional-risk-fslr-implementation.md`

- [ ] Record remote status and hashes before edits:

```bash
git status --short
sha256sum sparse_restore.py main_test.py tests/test_sparse_restore.py
```

- [ ] Create remote documentation directories, upload only the two documents, and verify local/remote SHA-256 equality.
- [ ] If remote policy permits, commit only the two documentation files; otherwise leave them uncommitted and report that boundary.

### Task 2: Add bounded directional-risk observation

**Files:**
- Modify remote: `sparse_restore.py`
- Test remote: `tests/test_sparse_restore.py`

**Interfaces:**
- Extend `PeriodicSparseRestorer` with opt-in `budget_mode="directional_risk"` and parameters `directional_risk_weight`, `directional_risk_ema_decay`, `directional_risk_on_threshold`, `directional_risk_off_threshold`, and `directional_risk_budget_gain`.
- Add `capture_pre_update() -> None` and `observe_post_update(updated: bool) -> None`.
- Add `_current_directional_risk() -> torch.Tensor` returning a flat CPU tensor.

- [ ] Add failing tests for outward motion, inward motion, zero-anchor stability, skipped updates, and unmatched snapshots.
- [ ] Run `python -m unittest tests.test_sparse_restore` and confirm only new tests fail.
- [ ] Validate finite ranges: nonnegative weight/gain, decay in `[0,1)`, and `0 <= off < on <= 1`.
- [ ] Implement a detached pre-update flat snapshot for the trainable BN affine coordinates.
- [ ] After a successful step, compute:

```python
before_delta = pre_update - flat_anchor
after_delta = current - flat_anchor
outward = (after_delta.abs() - before_delta.abs()).clamp_min(0.0)
scale = before_delta.abs() + (current - pre_update).abs()
eps = torch.finfo(torch.float32).eps * torch.maximum(
    torch.ones_like(scale), flat_anchor.abs()
)
observation = (outward / (scale + eps)).clamp(0.0, 1.0)
risk = decay * risk + (1.0 - decay) * observation.cpu()
```

- [ ] For `updated=False`, decay risk without treating the skipped update as outward motion; clear snapshots after every observation. Run the focused tests and commit only this isolated change if remote commit policy permits.

### Task 3: Add risk-aware ranking and continuous hysteretic budget

**Files:**
- Modify remote: `sparse_restore.py`
- Test remote: `tests/test_sparse_restore.py`

**Interfaces:**
- Add `_directional_global_risk(absolute_drift) -> float`.
- Add `_continuous_drift_ratio(drift) -> float` using knots `(0.10,0.00)`, `(0.20,0.02)`, `(0.35,0.05)`, `(0.50,0.10)`, capped at `0.10`.
- Add `_directional_ratio(drift, risk) -> float` with hysteresis and a 10% cap.

- [ ] Add failing tests for monotonic interpolation, 10% cap, ranking boost, and hysteresis activation/persistence/deactivation.
- [ ] Run focused tests and confirm the new tests fail before implementation.
- [ ] Compute `w = normalized_fisher * normalized_drift` and `R = sum(w*risk)/(sum(w)+eps)`, yielding `R` in `[0,1]`.
- [ ] Update hysteresis at scheduled restore events: enter at `R >= on`, exit at `R <= off`, otherwise keep the previous state.
- [ ] Compute:

```python
risk_increment = (
    budget_gain * max(0.0, R - off) / max(1e-12, 1.0 - off)
    if active else 0.0
)
ratio = min(0.10, continuous_drift_ratio(D) + risk_increment)
```

- [ ] In the new mode rank with `normalized_fisher * normalized_drift * (1 + weight * risk)`. With `weight=0`, recover the existing Fisher×drift ordering. Do not consult reliability for authorization or budget scaling in the new mode.
- [ ] Run focused tests and commit the isolated restorer change if permitted.

### Task 4: Integrate lifecycle hooks and CLI

**Files:**
- Modify remote: `main_test.py` argument parser, restorer construction, `DLRSAdapter.adapt`, logging, and result metadata
- Test remote: existing adapter tests plus `tests/test_sparse_restore.py`

- [ ] Add the five directional-risk CLI arguments and add `directional_risk` to `--restore_budget_mode`; keep default `drift_tiered`.
- [ ] Pass exact constructor names and values into `PeriodicSparseRestorer`; do not let `apply_official_baseline_config` overwrite them.
- [ ] In the FSLR `DLRSAdapter.adapt()` path, call `capture_pre_update()` immediately before every valid `optimizer.step()` and `observe_post_update(updated=True)` immediately after it. On non-finite/skip branches call `observe_post_update(updated=False)` only when a snapshot exists.
- [ ] Ensure exactly one observation per attempted FSLR optimizer update and no hooks for ASR/non-FSLR adapters.
- [ ] Record selection, budget, risk parameters, and the policy string `diagnostic_only_directional_risk_controls_budget`; do not claim reliability authorization in directional mode.
- [ ] Run:

```bash
python -m py_compile sparse_restore.py main_test.py tests/test_sparse_restore.py
python -m unittest tests.test_sparse_restore
```

### Task 5: Add diagnostics and frozen-evaluation checks

**Files:**
- Modify remote: `sparse_restore.py` stats property and `main_test.py` logging/result metadata
- Test remote: existing frozen-evaluation tests or a focused new invariant test

- [ ] Add JSON-safe fields for current/mean directional risk, active state, transitions, outward observations, risk budget increment, global risk, and directionally boosted selections. Preserve all existing drift/selection/optimizer diagnostics.
- [ ] Add per-layer drift energy, risk mass, selected count, and restoration-induced drift reduction.
- [ ] Snapshot parameters, optimizer state, restorer risk/counters, and adapter counters around frozen evaluation; assert no mutation and preserve existing `parameters_unchanged`/`adapter_counters_unchanged` semantics.
- [ ] Run `python -m unittest discover -s tests -p 'test*.py'` and `python main_test.py --help | grep directional_risk`.

### Task 6: Remote handoff and controlled validation boundary

**Files:**
- Remote modified: `sparse_restore.py`, `main_test.py`, `tests/test_sparse_restore.py`, documentation files

- [ ] Review `git diff -- sparse_restore.py main_test.py tests/test_sparse_restore.py`; verify unrelated pre-existing status is intact.
- [ ] Record post-change SHA-256 hashes and the exact opt-in command template with checkpoint path, `--anchor_key lookahead_anchor`, statistics path, and test commands.
- [ ] Stop before the three-checkpoint benchmark unless separately requested. Report design upload, code changes, tests, smoke status, and benchmark status separately.
