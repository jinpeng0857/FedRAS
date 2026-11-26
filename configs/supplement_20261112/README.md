# FedRAS supplementary experiment configuration

Design date: 2026-11-12  
Status: configuration design complete; full execution is blocked by missing checkpoints and the implementation gates below. No experiment was launched while writing this package.

This package translates the supplied E1–E5 plan into a versioned protocol and a machine-readable arm matrix. It does not replace or edit configs/revision_20261102.

experiment_design.json is a design specification, not a command manifest accepted directly by run_case.py. Any new hooks listed below are experimental controls only; the public FedDR/FedRAS path should remain unchanged.

## Files

- experiment_design.json — shared protocol, asset binding, arm definitions, status, and run accounting.
- run_order.csv — seeded, checkpoint-blocked execution order with the E2 reference dependency preserved.
- README.md — rationale, interpretation limits, and execution gates.

## Asset binding and immediate limitation

Use the revised evaluator as the code root and FEDEATA as the asset root:

| Role | Path |
|---|---|
| Code root | D:/dp1/FEDEATA-revision-20261018 |
| Runner | scripts/revision_20261018/run_case.py |
| Asset root | D:/dp1/FEDEATA |
| CIFAR-100-C | data/CIFAR-100-C |
| Clean CIFAR-100 | data/cifar100/cifar-100-python |
| Output root | output/fedras_supplement_20261112 |

The run1 FedACG ResNet-18 checkpoint is present, safe-loadable, and contains the required lookahead_anchor. It is configured for CIFAR-100, 100 classes, alpha 0.3, and FedACG. Its SHA-256 is recorded in experiment_design.json. The matching ordinary and EATA statistics are present. The run2 and run3 statistics are present, but their configured model checkpoints are missing. The local torchvision ResNet-18 cache is not a substitute for these FedACG checkpoints.

Therefore, the target design is paired over three stored checkpoints, but the local assets currently permit at most a one-checkpoint pilot. Treat pilot results as descriptive and do not present them as n=3 evidence.

## Locked shared evaluation protocol

| Setting | Value |
|---|---|
| Dataset and corruption | CIFAR-100-C, severity 5, all 15 corruption domains |
| Stream | 3 cycles, 15 domains per cycle, 45 blocks total |
| Batch size | 128 |
| Samples | Full stream; max_batches = -1 |
| Prediction | Predict-then-update |
| Evaluation seed | 0 for every treatment and checkpoint |
| Optimizer | SGD, learning rate 0.001, momentum 0.9, weight decay 0 |
| Adaptation parameters | BatchNorm affine parameters outside layer4 |
| Regional loss | ReCAP regional entropy |
| Anchor | FedACG lookahead_anchor |
| Within-arm state | Model and optimizer persist over the full 45-block stream |
| Between-arm state | Rebuild the model, optimizer, adapter, and restorer from the same checkpoint |
| Post-stream measures | Frozen Clean and Revisit evaluations |
| Revisit domains | gaussian_noise, shot_noise, impulse_noise, defocus_blur |
| Labels | Metrics and offline diagnostics only; never select or weight samples during adaptation |

The machine-readable configuration records the retained public FedDR settings explicitly. It also records the code/plan discrepancy in restoration ranking: the checked runtime uses fisher_drift for the public FedDR entrypoint, while the supplied plan describes normalized displacement. Do not relabel one as the other. Resolve this against the final manuscript and released method before calling the configuration paper-final.

The exact current public RGA and DSR values, including event thresholds, risk weights, and layer quotas, are recorded in experiment_design.json. For non-public ablation entrypoints, pass those values explicitly where they apply; do not rely on argparse defaults to reproduce the full method.

In the checked runtime, the public method name is feddr and it resolves to dlrs_fslr. The design uses FedRAS in paper-facing labels and records the exact runtime entrypoint for commands.

## E1 — Reliability signal and update-scale controls

The common E1 loss is per-sample ReCAP regional entropy at regional scale 1.2. DSR is disabled for all E1 arms. The primary comparison uses a per-batch count schedule captured from the anchor-disagreement arm:

1. On the anchor-disagreement reference stream, record k_t, the number of samples whose pre-update prediction differs from the frozen anchor prediction in each batch.
2. Entropy, confidence, and random controls use exactly k_t samples in that same batch.
3. Each selected sample receives weight 0.7; all other samples receive weight 1.0.
4. Stable ties use original sample index. The selector never reads labels.

For the uniform matched-scale control, set every sample weight to the batch mean of the anchor-disagreement weights. This matches the scalar mean weight before backpropagation; it does not assert identical gradients or parameter updates.

| Arm | Exact intervention | Current support |
|---|---|---|
| E1-A Uniform | All sample weights are 1 | Supported as the Base/ReCAP arm; identical run can serve E3-A after effective-config identity is checked |
| E1-B Entropy | Rank normalized entropy H(p)/log(100) descending; select k_t | Blocked: current entropy mode is continuous, not the specified matched-count binary mask |
| E1-C Confidence | Rank maximum softmax confidence ascending; select k_t | Blocked: no exact top-k confidence-mask branch |
| E1-D Random | Sample k_t without replacement; predeclared seed 20261112 plus checkpoint/stream-batch offsets | Blocked: no exact sample-weight random-mask branch |
| E1-E Anchor disagreement | Weight 0.7 iff current and frozen-anchor argmax differ | Supported through the anchor_only RGA ablation with beta 0.3; use DSR-free dlrs |
| E1-F Uniform matched scale | All weights equal mean(anchor-disagreement weights) for that batch | Blocked: needs a diagnostic loss-scaling control and per-batch logging |
| E1-G Normalized anchor disagreement | E1-E weights; weighted ReCAP loss divided by sum of weights | Supported through anchor_only, beta 0.3, and normalize_by_weight; verify the effective normalizer in saved config |

E1-E and E1-G test a binary anchor-disagreement control. They are not interchangeable with the full public FedDR RGA, which uses the composite sample and regional weighting path. Do not reuse E1-E/G as the E3 RGA-only result.

Report Cycle 1–3, Clean, Revisit, error-weight fraction/reduction, pre-update disagreement rate, actual mean weight, gradient norm, and the k_t schedule. Accuracy is not used to choose a threshold or change k_t.

## E2 — Matched restoration-policy comparison

Record the complete periodic/event-driven restoration plan from E3-D separately for each checkpoint. The plan binds the checkpoint, anchor, Fisher statistics, evaluator source, stream identity, and batch count. Replay the plan at the same batches and preserve each event's selected-coordinate count, per-tensor quota, restoration strength, and nominal dose.

| Arm | Intervention | Current support |
|---|---|---|
| E2-A No restoration | Same full RGA stream as E3-B; DSR absent | Reuse E3-B only after exact config and stream identity match |
| E2-B Random | Replay reference events/quotas/strength; select random coordinates from the reference eligible pool | Supported by revision runner ranking=random |
| E2-C Low displacement | Replay reference plan; within each tensor quota choose the smallest strictly positive normalized displacement, stable parameter-index ties | Blocked: low_drift ranking is not implemented |
| E2-D High displacement | Replay reference plan; within each tensor quota choose largest normalized displacement | Supported by revision runner ranking=drift_only |
| E2-Reference Public default | Public FedDR run that generated the reference plan | Reuse E3-D; current runtime uses fisher_drift ranking |

Random and displacement arms share the reference's event times, per-tensor quotas, restoration strengths, and eligible coordinate pool. The current replay tool already binds replay to checkpoint, anchor, Fisher statistics, sources, and stream. A replay that does not consume every event and every batch is invalid.

The supplied plan identifies high displacement with the public default. The checked source does not: the public default is Fisher×displacement. Keep E2-D as a separate drift-only arm and retain E2-Reference as the actual public method. This distinction adds three trajectories at the three-checkpoint target. Report actual coordinate count, layer quotas, strength, nominal dose, and drift even when a matched replay fails.

## E3 — 2×2 ablation of the deployed configuration

| Arm | Public entrypoint | RGA | DSR | Status |
|---|---|---:|---:|---|
| E3-A Base | base | Off | Off | Supported |
| E3-B RGA only | dlrs with the full public FedDR RGA settings | On | Off | Supported |
| E3-C DSR only | base_fslr with the public DSR settings and an explicit anchor-disagreement input | Off | On | Blocked until the adapter passes the planned anchor signal to DSR |
| E3-D Full | feddr | On | On | Supported; this is the current public runtime configuration |

Keep learning rate, optimizer, BatchNorm scope, ReCAP loss, anchor, stream order, and test seed fixed. For E3-C, the intended treatment must compute anchor disagreement for DSR risk without applying RGA sample weights. The current base_fslr adapter calls the restorer without this signal, so a naive base_fslr run is not the specified DSR-only arm.

For each checkpoint calculate the paired effects:

- RGA effect without DSR: E3-B minus E3-A.
- DSR effect without RGA: E3-C minus E3-A.
- Full method effect: E3-D minus E3-A.
- Factorial interaction: (E3-D − E3-C) − (E3-B − E3-A).

Use checkpoint-level paired differences. With three checkpoints, report mean, sample SD, and the prespecified exploratory paired interval; do not treat batches, domains, or cycles as independent replicates.

## E4 — Initialization provenance and prospective independent runs

First audit each existing checkpoint's training config, initial model, client-partition seed, client-sampling seed, optimizer seed, training log, and stored round. If a value cannot be recovered, record it as unknown. Existing run1/run2/run3 names do not prove independent FedACG training seeds.

If independence cannot be established and compute permits, train two additional FedACG models using predeclared seeds 20261112 and 20261113. Before training, bind the exact data split, FedACG implementation commit, all training hyperparameters, client sampling schedule, and checkpoint-selection rule. These proposed seeds are a prospective design choice; they do not repair unknown provenance for historical checkpoints.

On new checkpoints, run only Source, ReCAP, ASR, and full FedRAS under the same checkpoint-paired Domain-wise and Continual protocols. Training commands and hyperparameters remain a required input because the attached plan does not specify a complete FedACG retraining recipe.

## E5 — Additional dataset

Deferred. CIFAR-100-C and Tiny-ImageNet-C are already the planned coverage. Office-Home/PACS would require a compatible classifier, federated training, and a defensible continual stream; do not start E5 until E1–E4 are resolved.

## Run accounting and order

| Stage | Three-checkpoint trajectories before validated historical reuse |
|---|---:|
| E1 | 21 |
| E3 | 12 |
| E2 | 12 |
| Exact overlaps | −6: E1-A/E3-A and E2-A/E3-B |
| E2 public reference | Reuse E3-D |
| Unique core total | 39 |

The supplied estimate of 33 assumed that binary anchor-only RGA equals the full public RGA-only arm and that drift-only equals the public restoration ranking. The current implementation confirms neither assumption, so this package keeps those arms separate. If an existing result is reused, first verify checkpoint/statistics hashes, effective configuration, complete 45-block stream, sample order, prediction timing, and frozen post-stream evaluations.

With the files currently found, run1-only pilots can exercise supported E1-A/E/G, E2-B/D, and E3-A/B/D. E1-B/C/D/F, E2-C, and E3-C need the listed runtime hooks. Any run1-only result remains n=1.

run_order.csv contains a seeded random topological order (seed 20261112), blocked by checkpoint. It keeps each E2 replay after that checkpoint's E3-D reference. The run2/run3 rows are planning rows marked as blocked by missing model weights; they are not launchable entries.

Execution order:

1. Resolve the public-ranking discrepancy against the final manuscript/method source.
2. Recover or obtain run2 and run3 model checkpoints; validate all checkpoint/statistics pairs.
3. Implement and review the blocked E1/E2/E3 controls without changing the public FedDR path.
4. Run per-checkpoint preflight and save the effective configuration/source hashes.
5. Record the E2 reference schedule from E3-D, then run matched replay candidates.
6. Analyze paired endpoint differences and apply the acceptance gates below.

## Acceptance and output contract

Every formal row must have:

- the checkpoint and statistics SHA-256, source revision, config hash, and random seeds;
- all 45 corruption blocks, no truncation, finite losses and gradients;
- the actual effective learning rate, optimizer, BatchNorm scope, RGA mode, and DSR mode;
- the complete saved corruption/sample-order identity;
- frozen Clean and Revisit evaluations after the stream;
- actual weight means and/or DSR event count, coordinate count, layer quotas, strength, and dose;
- a paired row for every checkpoint with no imputation for missing pairs.

The execution status is process completion only. It is not semantic acceptance and does not force an improvement. Do not make an accuracy superiority claim unless the prespecified paired results and endpoint thresholds support it.
