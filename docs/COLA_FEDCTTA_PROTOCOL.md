# CoLA baseline under the FEDEATA protocol

This document fixes the boundary between the CoLA paper settings and the
shared FEDEATA evaluation protocol. It is the protocol used by
`scripts/run_cola_unified.py`.

## Baseline definition

The baseline is **ETA+CoLA**, the lifelong principal-agent variant reported by
the CoLA paper. The FedACG checkpoint is the deployment model, and the
`lookahead_anchor` state is loaded before test-time adaptation. This is a
single-stream adaptation baseline for FedCTTA: it preserves CoLA's domain
vector memory and shift-triggered vector saving, but does not pretend that a
live multi-device exchange is available inside the FedCTTA test stream.

The official CoLA repository is ViT-oriented. This project therefore uses the
paper's ResNet formulation: every affine BatchNorm scale/shift tensor is
represented as an anchor tensor plus the CoLA domain-vector mixture and the
current epsilon update. No labels, Fisher regularizer, FedDR restoration, or
FSLR event trigger is used by this baseline.

## Shared main-test protocol

- Dataset: Tiny-ImageNet-C, severity 5, all 15 corruption types.
- Backbone/checkpoint: the official FedACG-trained ResNet-18 checkpoint with
  `lookahead_anchor`.
- Domain size: 10,000 images per corruption.
- Batch size: 128.
- Within-domain order: sorted class/file list followed by the identical
  `np.random.RandomState(0).permutation` for every method and every cycle.
- Prediction timing: pre-update prediction; the current batch is adapted after
  its prediction is recorded.
- Continual protocol: the 15 corruptions in the main order, repeated for three
  cycles with model/optimizer/domain-vector state carried across boundaries.
- Post-stream checks: one clean evaluation and the four fixed revisit domains,
  after adaptation is frozen.
- Random seed: 0 for the evaluation run. The FedACG training checkpoint itself
  remains the user-provided seed-42 run.

## CoLA-specific settings

These values follow the official ETA+CoLA implementation and the paper's
lifelong settings:

- CoLA weight margin `w_m = 0.01`.
- First-stem Gaussian shift threshold `z = 0.1`.
- Domain-statistics EMA factor `lambda = 0.2`.
- Domain-vector cap `32`.
- Current epsilon update: SGD, learning rate `0.001`, momentum `0.9`, weight
  decay `0.4`.
- Domain-mixture `alpha` and temperature scale: AdamW, learning rate `0.1`,
  alpha weight decay `0.1`; the temperature scale has no weight decay.
- ETA entropy margin: `0.4 * ln(C)`; with Tiny-ImageNet, `C = 200`.
- ETA redundancy margin: `d_margin = 0.05`.
- All affine BatchNorm tensors are trainable; their running statistics are not
  used as a learned deployment state during TTA.

The follower-only temperature `T_f = 5` belongs to CoLA's separate follower
agent entrypoint. It is not injected into the ETA+CoLA principal-agent
baseline, because the FedCTTA stream has no independently saved follower
vectors to replay.

## Reproducibility outputs

Each run records the resolved configuration, a hash for each shuffled domain
order, and `cola_validated.json`. A formal run is accepted only when it has 15
domain-wise rows, 45 continual rows, 10,000 samples in every corruption row,
10,000 clean samples, and four complete revisit rows.
