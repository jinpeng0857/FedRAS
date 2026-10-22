# Figure 2 QA notes

## Figure contract

- Core conclusion: FedDR couples DLRS soft reliability adaptation with periodic FSLR restoration so that reliable target evidence updates BN parameters while anchor-guided recovery limits long-stream drift.
- Archetype: schematic-led horizontal workflow.
- Target: Knowledge-Based Systems double-column method figure.
- Final size: 180 mm × 82 mm.
- Structural reference level: style-only inheritance. The provided reference contributes the left-input, central-model, parallel-branch, and bidirectional-flow grammar; all scientific entities and transitions were rebuilt for FedDR.
- Evidence level: conceptual implementation diagram; it contains no experimental measurements or performance claims.

## Code-aligned flow

1. Federated pre-training provides the deployment anchor θ₀ and offline Fisher statistics.
2. The client receives an unlabeled shifted batch Bₜ and performs pre-update prediction.
3. DLRS computes continuous sample reliability rᵢ, reliability-controlled region scale, joint soft weight, and regional weighted loss L.
4. All samples remain in the batch; the weighted loss updates trainable BN affine parameters through the shown dashed gradient path.
5. The period decision is evaluated after each adaptation batch. Every 10 batches, FSLR measures drift dₜ, chooses a drift-tiered budget, ranks BN coordinates by Fisher × drift, and softly restores selected coordinates toward θ₀.
6. The resulting state is reused for later target batches. After the complete stream, clean and revisit domains are evaluated with adaptation frozen.

## Visual encoding

- Red arrows: federated source, deployment-anchor, Fisher-statistics, and anchor-restoration flow.
- Blue arrows: client target-test data and continual iteration.
- Black dashed arrows: gradient flow.
- Dashed rounded frames: major DLRS and FSLR modules.
- Diamond: periodic decision node.
- White/light-gray boxes, no gradients, shadows, transparency, texture, or 3D effects.
- Direct labels and spatially distinct paths retain meaning in grayscale printing.

## Export checks

- SVG: editable vector geometry with live text.
- PDF: one-page vector export at approximately 180 mm × 82 mm with searchable text.
- PNG and TIFF: 600 dpi at final physical size.
- Visual review: arrows, labels, module boundaries, decision text, and legend remain inside the canvas; no overlap changes the scientific meaning.

## Suggested caption

**Fig. 2. Overall framework of FedDR for federated continual test-time adaptation.** Federated pre-training supplies the deployment anchor θ₀ and offline Fisher statistics, while an unlabeled shifted target stream provides consecutive batches Bₜ. FedDR follows a predict-then-update protocol. The current model first produces pre-update predictions, after which DLRS estimates sample reliability rᵢ, controls the regional scale, and forms a joint soft-weighted regional loss L to update trainable BN affine parameters without hard sample rejection. Every 10 batches, FSLR measures the normalized drift dₜ from θ₀, assigns a drift-tiered restoration budget, ranks BN coordinates using Fisher and current drift, and softly restores selected coordinates toward the anchor. DLRS provides reliable adaptation evidence for recovery, whereas FSLR stabilizes subsequent predictions, forming a mutually reinforcing continual loop. Clean and revisit domains are evaluated with the final adapter state frozen after the stream.
