# FedDR macro framework figure: QA notes

## Figure contract

- Core conclusion: FedDR processes a continual unlabeled target stream through prediction, DLRS reliability weighting, and FSLR anchor-guided restoration, producing an adapted model for clean and historical-revisit evaluation.
- Archetype: schematic-led horizontal workflow.
- Target: Knowledge-Based Systems double-column method figure.
- Final size: 180 mm × 72 mm.
- Reference use: style-only inheritance from the provided horizontal framework figure. Its visual grammar was reused, while its scientific content, labels, modules, and causal relations were not reused.
- Evidence level: conceptual method diagram. No experimental result, statistic, threshold, or performance claim appears in the figure.

## Included macro flow

1. Federated Source Data → Federated Pre-training → Anchor Model θ₀.
2. Client Test Stream supplies consecutive unlabeled target batches Bₜ.
3. Bₜ → Feature Extractor → Predict → prediction ŷₜ.
4. Prediction → DLRS — Sample Reliability Weighting → FSLR — Drift-aware Sparse Restoration.
5. DLRS and FSLR are linked bidirectionally to show mutual stabilization and iterative cooperation.
6. The resulting Adapted Model is evaluated on the clean domain and historical revisit domain.
7. A blue feedback path returns the adapted state to the feature-extraction side for the next target batch.

## Deliberately excluded

- No period decision node or trigger interval.
- No reliability components, regional-loss construction, Fisher ranking, restoration budget, threshold, percentage, or optimizer detail.
- No loss equation or restoration equation.
- No experiment values, accuracy, drift result, or superiority claim.

## Visual encoding

- Red: federated source, anchor deployment, and anchor-guided restoration.
- Blue: client target batches, forward target flow, and continual feedback.
- Black dashed: gradient update.
- Rounded rectangles: functional modules; dashed rounded rectangles emphasize the two proposed core modules.
- White background and light-neutral module fill; no gradient, shadow, transparency, texture, cartoon, or 3D effect.
- Color is redundant with spatial paths, line styles, direct labels, and arrow direction for grayscale readability.

## Suggested caption

**Fig. 2. Overall framework of FedDR for federated continual test-time adaptation.** Federated source data are used to obtain the deployment anchor model θ₀, which is delivered to the client and provides the reference for restoration. Consecutive unlabeled target batches Bₜ are processed by the feature extractor and prediction module to obtain ŷₜ. DLRS suppresses unreliable target influence through sample reliability weighting, while FSLR restores drifted parameters toward the original anchor. Their bidirectional interaction forms a continual adaptation loop that produces an adapted model for clean-domain and historical-revisit-domain evaluation.
