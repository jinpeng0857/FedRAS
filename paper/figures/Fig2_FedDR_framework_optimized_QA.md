# Framework figure optimization QA

## Optimization contract

- Role: How FedDR operates.
- Core story: federated initialization → continual test-time adaptation → evaluation.
- Archetype: schematic-led horizontal workflow.
- Reference use: style-only adaptation of the ATPL Fig.2 horizontal data-flow grammar; the framework content is FedDR-specific.
- Final size: 180 mm × 72 mm.

## Changes applied

- The framework is split into three lightly labeled stages: **Federated initialization**, **Continual test-time adaptation**, and **Evaluation**.
- The main path is now a single readable line: `θ₀ → Test-time Model → DLRS → Model Update → FSLR → Adapted Model`.
- Anchor θ₀ is shown as an initialization and restoration reference, not as a data stream.
- Feature extraction and prediction are grouped into one compact test-time model block.
- DLRS and FSLR retain only their semantic roles: **Short-term reliability / sample weighting** and **Long-term drift control / sparse restoration**.
- The continual loop is expressed inside the target-stream sequence `B₁ → B₂ → … → Bₜ₊₁`; no oversized cross-figure return arc remains.
- Evaluation is visually secondary and reduced to **Clean / Revisit**.

## Export and semantics checks

- Red: source and anchor reference.
- Blue: target stream and continual flow.
- Black dashed: gradient update.
- No experimental numbers or unsupported performance claims.
- Vector SVG/PDF plus 600 dpi PNG/TIFF were exported and visually checked after rendering.

## Suggested caption

**Fig. 2. Overall framework of FedDR.** Federated source data first produce the deployment anchor θ₀. During continual test-time adaptation, each target batch Bₜ is processed by the test-time model and prediction module, followed by DLRS short-term reliability weighting, model update, and FSLR long-term drift control. The resulting adapted model is evaluated on clean and historical revisit domains.
