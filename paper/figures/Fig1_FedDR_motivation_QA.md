# Figure 1 QA notes

## Figure contract

- Core conclusion: FedDR jointly reduces unreliable-sample interference and long-stream parameter drift through DLRS soft reliability weighting and FSLR drift-aware sparse restoration.
- Archetype: schematic-led composite.
- Target: Knowledge-Based Systems-style double-column motivation figure.
- Final size: 180 mm × 72 mm.
- Panel map:
  - a: the federated deployment anchor has a domain gap and its source boundary misses the shifting target stream;
  - b-top: equal target-sample weighting lets unreliable samples steer adaptation;
  - b-bottom: long-stream BN adaptation drifts away from the deployment anchor;
  - c: DLRS and FSLR form a mutually reinforcing control loop.
- Evidence level: conceptual mechanism only; no experimental values or quantitative claims are shown.
- Reviewer risk addressed: DLRS is drawn as soft weighting with all samples retained, not as hard filtering.

## Adaptation and visual rules

- Reuse level: style-only inheritance from the provided three-panel reference; scientific content and mechanism were rebuilt for FedDR.
- Palette: red, blue, black, and white only; no gradients, shadows, transparency, or 3D effects.
- Encoding: color denotes source/target/reliability status; circle/triangle denotes class; dashed curves denote decision boundaries.
- Typography: Arial/Helvetica-compatible sans serif for labels and Times-compatible serif for θ₀, θₜ, rᵢ, and dₜ.
- Shared legend: one legend spans the three panels.

## Export checks

- SVG: editable vector geometry and live text.
- PDF: one page at 180 mm × 72 mm; vector line art and selectable text.
- PNG/TIFF: 600 dpi-class raster exports at the final physical size.
- Visual inspection: panel labels, legend, symbols, arrows, and boundary curves remain inside the canvas; no clipped text or overlap that changes meaning.

## Suggested caption

**Fig. 1. Motivation of FedDR for federated continual test-time adaptation.** (a) The federated deployment anchor θ₀ exhibits a domain gap, so its source-trained decision boundary does not cover the continuously shifting target stream. (b) Existing federated TTA can suffer from two coupled limitations: equal weighting allows unreliable samples to interfere with adaptation, while long-term BN updates increase the drift dₜ from the anchor and cause performance decay. (c) FedDR couples DLRS and FSLR in a positive loop. DLRS assigns sample-wise soft reliability rᵢ to reduce noisy update contributions without hard removal, and FSLR periodically restores drift-sensitive parameters toward θ₀ to stabilize subsequent predictions.
