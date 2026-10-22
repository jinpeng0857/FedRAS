# Strict motivation figure QA notes

## Figure contract

- Core conclusion: FedDR addresses the domain gap and the two dominant continual-TTA failure modes by reducing noisy update influence and returning drifted decision behavior toward the deployment anchor.
- Archetype: schematic-led three-panel scatter figure.
- Target: Knowledge-Based Systems / computer-vision journal motivation figure.
- Final size: 180 mm × 72 mm.
- Reference use: style-only inheritance from the provided ATPL Fig.1 layout; the source paper contributes only the three-panel scatter grammar, not scientific content.
- Evidence level: conceptual motivation only; no experimental values or quantitative claims are shown.

## Strict content constraints checked

- No rectangular algorithm-module boxes anywhere in the SVG.
- No `r_i`, `w_i`, `d_t`, loss equation, threshold, percentage, or internal algorithm parameter.
- Panel a shows red source samples, blue target samples, a black dashed anchor boundary θ₀, and a drifting target stream.
- Panel b contains only the two requested scatter limitations: equal-weight/noise-dominated updates and long-stream boundary drift away from θ₀.
- Panel c uses samples, dashed boundaries, arrows, and horizontal labels for DLRS noise suppression, FSLR anchor restoration, and the positive feedback loop.
- The shared legend is unboxed and contains only source, target, noise, decision boundary, and feedback.
- All arrow-side text is horizontal; no vertical annotations are used.

## Visual encoding

- Red: source samples and corrective/anchor-directed arrows.
- Blue: target samples and target-stream arrows.
- Black: noisy samples, decision boundaries, control arrows, and feedback.
- White background; no gradients, shadows, filters, transparency, 3D effects, or decorative icons.

## Export checks

- SVG: editable vector geometry and live text.
- PDF: one page at approximately 180 mm × 72 mm with searchable text.
- PNG/TIFF: 4252 × 1701 pixels, 600 dpi metadata.
- Visual QA: final PNG and PDF were inspected after rendering; panel separators, boundaries, points, arrows, legend, and labels remain within the canvas.

## Suggested caption

**Fig. 1. Motivation of FedDR for federated continual test-time adaptation.** (a) The federated anchor model θ₀ exhibits a domain gap: red source samples and the continuously shifting blue target stream are separated, so the anchor decision boundary cannot cover the target flow. (b) Existing federated TTA can suffer from equal-weight updates in which unreliable samples dominate adaptation, and from uncontrolled long-stream BN drift that moves the decision boundary away from the original anchor. (c) FedDR couples two complementary effects: DLRS reduces the influence of noisy target samples, while FSLR restores the drifted boundary toward θ₀. Their interaction forms a positive feedback loop for stable adaptation.
