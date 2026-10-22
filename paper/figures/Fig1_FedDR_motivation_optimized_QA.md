# Motivation figure optimization QA

## Optimization contract

- Role: Why FedDR is needed.
- Core story: domain gap → unreliable short-term update → accumulated long-term drift → dual reliability control.
- Archetype: minimal three-panel scatter schematic.
- Reference use: style-only adaptation of the ATPL Fig.1 three-panel scatter grammar; no research content or terminology was copied.
- Final size: 180 mm × 72 mm.

## Changes applied

- a was reduced to the domain-gap phenomenon and the θ₀ anchor boundary.
- b now uses only two risk labels: **Unreliable update** and **Accumulated drift**.
- c now maps the two risks to **Short-term reliability** (DLRS) and **Long-term stability** (FSLR), with a compact **Mutual reinforcement** arrow.
- Repeated explanatory sentences, redundant arrows, and internal method symbols were removed.
- The red/blue/black semantics are shared with the framework figure: red source/anchor, blue target, black noise/boundary/control.

## Exclusion checks

- No algorithm-module rectangles.
- No `r_i`, `w_i`, `d_t`, formulas, thresholds, percentages, or experimental values.
- No vertical annotation text.
- No gradients, shadows, filters, transparency, 3D effects, or decorative icons.

## Suggested caption

**Fig. 1. Motivation of FedDR.** (a) The federated anchor encounters a domain gap when the target stream shifts after deployment. (b) Existing federated TTA is vulnerable to unreliable short-term updates and accumulated long-term drift. (c) FedDR addresses these complementary risks through DLRS-based short-term reliability control and FSLR-based long-term stability control, whose interaction yields stable adaptation.
