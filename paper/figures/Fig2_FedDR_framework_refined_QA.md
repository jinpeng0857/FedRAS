# Refined framework figure QA

## Figure contract

- Role: core FedDR system framework (how the method runs).
- Archetype: schematic-led horizontal workflow.
- Core claim: each continual target batch is processed by the test-time model, DLRS reliability weighting, a reliability-aware update, and FSLR restoration, while the adapted state drives the next batch.
- Final canvas: 180 mm × 72 mm; white background; editable SVG source plus PDF/PNG/TIFF exports.

## Structural changes from the previous framework

- The main system path is now visually dominant: `Bₜ → Test-time Model → DLRS → Reliable Update → FSLR → Adapted Model θₜ₊₁`.
- Federated initialization is compressed into a vertical chain: `Federated Source Data → Federated Pre-training → Anchor Model θ₀`.
- Anchor use is separated: a solid red arrow initializes the test-time model, while a short red dashed arrow explicitly marks the `anchor reference θ₀` entering FSLR.
- Continuality is moved outside the stream box: a blue dashed state loop runs from `Adapted Model` back to the next target batch. The stream box now only shows `B₁ → B₂ → … → Bₜ`.
- `Model Update` is renamed `Reliable Update`; the black dashed parameter-update arrow now returns to the test-time model instead of duplicating the DLRS forward-flow arrow.
- FSLR is reduced to one role line plus one mechanism line: `Long-term drift control` and `(sparse restoration)`.
- Evaluation is reduced to a thin, secondary `Clean / Revisit` attachment and does not compete with the core method path.
- The title is now `Overall framework of FedDR` to match the figure's system-level role.

## Visual semantics

- Red: federated source and anchor-related state.
- Blue solid: target-batch forward data flow.
- Blue dashed: continual adapted-state loop to the next batch.
- Black dashed: parameter update acting on the test-time model.
- Dashed rounded boxes: DLRS, Reliable Update, and FSLR method components.

## Exclusion checks

- No experimental values, thresholds, percentages, or unsupported equations.
- No gradients, filters, shadows, 3D effects, or decorative icons.
- No oversized cross-figure anchor arc or duplicated DLRS-to-update gradient arrow.
- SVG XML parses successfully; text remains live in the SVG source.

## Suggested caption

**Fig. 2. Overall framework of FedDR.** Federated source data produce the anchor model θ₀, which initializes the test-time model and serves as the restoration reference. For each target batch Bₜ, the model predicts and passes the batch to DLRS for short-term reliability weighting, followed by a reliability-aware update and FSLR long-term drift control. The adapted state θₜ₊₁ is then carried to the next target batch; clean and historical revisit evaluation is shown as a secondary output.
