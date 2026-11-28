# FedRAS

FedRAS studies continual test-time adaptation of federated models using reliability-guided adaptation and sparse restoration. This repository contains the evaluation code, source-statistics preparation, and experiment launchers for FedRAS.

## Main evaluation

`main_test.py` runs the matched CIFAR-C / Tiny-ImageNet-C evaluation. Its default method is `fedras` (with `feddr` kept as a backward-compatible alias), combining RGA (DLRS) and DSR (FSLR). Use the corresponding trained checkpoint, source statistics, and dataset paths for your run.

```bash
python main_test.py --help
python prepare_main_statistics.py --help
```

The statistics script prepares source statistics from a trained checkpoint and source data. See [the evaluation entry point](main_test.py) for method options and [the training scripts](train/) for checkpoint generation.

## Paper experiments

Supplementary evaluations use [scripts/revision_20261018/run_case.py](scripts/revision_20261018/run_case.py) (and [scripts/run_tiny_ordered_revision_case.py](scripts/run_tiny_ordered_revision_case.py) for ordered Tiny-ImageNet-C runs).

- [Supplementary experiment configurations and command design](configs/supplement_20261112/README.md)
- [Additional supplementary experiments](supplement_experiments/README.md)

## Tests

Run the DLRS and sparse-restoration unit tests from the repository root.

```bash
python -m unittest tests.test_dlrs tests.test_sparse_restore
```

Additional regression tests are in [tests/](tests/).

## Summarize Results

Generate a Markdown summary table from `result.json` files:

```bash
python tools/summarize_results.py --input-dir outputs/fedras_pdf_supplement --output outputs/summary_tables.md
```
