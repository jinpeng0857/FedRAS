# FedRAS

FedRAS studies continual test-time adaptation of federated models using reliability-guided adaptation and sparse restoration. This repository contains the evaluation code, source-statistics preparation, and experiment launchers for the [FedRAS method](paper/fedras_method.tex).

## Main evaluation

`main_test.py` runs the matched CIFAR-C evaluation. Its default method is registered in the code as `feddr`, which combines DLRS and FSLR. Use the corresponding trained checkpoint, source statistics, and dataset paths for your run.

```bash
python main_test.py --help
python prepare_main_statistics.py --help
```

The statistics script prepares source statistics from a trained checkpoint and source data. See [the evaluation entry point](main_test.py) for method options and [the training scripts](train/) for checkpoint generation.

## Paper experiments

Preview the supplementary experiment commands from the repository root.

```bash
bash run_fedras_experiments.sh --help
bash run_fedras_experiments.sh components --dry-run
```

The launcher supports component ablations, Appendix F, random and fixed restoration controls, cost, and sensitivity experiments. It runs the bundled `scripts/revision_20261018/run_case.py`. Set `REVISION_ROOT` to use a compatible runtime from another checkout.

Appendix F and component runs write per-trial diagnostics. Restarting a run skips completed trials with the same arguments and archives unfinished trials with their logs. Cost experiments require CUDA and perform an in-process warm-up before each of the three measurements.

A full run requires CIFAR-100, CIFAR-100-C, and readable checkpoints with matching source-statistics files for the selected group. The launcher exposes path overrides through `--help` and writes one `result.json` per trial under `OUTPUT_ROOT`, which defaults to `output/fedras_pdf_supplement`. Dry runs print commands without launching evaluation or checking input files.

- [Paper ablation launcher](scripts/run_paper_ablations.sh)
- [Revised experiment configurations and command generation](configs/revision_20261102/README.md)
- [Additional supplementary experiments](supplement_experiments/README.md)

Run the Tiny-ImageNet-C normalization comparison arm with the FedACG ResNet-18 checkpoint and its matching source statistics.

```bash
bash run_fedras_experiments.sh tiny-batch-mean --dry-run
bash run_fedras_experiments.sh tiny-batch-mean
```

This runs one full evaluation with the mean of the weighted sample losses. For comparison, use a weight-normalized FedRAS result with the same checkpoint, data order, and evaluation settings. The launcher preserves the historical Tiny-ImageNet-C sample order and records its hashes. Use `TINY_CKPT`, `TINY_COV_PATH`, `TINY_DATA_ROOT`, and `TINY_CLEAN_ROOT` to point to the experiment assets.

## Tests

Run the DLRS and sparse-restoration unit tests from the repository root.

```bash
python -m unittest tests.test_dlrs tests.test_sparse_restore
```

Additional regression tests are in [tests/](tests/).
