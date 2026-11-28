#!/usr/bin/env python3
"""Experiment 3: direct validity check for RGA sample weights.

The four rows keep the adaptation protocol fixed and only change the weight
assigned to each sample update:

* Uniform: one weight for every sample;
* Entropy-only: entropy-derived sample reliability;
* ReCAP confidence: regional ReCAP confidence;
* Full RGA: full sample reliability multiplied by ReCAP confidence.

The runtime records correctness-conditioned weights from the pre-update
prediction, which makes the resulting table a validity check rather than an
accuracy claim.
"""

from __future__ import annotations

import argparse
import logging

try:
    from .common import (
        ROOT,
        add_data_args,
        add_runtime_args,
        build_main_task,
        official_protocol_flags,
        OFFICIAL_FEDRAS_BATCH_SIZE,
        preflight,
        raise_on_failures,
        rooted,
        run_tasks,
        summarize_tasks,
    )
except ImportError:
    from common import (
        ROOT,
        add_data_args,
        add_runtime_args,
        build_main_task,
        official_protocol_flags,
        OFFICIAL_FEDRAS_BATCH_SIZE,
        preflight,
        raise_on_failures,
        rooted,
        run_tasks,
        summarize_tasks,
    )


VARIANTS = {
    "uniform": ("uniform", "full", "Uniform"),
    "entropy_only": ("sample_only", "entropy", "Entropy-only"),
    "recap_confidence": ("region_only", "full", "ReCAP confidence"),
    "full_rga": ("product", "full", "Full RGA"),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--statistics_path", required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(VARIANTS),
        default=list(VARIANTS),
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/supplement_experiments/table3_rga_reliability_official",
    )
    add_data_args(
        parser,
        dataset="cifar100",
        corruption_default="data/CIFAR-100-C",
        clean_default="data",
    )
    add_runtime_args(parser, batch_size=OFFICIAL_FEDRAS_BATCH_SIZE)
    return parser.parse_args()


def rga_flags(args, weight_mode: str, sample_mode: str) -> list[str]:
    """Use the official main protocol and vary only the RGA weight variant."""
    return official_protocol_flags(args) + [
        "--tta_method",
        "rga",
        "--rga_weight_mode",
        weight_mode,
        "--rga_sample_reliability_mode",
        sample_mode,
    ]


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out_dir = ROOT / args.output_dir
    model_path, statistics_path = rooted(args.model_path), rooted(args.statistics_path)
    tasks = []
    for key in args.variants:
        weight_mode, sample_mode, _label = VARIANTS[key]
        tasks.append(
            build_main_task(
                name=f"rga_{key}",
                dataset="cifar100",
                federated_method="fedacg",
                model_path=model_path,
                anchor_key="lookahead_anchor",
                statistics_path=statistics_path,
                tta_method="rga",
                corruption_path=args.cifar_c_path,
                clean_root=args.clean_data_root,
                device=args.device,
                batch_size=OFFICIAL_FEDRAS_BATCH_SIZE,
                num_workers=args.num_workers,
                seed=args.seed,
                max_batches=args.max_batches,
                output=out_dir / f"{key}.json",
                log=out_dir / f"logs/{key}.log",
                extra_flags=rga_flags(args, weight_mode, sample_mode),
                metadata={
                    "dataset": "cifar100",
                    "backbone": "resnet18",
                    "required_inputs": [str(model_path), str(statistics_path)],
                },
            )
        )

    preflight(
        [
            {"role": "FedACG model", "path": str(model_path)},
            {"role": "statistics", "path": str(statistics_path)},
            {
                "role": "CIFAR-100-C",
                "path": args.cifar_c_path,
                "kind": "directory",
            },
            {
                "role": "CIFAR-clean",
                "path": args.clean_data_root,
                "kind": "directory",
            },
        ],
        out_dir / "preflight.json",
        allow_missing=args.allow_missing or args.dry_run,
    )
    failures = run_tasks(
        tasks,
        dry_run=args.dry_run,
        force_rerun=args.force_rerun,
        allow_missing=args.allow_missing or args.dry_run,
    )
    summarize_tasks(
        tasks,
        output=out_dir / "metrics.json",
        aliases={task.name: VARIANTS[task.name.removeprefix("rga_")][2] for task in tasks},
    )
    raise_on_failures(failures)


if __name__ == "__main__":
    main()
