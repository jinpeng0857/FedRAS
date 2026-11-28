#!/usr/bin/env python3
"""Experiment 2: CIFAR-100-C with a ResNet-34 backbone."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

try:
    from .common import (
        ROOT, add_data_args, add_runtime_args, build_main_task, preflight,
        raise_on_failures, run_tasks, summarize_tasks, rooted,
    )
except ImportError:
    from common import (
        ROOT, add_data_args, add_runtime_args, build_main_task, preflight,
        raise_on_failures, run_tasks, summarize_tasks, rooted,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True,
                        help="FedACG ResNet-34 checkpoint containing lookahead_anchor.")
    parser.add_argument("--statistics_path", required=True)
    add_data_args(parser, dataset="cifar100", corruption_default="data/CIFAR-100-C",
                  clean_default="data")
    parser.add_argument("--output_dir", default="outputs/supplement_experiments/exp2_resnet34_cifar100c")
    add_runtime_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out_dir = ROOT / args.output_dir
    task = build_main_task(
        name="cifar100c_resnet34_fedras",
        dataset="cifar100",
        federated_method="fedacg",
        model_path=rooted(args.model_path),
        anchor_key="lookahead_anchor",
        statistics_path=rooted(args.statistics_path),
        tta_method="fedras",
        corruption_path=args.cifar_c_path,
        clean_root=args.clean_data_root,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        max_batches=args.max_batches,
        output=out_dir / "cifar100c_resnet34_fedras.json",
        log=out_dir / "logs/cifar100c_resnet34_fedras.log",
        metadata={
            "dataset": "cifar100",
            "backbone": "resnet34",
            "required_inputs": [str(rooted(args.model_path)), str(rooted(args.statistics_path))],
        },
    )
    preflight([
        {"role": "model_checkpoint", "path": str(rooted(args.model_path))},
        {"role": "statistics", "path": str(rooted(args.statistics_path))},
        {"role": "CIFAR-100-C", "path": str(rooted(args.cifar_c_path)), "kind": "directory"},
        {"role": "CIFAR-clean", "path": str(rooted(args.clean_data_root)), "kind": "directory"},
    ], out_dir / "preflight.json", allow_missing=args.allow_missing or args.dry_run)
    failures = run_tasks([task], dry_run=args.dry_run,
                         force_rerun=args.force_rerun,
                         allow_missing=args.allow_missing or args.dry_run)
    summarize_tasks([task], output=out_dir / "metrics.json",
                    aliases={task.name: "CIFAR-100-C / ResNet34 / FedRAS"})
    raise_on_failures(failures)


if __name__ == "__main__":
    main()
