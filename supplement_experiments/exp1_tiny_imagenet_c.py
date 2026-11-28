#!/usr/bin/env python3
"""Experiment 1: Tiny-ImageNet-C with a ResNet-18 deployment checkpoint."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

try:
    from .common import (
        ROOT, add_data_args, add_runtime_args, build_main_task, preflight,
        raise_on_failures, run_tasks, summarize_tasks, rooted,
    )
except ImportError:  # direct ``python supplement_experiments/exp1_*.py``
    from common import (
        ROOT, add_data_args, add_runtime_args, build_main_task, preflight,
        raise_on_failures, run_tasks, summarize_tasks, rooted,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True,
                        help="FedACG ResNet-18 checkpoint containing lookahead_anchor.")
    parser.add_argument("--statistics_path", required=True,
                        help="Statistics generated from this exact checkpoint.")
    add_data_args(parser, dataset="tiny-imagenet", corruption_default="data/Tiny-ImageNet-C",
                  clean_default="data/tiny-imagenet-200")
    parser.add_argument("--output_dir", default="outputs/supplement_experiments/exp1_tiny_imagenet_c")
    add_runtime_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out_dir = ROOT / args.output_dir
    task = build_main_task(
        name="tiny_imagenet_resnet18_fedras",
        dataset=args.dataset,
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
        output=out_dir / "tiny_imagenet_resnet18_fedras.json",
        log=out_dir / "logs/tiny_imagenet_resnet18_fedras.log",
        metadata={
            "dataset": "tiny-imagenet",
            "backbone": "resnet18",
            "required_inputs": [str(rooted(args.model_path)), str(rooted(args.statistics_path))],
        },
    )
    preflight([
        {"role": "model_checkpoint", "path": str(rooted(args.model_path))},
        {"role": "statistics", "path": str(rooted(args.statistics_path))},
        {"role": "Tiny-ImageNet-C", "path": str(rooted(args.cifar_c_path)), "kind": "directory"},
        {"role": "Tiny-ImageNet-clean", "path": str(rooted(args.clean_data_root)), "kind": "directory"},
    ], out_dir / "preflight.json", allow_missing=args.allow_missing or args.dry_run)
    failures = run_tasks([task], dry_run=args.dry_run,
                         force_rerun=args.force_rerun,
                         allow_missing=args.allow_missing or args.dry_run)
    summarize_tasks([task], output=out_dir / "metrics.json",
                    aliases={task.name: "Tiny-ImageNet-C / ResNet18 / FedRAS"})
    raise_on_failures(failures)


if __name__ == "__main__":
    main()
