#!/usr/bin/env python3
"""Experiment 6: export the complete 45-block continual accuracy curve."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

try:
    from .common import (
        ROOT, add_data_args, add_runtime_args, build_main_task,
        extract_45_block_curve, preflight, protocol_flags, raise_on_failures,
        rooted, run_tasks, summarize_tasks, valid_result,
    )
except ImportError:
    from common import (
        ROOT, add_data_args, add_runtime_args, build_main_task,
        extract_45_block_curve, preflight, protocol_flags, raise_on_failures,
        rooted, run_tasks, summarize_tasks, valid_result,
    )


DEFAULT_MODEL = (
    "train/results/cifar100_100clients_alpha0.3_equal/fedacg_resnet18/"
    "cifar100_resnet18_run1/model_round499.pt"
)
DEFAULT_STATS = "test/outputs/fedacg_run1_statistics.pt"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument("--statistics_path", default=DEFAULT_STATS)
    parser.add_argument("--skip_run", action="store_true",
                        help="Only extract the curve from an existing JSON.")
    parser.add_argument("--result_path", default=None,
                        help="Existing continual JSON to extract (use with --skip_run).")
    parser.add_argument("--plot", action="store_true",
                        help="Also save a PNG when matplotlib is installed.")
    parser.add_argument("--output_dir", default="outputs/supplement_experiments/exp6_continual_curve")
    add_data_args(parser, dataset="cifar100", corruption_default="data/CIFAR-100-C",
                  clean_default="data")
    add_runtime_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out_dir = ROOT / args.output_dir
    model_path, statistics_path = rooted(args.model_path), rooted(args.statistics_path)
    if args.result_path and not args.skip_run:
        raise ValueError("--result_path is an extraction source and requires --skip_run")
    result_path = rooted(args.result_path) if args.result_path else out_dir / "cifar100c_feddr_continual.json"
    task = build_main_task(
        name="cifar100c_feddr_45_block",
        dataset="cifar100", federated_method="fedacg",
        model_path=model_path, anchor_key="lookahead_anchor",
        statistics_path=statistics_path, tta_method="feddr",
        corruption_path=args.cifar_c_path, clean_root=args.clean_data_root,
        device=args.device, batch_size=args.batch_size,
        num_workers=args.num_workers, seed=args.seed,
        max_batches=args.max_batches, output=result_path,
        log=out_dir / "logs/cifar100c_feddr_45_block.log",
        extra_flags=protocol_flags(args),
        metadata={"dataset": "cifar100", "backbone": "resnet18",
                  "required_inputs": [str(model_path), str(statistics_path)]},
    )
    if args.skip_run:
        inputs = [{"role": "existing_result", "path": str(result_path)}]
    else:
        inputs = [
            {"role": "FedACG model", "path": str(model_path)},
            {"role": "statistics", "path": str(statistics_path)},
            {"role": "CIFAR-100-C", "path": args.cifar_c_path, "kind": "directory"},
            {"role": "CIFAR-clean", "path": args.clean_data_root, "kind": "directory"},
        ]
    preflight(inputs, out_dir / "preflight.json",
              allow_missing=args.allow_missing or args.dry_run)
    tasks = [] if args.skip_run else [task]
    failures = run_tasks(tasks, dry_run=args.dry_run,
                         force_rerun=args.force_rerun,
                         allow_missing=args.allow_missing or args.dry_run)
    if valid_result(result_path, dataset="cifar100", backbone="resnet18"):
        extract_45_block_curve(
            result_path,
            out_dir / "continual_45_block.json",
            out_dir / "continual_45_block.csv",
            out_dir / "continual_45_block.png" if args.plot else None,
        )
    elif not args.dry_run:
        failures = list(failures) + ["extract_45_block_curve"]
    summarize_tasks(tasks or [task], output=out_dir / "metrics.json",
                    aliases={task.name: "CIFAR-100-C / FedDR / 45-block curve"})
    raise_on_failures(failures)


if __name__ == "__main__":
    main()
