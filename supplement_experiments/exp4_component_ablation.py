#!/usr/bin/env python3
"""Experiment 4: Base, Base+RGA, Base+DSR, and full FedRAS ablation.

This repository calls the two requested components RGA (region/sample
reliability, used here as RGA) and DSR (drift-sparse restoration, used here as
DSR).  The mapping is kept explicit in the output metadata and README.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

try:
    from .common import (
        ROOT, add_data_args, add_runtime_args, build_main_task, preflight,
        official_protocol_flags, OFFICIAL_FEDRAS_BATCH_SIZE,
        raise_on_failures, rooted, run_tasks, summarize_tasks,
    )
except ImportError:
    from common import (
        ROOT, add_data_args, add_runtime_args, build_main_task, preflight,
        official_protocol_flags, OFFICIAL_FEDRAS_BATCH_SIZE,
        raise_on_failures, rooted, run_tasks, summarize_tasks,
    )


VARIANTS = {
    "base": ("base", "Base"),
    "base_rga": ("rga", "Base+RGA"),
    "base_dsr": ("base_dsr", "Base+DSR"),
    "full": ("fedras", "FedRAS (FedRAS)"),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--statistics_path", required=True)
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument(
        "--output_dir",
        default="outputs/supplement_experiments/table2_component_ablation_official",
    )
    add_data_args(parser, dataset="cifar100", corruption_default="data/CIFAR-100-C",
                  clean_default="data")
    add_runtime_args(parser, batch_size=OFFICIAL_FEDRAS_BATCH_SIZE)
    return parser.parse_args()


def ablation_flags(args) -> list[str]:
    """Use the complete public main-experiment FedRAS protocol."""
    return official_protocol_flags(args)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out_dir = ROOT / args.output_dir
    model_path, statistics_path = rooted(args.model_path), rooted(args.statistics_path)
    tasks = []
    for key in args.variants:
        tta_method, _label = VARIANTS[key]
        # Every row shares the complete public main-experiment protocol. Only
        # the RGA/DSR switch represented by ``tta_method`` differs.
        extra = ablation_flags(args)
        tasks.append(build_main_task(
            name=f"component_{key}", dataset="cifar100", federated_method="fedacg",
            model_path=model_path, anchor_key="lookahead_anchor",
            statistics_path=statistics_path, tta_method=tta_method,
            corruption_path=args.cifar_c_path, clean_root=args.clean_data_root,
            device=args.device, batch_size=OFFICIAL_FEDRAS_BATCH_SIZE,
            num_workers=args.num_workers, seed=args.seed,
            max_batches=args.max_batches, output=out_dir / f"{key}.json",
            log=out_dir / f"logs/{key}.log",
            extra_flags=extra,
            metadata={"dataset": "cifar100", "backbone": "resnet18",
                      "required_inputs": [str(model_path), str(statistics_path)]},
        ))
    preflight([
        {"role": "FedACG model", "path": str(model_path)},
        {"role": "statistics", "path": str(statistics_path)},
        {"role": "CIFAR-100-C", "path": args.cifar_c_path, "kind": "directory"},
        {"role": "CIFAR-clean", "path": args.clean_data_root, "kind": "directory"},
    ], out_dir / "preflight.json", allow_missing=args.allow_missing or args.dry_run)
    failures = run_tasks(tasks, dry_run=args.dry_run,
                         force_rerun=args.force_rerun,
                         allow_missing=args.allow_missing or args.dry_run)
    summarize_tasks(tasks, output=out_dir / "metrics.json",
                    aliases={task.name: VARIANTS[task.name.removeprefix("component_")][1]
                             for task in tasks})
    raise_on_failures(failures)


if __name__ == "__main__":
    main()
