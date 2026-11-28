#!/usr/bin/env python3
"""Experiment 3: keep the deployment adapter fixed while changing FL initialization.

The canonical repository name is ``FedRAS`` (RGA+DSR).  The requested paper
label ``FedRAS`` is emitted as an alias in the summary, never passed to the
core evaluator.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

try:
    from .common import (
        ROOT, add_data_args, add_runtime_args, build_main_task,
        build_source_task, preflight, protocol_flags, raise_on_failures,
        rooted, run_tasks, summarize_tasks,
    )
except ImportError:
    from common import (
        ROOT, add_data_args, add_runtime_args, build_main_task,
        build_source_task, preflight, protocol_flags, raise_on_failures,
        rooted, run_tasks, summarize_tasks,
    )


DEFAULTS = {
    "fedacg": (
        "train/results/cifar100_100clients_alpha0.3_equal/fedacg_resnet18/"
        "cifar100_resnet18_run1/model_round499.pt",
        "test/outputs/fedacg_run1_statistics.pt",
        "lookahead_anchor",
    ),
    "fedavg": (
        "train/results/cifar100_100clients_alpha0.3_equal/fedavg_resnet18/"
        "run1_match/model_round499.pt",
        "test/outputs/fedavg_run1_statistics.pt",
        "adaptation_anchor",
    ),
    "fedprox": (
        "train/results/cifar100_100clients_alpha0.3_equal/fedprox_resnet18/"
        "run1_match/model_round499.pt",
        "test/outputs/fedprox_run1_statistics.pt",
        "adaptation_anchor",
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fedacg_model", default=DEFAULTS["fedacg"][0])
    parser.add_argument("--fedacg_statistics", default=DEFAULTS["fedacg"][1])
    parser.add_argument("--fedavg_model", default=DEFAULTS["fedavg"][0])
    parser.add_argument("--fedavg_statistics", default=DEFAULTS["fedavg"][1])
    parser.add_argument("--fedprox_model", default=DEFAULTS["fedprox"][0])
    parser.add_argument("--fedprox_statistics", default=DEFAULTS["fedprox"][1])
    parser.add_argument("--include_fedprox", action="store_true")
    parser.add_argument("--output_dir", default="outputs/supplement_experiments/exp3_fl_initialization")
    add_data_args(parser, dataset="cifar100", corruption_default="data/CIFAR-100-C",
                  clean_default="data")
    add_runtime_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out_dir = ROOT / args.output_dir
    methods = ["fedacg", "fedavg"] + (["fedprox"] if args.include_fedprox else [])
    paths = {
        "fedacg": (rooted(args.fedacg_model), rooted(args.fedacg_statistics)),
        "fedavg": (rooted(args.fedavg_model), rooted(args.fedavg_statistics)),
        "fedprox": (rooted(args.fedprox_model), rooted(args.fedprox_statistics)),
    }
    tasks = []
    for federated_method in methods:
        model_path, statistics_path = paths[federated_method]
        anchor_key = DEFAULTS[federated_method][2]
        prefix = federated_method
        tasks.append(build_source_task(
            name=f"{prefix}_source", dataset="cifar100", model_path=model_path,
            anchor_key=anchor_key, corruption_path=args.cifar_c_path,
            clean_root=args.clean_data_root, device=args.device,
            batch_size=args.batch_size, num_workers=args.num_workers, seed=args.seed,
            max_batches=args.max_batches, output=out_dir / f"{prefix}_source.json",
            log=out_dir / f"logs/{prefix}_source.log",
            metadata={"dataset": "cifar100", "backbone": "resnet18",
                      "required_inputs": [str(model_path)]},
        ))
        for tta_method, label in (("asr", "asr"), ("recap", "recap"), ("fedras", "fedras")):
            tasks.append(build_main_task(
                name=f"{prefix}_{label}", dataset="cifar100",
                federated_method=federated_method, model_path=model_path,
                anchor_key=anchor_key, statistics_path=statistics_path,
                tta_method=tta_method, corruption_path=args.cifar_c_path,
                clean_root=args.clean_data_root, device=args.device,
                batch_size=args.batch_size, num_workers=args.num_workers, seed=args.seed,
                max_batches=args.max_batches, output=out_dir / f"{prefix}_{label}.json",
                log=out_dir / f"logs/{prefix}_{label}.log",
                extra_flags=protocol_flags(args),
                metadata={"dataset": "cifar100", "backbone": "resnet18",
                          "required_inputs": [str(model_path), str(statistics_path)]},
            ))

    entries = []
    for method in methods:
        model_path, statistics_path = paths[method]
        entries.extend([
            {"role": f"{method}_model", "path": str(model_path)},
            {"role": f"{method}_statistics", "path": str(statistics_path)},
        ])
    entries.extend([
        {"role": "CIFAR-100-C", "path": args.cifar_c_path, "kind": "directory"},
        {"role": "CIFAR-clean", "path": args.clean_data_root, "kind": "directory"},
    ])
    preflight(entries, out_dir / "preflight.json",
              allow_missing=args.allow_missing or args.dry_run)
    failures = run_tasks(tasks, dry_run=args.dry_run,
                         force_rerun=args.force_rerun,
                         allow_missing=args.allow_missing or args.dry_run)
    aliases = {}
    for task in tasks:
        method, label = task.name.split("_", 1)
        display_method = method.upper() if method != "fedacg" else "FedACG"
        display_label = "FedRAS (FedRAS)" if label == "fedras" else label.upper()
        aliases[task.name] = f"{display_method}+{display_label}"
    summarize_tasks(tasks, output=out_dir / "metrics.json", aliases=aliases)
    raise_on_failures(failures)


if __name__ == "__main__":
    main()
