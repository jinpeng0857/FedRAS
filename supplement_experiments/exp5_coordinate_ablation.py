#!/usr/bin/env python3
"""Experiment 5: DSR/FSLR coordinate-selection ablation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

try:
    from .common import (
        ROOT, add_data_args, add_runtime_args, build_main_task, preflight,
        protocol_flags, raise_on_failures, rooted, run_tasks, summarize_tasks,
    )
except ImportError:
    from common import (
        ROOT, add_data_args, add_runtime_args, build_main_task, preflight,
        protocol_flags, raise_on_failures, rooted, run_tasks, summarize_tasks,
    )


CRITERIA = {
    "random": "Random",
    "drift_only": "Drift-only",
    "fisher_only": "Fisher-only",
    "fisher_drift": "Fisher×Drift",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--statistics_path", required=True)
    parser.add_argument("--criteria", nargs="+", choices=list(CRITERIA), default=list(CRITERIA))
    parser.add_argument("--random_seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--output_dir", default="outputs/supplement_experiments/exp5_coordinate_ablation")
    add_data_args(parser, dataset="cifar100", corruption_default="data/CIFAR-100-C",
                  clean_default="data")
    add_runtime_args(parser)
    return parser.parse_args()


def selection_flags(args, criterion: str, seed: int | None) -> list[str]:
    flags = protocol_flags(args) + [
        "--strict_mechanism_ablation",
        "--restore_budget_mode", "fixed",
        "--sparse_ratio", "0.05",
        "--restore_strength", "0.5",
        "--disable_fslr_event_trigger",
        "--disable_fslr_continuous_risk",
        "--disable_fslr_hierarchical_budget",
        "--disable_fslr_reliability_gate",
        "--fslr_selection_criterion", criterion,
        "--dlrs_region_scale_min", "1.2",
        "--dlrs_region_scale_max", "1.2",
        "--dlrs_hybrid_threshold", "0.0",
        "--dlrs_skip_update_tau_s", "0.0",
        "--dlrs_skip_update_tau_ess", "0.0",
        "--dlrs_normalize_by_weight",
        "--dlrs_weight_mode", "product",
        "--fslr_grad_clip_norm", "0.0",
    ]
    if seed is not None:
        flags.extend(["--fslr_selection_seed", str(seed)])
    return flags


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out_dir = ROOT / args.output_dir
    model_path, statistics_path = rooted(args.model_path), rooted(args.statistics_path)
    tasks = []
    for criterion in args.criteria:
        seeds = args.random_seeds if criterion == "random" else [None]
        for seed in seeds:
            suffix = f"_{seed}" if seed is not None else ""
            name = f"selection_{criterion}{suffix}"
            tasks.append(build_main_task(
                name=name, dataset="cifar100", federated_method="fedacg",
                model_path=model_path, anchor_key="lookahead_anchor",
                statistics_path=statistics_path, tta_method="dlrs_fslr",
                corruption_path=args.cifar_c_path, clean_root=args.clean_data_root,
                device=args.device, batch_size=args.batch_size,
                num_workers=args.num_workers, seed=args.seed,
                max_batches=args.max_batches, output=out_dir / f"{name}.json",
                log=out_dir / f"logs/{name}.log",
                extra_flags=selection_flags(args, criterion, seed),
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
    aliases = {}
    for task in tasks:
        key = task.name.removeprefix("selection_")
        criterion = next((item for item in CRITERIA if key == item or key.startswith(item + "_")), key)
        suffix = key[len(criterion):].lstrip("_")
        label = CRITERIA.get(criterion, criterion)
        aliases[task.name] = f"{label} (seed {suffix})" if suffix else label
    summarize_tasks(tasks, output=out_dir / "metrics.json", aliases=aliases)
    raise_on_failures(failures)


if __name__ == "__main__":
    main()
