"""Evaluate the frozen FedACG Lookahead checkpoint under the main protocol.

This is the frozen Source control for ``main_test.py``:

* an explicit deployment anchor (FedACG ``lookahead_anchor`` by default);
* CIFAR-C severity 5 and the same fixed 15-corruption order;
* separate Domain-wise and three-cycle Continual reports;
* no test-time gradient, parameter update, BN-statistics update, or restoration.

The model always stays in ``eval`` mode.  Therefore the Continual protocol is
an ordering/control check rather than an adaptive stream: all three cycles
should reproduce the same predictions.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from main_test import (
    CONTINUAL_CYCLES,
    CORRUPTIONS,
    DATASET_STATS,
    PROJECT_ROOT,
    REVISIT_CORRUPTIONS,
    SEVERITY,
    build_transform,
    evaluate_initial_clean,
    load_deployment_model,
    make_corruption_loader,
    resolve_device,
    summarize_corruptions,
)


LOGGER = logging.getLogger("source_only_test")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Frozen Source control under the main CIFAR-C protocol"
    )
    parser.add_argument(
        "--dataset", choices=["cifar10", "cifar100"], default="cifar100"
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument(
        "--anchor_key",
        choices=["lookahead_anchor", "adaptation_anchor"],
        default="lookahead_anchor",
    )
    parser.add_argument("--cifar_c_path", default=None)
    parser.add_argument("--clean_data_root", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max_batches",
        type=int,
        default=-1,
        help="Development-only cap per corruption; <=0 runs the full stream.",
    )
    parser.add_argument("--output_path", required=True)
    return parser.parse_args()


def build_frozen_model(args, device):
    model, backbone = load_deployment_model(
        args.model_path, device, anchor_key=args.anchor_key
    )
    model.requires_grad_(False)
    model.eval()

    classifier = getattr(model, "fc", None)
    num_classes = getattr(classifier, "out_features", None)
    expected_classes = DATASET_STATS[args.dataset]["num_classes"]
    if num_classes != expected_classes:
        raise ValueError(
            f"checkpoint has {num_classes} classes, expected {expected_classes}"
        )
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Source model unexpectedly has trainable parameters")

    return model, {
        "backbone": backbone,
        "num_classes": int(num_classes),
        "checkpoint_key": args.anchor_key,
        "model_mode": "eval",
        "trainable_parameters": 0,
        "bn_running_statistics": "frozen_source_statistics",
    }


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def evaluate_loader(model, loader, device, max_batches=-1):
    correct = 0
    total = 0
    batches = 0
    inference_seconds = 0.0
    for batch_index, (images, labels) in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        synchronize(device)
        start = time.perf_counter()
        logits = model(images)
        synchronize(device)
        inference_seconds += time.perf_counter() - start
        correct += logits.argmax(dim=1).eq(labels).sum().item()
        total += int(labels.size(0))
        batches += 1
    return {
        "accuracy": float(correct / total if total else 0.0),
        "correct": int(correct),
        "total": int(total),
        "batches": int(batches),
        "mean_inference_time_ms": float(
            1000.0 * inference_seconds / max(1, batches)
        ),
    }


def evaluate_corruption(args, model, corruption, device, cycle=None):
    dataset, loader = make_corruption_loader(args, corruption)
    try:
        metrics = evaluate_loader(model, loader, device, args.max_batches)
    finally:
        dataset.close()

    result = {
        "corruption": corruption,
        "severity": SEVERITY,
        **metrics,
        "adaptation_enabled": False,
        "updated_batches": 0,
        "restore_events": 0,
        "restored_assignments_total": 0,
    }
    if cycle is not None:
        result["cycle"] = int(cycle)
    print(
        f"{'cycle ' + str(cycle) + ' ' if cycle is not None else ''}"
        f"{corruption}: accuracy={100.0 * result['accuracy']:.2f}%"
    )
    return result


def resource_summary(results, wall_seconds, device):
    total_batches = sum(item["batches"] for item in results)
    weighted_inference_ms = sum(
        item["mean_inference_time_ms"] * item["batches"] for item in results
    )
    peak_memory = (
        float(torch.cuda.max_memory_allocated(device) / (1024.0**2))
        if device.type == "cuda"
        else None
    )
    return {
        "wall_time_seconds": float(wall_seconds),
        "mean_inference_time_ms": float(
            weighted_inference_ms / max(1, total_batches)
        ),
        "peak_cuda_memory_mb": peak_memory,
    }


def make_clean_loader(args):
    from torchvision import datasets

    clean_root = (
        Path(args.clean_data_root)
        if args.clean_data_root
        else PROJECT_ROOT / "data"
    )
    dataset_class = (
        datasets.CIFAR10 if args.dataset == "cifar10" else datasets.CIFAR100
    )
    dataset = dataset_class(
        root=str(clean_root),
        train=False,
        transform=build_transform(args.dataset),
        download=False,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda") and torch.cuda.is_available(),
    )


def snapshot_model_state(model):
    return {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }


def model_state_equal(model, snapshot):
    current = model.state_dict()
    return all(
        name in current and torch.equal(current[name].detach(), value)
        for name, value in snapshot.items()
    ) and len(current) == len(snapshot)


def run_domain_wise(args, device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    results = []
    metadata = None
    for corruption in CORRUPTIONS:
        model, current_metadata = build_frozen_model(args, device)
        if metadata is None:
            metadata = current_metadata
        elif current_metadata != metadata:
            raise RuntimeError("fresh Source models produced inconsistent metadata")
        results.append(evaluate_corruption(args, model, corruption, device))
        del model

    wall_seconds = time.perf_counter() - start
    return {
        "protocol": "reset_frozen_source_model_for_each_corruption",
        "state_continuity": False,
        "summary": summarize_corruptions(results),
        "resource": resource_summary(results, wall_seconds, device),
        "per_corruption": results,
    }, metadata


def run_continual(args, device):
    model, metadata = build_frozen_model(args, device)
    initial_state = snapshot_model_state(model)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()

    results = []
    cycle_summaries = []
    for cycle in range(1, CONTINUAL_CYCLES + 1):
        cycle_results = [
            evaluate_corruption(args, model, corruption, device, cycle=cycle)
            for corruption in CORRUPTIONS
        ]
        cycle_summary = summarize_corruptions(cycle_results)
        cycle_summary["cycle"] = cycle
        cycle_summaries.append(cycle_summary)
        results.extend(cycle_results)

    wall_seconds = time.perf_counter() - start
    stream_state_unchanged = model_state_equal(model, initial_state)
    if not stream_state_unchanged:
        raise RuntimeError(
            "Source parameters or BN buffers changed during the continual stream"
        )

    post_stream = evaluate_post_stream(args, model, results, device)
    overall_summary = summarize_corruptions(results)
    overall_summary.pop("second_half_definition")
    overall_summary.pop("second_half_mean_accuracy")
    cycle_means = [item["mean_accuracy"] for item in cycle_summaries]
    overall_summary.update(
        {
            "cycle_summaries": cycle_summaries,
            "cycle3_minus_cycle1": float(cycle_means[2] - cycle_means[0]),
            "cycle_accuracy_spread": float(max(cycle_means) - min(cycle_means)),
        }
    )
    return {
        "protocol": "same_frozen_source_model_over_15_corruptions_repeated_3_cycles",
        "state_continuity": True,
        "cycles": CONTINUAL_CYCLES,
        "model_state_unchanged": stream_state_unchanged,
        "summary": overall_summary,
        "resource": resource_summary(results, wall_seconds, device),
        "per_corruption": results,
        "post_stream": post_stream,
    }, metadata


def evaluate_post_stream(args, model, online_results, device):
    state_before = snapshot_model_state(model)
    clean = evaluate_loader(model, make_clean_loader(args), device, args.max_batches)

    first_pass = {}
    for item in online_results:
        first_pass.setdefault(item["corruption"], item)

    revisits = []
    for corruption in REVISIT_CORRUPTIONS:
        dataset, loader = make_corruption_loader(args, corruption)
        try:
            metrics = evaluate_loader(model, loader, device, args.max_batches)
        finally:
            dataset.close()
        first_accuracy = first_pass[corruption]["accuracy"]
        revisits.append(
            {
                "corruption": corruption,
                "accuracy": metrics["accuracy"],
                "correct": metrics["correct"],
                "total": metrics["total"],
                "first_pass_accuracy": float(first_accuracy),
                "forgetting": float(first_accuracy - metrics["accuracy"]),
            }
        )

    unchanged = model_state_equal(model, state_before)
    if not unchanged:
        raise RuntimeError(
            "frozen Source post-stream evaluation changed parameters or BN buffers"
        )

    return {
        "protocol": "frozen_source_model_no_adaptation",
        "model_state_unchanged": unchanged,
        "clean": {
            "accuracy": clean["accuracy"],
            "correct": clean["correct"],
            "total": clean["total"],
        },
        "revisit": revisits,
        "mean_revisit_accuracy": float(
            sum(item["accuracy"] for item in revisits) / len(revisits)
        ),
        "mean_forgetting": float(
            sum(item["forgetting"] for item in revisits) / len(revisits)
        ),
    }


def run():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    initial_clean = evaluate_initial_clean(args, device)
    # Keep the result-table label aligned with the FedDR comparison roster.
    LOGGER.info("method: Source")
    LOGGER.info("checkpoint key: %s", args.anchor_key)
    LOGGER.info("model mode: eval (source BN statistics frozen)")
    LOGGER.info("Domain-wise corruptions: %d", len(CORRUPTIONS))
    LOGGER.info(
        "Continual control stream: %d corruptions x %d cycles",
        len(CORRUPTIONS),
        CONTINUAL_CYCLES,
    )

    domain_wise, domain_metadata = run_domain_wise(args, device)
    continual, continual_metadata = run_continual(args, device)
    if domain_metadata != continual_metadata:
        raise RuntimeError("Domain-wise and Continual Source models are inconsistent")

    final_clean = continual["post_stream"]["clean"]
    domain_accuracy = {
        item["corruption"]: item["accuracy"]
        for item in domain_wise["per_corruption"]
    }
    protocol_differences = [
        abs(item["accuracy"] - domain_accuracy[item["corruption"]])
        for item in continual["per_corruption"]
    ]
    payload = {
        "schema_version": 1,
        "method": "Source",
        "comparison_role": "same_protocol_baseline",
        "dataset": args.dataset,
        "backbone": domain_metadata["backbone"],
        "model_path": args.model_path,
        "protocol": {
            "severity": SEVERITY,
            "corruptions": CORRUPTIONS,
            "domain_wise_and_continual_reported_separately": True,
            "continual_cycles": CONTINUAL_CYCLES,
            "prediction": "pre_update",
            "prediction_semantics": "frozen_forward_only_so_pre_and_post_are_identical",
            "seed": args.seed,
            "max_batches": args.max_batches,
        },
        "adaptation": {
            "enabled": False,
            "gradients": False,
            "optimizer": False,
            "bn_running_statistics_update": False,
        },
        "restoration": {"enabled": False},
        "model_metadata": domain_metadata,
        "initial_clean": initial_clean,
        "domain_wise": domain_wise,
        "continual": continual,
        "clean_retention": {
            "initial_clean_accuracy": float(initial_clean["accuracy"]),
            "final_continual_clean_accuracy": float(final_clean["accuracy"]),
            "clean_forgetting_initial_minus_final": float(
                initial_clean["accuracy"] - final_clean["accuracy"]
            ),
        },
        "source_only_invariance": {
            "expected": (
                "Domain-wise and all Continual passes use the identical frozen "
                "model and data order"
            ),
            "maximum_accuracy_difference": float(
                max(protocol_differences, default=0.0)
            ),
            "all_accuracies_identical": all(
                difference == 0.0 for difference in protocol_differences
            ),
        },
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(
        "Source Domain-wise CIFAR-C accuracy: "
        f"{100.0 * domain_wise['summary']['mean_accuracy']:.2f}%"
    )
    print(
        "Source Continual CIFAR-C accuracy: "
        f"{100.0 * continual['summary']['mean_accuracy']:.2f}% "
        f"(cycle spread="
        f"{100.0 * continual['summary']['cycle_accuracy_spread']:.3f}pp)"
    )
    print(
        "Source Clean accuracy: "
        f"initial={100.0 * initial_clean['accuracy']:.2f}% "
        f"final={100.0 * final_clean['accuracy']:.2f}%"
    )
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    run()
