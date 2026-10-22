"""Re-run the paper-facing FedRAS studies with an integrated reduced RGA.

The reduced RGA keeps patch-shuffle/PLPD, federated-anchor consistency, and
regional confidence. It removes predictive-entropy and class-margin sample
cues plus weak-view consistency; the reduced reliability is also the signal
passed to DSR. All other settings are taken from official_protocol_flags().
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import statistics
from pathlib import Path
from typing import Any

try:
    from .common import (
        ROOT,
        OFFICIAL_FEDDR_BATCH_SIZE,
        Task,
        build_main_task,
        official_protocol_flags,
        preflight,
        raise_on_failures,
        rooted,
        run_tasks,
        summarize_tasks,
    )
except ImportError:
    from common import (
        ROOT,
        OFFICIAL_FEDDR_BATCH_SIZE,
        Task,
        build_main_task,
        official_protocol_flags,
        preflight,
        raise_on_failures,
        rooted,
        run_tasks,
        summarize_tasks,
    )


LOGGER = logging.getLogger("reduced_fedras")
DEFAULT_OUTPUT = "outputs/supplement_experiments/fedras_reduced_rga_20260923"
REDUCED_FLAGS = ["--dlrs_cue_ablation", "reduced"]

CHECKPOINTS = [
    {
        "key": "cifar100_r18_fedacg_run1",
        "dataset": "cifar100",
        "backbone": "resnet18",
        "federated_method": "fedacg",
        "model": "train/results/cifar100_100clients_alpha0.3_equal/fedacg_resnet18/cifar100_resnet18_run1/model_round499.pt",
        "statistics": "test/outputs/fedacg_run1_statistics.pt",
        "anchor_key": "lookahead_anchor",
        "group": "main_cifar_r18",
    },
    {
        "key": "cifar100_r18_fedacg_run2",
        "dataset": "cifar100",
        "backbone": "resnet18",
        "federated_method": "fedacg",
        "model": "train/results/cifar100_100clients_alpha0.3_equal/fedacg_resnet18/run2/model_round499.pt",
        "statistics": "test/outputs/fedacg_run2_statistics.pt",
        "anchor_key": "lookahead_anchor",
        "group": "main_cifar_r18",
    },
    {
        "key": "cifar100_r18_fedacg_run3",
        "dataset": "cifar100",
        "backbone": "resnet18",
        "federated_method": "fedacg",
        "model": "train/results/cifar100_100clients_alpha0.3_equal/fedacg_resnet18/run3/model_round499.pt",
        "statistics": "test/outputs/fedacg_run3_statistics.pt",
        "anchor_key": "lookahead_anchor",
        "group": "main_cifar_r18",
    },
    {
        "key": "cifar100_r34_fedacg",
        "dataset": "cifar100",
        "backbone": "resnet34",
        "federated_method": "fedacg",
        "model": "train/results/cifar100_100clients_alpha0.3_equal/fedacg_resnet34/run1_match/model_round499.pt",
        "statistics": "test/outputs/cifar100_resnet34_unified_20260910/statistics.pt",
        "anchor_key": "lookahead_anchor",
        "group": "cifar_r34",
    },
    {
        "key": "cifar100_r18_fedavg",
        "dataset": "cifar100",
        "backbone": "resnet18",
        "federated_method": "fedavg",
        "model": "train/results/cifar100_100clients_alpha0.3_equal/fedavg_resnet18/run1_match/model_round499.pt",
        "statistics": "test/outputs/fedavg_run1_statistics.pt",
        "anchor_key": "adaptation_anchor",
        "group": "init_fedavg",
    },
    {
        "key": "cifar100_r18_fedprox",
        "dataset": "cifar100",
        "backbone": "resnet18",
        "federated_method": "fedprox",
        "model": "train/results/cifar100_100clients_alpha0.3_equal/fedprox_resnet18/run1_match/model_round499.pt",
        "statistics": "outputs/fedprox_tests/table1_protocol_20260903/fedprox_statistics.pt",
        "anchor_key": "adaptation_anchor",
        "group": "init_fedprox",
    },
    {
        "key": "tiny_imagenet_r18_fedacg",
        "dataset": "tiny-imagenet",
        "backbone": "resnet18",
        "federated_method": "fedacg",
        "model": "train/results/tiny-imagenet_100clients_alpha0.3_equal/fedacg_resnet18/tiny_imagenet_resnet18_fedacg_20260907_seed42/model_round499.pt",
        "statistics": "test/outputs/tiny_resnet18_unified_20260907/statistics.pt",
        "anchor_key": "lookahead_anchor",
        "group": "tiny_r18",
    },
    {
        "key": "tiny_imagenet_r34_fedacg",
        "dataset": "tiny-imagenet",
        "backbone": "resnet34",
        "federated_method": "fedacg",
        "model": "train/results/tiny-imagenet_100clients_alpha0.3_equal/fedacg_resnet34/tiny_imagenet_resnet34_cifar100_protocol/model_round499.pt",
        "statistics": "test/outputs/tiny_resnet34_unified_20260907/statistics.pt",
        "anchor_key": "lookahead_anchor",
        "group": "tiny_r34",
    },
]

PAPER_VALUES = {
    "cifar_r18": {
        "label": "CIFAR-100-C / ResNet18 / FedACG (3 checkpoints)",
        "keys": ["cifar100_r18_fedacg_run1", "cifar100_r18_fedacg_run2", "cifar100_r18_fedacg_run3"],
        "values": {"domain": 62.3, "cycle1": 63.5, "cycle2": 63.8, "cycle3": 63.7, "revisit": 58.0, "clean": 74.4},
    },
    "cifar_r34": {
        "label": "CIFAR-100-C / ResNet34 / FedACG",
        "keys": ["cifar100_r34_fedacg"],
        "values": {"domain": 61.8, "cycle1": 62.8, "cycle2": 63.2, "cycle3": 63.3, "revisit": 57.9, "clean": 74.5},
    },
    "tiny_r18": {
        "label": "Tiny-ImageNet-C / ResNet18 / FedACG",
        "keys": ["tiny_imagenet_r18_fedacg"],
        "values": {"domain": 34.9, "cycle1": 36.2, "cycle2": 36.7, "cycle3": 36.6, "revisit": 30.6, "clean": 58.6},
    },
    "tiny_r34": {
        "label": "Tiny-ImageNet-C / ResNet34 / FedACG",
        "keys": ["tiny_imagenet_r34_fedacg"],
        "values": {"domain": 37.0, "cycle1": 37.6, "cycle2": 37.9, "cycle3": 37.8, "revisit": 32.1, "clean": 59.7},
    },
    "init_fedavg": {
        "label": "CIFAR-100-C / ResNet18 / FedAvg init",
        "keys": ["cifar100_r18_fedavg"],
        "values": {"cycle2": 59.98, "cycle3": 60.04},
    },
    "init_fedprox": {
        "label": "CIFAR-100-C / ResNet18 / FedProx init",
        "keys": ["cifar100_r18_fedprox"],
        "values": {"cycle2": 60.55, "cycle3": 60.57},
    },
    "init_fedacg": {
        "label": "CIFAR-100-C / ResNet18 / FedACG init (run1)",
        "keys": ["cifar100_r18_fedacg_run1"],
        "values": {"cycle2": 63.88, "cycle3": 63.70},
    },
}

PAPER_COMPONENTS = {
    "Base": {"domain": 62.89, "cycle1": 63.26, "cycle2": 62.42, "cycle3": 61.61, "revisit": 56.80, "clean": 70.72},
    "Base + RGA": {"domain": 62.24, "cycle1": 62.96, "cycle2": 62.54, "cycle3": 62.11, "revisit": 57.42, "clean": 71.53},
    "Base + DSR": {"domain": 62.91, "cycle1": 63.83, "cycle2": 63.82, "cycle3": 63.73, "revisit": 58.20, "clean": 73.70},
    "FedRAS": {"domain": 62.23, "cycle1": 63.62, "cycle2": 63.88, "cycle3": 63.70, "revisit": 58.38, "clean": 74.50},
}

PAPER_SENSITIVITY = [
    ("R", "5", "sensitivity_R5_cifar100_r18_run1", {"cycle1": 63.48, "cycle3": 63.55, "revisit": 58.12, "clean": 74.65}),
    ("R", "10 default", "cifar100_r18_fedacg_run1", {"cycle1": 63.62, "cycle3": 63.70, "revisit": 58.38, "clean": 74.50}),
    ("R", "20", "sensitivity_R20_cifar100_r18_run1", {"cycle1": 63.65, "cycle3": 63.15, "revisit": 57.42, "clean": 73.80}),
    ("tau_D", "0.25", "sensitivity_tauD025_cifar100_r18_run1", {"cycle1": 63.30, "cycle3": 63.45, "revisit": 58.20, "clean": 74.85}),
    ("tau_D", "0.35 default", "cifar100_r18_fedacg_run1", {"cycle1": 63.62, "cycle3": 63.70, "revisit": 58.38, "clean": 74.50}),
    ("tau_D", "0.50", "sensitivity_tauD050_cifar100_r18_run1", {"cycle1": 63.60, "cycle3": 62.85, "revisit": 56.95, "clean": 73.10}),
    ("rho_max", "0.05", "sensitivity_rhoMax005_cifar100_r18_run1", {"cycle1": 63.58, "cycle3": 63.22, "revisit": 57.35, "clean": 73.65}),
    ("rho_max", "0.10 default", "cifar100_r18_fedacg_run1", {"cycle1": 63.62, "cycle3": 63.70, "revisit": 58.38, "clean": 74.50}),
    ("rho_max", "0.20", "sensitivity_rhoMax020_cifar100_r18_run1", {"cycle1": 63.15, "cycle3": 63.40, "revisit": 58.45, "clean": 74.92}),
    ("tau_H", "0.10", "sensitivity_tauH010_cifar100_r18_run1", {"cycle1": 63.40, "cycle3": 63.32, "revisit": 57.60, "clean": 73.85}),
    ("tau_H", "0.20 default", "cifar100_r18_fedacg_run1", {"cycle1": 63.62, "cycle3": 63.70, "revisit": 58.38, "clean": 74.50}),
    ("tau_H", "0.30", "sensitivity_tauH030_cifar100_r18_run1", {"cycle1": 63.25, "cycle3": 63.50, "revisit": 58.15, "clean": 74.30}),
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--cifar_c_path", default="data/CIFAR-100-C")
    parser.add_argument("--cifar_clean_root", default="data")
    parser.add_argument("--tiny_c_path", default="/mnt/sda/PythonProject/Jiang/dataset/Tiny-ImageNet-C")
    parser.add_argument("--tiny_clean_root", default="/mnt/sda/PythonProject/Jiang/dataset/tiny-imagenet-200")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch_size", type=int, default=OFFICIAL_FEDDR_BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=-1)
    parser.add_argument("--skip_sensitivity", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--force_rerun", action="store_true")
    parser.add_argument("--allow_missing", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _data_paths(args, dataset: str) -> tuple[str, str]:
    if dataset == "tiny-imagenet":
        return args.tiny_c_path, args.tiny_clean_root
    return args.cifar_c_path, args.cifar_clean_root


def _build_task(args, out_dir: Path, spec: dict[str, Any], *,
                name: str | None = None, tta_method: str = "feddr",
                extra_flags: list[str] | None = None,
                include_reduced: bool = True) -> Task:
    model_path = rooted(spec["model"])
    stats_path = rooted(spec["statistics"])
    corruption_path, clean_root = _data_paths(args, spec["dataset"])
    task_name = name or spec["key"]
    flags = official_protocol_flags(args) + (REDUCED_FLAGS if include_reduced else []) + (extra_flags or [])
    return build_main_task(
        name=task_name,
        dataset=spec["dataset"],
        federated_method=spec["federated_method"],
        model_path=model_path,
        anchor_key=spec["anchor_key"],
        statistics_path=stats_path,
        tta_method=tta_method,
        corruption_path=corruption_path,
        clean_root=clean_root,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        max_batches=args.max_batches,
        output=out_dir / f"{task_name}.json",
        log=out_dir / "logs" / f"{task_name}.log",
        extra_flags=flags,
        metadata={
            "dataset": spec["dataset"],
            "backbone": spec["backbone"],
            "required_inputs": [str(model_path), str(stats_path)],
            "checkpoint_key": spec["key"],
            "anchor_key": spec["anchor_key"],
            "rga_variant": "reduced_entropy_margin_weak_view" if include_reduced else "not_applicable",
        },
    )


def _paper_metrics(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    domain = payload["domain_wise"]["summary"]["mean_accuracy"]
    cycles = payload["continual"]["summary"]["cycle_summaries"]
    post = payload["continual"]["post_stream"]
    clean = post.get("clean", {}).get("accuracy")
    if clean is None:
        clean = payload.get("clean_retention", {}).get("final_continual_clean_accuracy")
    return {
        "domain": 100.0 * float(domain),
        "cycle1": 100.0 * float(cycles[0]["mean_accuracy"]),
        "cycle2": 100.0 * float(cycles[1]["mean_accuracy"]),
        "cycle3": 100.0 * float(cycles[2]["mean_accuracy"]),
        "revisit": 100.0 * float(post["mean_revisit_accuracy"]),
        "clean": 100.0 * float(clean),
    }


def _drift_by_cycle(path: Path) -> list[float | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    blocks = payload.get("continual", {}).get("per_corruption", [])
    drifts: list[float | None] = []
    for cycle in (1, 2, 3):
        cycle_blocks = [item for item in blocks if int(item.get("cycle", 0)) == cycle]
        drift = cycle_blocks[-1].get("final_bn_affine_drift") if cycle_blocks else None
        drifts.append(None if drift is None else float(drift))
    return drifts


def _rga_reliability(path: Path) -> dict[str, float] | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("domain_wise", {}).get("per_corruption", [])
    diagnostics = [item["rga_reliability"] for item in records if item.get("rga_reliability")]
    if not diagnostics:
        return None
    correct_count = sum(item["correct_samples"] for item in diagnostics)
    wrong_count = sum(item["wrong_samples"] for item in diagnostics)
    total_count = sum(item["total_samples"] for item in diagnostics)
    correct_sum = sum(item["correct_weight_sum"] for item in diagnostics)
    wrong_sum = sum(item["wrong_weight_sum"] for item in diagnostics)
    total_sum = sum(item["total_weight_sum"] for item in diagnostics)
    applied_count = sum(item["applied_samples"] for item in diagnostics)
    correct_avg = correct_sum / max(1, correct_count)
    wrong_avg = wrong_sum / max(1, wrong_count)
    return {
        "correct_avg": correct_avg,
        "wrong_avg": wrong_avg,
        "gap": correct_avg - wrong_avg,
        "wrong_weight_pct": 100.0 * wrong_sum / max(total_sum, 1e-12),
        "update_retention_pct": 100.0 * applied_count / max(1, total_count),
    }


def _fmt(value: float | None) -> str:
    return "—" if value is None or not math.isfinite(value) else f"{value:.2f}"


def write_comparison(tasks: list[Task], output: Path):
    by_name = {task.name: task for task in tasks}
    task_values: dict[str, dict[str, float]] = {}
    for name, task in by_name.items():
        if task.output.is_file():
            try:
                task_values[name] = _paper_metrics(task.output)
            except (OSError, ValueError, KeyError, TypeError):
                continue

    lines = [
        "# Reduced-RGA FedRAS: comparison with the prior paper results",
        "",
        "Variant: remove sample-entropy, class-margin, and weak-view consistency; retain patch-shuffle/PLPD, federated-anchor consistency, and regional confidence. The same reduced reliability is passed to DSR. Regional entropy in the ReCAP objective is unchanged.",
        "",
        "All previous-paper values below are transcribed from the user-designated `main.tex`; they are not rerun baselines. New values are from the JSON artifacts in this run directory. Positive Δ means the reduced variant is higher.",
        "",
        "| Evaluation | Metric | Paper FedRAS | Reduced RGA FedRAS | Δ (pp) |",
        "|---|---:|---:|---:|---:|",
    ]
    for group, definition in PAPER_VALUES.items():
        available = [task_values[key] for key in definition["keys"] if key in task_values]
        label = definition["label"]
        if not available:
            for metric, old in definition["values"].items():
                lines.append(f"| {label} | {metric} | {_fmt(old)} | not completed | — |")
            continue
        for metric, old in definition["values"].items():
            vals = [row[metric] for row in available]
            mean = statistics.mean(vals)
            spread = statistics.stdev(vals) if len(vals) > 1 else None
            new_display = f"{mean:.2f} ± {spread:.2f}" if spread is not None else _fmt(mean)
            delta = mean - old
            lines.append(f"| {label} | {metric} | {_fmt(old)} | {new_display} | {delta:+.2f} |")
    lines += [
        "",
        "## Checkpoint-level results",
        "",
        "| Checkpoint | DW | C1 | C2 | C3 | Revisit | Clean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for spec in CHECKPOINTS:
        values = task_values.get(spec["key"])
        if values is None:
            continue
        lines.append(
            "| {key} | {domain:.2f} | {cycle1:.2f} | {cycle2:.2f} | {cycle3:.2f} | {revisit:.2f} | {clean:.2f} |".format(
                key=spec["key"], **values
            )
        )
    lines += [
        "",
        "## Component ablation comparison (CIFAR-100-C, run1)",
        "",
        "| Variant | Source table DW | New DW | Source table C1 | New C1 | Source table C3 | New C3 | Source table Revisit | New Revisit | Source table Clean | New Clean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    component_task = {
        "Base": "component_base_run1",
        "Base + RGA": "component_reduced_rga_only_run1",
        "Base + DSR": "component_dsr_only_run1",
        "FedRAS": "cifar100_r18_fedacg_run1",
    }
    for label, source_values in PAPER_COMPONENTS.items():
        values = task_values.get(component_task[label])
        if values is None:
            lines.append(f"| {label} | {source_values['domain']:.2f} | not completed | {source_values['cycle1']:.2f} | — | {source_values['cycle3']:.2f} | — | {source_values['revisit']:.2f} | — | {source_values['clean']:.2f} | — |")
        else:
            lines.append(
                f"| {label} | {source_values['domain']:.2f} | {values['domain']:.2f} | {source_values['cycle1']:.2f} | {values['cycle1']:.2f} | {source_values['cycle3']:.2f} | {values['cycle3']:.2f} | {source_values['revisit']:.2f} | {values['revisit']:.2f} | {source_values['clean']:.2f} | {values['clean']:.2f} |"
            )
    lines += [
        "",
        "## RGA reliability validity (CIFAR-100-C domain-wise, run1)",
        "",
        "| Method | Correct Avg. Weight | Wrong Avg. Weight | Gap | Wrong Weight (%) | Update Retention (%) |",
        "|---|---:|---:|---:|---:|---:|",
        "| Prior paper: Full RGA | 0.871 | 0.168 | 0.703 | 10.480 | 75.380 |",
    ]
    diagnostic_task = by_name.get("component_reduced_rga_only_run1")
    if diagnostic_task is not None and diagnostic_task.output.is_file():
        diagnostic = _rga_reliability(diagnostic_task.output)
        if diagnostic:
            lines.append(
                "| Reduced RGA | {correct_avg:.3f} | {wrong_avg:.3f} | {gap:.3f} | {wrong_weight_pct:.3f} | {update_retention_pct:.3f} |".format(**diagnostic)
            )
        else:
            lines.append("| Reduced RGA | diagnostics missing | — | — | — | — |")
    else:
        lines.append("| Reduced RGA | not completed | — | — | — | — |")
    lines += [
        "",
        "## DSR drift check (CIFAR-100-C, run1)",
        "",
        "| Method | Drift C1 | Drift C2 | Drift C3 | Cycle 3 accuracy (%) | Clean (%) |",
        "|---|---:|---:|---:|---:|---:|",
        "| Prior paper: FedRAS | 0.195 | 0.217 | 0.227 | 63.70 | 74.50 |",
    ]
    full_task = by_name.get("cifar100_r18_fedacg_run1")
    if full_task is not None and full_task.output.is_file():
        drifts = _drift_by_cycle(full_task.output)
        values = task_values.get(full_task.name, {})
        lines.append(
            "| Reduced RGA + DSR | {d1} | {d2} | {d3} | {c3} | {clean} |".format(
                d1=_fmt(drifts[0]), d2=_fmt(drifts[1]), d3=_fmt(drifts[2]),
                c3=_fmt(values.get("cycle3")), clean=_fmt(values.get("clean")),
            )
        )
    else:
        lines.append("| Reduced RGA + DSR | not completed | — | — | — | — |")
    lines += [
        "",
        "## Sensitivity comparison",
        "",
        "| Parameter | Value | Paper C1 | New C1 | Paper C3 | New C3 | Paper Revisit | New Revisit | Paper Clean | New Clean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for parameter, value, task_name, old in PAPER_SENSITIVITY:
        new = task_values.get(task_name)
        lines.append(
            "| {parameter} | {value} | {old_c1:.2f} | {new_c1} | {old_c3:.2f} | {new_c3} | {old_revisit:.2f} | {new_revisit} | {old_clean:.2f} | {new_clean} |".format(
                parameter=parameter, value=value,
                old_c1=old["cycle1"], old_c3=old["cycle3"],
                old_revisit=old["revisit"], old_clean=old["clean"],
                new_c1=_fmt(new["cycle1"] if new else None),
                new_c3=_fmt(new["cycle3"] if new else None),
                new_revisit=_fmt(new["revisit"] if new else None),
                new_clean=_fmt(new["clean"] if new else None),
            )
        )
    lines += [
        "",
        "## Protocol and interpretation limits",
        "",
        "- Shared protocol: severity 5, canonical 15-corruption order, 3 uninterrupted cycles, predict-then-update, batch size 128, learning rate 1e-3, and the explicit official FedRAS flags in `common.py`; checkpoints and statistics are paired as listed in `run_manifest.json`.",
        "- This is a method-change comparison, not a paired rerun of Tent/EATA/SAR/ASR/ReCAP or other baselines. Their paper numbers remain historical comparison values.",
        "- The three-checkpoint CIFAR ResNet-18 row uses mean ± sample standard deviation across the same three named trained checkpoints. Single-checkpoint rows are shown without a fabricated variance.",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out_dir = rooted(args.output_dir)
    tasks = [
        _build_task(args, out_dir, spec)
        for spec in CHECKPOINTS
    ]

    # The updated component table needs a new RGA-only row; the full reduced
    # FedRAS run for run1 is already part of the main three-checkpoint group.
    rga_only_spec = next(item for item in CHECKPOINTS if item["key"] == "cifar100_r18_fedacg_run1")
    tasks.append(_build_task(
        args,
        out_dir,
        rga_only_spec,
        name="component_reduced_rga_only_run1",
        tta_method="dlrs",
    ))
    tasks.append(_build_task(
        args,
        out_dir,
        rga_only_spec,
        name="component_base_run1",
        tta_method="base",
        include_reduced=False,
    ))
    tasks.append(_build_task(
        args,
        out_dir,
        rga_only_spec,
        name="component_dsr_only_run1",
        tta_method="base_fslr",
        include_reduced=False,
    ))

    sensitivity_specs = [
        ("R5", ["--recovery_interval", "5"]),
        ("R20", ["--recovery_interval", "20"]),
        ("tauD025", ["--fslr_risk_tau_drift", "0.25"]),
        ("tauD050", ["--fslr_risk_tau_drift", "0.50"]),
        ("rhoMax005", ["--fslr_risk_rho_max", "0.05"]),
        ("rhoMax020", ["--fslr_risk_rho_max", "0.20"]),
        ("tauH010", ["--dlrs_hybrid_threshold", "0.10"]),
        ("tauH030", ["--dlrs_hybrid_threshold", "0.30"]),
    ]
    if not args.skip_sensitivity:
        for label, flags in sensitivity_specs:
            tasks.append(_build_task(
                args,
                out_dir,
                rga_only_spec,
                name=f"sensitivity_{label}_cifar100_r18_run1",
                extra_flags=flags,
            ))

    preflight_entries = []
    seen = set()
    for spec in CHECKPOINTS:
        for role, relative in (("checkpoint", spec["model"]), ("statistics", spec["statistics"])):
            absolute = rooted(relative)
            if absolute in seen:
                continue
            seen.add(absolute)
            preflight_entries.append({"role": f"{spec['key']} {role}", "path": str(absolute)})
        corruption, clean = _data_paths(args, spec["dataset"])
        for role, path in ((f"{spec['dataset']} corruption data", corruption), (f"{spec['dataset']} clean data", clean)):
            absolute = rooted(path)
            if absolute not in seen:
                seen.add(absolute)
                preflight_entries.append({"role": role, "path": str(absolute), "kind": "directory"})
    preflight(preflight_entries, out_dir / "preflight.json", allow_missing=args.allow_missing or args.dry_run)

    manifest = {
        "schema_version": 1,
        "variant": {
            "name": "reduced_rga_integrated_fedras",
            "removed_sample_cues": ["predictive_entropy", "class_margin", "weak_view_consistency"],
            "retained_cues": ["patch_shuffle_plpd", "federated_anchor_consistency", "regional_confidence"],
            "regional_entropy_objective_retained": True,
            "dsr_reliability_source": "same_reduced_rga_signal",
        },
        "protocol": {
            "device": args.device,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "max_batches": args.max_batches,
            "official_flags": official_protocol_flags(args),
            "reduced_flags": REDUCED_FLAGS,
        },
        "tasks": [],
    }
    for task in tasks:
        manifest["tasks"].append({
            "name": task.name,
            "command": task.command,
            "output": str(task.output),
            "log": str(task.log),
            "metadata": task.metadata,
        })
    manifest["code_sha256"] = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in (ROOT / "adapters/dlrs.py", ROOT / "main_test.py", ROOT / "evaluator/engine.py", Path(__file__).resolve())
    }
    manifest["input_sha256"] = {
        path: sha256_file(rooted(path))
        for spec in CHECKPOINTS
        for path in (spec["model"], spec["statistics"])
        if rooted(path).is_file()
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    failures = run_tasks(
        tasks,
        dry_run=args.dry_run,
        force_rerun=args.force_rerun,
        allow_missing=args.allow_missing or args.dry_run,
    )
    summarize_tasks(tasks, output=out_dir / "metrics.json", aliases={task.name: task.name for task in tasks})
    write_comparison(tasks, out_dir / "comparison_with_paper.md")
    raise_on_failures(failures)


if __name__ == "__main__":
    main()
