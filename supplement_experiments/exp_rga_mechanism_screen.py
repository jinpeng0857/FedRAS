#!/usr/bin/env python3
"""Five-arm RGA mechanism screen with DSR off and on, on one FedACG checkpoint.

The four simplified arms use the same regional-entropy objective, fixed
ReCAP region scale, shared ESS/weight-normalization/gradient protections, and
the same downstream protocol. Their RGA weights are respectively uniform,
regional confidence, anchor agreement, or their product. ``full_rga`` retains
the public full RGA configuration. FSLR/DSR receives the same full reliability
signal in every row, so the factorial comparison changes RGA weighting while
holding DSR's reliability input definition fixed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

try:
    from .common import (
        ROOT,
        OFFICIAL_FEDDR_BATCH_SIZE,
        add_data_args,
        add_runtime_args,
        build_main_task,
        official_protocol_flags,
        preflight,
        raise_on_failures,
        rooted,
        run_tasks,
    )
except ImportError:
    from common import (
        ROOT,
        OFFICIAL_FEDDR_BATCH_SIZE,
        add_data_args,
        add_runtime_args,
        build_main_task,
        official_protocol_flags,
        preflight,
        raise_on_failures,
        rooted,
        run_tasks,
    )


VARIANTS = {
    "uniform_recap": ("uniform_recap", "Unweighted ReCAP"),
    "regional_only": ("regional_only", "Regional only"),
    "anchor_only": ("anchor_only", "Anchor only"),
    "regional_anchor": ("regional_anchor", "Regional + anchor"),
    "full_rga": ("full", "Full RGA"),
}
DSR_MODES = ("off", "on")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--statistics_path", required=True)
    parser.add_argument(
        "--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS)
    )
    parser.add_argument(
        "--dsr_modes", nargs="+", choices=list(DSR_MODES), default=list(DSR_MODES)
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/supplement_experiments/rga_mechanism_screen_run1",
    )
    add_data_args(
        parser,
        dataset="cifar100",
        corruption_default="data/CIFAR-100-C",
        clean_default="data",
    )
    add_runtime_args(parser, device="cuda:1", batch_size=OFFICIAL_FEDDR_BATCH_SIZE)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def result_row(task, label: str, dsr_mode: str) -> dict[str, Any]:
    if not task.output.is_file():
        return {
            "DSR": dsr_mode,
            "Variant": label,
            "status": "missing",
        }
    payload = json.loads(task.output.read_text(encoding="utf-8"))
    domain = payload["domain_wise"]["summary"]
    continual = payload["continual"]
    cycles = continual["summary"]["cycle_summaries"]
    post = continual["post_stream"]
    reliability = domain.get("rga_reliability") or {}
    percent = lambda value: round(100.0 * float(value), 4)
    return {
        "DSR": dsr_mode,
        "Variant": label,
        "status": "ok",
        "DW (%)": percent(domain["mean_accuracy"]),
        "Cycle 1 (%)": percent(cycles[0]["mean_accuracy"]),
        "Cycle 2 (%)": percent(cycles[1]["mean_accuracy"]),
        "Cycle 3 (%)": percent(cycles[2]["mean_accuracy"]),
        "Revisit (%)": percent(post["mean_revisit_accuracy"]),
        "Clean (%)": percent(post["clean"]["accuracy"]),
        "Correct avg weight": round(float(reliability.get("correct_avg_weight", 0.0)), 6),
        "Wrong avg weight": round(float(reliability.get("wrong_avg_weight", 0.0)), 6),
        "Weight gap": round(float(reliability.get("separation_gap", 0.0)), 6),
        "Wrong weight ratio (%)": round(
            100.0 * float(reliability.get("wrong_sample_weight_ratio", 0.0)), 4
        ),
        "Update retention (%)": round(
            100.0 * float(reliability.get("update_retention", 0.0)), 4
        ),
        "Weight mass retention (%)": round(
            100.0 * float(reliability.get("update_weight_mass_retention", 0.0)), 4
        ),
        "result": str(task.output),
    }


def write_summary(rows: list[dict[str, Any]], out_dir: Path) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (out_dir / "rga_mechanism_screen.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# RGA mechanism screen: one FedACG checkpoint",
        "",
        "Single-checkpoint screening results on CIFAR-100-C severity 5, 15 corruptions, three uninterrupted cycles. This is a screening run, not a multi-seed estimate.",
        "",
        "| DSR | RGA variant | DW | C1 | C2 | C3 | Revisit | Clean | Weight gap | Wrong-weight ratio |",
        "|:---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row.get("status") != "ok":
            lines.append(f"| {row['DSR']} | {row['Variant']} | {row['status']} | - | - | - | - | - | - | - |")
            continue
        lines.append(
            f"| {row['DSR']} | {row['Variant']} | {row['DW (%)']:.2f} | "
            f"{row['Cycle 1 (%)']:.2f} | {row['Cycle 2 (%)']:.2f} | "
            f"{row['Cycle 3 (%)']:.2f} | {row['Revisit (%)']:.2f} | "
            f"{row['Clean (%)']:.2f} | {row['Weight gap']:.4f} | "
            f"{row['Wrong weight ratio (%)']:.2f}% |"
        )
    lines += [
        "",
        "Variant definitions: Unweighted ReCAP uses unit weights; Regional only uses ReCAP regional confidence; Anchor only uses `1 - 0.3 × disagreement`; Regional + anchor multiplies those two terms; Full RGA uses the retained official composite reliability, threshold, and reliability-controlled region scale.",
        "",
        "The simplified arms fix the region scale at 1.2 and disable the sample hard threshold; all arms retain ESS skip, capped weight normalization, gradient clipping, and the same DSR reliability input definition.",
        "",
        "One checkpoint gives a paired protocol comparison but does not estimate between-checkpoint or seed variability.",
    ]
    (out_dir / "rga_mechanism_screen.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out_dir = ROOT / args.output_dir
    model_path = rooted(args.model_path)
    statistics_path = rooted(args.statistics_path)
    tasks = []
    task_meta = {}

    for variant_key in args.variants:
        for dsr_mode in args.dsr_modes:
            tta_method = "dlrs_fslr" if dsr_mode == "on" else "dlrs"
            variant_value, label = VARIANTS[variant_key]
            extra_flags = official_protocol_flags(args) + [
                "--tta_method",
                tta_method,
                "--dlrs_ablation_variant",
                variant_value,
            ]
            if variant_value != "full":
                extra_flags += [
                    "--dlrs_region_scale_min",
                    "1.2",
                    "--dlrs_region_scale_max",
                    "1.2",
                    "--dlrs_hybrid_threshold",
                    "0.0",
                ]
            name = f"rga_{dsr_mode}_{variant_key}"
            output = out_dir / dsr_mode / f"{variant_key}.json"
            task = build_main_task(
                name=name,
                dataset="cifar100",
                federated_method="fedacg",
                model_path=model_path,
                anchor_key="lookahead_anchor",
                statistics_path=statistics_path,
                tta_method=tta_method,
                corruption_path=args.cifar_c_path,
                clean_root=args.clean_data_root,
                device=args.device,
                batch_size=OFFICIAL_FEDDR_BATCH_SIZE,
                num_workers=args.num_workers,
                seed=args.seed,
                max_batches=args.max_batches,
                output=output,
                log=out_dir / "logs" / f"{name}.log",
                extra_flags=extra_flags,
                metadata={
                    "dataset": "cifar100",
                    "backbone": "resnet18",
                    "required_inputs": [str(model_path), str(statistics_path)],
                },
            )
            tasks.append(task)
            task_meta[name] = (label, dsr_mode)

    checks = preflight(
        [
            {"role": "FedACG run1 checkpoint", "path": str(model_path)},
            {"role": "matching ReCAP/Fisher statistics", "path": str(statistics_path)},
            {"role": "CIFAR-100-C severity 5", "path": args.cifar_c_path, "kind": "directory"},
            {"role": "CIFAR-100 clean data", "path": args.clean_data_root, "kind": "directory"},
        ],
        out_dir / "preflight.json",
        allow_missing=args.allow_missing or args.dry_run,
    )
    manifest = {
        "design": "5 RGA variants x DSR off/on; one checkpoint; seed fixed",
        "checkpoint": str(model_path),
        "checkpoint_sha256": sha256_file(model_path) if model_path.is_file() else None,
        "statistics": str(statistics_path),
        "statistics_sha256": sha256_file(statistics_path) if statistics_path.is_file() else None,
        "dataset": "FedACG ResNet-18 / CIFAR-100-C severity 5",
        "seed": args.seed,
        "batch_size": OFFICIAL_FEDDR_BATCH_SIZE,
        "device": args.device,
        "max_batches": args.max_batches,
        "preflight": checks,
        "variants": {key: {"implementation": value, "label": label} for key, (value, label) in VARIANTS.items()},
        "dsr_modes": list(args.dsr_modes),
        "limitations": ["single checkpoint; not an estimate of checkpoint/seed variability"],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    failures = run_tasks(
        tasks,
        dry_run=args.dry_run,
        force_rerun=args.force_rerun,
        allow_missing=args.allow_missing or args.dry_run,
    )
    rows = [result_row(task, *task_meta[task.name]) for task in tasks]
    write_summary(rows, out_dir)
    raise_on_failures(failures)


if __name__ == "__main__":
    main()
