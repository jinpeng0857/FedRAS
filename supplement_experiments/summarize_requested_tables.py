#!/usr/bin/env python3
"""Format the three image-requested tables from completed JSON results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


COMPONENT_ROWS = [
    ("base", "Base", "✗", "✗"),
    ("base_rga", "Base + RGA", "✓", "✗"),
    ("base_dsr", "Base + DSR", "✗", "✓"),
    ("full", "FedRAS", "✓", "✓"),
]
DRIFT_ROWS = [
    ("base", "Base"),
    ("base_rga", "RGA only"),
    ("base_dsr", "DSR only"),
    ("full", "FedRAS"),
]
RGA_ROWS = [
    ("uniform", "Uniform"),
    ("entropy_only", "Entropy-only"),
    ("recap_confidence", "ReCAP confidence"),
    ("full_rga", "Full RGA"),
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--component_dir",
        default="outputs/supplement_experiments/table2_component_ablation_official",
    )
    parser.add_argument(
        "--rga_dir",
        default="outputs/supplement_experiments/table3_rga_reliability_official",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/supplement_experiments/requested_tables_official",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing result: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"result is not a JSON object: {path}")
    return payload


def percent(value: float) -> float:
    return round(100.0 * float(value), 4)


def result_metrics(payload: dict[str, Any]) -> dict[str, float]:
    domain = payload.get("domain_wise", {}).get("summary", {})
    cycles = payload.get("continual", {}).get("summary", {}).get(
        "cycle_summaries", []
    )
    if len(cycles) != 3:
        raise ValueError("result must contain exactly three continual cycles")
    post_stream = payload.get("continual", {}).get("post_stream", {})
    clean = post_stream.get("clean", {})
    revisit = post_stream.get("mean_revisit_accuracy")
    if revisit is None or "accuracy" not in clean:
        raise ValueError("result is missing revisit or clean-retention metrics")
    return {
        "domain": percent(domain["mean_accuracy"]),
        "cycle1": percent(cycles[0]["mean_accuracy"]),
        "cycle2": percent(cycles[1]["mean_accuracy"]),
        "cycle3": percent(cycles[2]["mean_accuracy"]),
        "revisit": percent(revisit),
        "clean": percent(clean["accuracy"]),
        "drift1": float(cycles[0]["final_bn_affine_drift"]["total"]),
        "drift2": float(cycles[1]["final_bn_affine_drift"]["total"]),
        "drift3": float(cycles[2]["final_bn_affine_drift"]["total"]),
    }


def control_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    config = payload.get("config", {})
    adaptation = payload.get("adaptation", {})
    protocol = payload.get("protocol", {})
    restoration = payload.get("restoration", {})
    return {
        "checkpoint": payload.get("model_path"),
        "seed": protocol.get("seed"),
        "batch_size": protocol.get("batch_size"),
        "corruptions": tuple(protocol.get("corruptions", [])),
        "bn_scope": adaptation.get("parameter_scope"),
        "optimizer": adaptation.get("optimizer"),
        "learning_rate": adaptation.get("learning_rate"),
        "momentum": adaptation.get("momentum"),
        "weight_decay": adaptation.get("weight_decay"),
        "region_scale_min": config.get("dlrs_region_scale_min"),
        "region_scale_max": config.get("dlrs_region_scale_max"),
        "gradient_clip": config.get("fslr_grad_clip_norm"),
        "weak_view": config.get("dlrs_view_consistency"),
        "recovery_interval": restoration.get("interval"),
        "restore_strength": restoration.get("restore_strength"),
        "drift_thresholds": tuple(restoration.get("drift_thresholds", [])),
        "drift_ratios": tuple(restoration.get("drift_ratios", [])),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    component_dir = Path(args.component_dir)
    rga_dir = Path(args.rga_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    component_payloads = {
        key: load_json(component_dir / f"{key}.json")
        for key, _label, _rga, _dsr in COMPONENT_ROWS
    }
    rga_payloads = {
        key: load_json(rga_dir / f"{key}.json") for key, _label in RGA_ROWS
    }

    snapshots = {key: control_snapshot(payload) for key, payload in component_payloads.items()}
    first_snapshot = snapshots[COMPONENT_ROWS[0][0]]
    controls_equal = all(snapshot == first_snapshot for snapshot in snapshots.values())
    if not controls_equal:
        raise ValueError(
            "Table 2 controls are not identical; inspect controls_audit.json"
        )

    table2_rows = []
    table4_rows = []
    for key, label, rga, dsr in COMPONENT_ROWS:
        metrics = result_metrics(component_payloads[key])
        table2_rows.append(
            {
                "Configuration": label,
                "RGA": rga,
                "DSR": dsr,
                "DW": metrics["domain"],
                "Cycle 1": metrics["cycle1"],
                "Cycle 2": metrics["cycle2"],
                "Cycle 3": metrics["cycle3"],
                "Revisit": metrics["revisit"],
                "Clean": metrics["clean"],
            }
        )

    for key, label in DRIFT_ROWS:
        metrics = result_metrics(component_payloads[key])
        table4_rows.append(
            {
                "Method": label,
                "Drift after C1": metrics["drift1"],
                "Drift after C2": metrics["drift2"],
                "Drift after C3": metrics["drift3"],
                "Cycle 3 Acc": metrics["cycle3"],
                "Clean": metrics["clean"],
            }
        )

    table3_rows = []
    for key, label in RGA_ROWS:
        payload = rga_payloads[key]
        reliability = payload.get("domain_wise", {}).get("summary", {}).get(
            "rga_reliability"
        )
        if not reliability:
            raise ValueError(f"missing domain-wise RGA metrics in {key}.json")
        table3_rows.append(
            {
                "Method": label,
                "Correct Avg. Weight": round(
                    float(reliability["correct_avg_weight"]), 6
                ),
                "Wrong Avg. Weight": round(
                    float(reliability["wrong_avg_weight"]), 6
                ),
                "Gap": round(float(reliability["separation_gap"]), 6),
                "Wrong-sample Weight Ratio": round(
                    100.0 * float(reliability["wrong_sample_weight_ratio"]),
                    4,
                ),
                "Update Retention": round(
                    100.0 * float(reliability["update_retention"]), 4
                ),
            }
        )

    write_csv(
        output_dir / "table2_component_ablation.csv",
        table2_rows,
        list(table2_rows[0]),
    )
    write_csv(
        output_dir / "table3_rga_reliability.csv",
        table3_rows,
        list(table3_rows[0]),
    )
    write_csv(
        output_dir / "table4_dsr_drift.csv",
        table4_rows,
        list(table4_rows[0]),
    )

    audit = {
        "controls_equal": controls_equal,
        "control_snapshot": {
            key: {
                name: list(value) if isinstance(value, tuple) else value
                for name, value in snapshot.items()
            }
            for key, snapshot in snapshots.items()
        },
        "table3_definition": {
            "scope": "domain-wise CIFAR-100-C severity 5, pooled over 15 corruptions",
            "correct": "pre-update argmax equals ground-truth label",
            "wrong": "pre-update argmax differs from ground-truth label",
            "wrong_sample_weight_ratio": "sum(weight on wrong samples) / sum(all sample weights)",
            "update_retention": "samples with positive applied update weight / all samples",
        },
        "table4_definition": {
            "drift": "||theta_t,A - theta_0,A||_2 / (||theta_0,A||_2 + epsilon)",
            "theta": "trainable BN affine parameters",
            "cycle_boundary": "after all 15 corruptions in the continual cycle",
        },
    }
    (output_dir / "controls_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        "# Official-protocol RGA × DSR ablation tables",
        "",
        "All numbers below are pooled from one fixed FedACG ResNet-18 checkpoint under the public main FedDR protocol: batch size 128, learning rate 1e-3, excluding-layer4 BN scope, and the retained FSLR risk/restoration settings. Table 2 uses identical controls outside the RGA/DSR switches; the audit is in `controls_audit.json`.",
        "",
        "## Table 2 — Official-protocol RGA × DSR ablation",
        "",
        "| Configuration | RGA | DSR | DW (%) | Cycle 1 (%) | Cycle 2 (%) | Cycle 3 (%) | Revisit (%) | Clean (%) |",
        "|---|:---:|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table2_rows:
        lines.append(
            f"| {row['Configuration']} | {row['RGA']} | {row['DSR']} | "
            f"{row['DW']:.2f} | {row['Cycle 1']:.2f} | {row['Cycle 2']:.2f} | "
            f"{row['Cycle 3']:.2f} | {row['Revisit']:.2f} | {row['Clean']:.2f} |"
        )
    lines += [
        "",
        "## Table 3 — RGA reliability validity",
        "",
        "Correct/wrong is assigned from the pre-update prediction against the ground-truth label. The wrong-sample ratio is a weight share, not an error rate.",
        "",
        "| Method | Correct Avg. Weight | Wrong Avg. Weight | Gap ↑ | Wrong-sample Weight Ratio ↓ (%) | Update Retention (%) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in table3_rows:
        lines.append(
            f"| {row['Method']} | {row['Correct Avg. Weight']:.4f} | "
            f"{row['Wrong Avg. Weight']:.4f} | {row['Gap']:.4f} | "
            f"{row['Wrong-sample Weight Ratio']:.2f} | {row['Update Retention']:.2f} |"
        )
    lines += [
        "",
        "## Table 4 — DSR drift verification",
        "",
        "Drift is measured on trainable BN affine parameters relative to the deployment anchor at each continual-cycle boundary.",
        "",
        "| Method | Drift after C1 ↓ | Drift after C2 ↓ | Drift after C3 ↓ | Cycle 3 Acc ↑ (%) | Clean ↑ (%) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in table4_rows:
        lines.append(
            f"| {row['Method']} | {row['Drift after C1']:.6f} | "
            f"{row['Drift after C2']:.6f} | {row['Drift after C3']:.6f} | "
            f"{row['Cycle 3 Acc']:.2f} | {row['Clean']:.2f} |"
        )

    (output_dir / "requested_tables.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (output_dir / "requested_tables.json").write_text(
        json.dumps(
            {
                "table2": table2_rows,
                "table3": table3_rows,
                "table4": table4_rows,
                "controls_equal": controls_equal,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output_dir / "requested_tables.md")


if __name__ == "__main__":
    main()
