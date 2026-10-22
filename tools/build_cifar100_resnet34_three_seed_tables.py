#!/usr/bin/env python3
"""Aggregate three independent CIFAR-100-C / ResNet34 runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev


METHODS = ["Source", "Tent", "EATA", "SAR", "ASR", "ReCAP", "FedDR"]
FILES = {
    "Source": "source.json",
    "Tent": "tent.json",
    "EATA": "eata.json",
    "SAR": "sar.json",
    "ASR": "asr.json",
    "ReCAP": "recap.json",
    "FedDR": "feddr.json",
}
NOISE = {"gaussian_noise", "shot_noise", "impulse_noise"}
BLUR = {"defocus_blur", "glass_blur", "motion_blur", "zoom_blur"}
WEATHER = {"snow", "frost", "fog", "brightness"}
DIGITAL = {"contrast", "elastic_transform", "pixelate", "jpeg_compression"}
CATEGORIES = {"Noise": NOISE, "Blur": BLUR, "Weather": WEATHER, "Digital": DIGITAL}
CORRUPTIONS = sorted(set().union(*CATEGORIES.values()))


def load(root: Path, method: str) -> dict:
    with (root / FILES[method]).open(encoding="utf-8") as handle:
        return json.load(handle)


def cycle_summaries(result: dict) -> list[dict]:
    return result["continual"]["summary"]["cycle_summaries"]


def final_clean(result: dict) -> float:
    clean = result["clean_retention"]
    if "final_clean" in clean:
        return clean["final_clean"]["accuracy"]
    return clean["final_continual_clean_accuracy"]


def initial_clean(result: dict) -> float:
    clean = result["clean_retention"]
    if "initial_clean" in clean:
        return clean["initial_clean"]["accuracy"]
    return clean["initial_clean_accuracy"]


def revisit(result: dict) -> float:
    return result["continual"]["post_stream"]["mean_revisit_accuracy"]


def validate(result: dict, method: str, expected_seed: int) -> list[str]:
    errors: list[str] = []
    domains = result["domain_wise"]["per_corruption"]
    cycles = cycle_summaries(result)
    if len(domains) != 15:
        errors.append(f"domains={len(domains)}")
    if {item["corruption"] for item in domains} != set(CORRUPTIONS):
        errors.append("corruption roster mismatch")
    if len(cycles) != 3:
        errors.append(f"cycles={len(cycles)}")
    if result.get("backbone") != "resnet34":
        errors.append(f"backbone={result.get('backbone')!r}")
    if result.get("dataset") != "cifar100":
        errors.append(f"dataset={result.get('dataset')!r}")
    if result["protocol"].get("seed") != expected_seed:
        errors.append(f"seed={result['protocol'].get('seed')!r}")
    if result["protocol"].get("severity") != 5:
        errors.append(f"severity={result['protocol'].get('severity')!r}")
    if result["protocol"].get("prediction") != "pre_update":
        errors.append(f"prediction={result['protocol'].get('prediction')!r}")
    if result["domain_wise"]["summary"]["total_samples"] != 150000:
        errors.append("domain samples != 150000")
    if result["continual"]["summary"]["total_samples"] != 450000:
        errors.append("continual samples != 450000")
    return errors


def fmt(value: float) -> str:
    return f"{100.0 * value:.1f}"


def fmt_pm(values: list[float]) -> str:
    avg = mean(values)
    spread = stdev(values) if len(values) > 1 else 0.0
    return f"{100.0 * avg:.1f} ± {100.0 * spread:.1f}"


def domain_values(result: dict) -> dict[str, float]:
    by_name = {item["corruption"]: item["accuracy"] for item in result["domain_wise"]["per_corruption"]}
    values = {category: mean(by_name[name] for name in names) for category, names in CATEGORIES.items()}
    values["Overall Mean"] = result["domain_wise"]["summary"]["mean_accuracy"]
    return values


def continual_values(result: dict) -> dict[str, float]:
    cycles = cycle_summaries(result)
    return {
        "DW": result["domain_wise"]["summary"]["mean_accuracy"],
        "Cycle 1": cycles[0]["mean_accuracy"],
        "Cycle 2": cycles[1]["mean_accuracy"],
        "Cycle 3": cycles[2]["mean_accuracy"],
        "Revisit": revisit(result),
        "Clean": final_clean(result),
        "Clean initial": initial_clean(result),
        "Clean forgetting": initial_clean(result) - final_clean(result),
        "Cycle3-Cycle1": cycles[2]["mean_accuracy"] - cycles[0]["mean_accuracy"],
    }


def stats_rows(per_seed: dict[int, dict[str, dict[str, float]]], keys: list[str]) -> list[dict]:
    rows = []
    for method in METHODS:
        row = {"Method": method}
        for key in keys:
            row[key] = [per_seed[seed][method][key] for seed in sorted(per_seed)]
        rows.append(row)
    return rows


def md_table(rows: list[dict], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        cells = [row["Method"]]
        for key in columns[1:]:
            cells.append(fmt_pm(row[key]))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("outputs/cifar100_resnet34_unified_20260910"))
    args = parser.parse_args()
    roots = {0: args.base, 1: args.base.parent / (args.base.name + "_seed1"), 2: args.base.parent / (args.base.name + "_seed2")}
    results: dict[int, dict[str, dict]] = {}
    validation = []
    for seed, root in roots.items():
        results[seed] = {}
        for method in METHODS:
            result = load(root, method)
            results[seed][method] = result
            validation.append({"seed": seed, "method": method, "root": str(root), "errors": validate(result, method, seed)})

    per_seed_domain = {seed: {method: domain_values(results[seed][method]) for method in METHODS} for seed in roots}
    per_seed_continual = {seed: {method: continual_values(results[seed][method]) for method in METHODS} for seed in roots}
    table1_keys = ["Noise", "Blur", "Weather", "Digital", "Overall Mean"]
    table2_keys = ["DW", "Cycle 1", "Cycle 2", "Cycle 3", "Revisit", "Clean"]
    table1 = stats_rows(per_seed_domain, table1_keys)
    table2 = stats_rows(per_seed_continual, table2_keys)
    diagnostics = stats_rows(per_seed_continual, ["Clean initial", "Clean forgetting", "Cycle3-Cycle1"])

    summary = {
        "dataset": "cifar100-c",
        "backbone": "resnet34",
        "seeds": sorted(roots),
        "sample_standard_deviation": True,
        "category_mapping": {key: sorted(value) for key, value in CATEGORIES.items()},
        "validation": validation,
        "table_domain_wise": table1,
        "table_continual": table2,
        "diagnostics": diagnostics,
    }
    out = args.base
    (out / "CIFAR100C_RESNET34_THREE_SEED_SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    with (out / "CIFAR100C_RESNET34_THREE_SEED_TABLES.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Table I: domain-wise mean +/- std (%)"] + table1_keys)
        for row in table1:
            writer.writerow([row["Method"]] + [fmt_pm(row[key]) for key in table1_keys])
        writer.writerow([])
        writer.writerow(["Table II: continual mean +/- std (%)"] + table2_keys)
        for row in table2:
            writer.writerow([row["Method"]] + [fmt_pm(row[key]) for key in table2_keys])

    protocol = results[0]["Source"].get("protocol", {})
    lines = [
        "# CIFAR-100-C / ResNet34 three-seed results",
        "",
        "Each value is the mean ± sample standard deviation over independent seeds 0, 1 and 2.",
        "",
        "- Same ResNet34 FedACG checkpoint for all methods and seeds.",
        f"- CIFAR-100-C severity {protocol.get('severity', 5)}, 15 corruptions, 10,000 images per corruption.",
        "- Domain-wise results use an independent reset for every corruption.",
        "- Continual results carry state through the fixed 15-domain order for three cycles; prediction is pre-update.",
        "- Revisit is the mean over gaussian_noise, shot_noise, impulse_noise and defocus_blur after the stream, with adaptation disabled.",
        "- Clean is the final clean CIFAR-100 accuracy after the continual stream.",
        "- Batch size 128, two workers, full data, CUDA; category mapping is Noise (Gaussian/Shot/Impulse), Blur (Defocus/Glass/Motion/Zoom), Weather (Snow/Frost/Fog/Brightness), Digital (Contrast/Elastic/Pixelate/JPEG).",
        "",
        "## Table I. Domain-wise results under independent resets",
        "",
        md_table(table1, ["Method"] + table1_keys),
        "",
        "## Table II. Continual evaluation across three cycles and retention",
        "",
        md_table(table2, ["Method"] + table2_keys),
        "",
        "## Validation",
        "",
        "| Seed | Method | Domains | Cycles | Status |",
        "|---:|---|---:|---:|---|",
    ]
    for item in validation:
        status = "PASS" if not item["errors"] else "FAIL: " + "; ".join(item["errors"])
        lines.append(f"| {item['seed']} | {item['method']} | 15 | 3 | {status} |")
    (out / "CIFAR100C_RESNET34_THREE_SEED_TABLES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out / "CIFAR100C_RESNET34_THREE_SEED_TABLES.md")
    print("validation:", "PASS" if all(not item["errors"] for item in validation) else "FAIL")


if __name__ == "__main__":
    main()
