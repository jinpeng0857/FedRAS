#!/usr/bin/env python3
"""Build paper-style CIFAR-100-C tables from unified result JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


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


def pct(value: float | None) -> str:
    return "—" if value is None else f"{100.0 * value:.1f}"


def pct_signed(value: float | None) -> str:
    return "—" if value is None else f"{100.0 * value:+.1f}"


def load_result(root: Path, method: str) -> dict:
    path = root / FILES[method]
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def domain_entries(result: dict) -> list[dict]:
    return result["domain_wise"]["per_corruption"]


def continual_cycles(result: dict) -> list[dict]:
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


def revisit_mean(result: dict) -> float:
    return result["continual"]["post_stream"]["mean_revisit_accuracy"]


def validate_result(result: dict, method: str) -> dict:
    domains = domain_entries(result)
    cycles = continual_cycles(result)
    names = [entry["corruption"] for entry in domains]
    expected = set().union(*CATEGORIES.values())
    errors: list[str] = []
    if len(domains) != 15:
        errors.append(f"domain entries={len(domains)} (expected 15)")
    if set(names) != expected:
        errors.append("corruption roster mismatch")
    if len(cycles) != 3:
        errors.append(f"cycle summaries={len(cycles)} (expected 3)")
    if result.get("backbone") != "resnet34":
        errors.append(f"backbone={result.get('backbone')!r}")
    if result.get("dataset") != "cifar100":
        errors.append(f"dataset={result.get('dataset')!r}")
    domain_total = result["domain_wise"]["summary"]["total_samples"]
    continual_total = result["continual"]["summary"]["total_samples"]
    if domain_total != 150000:
        errors.append(f"domain samples={domain_total} (expected 150000)")
    if continual_total != 450000:
        errors.append(f"continual samples={continual_total} (expected 450000)")
    return {"method": method, "errors": errors, "domain_samples": domain_total,
            "continual_samples": continual_total, "num_domains": len(domains),
            "num_cycles": len(cycles), "model_path": result.get("model_path"),
            "backbone": result.get("backbone"), "dataset": result.get("dataset"),
            "protocol": result.get("protocol", {})}


def make_rows(results: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    table1: list[dict] = []
    table2: list[dict] = []
    for method in METHODS:
        result = results[method]
        by_name = {entry["corruption"]: entry["accuracy"] for entry in domain_entries(result)}
        row1 = {"Method": method}
        for category, names in CATEGORIES.items():
            row1[category] = mean(by_name[name] for name in names)
        row1["Overall Mean"] = result["domain_wise"]["summary"]["mean_accuracy"]
        table1.append(row1)

        cycles = continual_cycles(result)
        row2 = {
            "Method": method,
            "DW": result["domain_wise"]["summary"]["mean_accuracy"],
            "Cycle 1": cycles[0]["mean_accuracy"],
            "Cycle 2": cycles[1]["mean_accuracy"],
            "Cycle 3": cycles[2]["mean_accuracy"],
            "Revisit": revisit_mean(result),
            "Clean": final_clean(result),
            "Clean initial": initial_clean(result),
            "Clean forgetting": initial_clean(result) - final_clean(result),
            "Cycle3-Cycle1": cycles[2]["mean_accuracy"] - cycles[0]["mean_accuracy"],
        }
        table2.append(row2)
    return table1, table2


def markdown_table(rows: list[dict], columns: list[str], signed: set[str] | None = None) -> str:
    signed = signed or set()
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        vals = []
        for column in columns:
            value = row[column]
            if isinstance(value, str):
                vals.append(value)
            elif column in signed:
                vals.append(pct_signed(value))
            else:
                vals.append(pct(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_outputs(root: Path, results: dict[str, dict], audits: list[dict], table1: list[dict], table2: list[dict]) -> None:
    summary = {
        "dataset": "cifar100-c",
        "backbone": "resnet34",
        "methods": METHODS,
        "category_mapping": {key: sorted(value) for key, value in CATEGORIES.items()},
        "validation": audits,
        "table_domain_wise": table1,
        "table_continual": table2,
    }
    (root / "CIFAR100C_RESNET34_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    with (root / "CIFAR100C_RESNET34_TABLES.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Table I: domain-wise", "Noise", "Blur", "Weather", "Digital", "Overall Mean"])
        for row in table1:
            writer.writerow([row["Method"]] + [100 * row[key] for key in ["Noise", "Blur", "Weather", "Digital", "Overall Mean"]])
        writer.writerow([])
        writer.writerow(["Table II: continual", "DW", "Cycle 1", "Cycle 2", "Cycle 3", "Revisit", "Clean", "Clean initial", "Clean forgetting", "Cycle3-Cycle1"])
        for row in table2:
            writer.writerow([row["Method"]] + [100 * row[key] for key in ["DW", "Cycle 1", "Cycle 2", "Cycle 3", "Revisit", "Clean", "Clean initial", "Clean forgetting", "Cycle3-Cycle1"]])

    protocol = results["Source"].get("protocol", {})
    lines = [
        "# CIFAR-100-C / ResNet34 unified evaluation",
        "",
        "The seven methods use the same ResNet34 FedACG checkpoint and the fixed unified protocol.",
        "",
        f"- Dataset: CIFAR-100-C, severity {protocol.get('severity', 5)}, 15 corruptions, 10,000 images per corruption.",
        "- Corruption order: gaussian_noise, shot_noise, impulse_noise, defocus_blur, glass_blur, motion_blur, zoom_blur, snow, frost, fog, brightness, contrast, elastic_transform, pixelate, jpeg_compression.",
        "- Domain-wise evaluation: each corruption starts from a fresh copy of the source checkpoint (independent reset).",
        "- Continual evaluation: the model state is carried through the same 15-corruption order for three cycles; prediction is measured before the batch update.",
        "- Clean: clean CIFAR-100 test accuracy after the continual stream; initial clean accuracy is measured from the untouched lookahead anchor.",
        "- Revisit: mean accuracy on the four protocol revisit domains (gaussian_noise, shot_noise, impulse_noise, defocus_blur) after the continual stream, with adaptation disabled.",
        "- Batch size 128, two data-loader workers, seed 0, CUDA, full dataset (`max_batches=-1`).",
        "- Category mapping: Noise = Gaussian/Shot/Impulse; Blur = Defocus/Glass/Motion/Zoom; Weather = Snow/Frost/Fog/Brightness; Digital = Contrast/Elastic/Pixelate/JPEG.",
        "",
        "## Table I. Domain-wise results under independent resets",
        "",
        markdown_table(table1, ["Method", "Noise", "Blur", "Weather", "Digital", "Overall Mean"]),
        "",
        "All values are accuracy percentages; this is a single fixed-seed run, so no standard deviation is shown.",
        "",
        "## Table II. Continual evaluation across three cycles and retention",
        "",
        markdown_table(table2, ["Method", "DW", "Cycle 1", "Cycle 2", "Cycle 3", "Revisit", "Clean"]),
        "",
        "DW is the domain-wise overall mean. Clean is the final clean accuracy after the continual stream.",
        "",
        "## Validation",
        "",
        "| Method | Domains | Cycles | Domain samples | Continual samples | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for audit in audits:
        status = "PASS" if not audit["errors"] else "FAIL: " + "; ".join(audit["errors"])
        lines.append(f"| {audit['method']} | {audit['num_domains']} | {audit['num_cycles']} | {audit['domain_samples']} | {audit['continual_samples']} | {status} |")
    (root / "CIFAR100C_RESNET34_TABLES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?", default=Path("outputs/cifar100_resnet34_unified_20260910"))
    args = parser.parse_args()
    results = {method: load_result(args.root, method) for method in METHODS}
    audits = [validate_result(results[method], method) for method in METHODS]
    table1, table2 = make_rows(results)
    write_outputs(args.root, results, audits, table1, table2)
    print(args.root / "CIFAR100C_RESNET34_TABLES.md")
    print("validation:", "PASS" if all(not item["errors"] for item in audits) else "FAIL")


if __name__ == "__main__":
    main()
