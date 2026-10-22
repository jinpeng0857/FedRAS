#!/usr/bin/env python3
"""Create publication-ready FedACG comparison tables from result JSON files.

The three FedACG weight runs are treated as the replicate axis. CoTTA is
explicitly excluded from both tables at the user's request.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm, to_hex
from matplotlib.patches import Rectangle


RUNS = ("fedacg_run1", "fedacg_run2", "fedacg_run3")
METHOD_ORDER = (
    "Source",
    "Tent",
    "EATA",
    "SAR",
    "ASR",
    "RoTTA",
    "PeTTA",
    "DeYO",
    "ReCAP",
    "FedDR",
)
EXCLUDED_METHODS = ("CoTTA",)

CORRUPTION_LABELS = {
    "gaussian_noise": "Gauss.\nnoise",
    "shot_noise": "Shot\nnoise",
    "impulse_noise": "Impulse\nnoise",
    "defocus_blur": "Defocus\nblur",
    "glass_blur": "Glass\nblur",
    "motion_blur": "Motion\nblur",
    "zoom_blur": "Zoom\nblur",
    "snow": "Snow",
    "frost": "Frost",
    "fog": "Fog",
    "brightness": "Brightness",
    "contrast": "Contrast",
    "elastic_transform": "Elastic\ntransform",
    "pixelate": "Pixelate",
    "jpeg_compression": "JPEG\ncompression",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Directory containing fedacg_run1, fedacg_run2 and fedacg_run3.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def load_results(root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    results: dict[str, dict[str, dict[str, Any]]] = {}
    for run in RUNS:
        run_dir = root / run
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Missing result directory: {run_dir}")
        by_method: dict[str, dict[str, Any]] = {}
        for path in sorted(run_dir.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            method = data.get("method")
            if method in EXCLUDED_METHODS:
                continue
            if method in by_method:
                raise ValueError(f"Duplicate method {method!r} in {run_dir}")
            by_method[method] = data
        results[run] = by_method

    missing = {
        method: [run for run in RUNS if method not in results[run]]
        for method in METHOD_ORDER
    }
    missing = {method: runs for method, runs in missing.items() if runs}
    if missing:
        raise ValueError(f"Missing required method results: {missing}")
    return results


def get_domain_values(
    results: dict[str, dict[str, dict[str, Any]]], method: str
) -> tuple[list[str], dict[str, list[float]], list[float]]:
    corruption_order: list[str] | None = None
    values: dict[str, list[float]] = {}
    averages: list[float] = []
    for run in RUNS:
        block = results[run][method]["domain_wise"]
        per_corruption = block["per_corruption"]
        names = [item["corruption"] for item in per_corruption]
        if corruption_order is None:
            corruption_order = names
        elif names != corruption_order:
            raise ValueError(f"Corruption order differs for {method} in {run}")
        for item in per_corruption:
            values.setdefault(item["corruption"], []).append(float(item["accuracy"]))
        averages.append(float(block["summary"]["mean_accuracy"]))
    assert corruption_order is not None
    if set(corruption_order) != set(CORRUPTION_LABELS):
        raise ValueError(
            f"Expected 15 CIFAR-100-C corruptions, found {corruption_order}"
        )
    return corruption_order, values, averages


def get_continual_values(
    results: dict[str, dict[str, dict[str, Any]]], method: str
) -> dict[str, list[float]]:
    metrics = {key: [] for key in ("cycle1", "cycle3", "delta", "revisit", "clean")}
    for run in RUNS:
        continual = results[run][method]["continual"]
        cycles = continual["summary"]["cycle_summaries"]
        cycle1 = float(cycles[0]["mean_accuracy"])
        cycle3 = float(cycles[2]["mean_accuracy"])
        metrics["cycle1"].append(cycle1)
        metrics["cycle3"].append(cycle3)
        metrics["delta"].append(cycle3 - cycle1)
        metrics["revisit"].append(float(continual["post_stream"]["mean_revisit_accuracy"]))
        metrics["clean"].append(float(continual["post_stream"]["clean"]["accuracy"]))
    return metrics


def mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) != 3:
        raise ValueError(f"Expected n=3 values, found {len(values)}")
    return float(np.mean(values)), float(np.std(values, ddof=1))


def format_pm(mean: float, std: float, *, signed: bool = False) -> str:
    mean_pct = mean * 100.0
    std_pct = std * 100.0
    if signed:
        return f"{mean_pct:+.1f}±{std_pct:.1f}"
    return f"{mean_pct:.1f}±{std_pct:.1f}"


def write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_latex_table(
    path: Path,
    caption: str,
    label: str,
    headers: list[str],
    rows: list[list[str]],
    first_col_width: str = "1.25cm",
) -> None:
    def tex_escape(value: str) -> str:
        replacements = {
            "&": r"\&",
            "%": r"\%",
            "#": r"\#",
            "_": r"\_",
        }
        return "".join(replacements.get(char, char) for char in value)

    column_spec = "p{" + first_col_width + "}" + "c" * (len(headers) - 1)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        " & ".join(tex_escape(header) for header in headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        escaped = [tex_escape(str(item)).replace("±", r"$\pm$") for item in row]
        lines.append(" & ".join(escaped) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def save_publication_figure(fig: plt.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def domain_table(
    output: Path,
    results: dict[str, dict[str, dict[str, Any]]],
) -> tuple[list[list[str]], list[list[Any]]]:
    all_values: dict[str, dict[str, tuple[float, float]]] = {}
    raw_rows: list[list[Any]] = []
    corruption_order: list[str] | None = None
    for method in METHOD_ORDER:
        order, values, averages = get_domain_values(results, method)
        corruption_order = order if corruption_order is None else corruption_order
        all_values[method] = {
            name: mean_std(vals) for name, vals in values.items()
        }
        avg_mean, avg_std = mean_std(averages)
        all_values[method]["mean_accuracy"] = (avg_mean, avg_std)
        for run, avg in zip(RUNS, averages):
            raw = {item["corruption"]: item["accuracy"] for item in results[run][method]["domain_wise"]["per_corruption"]}
            for name in order:
                raw_rows.append([run, method, name, raw[name]])
            raw_rows.append([run, method, "mean_accuracy", avg])
    assert corruption_order is not None

    display_columns = corruption_order + ["mean_accuracy"]
    display_headers = [CORRUPTION_LABELS[name] for name in corruption_order] + ["Mean"]
    display_rows: list[list[str]] = []
    means = np.zeros((len(METHOD_ORDER), len(display_columns)), dtype=float)
    stds = np.zeros_like(means)
    for row_idx, method in enumerate(METHOD_ORDER):
        row = [method]
        for col_idx, name in enumerate(display_columns):
            mean, std = all_values[method][name]
            means[row_idx, col_idx] = mean * 100.0
            stds[row_idx, col_idx] = std * 100.0
            row.append(format_pm(mean, std))
        display_rows.append(row)

    write_csv(
        output / "domain_wise_mean_std.csv",
        ["method", "corruption", "mean_accuracy", "std_accuracy", "n", "run1", "run2", "run3"],
        [
            [
                method,
                name,
                all_values[method][name][0],
                all_values[method][name][1],
                3,
                *[
                    next(
                        item["accuracy"]
                        for item in results[run][method]["domain_wise"]["per_corruption"]
                        if item["corruption"] == name
                    )
                    if name != "mean_accuracy"
                    else results[run][method]["domain_wise"]["summary"]["mean_accuracy"]
                    for run in RUNS
                ],
            ]
            for method in METHOD_ORDER
            for name in display_columns
        ],
    )

    write_csv(
        output / "domain_wise_raw_long.csv",
        ["run", "method", "corruption", "accuracy"],
        raw_rows,
    )

    latex_rows = display_rows
    write_latex_table(
        output / "domain_wise_table.tex",
        "Domain-wise CIFAR-100-C accuracy. Values are mean $\\pm$ s.d. across the three FedACG weight runs (n=3).",
        "tab:domain-wise-fedacg",
        ["Method"] + [h.replace("\n", " ") for h in display_headers],
        latex_rows,
        first_col_width="1.35cm",
    )

    fig_width = 16.2
    fig, ax = plt.subplots(figsize=(fig_width, 5.6))
    cmap = mpl.colormaps["YlGnBu"]
    norm = Normalize(vmin=max(35.0, float(np.nanmin(means)) - 1.0), vmax=min(85.0, float(np.nanmax(means)) + 1.0))
    ax.imshow(means, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(display_columns)))
    ax.set_xticklabels(display_headers, rotation=0, ha="center")
    ax.set_yticks(np.arange(len(METHOD_ORDER)))
    ax.set_yticklabels(METHOD_ORDER)
    ax.tick_params(length=0, pad=4)
    ax.set_xlabel("CIFAR-100-C corruption type", labelpad=9)
    ax.set_ylabel("Adaptation method", labelpad=8)
    ax.set_title("Domain-wise performance across CIFAR-100-C corruptions", pad=12, loc="left", fontweight="bold")
    for row_idx in range(len(METHOD_ORDER)):
        for col_idx in range(len(display_columns)):
            luminance = np.mean(cmap(norm(means[row_idx, col_idx]))[:3])
            color = "white" if luminance < 0.56 else "#163247"
            ax.text(
                col_idx,
                row_idx,
                f"{means[row_idx, col_idx]:.1f}±{stds[row_idx, col_idx]:.1f}",
                ha="center",
                va="center",
                color=color,
                fontsize=7.1,
            )
    feddr_idx = METHOD_ORDER.index("FedDR")
    ax.add_patch(Rectangle((-0.5, feddr_idx - 0.5), len(display_columns), 1, fill=False, edgecolor="#c2410c", linewidth=2.0))
    ax.axvline(len(display_columns) - 1.5, color="white", linewidth=1.5)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.018, pad=0.012)
    cbar.set_label("Accuracy (%)", rotation=90, labelpad=8)
    cbar.outline.set_linewidth(0.5)
    fig.text(0.01, 0.005, "Cells: mean ± s.d. across FedACG run1–run3 (n=3; sample s.d.). CoTTA excluded.", fontsize=7, color="#4b5563")
    fig.subplots_adjust(left=0.085, right=0.965, top=0.86, bottom=0.18)
    save_publication_figure(fig, output / "domain_wise_table")
    return display_rows, raw_rows


def continual_table(
    output: Path,
    results: dict[str, dict[str, dict[str, Any]]],
) -> tuple[list[list[str]], list[list[Any]]]:
    metric_order = ("cycle1", "cycle3", "delta", "revisit", "clean")
    headers = ["Method", "Cycle 1 (%)", "Cycle 3 (%)", "Δ C3−C1 (p.p.)", "Revisit (%)", "Clean (%)"]
    aggregate: dict[str, dict[str, tuple[float, float]]] = {}
    raw_rows: list[list[Any]] = []
    for method in METHOD_ORDER:
        metrics = get_continual_values(results, method)
        aggregate[method] = {metric: mean_std(metrics[metric]) for metric in metric_order}
        for run_idx, run in enumerate(RUNS):
            raw_rows.append([run, method, *[metrics[metric][run_idx] for metric in metric_order]])

    display_rows: list[list[str]] = []
    means = np.zeros((len(METHOD_ORDER), len(metric_order)), dtype=float)
    stds = np.zeros_like(means)
    for row_idx, method in enumerate(METHOD_ORDER):
        row = [method]
        for col_idx, metric in enumerate(metric_order):
            mean, std = aggregate[method][metric]
            means[row_idx, col_idx] = mean * 100.0
            stds[row_idx, col_idx] = std * 100.0
            row.append(format_pm(mean, std, signed=(metric == "delta")))
        display_rows.append(row)

    write_csv(
        output / "continual_mean_std.csv",
        ["method", "cycle1_mean", "cycle1_std", "cycle3_mean", "cycle3_std", "delta_mean", "delta_std", "revisit_mean", "revisit_std", "clean_mean", "clean_std", "n"],
        [
            [
                method,
                aggregate[method]["cycle1"][0], aggregate[method]["cycle1"][1],
                aggregate[method]["cycle3"][0], aggregate[method]["cycle3"][1],
                aggregate[method]["delta"][0], aggregate[method]["delta"][1],
                aggregate[method]["revisit"][0], aggregate[method]["revisit"][1],
                aggregate[method]["clean"][0], aggregate[method]["clean"][1],
                3,
            ]
            for method in METHOD_ORDER
        ],
    )
    write_csv(
        output / "continual_raw.csv",
        ["run", "method", *metric_order],
        raw_rows,
    )
    write_latex_table(
        output / "continual_table.tex",
        "Continual CIFAR-100-C performance and retention. Values are mean $\\pm$ s.d. across the three FedACG weight runs (n=3).",
        "tab:continual-fedacg",
        headers,
        display_rows,
        first_col_width="1.35cm",
    )

    fig, ax = plt.subplots(figsize=(10.7, 5.5))
    ax.axis("off")
    table_data = display_rows
    table = ax.table(
        cellText=table_data,
        rowLabels=None,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        bbox=[0.015, 0.18, 0.97, 0.70],
        colWidths=[0.16, 0.15, 0.15, 0.20, 0.15, 0.15],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.1)
    table.scale(1.0, 1.55)
    metric_norms: list[Any] = [
        Normalize(vmin=40, vmax=80),
        Normalize(vmin=40, vmax=80),
        TwoSlopeNorm(vmin=-8, vcenter=0, vmax=4),
        Normalize(vmin=35, vmax=75),
        Normalize(vmin=40, vmax=85),
    ]
    cmaps = [mpl.colormaps["YlGnBu"], mpl.colormaps["YlGnBu"], mpl.colormaps["RdYlGn"], mpl.colormaps["YlGnBu"], mpl.colormaps["YlGnBu"]]
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d1d5db")
        cell.set_linewidth(0.45)
        if row == 0:
            cell.set_facecolor("#173f5f")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
            cell.set_height(0.085)
        elif col == -1:
            cell.set_facecolor("#f3f4f6")
            cell.get_text().set_weight("bold")
        else:
            metric_idx = col - 1
            value = means[row - 1, metric_idx]
            face = cmaps[metric_idx](metric_norms[metric_idx](value))
            cell.set_facecolor(face)
            cell.get_text().set_color("#163247" if np.mean(face[:3]) > 0.56 else "white")
            if row - 1 == METHOD_ORDER.index("FedDR"):
                cell.set_edgecolor("#c2410c")
                cell.set_linewidth(1.5)
    ax.set_title("Continual adaptation, revisit and clean retention", pad=18, loc="left", fontweight="bold")
    fig.text(0.015, 0.105, "Δ = Cycle 3 − Cycle 1; positive values indicate late-stream improvement.", fontsize=7.5, color="#4b5563")
    fig.text(0.015, 0.055, "Cells: mean ± s.d. across FedACG run1–run3 (n=3; sample s.d.). CoTTA excluded.", fontsize=7, color="#4b5563")
    fig.subplots_adjust(left=0.02, right=0.99, top=0.92, bottom=0.05)
    save_publication_figure(fig, output / "continual_table")
    return display_rows, raw_rows


def write_metadata(output: Path, root: Path) -> None:
    metadata = {
        "source_root": str(root),
        "replicate_axis": list(RUNS),
        "n": 3,
        "standard_deviation": "sample standard deviation (ddof=1)",
        "excluded_methods": list(EXCLUDED_METHODS),
        "methods": list(METHOD_ORDER),
        "domain_wise_metric": "domain_wise.summary.mean_accuracy and per_corruption.accuracy",
        "continual_metrics": {
            "cycle1": "continual.summary.cycle_summaries[0].mean_accuracy",
            "cycle3": "continual.summary.cycle_summaries[2].mean_accuracy",
            "delta": "cycle3 - cycle1, percentage-point scale in display",
            "revisit": "continual.post_stream.mean_revisit_accuracy",
            "clean": "continual.post_stream.clean.accuracy",
        },
        "note": "run1-run3 are FedACG weight runs; they are not claimed to be independent random seeds.",
    }
    (output / "README.txt").write_text(
        "FedACG result-table package\n\n"
        "Statistics: mean ± sample standard deviation across the three FedACG weight runs (run1, run2, run3), n=3.\n"
        "CoTTA is intentionally excluded from both tables.\n"
        "The three runs are weight/checkpoint runs, not independent random seeds.\n\n"
        "Outputs: SVG/PDF/TIFF/PNG figures, CSV source data, and LaTeX tables.\n",
        encoding="utf-8",
    )
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    args.output.mkdir(parents=True, exist_ok=True)
    results = load_results(args.root)
    domain_table(args.output, results)
    continual_table(args.output, results)
    write_metadata(args.output, args.root)
    print(f"Wrote publication tables to {args.output}")


if __name__ == "__main__":
    main()
