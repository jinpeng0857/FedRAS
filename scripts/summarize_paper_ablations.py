#!/usr/bin/env python3
"""Summarize and Format FedDR Paper Ablation Results.

Parses JSON outputs produced by `run_paper_ablations.py`, aggregates results
across FedACG checkpoints (mean +- sample SD), and generates:
  - Clean Markdown summary tables
  - Ready-to-use LaTeX code formatted for Table 3 in the manuscript.
"""

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

# Ensure UTF-8 output on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize FedDR Paper Ablation Results (Blocks A, B, C)"
    )
    parser.add_argument(
        "--output_dir",
        default="test/outputs/paper_ablations",
        help="Root directory where ablation outputs are stored.",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=["run1", "run2", "run3"],
        help="Run names to aggregate.",
    )
    parser.add_argument(
        "--random_seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
        help="Random seeds used for Block C random selection ablation.",
    )
    parser.add_argument(
        "--latex_out",
        default=None,
        help="Optional path to save formatted LaTeX table snippet.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise error and exit with code 1 if any expected run or seed output is missing.",
    )
    return parser.parse_args()


def load_metrics_from_json(json_path: Path) -> Optional[Dict[str, float]]:
    """Extract standard evaluation metrics from schema v9 output JSON."""
    if not json_path.exists() or json_path.stat().st_size == 0:
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 1. Domain-wise average (15 corruptions) -> convert decimal to percentage
        domain_wise_acc = None
        if "domain_wise" in data and "summary" in data["domain_wise"]:
            raw_dom = data["domain_wise"]["summary"].get("mean_accuracy")
            if raw_dom is not None and math.isfinite(float(raw_dom)):
                domain_wise_acc = float(raw_dom) * 100.0

        # 2. Continual evaluation metrics -> convert decimal to percentage
        continual_data = data.get("continual", {})
        summary_data = continual_data.get("summary", {})
        cycle_summaries = summary_data.get("cycle_summaries", [])

        c1 = None
        c3 = None
        if len(cycle_summaries) >= 3:
            c1_raw = cycle_summaries[0].get("mean_accuracy")
            c3_raw = cycle_summaries[2].get("mean_accuracy")
            if c1_raw is not None and math.isfinite(float(c1_raw)):
                c1 = float(c1_raw) * 100.0
            if c3_raw is not None and math.isfinite(float(c3_raw)):
                c3 = float(c3_raw) * 100.0

        delta = None
        raw_delta = summary_data.get("late_delta_cycle3_minus_cycle1")
        if raw_delta is not None and math.isfinite(float(raw_delta)):
            delta = float(raw_delta) * 100.0
        elif c1 is not None and c3 is not None:
            delta = c3 - c1

        # 3. Post-stream Revisit accuracy -> convert decimal to percentage
        revisit = None
        post_stream = continual_data.get("post_stream", {})
        raw_rev = post_stream.get("mean_revisit_accuracy")
        if raw_rev is not None and math.isfinite(float(raw_rev)):
            revisit = float(raw_rev) * 100.0

        # 4. Clean Retention -> convert decimal to percentage
        clean = None
        clean_retention = data.get("clean_retention", {})
        raw_clean = clean_retention.get("final_continual_clean_accuracy")
        if raw_clean is None:
            raw_clean = clean_retention.get("final_clean", {}).get("accuracy")
        if raw_clean is not None and math.isfinite(float(raw_clean)):
            clean = float(raw_clean) * 100.0

        # 5. Drift reduction
        drift_reduction = None
        restoration_data = continual_data.get("restoration", {})
        if "total_drift_reduction" in restoration_data:
            drift_reduction = float(restoration_data["total_drift_reduction"])

        return {
            "domain_wise": domain_wise_acc if domain_wise_acc is not None else float("nan"),
            "cycle1": float(c1) if c1 is not None else float("nan"),
            "cycle3": float(c3) if c3 is not None else float("nan"),
            "delta": float(delta) if delta is not None else float("nan"),
            "revisit": float(revisit) if revisit is not None else float("nan"),
            "clean": float(clean) if clean is not None else float("nan"),
            "drift_reduction": drift_reduction,
        }
    except Exception as e:
        print(f"Warning: Failed to parse {json_path}: {e}")
        return None


def format_mean_std(values: List[float], is_delta: bool = False, precision: int = 1) -> str:
    """Format array of floats as 'Mean +- Std' for LaTeX."""
    valid_vals = [v for v in values if math.isfinite(v)]
    if not valid_vals:
        return "--"
    mean = float(np.mean(valid_vals))
    std = float(np.std(valid_vals, ddof=1)) if len(valid_vals) > 1 else 0.0

    if is_delta:
        sign = "+" if mean > 0 else ""
        return f"{sign}{mean:.{precision}f} $\\pm$ {std:.{precision}f}"
    return f"{mean:.{precision}f} $\\pm$ {std:.{precision}f}"


def format_mean_std_md(values: List[float], is_delta: bool = False, precision: int = 1) -> str:
    """Format array of floats as 'Mean +- Std' for Markdown."""
    valid_vals = [v for v in values if math.isfinite(v)]
    if not valid_vals:
        return "--"
    mean = float(np.mean(valid_vals))
    std = float(np.std(valid_vals, ddof=1)) if len(valid_vals) > 1 else 0.0

    if is_delta:
        sign = "+" if mean > 0 else ""
        return f"{sign}{mean:.{precision}f} ± {std:.{precision}f}"
    return f"{mean:.{precision}f} ± {std:.{precision}f}"


def summarize_block_a(
    output_root: Path, runs: List[str], missing_files: List[str]
) -> Tuple[str, str, bool]:
    """Summarize Block A: Core Module Decomposition."""
    block_dir = output_root / "block_a_core"
    variants = [
        ("Base", "block_a_base", "✗", "✗"),
        ("DLRS-only", "block_a_dlrs_only", "✓", "✗"),
        ("FSLR-only", "block_a_fslr_only", "✗", "✓"),
        ("FedDR-Core", "block_a_feddr_core", "✓", "✓"),
        ("FedDR-Full", "block_a_feddr_full", "✓", "✓"),
    ]

    md_lines = [
        "### Block A: Core Module Decomposition",
        "| Method | DLRS | FSLR | Domain (%) | Cycle 1 (%) | Cycle 3 (%) | Δ C3-C1 (p.p.) | Revisit (%) | Clean (%) |",
        "|---|:---:|:---:|---:|---:|---:|---:|---:|---:|",
    ]

    latex_lines = [
        r"\multicolumn{9}{l}{\textit{(A) Core Module Decomposition}} \\",
    ]

    is_complete = True

    for label, prefix, dlrs_mark, fslr_mark in variants:
        domain_list, c1_list, c3_list, delta_list, rev_list, clean_list = [], [], [], [], [], []
        for r in runs:
            f_path = block_dir / f"{prefix}_{r}.json"
            m = load_metrics_from_json(f_path)
            if m and math.isfinite(m["cycle3"]):
                domain_list.append(m["domain_wise"])
                c1_list.append(m["cycle1"])
                c3_list.append(m["cycle3"])
                delta_list.append(m["delta"])
                rev_list.append(m["revisit"])
                clean_list.append(m["clean"])
            else:
                missing_files.append(str(f_path))
                is_complete = False

        dom_s_md = format_mean_std_md(domain_list)
        c1_s_md = format_mean_std_md(c1_list)
        c3_s_md = format_mean_std_md(c3_list)
        del_s_md = format_mean_std_md(delta_list, is_delta=True)
        rev_s_md = format_mean_std_md(rev_list)
        cln_s_md = format_mean_std_md(clean_list)

        dom_s_tex = format_mean_std(domain_list)
        c1_s_tex = format_mean_std(c1_list)
        c3_s_tex = format_mean_std(c3_list)
        del_s_tex = format_mean_std(delta_list, is_delta=True)
        rev_s_tex = format_mean_std(rev_list)
        cln_s_tex = format_mean_std(clean_list)

        md_lines.append(
            f"| {label} | {dlrs_mark} | {fslr_mark} | {dom_s_md} | {c1_s_md} | {c3_s_md} | {del_s_md} | {rev_s_md} | {cln_s_md} |"
        )

        name_tex = f"\\textbf{{{label}}}" if "Full" in label else label
        d_tex = r"\checkmark" if dlrs_mark == "✓" else ""
        f_tex = r"\checkmark" if fslr_mark == "✓" else ""
        latex_lines.append(
            f"{name_tex} & {d_tex} & {f_tex} & {dom_s_tex} & {c1_s_tex} & {c3_s_tex} & {del_s_tex} & {rev_s_tex} & {cln_s_tex} \\\\"
        )

    return "\n".join(md_lines), "\n".join(latex_lines), is_complete


def summarize_block_b(
    output_root: Path, runs: List[str], missing_files: List[str]
) -> Tuple[str, str, bool]:
    """Summarize Block B: DLRS Joint Weighting Ablation."""
    block_dir = output_root / "block_b_weighting"
    variants = [
        ("Uniform", "block_b_weight_uniform"),
        ("Sample-only", "block_b_weight_sample_only"),
        ("Region-only", "block_b_weight_region_only"),
        ("Product", "block_b_weight_product"),
    ]

    md_lines = [
        "### Block B: DLRS Joint Weighting Ablation (Fixed Radius η=1.2)",
        "| DLRS weighting | Domain (%) | Cycle 1 (%) | Cycle 3 (%) | Δ C3-C1 (p.p.) | Revisit (%) | Clean (%) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    latex_lines = [
        r"\multicolumn{9}{l}{\textit{(B) DLRS Joint Weighting Formulation (Fixed Search Radius $\eta=1.2$)}} \\",
    ]

    is_complete = True

    for label, prefix in variants:
        domain_list, c1_list, c3_list, delta_list, rev_list, clean_list = [], [], [], [], [], []
        for r in runs:
            f_path = block_dir / f"{prefix}_{r}.json"
            m = load_metrics_from_json(f_path)
            if m and math.isfinite(m["cycle3"]):
                domain_list.append(m["domain_wise"])
                c1_list.append(m["cycle1"])
                c3_list.append(m["cycle3"])
                delta_list.append(m["delta"])
                rev_list.append(m["revisit"])
                clean_list.append(m["clean"])
            else:
                missing_files.append(str(f_path))
                is_complete = False

        dom_s_md = format_mean_std_md(domain_list)
        c1_s_md = format_mean_std_md(c1_list)
        c3_s_md = format_mean_std_md(c3_list)
        del_s_md = format_mean_std_md(delta_list, is_delta=True)
        rev_s_md = format_mean_std_md(rev_list)
        cln_s_md = format_mean_std_md(clean_list)

        dom_s_tex = format_mean_std(domain_list)
        c1_s_tex = format_mean_std(c1_list)
        c3_s_tex = format_mean_std(c3_list)
        del_s_tex = format_mean_std(delta_list, is_delta=True)
        rev_s_tex = format_mean_std(rev_list)
        cln_s_tex = format_mean_std(clean_list)

        md_lines.append(
            f"| {label} | {dom_s_md} | {c1_s_md} | {c3_s_md} | {del_s_md} | {rev_s_md} | {cln_s_md} |"
        )

        name_tex = f"\\textbf{{{label}}}" if label == "Product" else label
        latex_lines.append(
            f"{name_tex} & & & {dom_s_tex} & {c1_s_tex} & {c3_s_tex} & {del_s_tex} & {rev_s_tex} & {cln_s_tex} \\\\"
        )

    return "\n".join(md_lines), "\n".join(latex_lines), is_complete


def summarize_block_c(
    output_root: Path,
    runs: List[str],
    random_seeds: List[int],
    missing_files: List[str],
) -> Tuple[str, str, bool]:
    """Summarize Block C: FSLR Coordinate Selection Ablation."""
    block_dir = output_root / "block_c_selection"
    variants = [
        ("Random", "random"),
        ("Drift-only", "block_c_select_drift_only"),
        ("Fisher-only", "block_c_select_fisher_only"),
        ("Fisher × Drift", "block_c_select_fisher_drift"),
    ]

    md_lines = [
        "### Block C: FSLR Coordinate Selection Ablation (Fixed Budget 5%)",
        "| FSLR selection | Domain (%) | Cycle 1 (%) | Cycle 3 (%) | Δ C3-C1 (p.p.) | Revisit (%) | Clean (%) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    latex_lines = [
        r"\multicolumn{9}{l}{\textit{(C) FSLR Coordinate Selection Criterion (Fixed Budget $\rho=5\%$)}} \\",
    ]

    is_complete = True

    for label, prefix in variants:
        domain_list, c1_list, c3_list, delta_list, rev_list, clean_list = [], [], [], [], [], []

        if prefix == "random":
            # For each run, average over the random seeds first
            for r in runs:
                seed_dom, seed_c1, seed_c3, seed_del, seed_rev, seed_cln = [], [], [], [], [], []
                for seed in random_seeds:
                    f_path = block_dir / f"block_c_select_random_s{seed}_{r}.json"
                    m = load_metrics_from_json(f_path)
                    if m and math.isfinite(m["cycle3"]):
                        seed_dom.append(m["domain_wise"])
                        seed_c1.append(m["cycle1"])
                        seed_c3.append(m["cycle3"])
                        seed_del.append(m["delta"])
                        seed_rev.append(m["revisit"])
                        seed_cln.append(m["clean"])
                    else:
                        missing_files.append(str(f_path))
                        is_complete = False

                if seed_c3:
                    domain_list.append(float(np.nanmean(seed_dom)))
                    c1_list.append(float(np.nanmean(seed_c1)))
                    c3_list.append(float(np.nanmean(seed_c3)))
                    delta_list.append(float(np.nanmean(seed_del)))
                    rev_list.append(float(np.nanmean(seed_rev)))
                    clean_list.append(float(np.nanmean(seed_cln)))
        else:
            for r in runs:
                f_path = block_dir / f"{prefix}_{r}.json"
                m = load_metrics_from_json(f_path)
                if m and math.isfinite(m["cycle3"]):
                    domain_list.append(m["domain_wise"])
                    c1_list.append(m["cycle1"])
                    c3_list.append(m["cycle3"])
                    delta_list.append(m["delta"])
                    rev_list.append(m["revisit"])
                    clean_list.append(m["clean"])
                else:
                    missing_files.append(str(f_path))
                    is_complete = False

        dom_s_md = format_mean_std_md(domain_list)
        c1_s_md = format_mean_std_md(c1_list)
        c3_s_md = format_mean_std_md(c3_list)
        del_s_md = format_mean_std_md(delta_list, is_delta=True)
        rev_s_md = format_mean_std_md(rev_list)
        cln_s_md = format_mean_std_md(clean_list)

        dom_s_tex = format_mean_std(domain_list)
        c1_s_tex = format_mean_std(c1_list)
        c3_s_tex = format_mean_std(c3_list)
        del_s_tex = format_mean_std(delta_list, is_delta=True)
        rev_s_tex = format_mean_std(rev_list)
        cln_s_tex = format_mean_std(clean_list)

        md_lines.append(
            f"| {label} | {dom_s_md} | {c1_s_md} | {c3_s_md} | {del_s_md} | {rev_s_md} | {cln_s_md} |"
        )

        name_tex = f"\\textbf{{{label}}}" if "Fisher" in label and "Drift" in label else label
        latex_lines.append(
            f"{name_tex} & & & {dom_s_tex} & {c1_s_tex} & {c3_s_tex} & {del_s_tex} & {rev_s_tex} & {cln_s_tex} \\\\"
        )

    return "\n".join(md_lines), "\n".join(latex_lines), is_complete


def main():
    args = parse_args()
    output_root = Path(args.output_dir)

    print("=" * 80)
    print("FedDR Main Paper Ablation Study Summary (Blocks A, B, C)")
    print(f"Source Directory: {output_root}")
    print(f"Runs Included: {args.runs}")
    print(f"Random Seeds: {args.random_seeds}")
    print("=" * 80)

    missing_files: List[str] = []
    md_a, tex_a, comp_a = summarize_block_a(output_root, args.runs, missing_files)
    md_b, tex_b, comp_b = summarize_block_b(output_root, args.runs, missing_files)
    md_c, tex_c, comp_c = summarize_block_c(output_root, args.runs, args.random_seeds, missing_files)

    all_complete = comp_a and comp_b and comp_c

    full_md = f"# FedDR Ablation Summary\n\n{md_a}\n\n{md_b}\n\n{md_c}\n"
    print(full_md)

    if missing_files:
        print(f"Notice: {len(missing_files)} expected JSON output(s) are missing or incomplete.")
        if len(missing_files) <= 10:
            for mf in missing_files:
                print(f"  - Missing: {mf}")
        else:
            print(f"  (First 5 missing: {missing_files[:5]})")

    if args.latex_out:
        if args.strict and not all_complete:
            print(f"Error: Strict mode enabled and {len(missing_files)} files are missing. Refusing to write LaTeX table.", file=sys.stderr)
            sys.exit(1)

        full_latex = (
            r"\begin{table}[t]" + "\n"
            r"\centering" + "\n"
            r"\small" + "\n"
            r"\setlength{\tabcolsep}{4.5pt}" + "\n"
            r"\caption{\textbf{Comprehensive ablation study of FedDR core components on CIFAR-100-C (severity 5).} "
            r"Block~(A) evaluates standardized factorial decomposition and final full configuration. "
            r"Block~(B) evaluates loss weighting formulations under fixed perturbation radius $\eta=1.2$. "
            r"Block~(C) evaluates coordinate selection criteria under fixed budget $\rho=5\%$. "
            r"Reported values are mean $\pm$ sample s.d. across three \FedACG weight checkpoints ($n=3$).}" + "\n"
            r"\label{tab:ablation}" + "\n"
            r"\begin{tabular}{l cc cccc cc}" + "\n"
            r"\toprule" + "\n"
            r"\textbf{Configuration / Variant} & \textbf{DLRS} & \textbf{FSLR} & \textbf{Domain} & \textbf{Cycle 1} & \textbf{Cycle 3} & $\boldsymbol{\Delta_{3-1}}$ & \textbf{Revisit} & \textbf{Clean} \\" + "\n"
            r"\midrule" + "\n"
            + tex_a + "\n"
            r"\midrule" + "\n"
            + tex_b + "\n"
            r"\midrule" + "\n"
            + tex_c + "\n"
            r"\bottomrule" + "\n"
            r"\end{tabular}" + "\n"
            r"\end{table}" + "\n"
        )

        out_file = Path(args.latex_out)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(full_latex, encoding="utf-8")
        print(f"LaTeX snippet successfully written to: {out_file}")

    if args.strict and not all_complete:
        sys.exit(1)


if __name__ == "__main__":
    main()
