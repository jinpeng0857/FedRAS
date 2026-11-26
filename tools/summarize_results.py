#!/usr/bin/env python3
"""Summarize FedRAS experiment result.json files into Markdown tables."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _pick(data: Dict[str, Any], *keys: str) -> Optional[float]:
    summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
    for container in (summary, data):
        for key in keys:
            val = container.get(key)
            if isinstance(val, (int, float)):
                return float(val)
    return None


def _fmt(val: Optional[float], signed: bool = False) -> str:
    if val is None:
        return "-"
    return f"{val:+.2f}" if signed else f"{val:.2f}"


def collect_results(input_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(input_dir.rglob("result.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows.append(
            {
                "run": path.parent.name,
                "method": data.get("method") or data.get("public_tta_entrypoint") or "-",
                "dataset": data.get("dataset", "-"),
                "backbone": data.get("backbone", "-"),
                "domain_acc": _pick(data, "overall_accuracy", "domain_wise_accuracy", "mean_acc"),
                "cycle_1": _pick(data, "cycle_1_accuracy", "cycle1_acc", "cycle_1"),
                "cycle_3": _pick(data, "cycle_3_accuracy", "cycle3_acc", "cycle_3"),
                "delta_3_1": _pick(data, "delta_3_1", "cycle_3_minus_1"),
                "revisit": _pick(data, "revisit_accuracy", "revisit_acc", "revisit"),
                "clean": _pick(data, "clean_accuracy", "clean_acc", "clean"),
            }
        )
    return rows


def render_markdown(rows: List[Dict[str, Any]]) -> str:
    lines = [
        "# FedRAS Experiment Summary",
        "",
        "| Run | Method | Dataset | Backbone | Domain-wise ↑ | Cycle 1 ↑ | Cycle 3 ↑ | Δ3−1 ↑ | Revisit ↑ | Clean ↑ |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['run']} | {r['method']} | {r['dataset']} | {r['backbone']} "
            f"| {_fmt(r['domain_acc'])} | {_fmt(r['cycle_1'])} | {_fmt(r['cycle_3'])} "
            f"| {_fmt(r['delta_3_1'], signed=True)} | {_fmt(r['revisit'])} | {_fmt(r['clean'])} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize result.json files into Markdown.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/fedras_pdf_supplement"),
        help="Directory to recursively search for result.json files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/summary_tables.md"),
        help="Output Markdown file path.",
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise SystemExit(f"Input directory not found: {args.input_dir}")

    rows = collect_results(args.input_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(rows), encoding="utf-8")
    print(f"Summarized {len(rows)} result(s) -> {args.output}")


if __name__ == "__main__":
    main()
