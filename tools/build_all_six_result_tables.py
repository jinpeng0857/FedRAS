#!/usr/bin/env python3
"""Create one Markdown document containing the six requested result tables."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean


ROOT = Path("outputs")
METHODS = ["Source", "Tent", "EATA", "SAR", "ASR", "ReCAP", "FedDR"]
NOISE = {"gaussian_noise", "shot_noise", "impulse_noise"}
BLUR = {"defocus_blur", "glass_blur", "motion_blur", "zoom_blur"}
WEATHER = {"snow", "frost", "fog", "brightness"}
DIGITAL = {"contrast", "elastic_transform", "pixelate", "jpeg_compression"}
CATEGORIES = {"Noise": NOISE, "Blur": BLUR, "Weather": WEATHER, "Digital": DIGITAL}


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def tiny_paths(backbone: str) -> dict[str, Path]:
    base = ROOT / f"tiny_imagenet_{backbone}_20260907"
    tta = ROOT / f"tiny_imagenet_{backbone}_tta5_20260908"
    return {
        "Source": base / "source.json",
        "Tent": tta / "tent.json",
        "EATA": tta / "eata.json",
        "SAR": tta / "sar.json",
        "ASR": tta / "asr.json",
        "ReCAP": tta / "recap.json",
        "FedDR": base / "feddr.json",
    }


def final_clean(result: dict) -> float:
    clean = result["clean_retention"]
    return clean["final_clean"]["accuracy"] if "final_clean" in clean else clean["final_continual_clean_accuracy"]


def domain_table_for_single_seed(paths: dict[str, Path]) -> list[dict[str, float | str]]:
    rows = []
    for method in METHODS:
        result = read(paths[method])
        by_name = {entry["corruption"]: entry["accuracy"] for entry in result["domain_wise"]["per_corruption"]}
        row: dict[str, float | str] = {"Method": method}
        for category, names in CATEGORIES.items():
            row[category] = mean(by_name[name] for name in names)
        row["Overall Mean"] = result["domain_wise"]["summary"]["mean_accuracy"]
        rows.append(row)
    return rows


def continual_table_for_single_seed(paths: dict[str, Path]) -> list[dict[str, float | str]]:
    rows = []
    for method in METHODS:
        result = read(paths[method])
        cycles = result["continual"]["summary"]["cycle_summaries"]
        rows.append({
            "Method": method,
            "DW": result["domain_wise"]["summary"]["mean_accuracy"],
            "Cycle 1": cycles[0]["mean_accuracy"],
            "Cycle 2": cycles[1]["mean_accuracy"],
            "Cycle 3": cycles[2]["mean_accuracy"],
            "Revisit": result["continual"]["post_stream"]["mean_revisit_accuracy"],
            "Clean": final_clean(result),
        })
    return rows


def fmt_single(value: float) -> str:
    return f"{100.0 * value:.1f}"


def fmt_cifar(values: list[float]) -> str:
    avg = mean(values)
    spread = 0.0
    if len(values) > 1:
        n = len(values)
        spread = (sum((value - avg) ** 2 for value in values) / (n - 1)) ** 0.5
    return f"{100.0 * avg:.1f} ± {100.0 * spread:.1f}"


def markdown_single(rows: list[dict], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        lines.append("| " + " | ".join([row["Method"]] + [fmt_single(row[key]) for key in columns[1:]]) + " |")
    return "\n".join(lines)


def markdown_cifar(rows: list[dict], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        cells = [row["Method"]] + [fmt_cifar(row[key]) for key in columns[1:]]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def load_cifar_rows() -> tuple[list[dict], list[dict]]:
    summary = read(ROOT / "cifar100_resnet34_unified_20260910" / "CIFAR100C_RESNET34_THREE_SEED_SUMMARY.json")
    return summary["table_domain_wise"], summary["table_continual"]


def main() -> None:
    cifar_domain, cifar_continual = load_cifar_rows()
    tiny18 = tiny_paths("resnet18")
    tiny34 = tiny_paths("resnet34")
    sections = [
        (
            "表 1：ResNet34 + CIFAR-100-C：逐域重置",
            "cifar",
            cifar_domain,
            ["Method", "Noise", "Blur", "Weather", "Digital", "Overall Mean"],
        ),
        (
            "表 2：ResNet34 + CIFAR-100-C：三轮连续适应及 Clean/Revisit",
            "cifar",
            cifar_continual,
            ["Method", "DW", "Cycle 1", "Cycle 2", "Cycle 3", "Revisit", "Clean"],
        ),
        (
            "表 3：ResNet18 + Tiny-ImageNet-C：逐域重置",
            "tiny",
            domain_table_for_single_seed(tiny18),
            ["Method", "Noise", "Blur", "Weather", "Digital", "Overall Mean"],
        ),
        (
            "表 4：ResNet18 + Tiny-ImageNet-C：三轮连续适应及 Clean/Revisit",
            "tiny",
            continual_table_for_single_seed(tiny18),
            ["Method", "DW", "Cycle 1", "Cycle 2", "Cycle 3", "Revisit", "Clean"],
        ),
        (
            "表 5：ResNet34 + Tiny-ImageNet-C：逐域重置",
            "tiny",
            domain_table_for_single_seed(tiny34),
            ["Method", "Noise", "Blur", "Weather", "Digital", "Overall Mean"],
        ),
        (
            "表 6：ResNet34 + Tiny-ImageNet-C：三轮连续适应及 Clean/Revisit",
            "tiny",
            continual_table_for_single_seed(tiny34),
            ["Method", "DW", "Cycle 1", "Cycle 2", "Cycle 3", "Revisit", "Clean"],
        ),
    ]

    lines = [
        "# ResNet/CIFAR-100-C 与 Tiny-ImageNet-C 测试结果汇总",
        "",
        "六张表均包含 Source、Tent、EATA、SAR、ASR、ReCAP 和 FedDR 七个算法。",
        "",
        "统一测试协议：severity=5；15 个 corruption；每个 corruption 使用 10,000 张图像；逐域重置从同一 source checkpoint 独立开始；连续流按固定 15 域顺序连续适应 3 轮；预测采用 pre-update；batch size=128；完整测试集；Clean 为连续流结束后的 CIFAR-100/Tiny-ImageNet clean 测试准确率；Revisit 为连续流结束后冻结模型在固定四个域上的平均准确率。",
        "",
        "表中类别划分为：Noise = Gaussian/Shot/Impulse；Blur = Defocus/Glass/Motion/Zoom；Weather = Snow/Frost/Fog/Brightness；Digital = Contrast/Elastic/Pixelate/JPEG。",
        "",
        "CIFAR-100-C 的 ResNet34 结果为 seed=0、1、2 三个独立 seed 的均值 ± 样本标准差；Tiny-ImageNet-C 的 ResNet18 和 ResNet34 结果为当前已完成的 seed=0 单次结果，因此不显示标准差。",
        "",
    ]
    for title, kind, rows, columns in sections:
        lines.append(f"## {title}")
        lines.append("")
        lines.append(markdown_cifar(rows, columns) if kind == "cifar" else markdown_single(rows, columns))
        lines.append("")

    lines.extend([
        "## 结果文件",
        "",
        "- ResNet34 + CIFAR-100-C（三 seed 汇总）：`outputs/cifar100_resnet34_unified_20260910/CIFAR100C_RESNET34_THREE_SEED_SUMMARY.json`",
        "- ResNet18 + Tiny-ImageNet-C：`outputs/tiny_imagenet_resnet18_tta5_20260908/TTA5_RESULTS_WITH_FEDDR.md`",
        "- ResNet34 + Tiny-ImageNet-C：`outputs/tiny_imagenet_resnet34_tta5_20260908/TTA5_RESULTS_WITH_FEDDR.md`",
        "",
        "CIFAR-100-C 三个 seed 的原始结果分别保存在 `outputs/cifar100_resnet34_unified_20260910/`、`outputs/cifar100_resnet34_unified_20260910_seed1/` 和 `outputs/cifar100_resnet34_unified_20260910_seed2/`。",
    ])
    out = ROOT / "ALL_RESNET_TINY_CIFAR_C_RESULTS.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
