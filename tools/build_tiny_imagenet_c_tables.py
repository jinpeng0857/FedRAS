import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "TINY_IMAGENET_C_RESULTS_TABLES.md"


METHOD_LABELS = {
    "source": "Source",
    "feddr": "FedDR",
    "tent": "Tent",
    "eata": "EATA",
    "sar": "SAR",
    "asr": "ASR",
    "recap": "ReCAP",
}
METHOD_ORDER = ["source", "feddr", "tent", "eata", "sar", "asr", "recap"]
TTA_METHODS = ["tent", "eata", "sar", "asr", "recap"]


RUNS = {
    "ResNet-18": {
        "source": ROOT / "outputs/tiny_imagenet_resnet18_20260907/source.json",
        "feddr": ROOT / "outputs/tiny_imagenet_resnet18_20260907/feddr.json",
        **{
            method: ROOT / f"outputs/tiny_imagenet_resnet18_tta5_20260908/{method}.json"
            for method in TTA_METHODS
        },
    },
    "ResNet-34": {
        "source": ROOT / "outputs/tiny_imagenet_resnet34_20260907/source.json",
        "feddr": ROOT / "outputs/tiny_imagenet_resnet34_20260907/feddr.json",
        **{
            method: ROOT / f"outputs/tiny_imagenet_resnet34_tta5_20260908/{method}.json"
            for method in TTA_METHODS
        },
    },
}


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value):
    return f"{100.0 * float(value):.2f}"


def clean_values(data):
    clean = data["clean_retention"]
    if "initial_clean" in clean:
        initial = clean["initial_clean"]["accuracy"]
        final = clean["final_clean"]["accuracy"]
    else:
        initial = clean["initial_clean_accuracy"]
        final = clean["final_continual_clean_accuracy"]
    return initial, final, final - initial


def revisit_mean(data):
    revisit = data["continual"]["post_stream"]["revisit"]
    return sum(float(item["accuracy"]) for item in revisit) / len(revisit)


def summary(data):
    initial_clean, final_clean, clean_delta = clean_values(data)
    cycles = data["continual"]["summary"].get("cycle_summaries", [])
    cycle_values = [item["mean_accuracy"] for item in cycles]
    cycle_delta = data["continual"]["summary"].get("late_delta_cycle3_minus_cycle1")
    if cycle_delta is None:
        cycle_delta = data["continual"]["summary"].get("cycle3_minus_cycle1", cycle_values[-1] - cycle_values[0])
    drift = data["continual"]["summary"].get("final_bn_affine_drift")
    return {
        "domain": data["domain_wise"]["summary"]["mean_accuracy"],
        "cycles": cycle_values,
        "cycle_delta": cycle_delta,
        "initial_clean": initial_clean,
        "final_clean": final_clean,
        "clean_delta": clean_delta,
        "revisit": revisit_mean(data),
        "bn_drift": None if drift is None else drift.get("total"),
        "domain_samples": data["domain_wise"]["summary"]["total_samples"],
        "continual_samples": data["continual"]["summary"]["total_samples"],
    }


def corruption_map(data):
    return {
        item["corruption"]: item["accuracy"]
        for item in data["domain_wise"]["per_corruption"]
    }


def row(values):
    return "| " + " | ".join(values) + " |"


def build_model_section(model_name, paths):
    data = {method: load(path) for method, path in paths.items()}
    summaries = {method: summary(value) for method, value in data.items()}
    maps = {method: corruption_map(value) for method, value in data.items()}
    corruptions = list(maps["source"])

    lines = [f"## {model_name}", "", "### 汇总表", ""]
    lines.append(row(["方法", "逐域重置", "C1", "C2", "C3", "C3-C1", "初始 Clean", "最终 Clean", "Clean 变化", "Revisit", "最终 BN 漂移"]))
    lines.append(row(["---"] + ["---:"] * 10))
    for method in METHOD_ORDER:
        item = summaries[method]
        cycles = item["cycles"] + [None] * (3 - len(item["cycles"]))
        lines.append(row([
            METHOD_LABELS[method],
            pct(item["domain"]),
            pct(cycles[0]),
            pct(cycles[1]),
            pct(cycles[2]),
            f"{100 * item['cycle_delta']:.2f}",
            pct(item["initial_clean"]),
            pct(item["final_clean"]),
            f"{100 * item['clean_delta']:.2f}",
            pct(item["revisit"]),
            "—" if item["bn_drift"] is None else f"{item['bn_drift']:.3f}",
        ]))

    lines += ["", "### 逐扰动逐域重置表", ""]
    columns = ["扰动"] + [METHOD_LABELS[method] for method in METHOD_ORDER]
    lines.append(row(columns))
    lines.append(row(["---"] + ["---:"] * len(METHOD_ORDER)))
    for corruption in corruptions:
        lines.append(row([corruption] + [pct(maps[method][corruption]) for method in METHOD_ORDER]))

    lines += ["", "### 数据计数核对", ""]
    lines.append(row(["方法", "逐域样本", "连续流样本", "逐域数量", "连续域数量"]))
    lines.append(row(["---", "---:", "---:", "---:", "---:"]))
    for method in METHOD_ORDER:
        item = summaries[method]
        lines.append(row([METHOD_LABELS[method], str(item["domain_samples"]), str(item["continual_samples"]), "15", "45"]))
    return lines


def main():
    protocol_lines = [
        "统一协议：Tiny-ImageNet-C severity=5，15 个 corruption，每域 10,000 张图像，三轮连续适应，固定 RandomState(0) 域内顺序，batch size=128，seed=0。逐域重置从同一 checkpoint 独立开始；连续流为同一模型连续经过 15 个域三轮；Clean 和四域 Revisit 在连续流结束后冻结评估。准确率和差值单位均为百分比（百分点）。",
        "",
        "FedDR 行使用原始 FedDR 协议结果；Tent、EATA、SAR、ASR、ReCAP 行来自统一 TTA5 测试。最终 BN 漂移仅在结果文件提供该诊断时填写。",
    ]
    lines = [
        "# Tiny-ImageNet-C ResNet-18 / ResNet-34 测试结果表",
        "",
        *protocol_lines,
        "",
    ]
    for index, (model_name, paths) in enumerate(RUNS.items()):
        if index:
            lines += ["", "---", ""]
        section = build_model_section(model_name, paths)
        lines += section
        slug = model_name.replace("-", "").upper()
        individual = [f"# Tiny-ImageNet-C {model_name} 测试结果表", "", *protocol_lines, ""]
        individual += section
        individual += [
            "",
            "## 结果文件",
            "",
            *[f"- {METHOD_LABELS[method]}：`{path.relative_to(ROOT)}`" for method, path in paths.items()],
        ]
        (OUT.parent / f"TINY_IMAGENET_C_{slug}_RESULTS.md").write_text("\n".join(individual) + "\n", encoding="utf-8")

    lines += [
        "",
        "## 结果文件",
        "",
        "- ResNet-18 Source/FedDR：`outputs/tiny_imagenet_resnet18_20260907/`",
        "- ResNet-18 Tent/EATA/SAR/ASR/ReCAP：`outputs/tiny_imagenet_resnet18_tta5_20260908/`",
        "- ResNet-34 Source/FedDR：`outputs/tiny_imagenet_resnet34_20260907/`",
        "- ResNet-34 Tent/EATA/SAR/ASR/ReCAP：`outputs/tiny_imagenet_resnet34_tta5_20260908/`",
    ]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
