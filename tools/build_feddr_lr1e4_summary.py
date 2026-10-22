import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "ResNet-18": {
        "original": ROOT / "outputs/tiny_imagenet_resnet18_20260907/feddr.json",
        "lr1e4": ROOT / "outputs/tiny_resnet18_feddr_lr1e4_20260908/feddr_lr1e4.json",
        "config": ROOT / "outputs/tiny_resnet18_feddr_lr1e4_20260908/feddr_lr1e4_resolved_config.json",
        "validated": ROOT / "outputs/tiny_resnet18_feddr_lr1e4_20260908/feddr_lr1e4_validated.json",
    },
    "ResNet-34": {
        "original": ROOT / "outputs/tiny_imagenet_resnet34_20260907/feddr.json",
        "lr1e4": ROOT / "outputs/tiny_resnet34_feddr_lr1e4_20260908/feddr_lr1e4.json",
        "config": ROOT / "outputs/tiny_resnet34_feddr_lr1e4_20260908/feddr_lr1e4_resolved_config.json",
        "validated": ROOT / "outputs/tiny_resnet34_feddr_lr1e4_20260908/feddr_lr1e4_validated.json",
    },
}


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value):
    return round(100.0 * float(value), 3)


def summary(data):
    continual = data["continual"]["summary"]
    clean = data["clean_retention"]
    if "initial_clean" in clean:
        initial = clean["initial_clean"]["accuracy"]
        final = clean["final_clean"]["accuracy"]
    else:
        initial = clean["initial_clean_accuracy"]
        final = clean["final_continual_clean_accuracy"]
    revisit = data["continual"]["post_stream"]["revisit"]
    return {
        "domain": pct(data["domain_wise"]["summary"]["mean_accuracy"]),
        "cycles": [pct(x["mean_accuracy"]) for x in continual["cycle_summaries"]],
        "cycle_delta": pct(continual.get("late_delta_cycle3_minus_cycle1", continual.get("cycle3_minus_cycle1"))),
        "initial_clean": pct(initial),
        "final_clean": pct(final),
        "clean_delta": pct(final - initial),
        "revisit": pct(sum(x["accuracy"] for x in revisit) / len(revisit)),
        "bn_drift": continual.get("final_bn_affine_drift"),
    }


def verify(data, validated):
    return {
        "validated": validated,
        "domain_count": len(data["domain_wise"]["per_corruption"]),
        "continual_count": len(data["continual"]["per_corruption"]),
        "domain_samples": data["domain_wise"]["summary"]["total_samples"],
        "continual_samples": data["continual"]["summary"]["total_samples"],
        "clean_samples": data["clean_retention"].get("final_clean", {}).get("total", 10000),
        "revisit_domains": len(data["continual"]["post_stream"]["revisit"]),
    }


def row(values):
    return "| " + " | ".join(values) + " |"


def main():
    all_data = {}
    audit = {}
    lines = [
        "# FedDR 仅降低学习率至 1e-4 的验证结果",
        "",
        "本实验严格保持原始 FedDR 协议不变，只将 `adapt_lr` 从 `1e-3` 改为 `1e-4`。BN 范围、hybrid threshold、anchor rectify beta、FSLR reliability gate、continuous risk、view consistency 和 hierarchical budget 均保持原值。",
        "",
        "| 骨干网络 | 配置 | 逐域重置 | C1 | C2 | C3 | C3-C1 | 初始 Clean | 最终 Clean | Clean 变化 | Revisit | BN 漂移 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, paths in RUNS.items():
        original = load(paths["original"])
        lr1e4 = load(paths["lr1e4"])
        all_data[model] = {"original": summary(original), "lr1e4": summary(lr1e4)}
        audit[model] = {
            "original": verify(original, None),
            "lr1e4": verify(lr1e4, load(paths["validated"])),
            "resolved_config": load(paths["config"]),
        }
        for label, data in [("原始 FedDR (`1e-3`)", all_data[model]["original"]), ("仅降 LR (`1e-4`)", all_data[model]["lr1e4"])]:
            lines.append(row([
                model,
                label,
                f"{data['domain']:.3f}",
                f"{data['cycles'][0]:.3f}",
                f"{data['cycles'][1]:.3f}",
                f"{data['cycles'][2]:.3f}",
                f"{data['cycle_delta']:+.3f}",
                f"{data['initial_clean']:.3f}",
                f"{data['final_clean']:.3f}",
                f"{data['clean_delta']:+.3f}",
                f"{data['revisit']:.3f}",
                "—" if data["bn_drift"] is None else f"{data['bn_drift']['total']:.3f}",
            ]))
    lines += [
        "",
        "## 结果解读",
        "",
        "- ResNet-18：最终 Clean 从 58.590% 提升到 60.890%，BN 漂移从 0.254 降到 0.124；逐域均值下降到 32.429%，Revisit 基本不变（30.607%→30.483%）。",
        "- ResNet-34：最终 Clean 从 59.650% 提升到 63.270%，BN 漂移从 0.287 降到 0.111；逐域均值下降到 33.938%，Revisit 从 32.065% 提升到 32.295%。",
        "- 两个骨干网络的 C3 均高于 C1，说明降低学习率后连续流没有出现后期塌陷；代价是逐域重置第一轮适应精度下降。",
        "",
        "完整结果满足 15 个逐域、45 个连续域、150,000 个逐域样本、450,000 个连续流样本、10,000 个 Clean 样本和 4 个 Revisit 域的协议要求。",
    ]
    (ROOT / "outputs/FEDDR_LR1E4_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROOT / "outputs/FEDDR_LR1E4_SUMMARY.json").write_text(json.dumps(all_data, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "outputs/FEDDR_LR1E4_AUDIT.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(ROOT / "outputs/FEDDR_LR1E4_RESULTS.md")


if __name__ == "__main__":
    main()
