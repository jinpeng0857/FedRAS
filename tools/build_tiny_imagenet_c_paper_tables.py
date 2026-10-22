import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METHODS = ["source", "tent", "eata", "sar", "asr", "recap", "feddr"]
LABELS = {
    "source": "Source",
    "tent": "Tent",
    "eata": "EATA",
    "sar": "SAR",
    "asr": "ASR",
    "recap": "ReCAP",
    "feddr": "FedDR",
}
CATEGORIES = {
    "Noise": ["gaussian_noise", "shot_noise", "impulse_noise"],
    "Blur": ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"],
    "Weather": ["snow", "frost", "fog"],
    "Digital": ["brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression"],
}


def runs(model):
    prefix = f"outputs/tiny_imagenet_{model}"
    tta = ROOT / f"{prefix}_tta5_20260908"
    base = ROOT / f"{prefix}_20260907"
    return {
        "source": base / "source.json",
        "feddr": base / "feddr.json",
        "tent": tta / "tent.json",
        "eata": tta / "eata.json",
        "sar": tta / "sar.json",
        "asr": tta / "asr.json",
        "recap": tta / "recap.json",
    }


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def pct(x):
    return 100.0 * float(x)


def domain_map(data):
    return {item["corruption"]: float(item["accuracy"]) for item in data["domain_wise"]["per_corruption"]}


def clean_final(data):
    clean = data["clean_retention"]
    if "final_clean" in clean:
        return float(clean["final_clean"]["accuracy"])
    return float(clean["final_continual_clean_accuracy"])


def summary(data):
    domain = float(data["domain_wise"]["summary"]["mean_accuracy"])
    cycles = [float(x["mean_accuracy"]) for x in data["continual"]["summary"]["cycle_summaries"]]
    revisit = data["continual"]["post_stream"]["revisit"]
    revisit_mean = sum(float(x["accuracy"]) for x in revisit) / len(revisit)
    return [domain, *cycles, revisit_mean, clean_final(data)]


def cell(value, bold=False):
    text = f"{pct(value):.1f} ± —"
    return f"**{text}**" if bold else text


def table_row(values):
    return "| " + " | ".join(values) + " |"


def build(model):
    paths = runs(model)
    data = {method: load(path) for method, path in paths.items()}
    maps = {method: domain_map(item) for method, item in data.items()}
    summaries = {method: summary(item) for method, item in data.items()}

    lines = [
        f"# Tiny-ImageNet-C {model.replace('resnet', 'ResNet-')} Paper-Style Tables",
        "",
        "## TABLE I",
        "",
        f"**DOMAIN-WISE RESULTS UNDER INDEPENDENT RESETS ON TINY-IMAGENET-C (SEVERITY 5).** Values are from one fixed evaluation run; `± —` indicates that three-independent-run standard deviations are unavailable.",
        "",
        table_row(["Method", "Noise", "Blur", "Weather", "Digital", "Overall Mean"]),
        table_row(["---", "---:", "---:", "---:", "---:", "---:"]),
    ]
    category_values = {}
    for method in METHODS:
        values = []
        for corruptions in CATEGORIES.values():
            values.append(sum(maps[method][name] for name in corruptions) / len(corruptions))
        values.append(summaries[method][0])
        category_values[method] = values
    maxima = [max(category_values[method][i] for method in METHODS) for i in range(5)]
    for method in METHODS:
        values = category_values[method]
        lines.append(table_row([LABELS[method]] + [cell(value, abs(value - maxima[i]) < 1e-12) for i, value in enumerate(values)]))

    lines += [
        "",
        "Category mapping: Noise = Gaussian/Shot/Impulse; Blur = Defocus/Glass/Motion/Zoom; Weather = Snow/Frost/Fog; Digital = Brightness/Contrast/Elastic/Pixelate/JPEG.",
        "",
        "## TABLE II",
        "",
        f"**CONTINUAL EVALUATION ACROSS 3 CYCLES (45 BLOCKS) AND RETENTION RESULTS ON TINY-IMAGENET-C.** Values are from one fixed evaluation run; `± —` indicates unavailable multi-run standard deviations.",
        "",
        table_row(["Method", "DW", "Cycle 1", "Cycle 2", "Cycle 3", "Revisit", "Clean"]),
        table_row(["---", "---:", "---:", "---:", "---:", "---:", "---:"]),
    ]
    maxima = [max(summaries[method][i] for method in METHODS) for i in range(6)]
    for method in METHODS:
        values = summaries[method]
        lines.append(table_row([LABELS[method]] + [cell(value, abs(value - maxima[i]) < 1e-12) for i, value in enumerate(values)]))

    lines += [
        "",
        "Protocol: 15 corruptions, severity=5, 10,000 samples per corruption, fixed RandomState(0) order, batch size=128, seed=0. Revisit is the mean of Gaussian, Shot, Impulse, and Defocus; Clean is measured after the third continuous cycle with adaptation frozen.",
        "",
        "FedDR uses the original FedDR protocol. The other adaptation methods use the fixed TTA5 protocol. No multi-seed mean ± standard deviation is reported because the available results contain one evaluation run per method.",
    ]
    output = ROOT / f"outputs/TINY_IMAGENET_C_{model.upper()}_PAPER_TABLES.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def main():
    for model in ["resnet18", "resnet34"]:
        print(build(model))


if __name__ == "__main__":
    main()
