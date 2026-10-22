"""Prepare ReCAP and Fisher statistics for one deployment anchor.

The output contains ReCAP feature moments and BN-affine diagonal Fisher values
for either a FedACG ``lookahead_anchor`` or FedAvg ``adaptation_anchor``.
Source images are used only in this offline preparation step.
"""

import argparse
import logging
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from main_test import (
    DATASET_STATS,
    PROJECT_ROOT,
    build_transform,
    configure_official_bn_adaptation,
    get_linear_classifier,
    load_deployment_model,
    resolve_device,
)


LOGGER = logging.getLogger("prepare_main_statistics")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare ReCAP feature statistics and BN Fisher together"
    )
    parser.add_argument(
        "--dataset",
        choices=["cifar10", "cifar100", "tiny-imagenet"],
        default="cifar100",
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument(
        "--anchor_key",
        choices=["lookahead_anchor", "adaptation_anchor"],
        default="lookahead_anchor",
    )
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--feature_samples", type=int, default=500)
    parser.add_argument("--fisher_max_batches", type=int, default=-1)
    parser.add_argument(
        "--fisher_mode",
        choices=["main", "eata_official"],
        default="main",
        help="Keep the main-method Fisher or reproduce the official EATA Fisher.",
    )
    parser.add_argument(
        "--fisher_samples",
        type=int,
        default=-1,
        help="Number of source samples for Fisher; <=0 uses the full train set.",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output_path", required=True)
    return parser.parse_args()


def build_source_dataset(args):
    from torchvision import datasets

    root = (
        Path(args.data_root)
        if args.data_root
        else PROJECT_ROOT / "data"
    )
    if args.dataset == "tiny-imagenet":
        root = root / "train"
        if not root.is_dir():
            raise FileNotFoundError(
                f"Tiny-ImageNet training directory not found: {root}"
            )
        return datasets.ImageFolder(
            root=str(root),
            transform=build_transform(args.dataset),
        )
    dataset_class = datasets.CIFAR10 if args.dataset == "cifar10" else datasets.CIFAR100
    return dataset_class(
        root=str(root),
        train=True,
        transform=build_transform(args.dataset),
        download=args.download,
    )


def make_loader(dataset, args):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda") and torch.cuda.is_available(),
    )


@torch.no_grad()
def compute_feature_moments(model, loader, device):
    classifier_name, classifier = get_linear_classifier(model)
    captured = {}

    def capture_input(_module, inputs):
        if not inputs:
            raise RuntimeError("classifier hook received no input")
        captured["features"] = inputs[0].detach()

    hook = classifier.register_forward_pre_hook(capture_input)
    count = 0
    feature_sum = None
    feature_square_sum = None
    feature_chunks = []
    label_chunks = []
    model.eval()
    try:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            captured.clear()
            logits = model(images)
            features = captured.get("features")
            if features is None:
                raise RuntimeError("failed to capture classifier input features")
            features = features.reshape(features.size(0), -1).cpu().double()
            if feature_sum is None:
                feature_sum = torch.zeros(features.size(1), dtype=torch.float64)
                feature_square_sum = torch.zeros_like(feature_sum)
            feature_sum += features.sum(dim=0)
            feature_square_sum += features.square().sum(dim=0)
            feature_chunks.append(features)
            label_chunks.append(labels.cpu())
            count += int(features.size(0))
    finally:
        hook.remove()

    if count == 0 or feature_sum is None:
        raise RuntimeError("no source features were collected")
    mean = feature_sum / count
    variance = (feature_square_sum / count - mean.square()).clamp_min(0.0)
    if variance.numel() != classifier.in_features:
        raise RuntimeError("feature dimension does not match the classifier")
    all_features = torch.cat(feature_chunks, dim=0)
    all_labels = torch.cat(label_chunks, dim=0)
    num_classes = int(classifier.out_features)
    class_mean = torch.zeros(num_classes, all_features.size(1), dtype=torch.float32)
    class_variance = torch.ones_like(class_mean)
    for class_id in range(num_classes):
        selected = all_features[all_labels == class_id]
        if selected.numel() == 0:
            # Keep deployment finite when the small preparation subset misses a class.
            class_mean[class_id] = mean.float()
            class_variance[class_id] = variance.float().clamp_min(1e-6)
        else:
            class_mean[class_id] = selected.mean(dim=0).float()
            class_variance[class_id] = selected.var(dim=0, unbiased=False).float().clamp_min(1e-6)
    return {
        "classifier_name": classifier_name,
        "feature_mean": mean.float(),
        "feature_variance": variance.float(),
        "class_feature_mean": class_mean,
        "class_feature_variance": class_variance,
        "feature_dim": int(classifier.in_features),
        "num_classes": int(classifier.out_features),
        "num_feature_samples": count,
    }


def collect_bn_affine(model: nn.Module):
    params = []
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.BatchNorm2d):
            continue
        for param_name, param in module.named_parameters(recurse=False):
            if param_name in ("weight", "bias") and param is not None:
                params.append((f"{module_name}.{param_name}", param))
    return params


def compute_fishers(model, loader, device, max_batches, mode="main"):
    if mode == "eata_official":
        configure_official_bn_adaptation(model)
        model.train()
    elif mode == "main":
        model.eval()
        model.requires_grad_(False)
    else:
        raise ValueError(f"unsupported Fisher mode: {mode}")
    named_params = collect_bn_affine(model)
    if not named_params:
        raise RuntimeError("no BN affine parameters were found")
    for _name, param in named_params:
        param.requires_grad_(True)

    accumulated = {
        name: torch.zeros_like(param, dtype=torch.float32, device="cpu")
        for name, param in named_params
    }
    anchors = {
        name: param.detach().cpu().clone() for name, param in named_params
    }
    criterion = nn.CrossEntropyLoss()
    samples = 0
    batches = 0

    for batch_index, (images, labels) in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        model.zero_grad(set_to_none=True)
        logits = model(images)
        if mode == "eata_official":
            fisher_targets = logits.detach().argmax(dim=1)
        else:
            fisher_targets = labels.to(device, non_blocking=True)
        loss = criterion(logits, fisher_targets)
        loss.backward()

        batch_size = int(images.size(0))
        for name, param in named_params:
            if param.grad is not None:
                contribution = param.grad.detach().cpu().float().square()
                if mode == "main":
                    contribution = contribution * batch_size
                accumulated[name] += contribution
        samples += batch_size
        batches += 1

    model.zero_grad(set_to_none=True)
    if samples == 0:
        raise RuntimeError("no source samples were used for Fisher")
    denominator = batches if mode == "eata_official" else samples
    fishers = {
        name: (accumulated[name] / denominator, anchors[name])
        for name in accumulated
    }
    return fishers, samples, batches


def run():
    args = parse_args()
    if args.feature_samples <= 0:
        raise ValueError("feature_samples must be positive")
    if args.fisher_samples == 0:
        raise ValueError("fisher_samples must be positive or <=0 for all samples")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)

    model, backbone = load_deployment_model(
        args.model_path, device, anchor_key=args.anchor_key
    )
    dataset = build_source_dataset(args)
    if args.feature_samples > len(dataset):
        raise ValueError(
            f"feature_samples={args.feature_samples} exceeds dataset size={len(dataset)}"
        )

    generator = torch.Generator().manual_seed(args.seed)
    feature_indices = torch.randperm(len(dataset), generator=generator)[
        : args.feature_samples
    ].tolist()
    feature_loader = make_loader(Subset(dataset, feature_indices), args)
    if args.fisher_samples > 0:
        if args.fisher_samples > len(dataset):
            raise ValueError(
                f"fisher_samples={args.fisher_samples} exceeds dataset size={len(dataset)}"
            )
        fisher_generator = torch.Generator().manual_seed(args.seed)
        fisher_indices = torch.randperm(
            len(dataset), generator=fisher_generator
        )[: args.fisher_samples].tolist()
        fisher_dataset = Subset(dataset, fisher_indices)
    else:
        fisher_dataset = dataset
    fisher_loader = make_loader(fisher_dataset, args)

    moments = compute_feature_moments(model, feature_loader, device)
    fishers, fisher_samples, fisher_batches = compute_fishers(
        model,
        fisher_loader,
        device,
        args.fisher_max_batches,
        mode=args.fisher_mode,
    )

    expected_classes = DATASET_STATS[args.dataset]["num_classes"]
    if moments["num_classes"] != expected_classes:
        raise ValueError(
            f"checkpoint has {moments['num_classes']} classes, "
            f"expected {expected_classes}"
        )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "feature_mean": moments["feature_mean"],
            "feature_variance": moments["feature_variance"],
            "class_feature_mean": moments["class_feature_mean"],
            "class_feature_variance": moments["class_feature_variance"],
            "fishers": fishers,
            "meta": {
                "dataset": args.dataset,
                "backbone": backbone,
                "model_path": str(Path(args.model_path).expanduser().resolve()),
                "actual_ckpt_key": args.anchor_key,
                "classifier_name": moments["classifier_name"],
                "feature_dim": moments["feature_dim"],
                "num_classes": moments["num_classes"],
                "num_feature_samples": moments["num_feature_samples"],
                "num_fisher_samples": fisher_samples,
                "num_fisher_batches": fisher_batches,
                "fisher_estimator": "batch_mean_gradient_square",
                "fisher_mode": args.fisher_mode,
                "fisher_label_source": (
                    "pseudo_label"
                    if args.fisher_mode == "eata_official"
                    else "ground_truth"
                ),
                "fisher_bn_mode": (
                    "target_batch"
                    if args.fisher_mode == "eata_official"
                    else "source_running_statistics"
                ),
                "fisher_denominator": (
                    "batches"
                    if args.fisher_mode == "eata_official"
                    else "samples"
                ),
                "fisher_scope": "all_bn_affine",
                "fisher_ranking": "global_scalar",
                "fisher_includes_weight_and_bias": True,
                "fisher_uses_batch_mean": True,
                "seed": args.seed,
            },
        },
        output_path,
    )
    LOGGER.info("backbone: %s", backbone)
    LOGGER.info("feature samples: %d", moments["num_feature_samples"])
    LOGGER.info("Fisher samples: %d", fisher_samples)
    LOGGER.info("BN Fisher tensors: %d", len(fishers))
    LOGGER.info("saved statistics: %s", output_path)


if __name__ == "__main__":
    run()
