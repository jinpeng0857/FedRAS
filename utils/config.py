"""Argparse configuration for FedACG and FedAvg federated training."""

import argparse

parser = argparse.ArgumentParser(description="FEDEATA federated source training")

# ========== Basic ==========
parser.add_argument("--debug", action="store_true", default=True,
                    help="Run in debug mode")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--step", type=str, default="train", choices=["train"])

# ========== Method ==========
parser.add_argument(
    "--method",
    type=str,
    default="fedacg",
    choices=[
        "fedacg",
        "fedavg",
        "fedprox",
        "scaffold",
        "feddyn",
        "fedadam",
    ],
)
parser.add_argument("--fedacg_mode", type=str, default="fedacg", choices=["fedacg"],
                    help="FedACG mode; ignored by FedAvg.")

# ========== Model ==========
parser.add_argument("--backbone", type=str, default="resnet18",
                    choices=["resnet18", "resnet34", "mobilenetv2"],
                    help="Backbone architecture.")
parser.add_argument("--num_groups", type=int, default=32,
                    help="GroupNorm groups when --use_bn_layer is disabled.")
parser.add_argument("--use_bn_layer", action="store_true", default=True,
                    help="Use BatchNorm layers. Keep enabled for the main BN-affine test.")
parser.add_argument("--pretrained", action="store_true", default=False)

# ========== Dataset ==========
parser.add_argument("--dataset", type=str, default="cifar10",
                    choices=["cifar10", "cifar100", "cinic", "imagenet"])
parser.add_argument("--dataset_path", type=str, default=None,
                    help="Override default dataset path")
parser.add_argument("--num_client", type=int, default=100)
parser.add_argument("--alpha", type=float, default=0.3,
                    help="Dirichlet distribution alpha")
parser.add_argument("--equal_size", action="store_true", default=True,
                    help="Each client gets exactly equal samples with Dirichlet class skew")
parser.add_argument("--num_class", type=int, default=100)
parser.add_argument("--dataset_seed", type=int, default=21)
parser.add_argument("--new_dataset_seed", type=int, default=30)

# ========== FL Training ==========
parser.add_argument("--join_ratio", type=float, default=0.15)
parser.add_argument("--global_rounds", type=int, default=500)
parser.add_argument("--local_epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--learning_rate", type=float, default=0.1)
parser.add_argument("--momentum", type=float, default=0.0)
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument("--num_workers", type=int, default=0,
                    help="DataLoader num_workers (0 = main process)")
parser.add_argument("--local_lr_decay", type=float, default=0.998,
                    help="Per-round local learning-rate decay")
parser.add_argument("--label_smoothing", type=float, default=0.0,
                    help="Label smoothing passed to cross-entropy loss")
parser.add_argument("--use_randaugment", action="store_true",
                    help="Use RandAugment in CIFAR-100 train transform")
parser.add_argument("--randaugment_n", type=int, default=2,
                    help="Number of RandAugment operations")
parser.add_argument("--randaugment_m", type=int, default=9,
                    help="RandAugment magnitude")
parser.add_argument("--lr_scheduler", type=str, default="cosine",
                    choices=["none", "cosine", "multistep"],
                    help="Per-round learning-rate schedule")
parser.add_argument("--eval_clean_every", type=int, default=50,
                    help="Evaluate clean test accuracy every N rounds; <=0 disables")
parser.add_argument("--save_best_clean", action="store_true",
                    help="Save best_clean.pt when clean accuracy improves")
parser.add_argument("--amp", action="store_true", default=True, dest="use_amp",
                    help="Enable automatic mixed precision")
parser.add_argument("--no-amp", action="store_false", dest="use_amp",
                    help="Disable automatic mixed precision")
parser.add_argument("--checkpoint_path", type=str, default="default")
parser.add_argument("--save_freq", type=int, default=50)
parser.add_argument("--eval_freq", type=int, default=10)
parser.add_argument("--task_name", type=str, default="default")

# ========== FedACG Server ==========
parser.add_argument("--server_momentum", type=float, default=0.85,
                    help="FedACG lookahead/server momentum coefficient")
parser.add_argument("--server_lr", type=float, default=1.0,
                    help="Reserved for compatibility; pure FedACG uses weighted averaging plus momentum lookahead")
parser.add_argument("--ema_decay", type=float, default=0.999,
                    help="EMA decay for model weight smoothing (0.0 = disable)")

# ========== FedACG Client ==========
parser.add_argument("--ce_weight", type=float, default=1.0,
                    help="Cross-entropy loss weight")
parser.add_argument("--prox_weight", type=float, default=0.01,
                    help="Optional FedACG proximal regularization weight; set 0 to disable")
parser.add_argument(
    "--fedprox_mu",
    type=float,
    default=0.01,
    help="FedProx proximal coefficient mu; used only by --method fedprox.",
)

# ========== Additional train/ baselines ==========
parser.add_argument(
    "--feddyn_alpha",
    type=float,
    default=0.01,
    help="FedDyn dynamic-regularization coefficient.",
)
parser.add_argument(
    "--scaffold_server_lr",
    type=float,
    default=1.0,
    help="PFLlib SCAFFOLD server learning rate.",
)
parser.add_argument("--fedadam_beta1", type=float, default=0.9)
parser.add_argument("--fedadam_beta2", type=float, default=0.999)
parser.add_argument(
    "--fedadam_tau",
    type=float,
    default=0.001,
    help="FedAdam numerical stability constant tau.",
)
parser.add_argument(
    "--fedopt_server_lr",
    type=float,
    default=1.0,
    help="Server step size for FedAdam.",
)
