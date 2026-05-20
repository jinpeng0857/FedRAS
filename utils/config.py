"""
Argparse configuration for FEDEATA.
All parameters for FL training + EATA-C TTA evaluation in one place.
"""

import argparse

parser = argparse.ArgumentParser(description="FEDEATA: Federated EATA-C")

# ========== Basic ==========
parser.add_argument("--debug", action="store_true", default=True,
                    help="Run in debug mode (no wandb)")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--step", type=str, default="train",
                    choices=["train", "test", "fisher"])

# ========== Method ==========
parser.add_argument("--method", type=str, default="fedacg",
                    choices=["fedacg", "fedavg"])
parser.add_argument("--fedacg_mode", type=str, default="fedacg",
                    choices=["fedavg", "fedprox", "fedlc", "feddecorr",
                             "fedmlb", "fedntd", "feddyn", "fedacg", "fedadam"])

# ========== Model ==========
parser.add_argument("--backbone", type=str, default="eatac_resnet18",
                    choices=["resnet18", "eatac_resnet18", "eatac_resnet34",
                             "eatac_resnet50", "eatac_resnet101"])
parser.add_argument("--num_groups", type=int, default=32,
                    help="GroupNorm groups (only when use_bn_layer=False)")
parser.add_argument("--use_bn_layer", action="store_true", default=True,
                    help="Replace GroupNorm with BatchNorm (required for TTA)")
parser.add_argument("--subnet_drop_rate", type=float, default=0.2,
                    help="EATAC sub-network stochastic depth drop rate")
parser.add_argument("--subnet_train_prob", type=float, default=0.3,
                    help="Probability of using sub-network dropout during local training (0=off)")
parser.add_argument("--pretrained", action="store_true", default=False)

# ========== Dataset ==========
parser.add_argument("--dataset", type=str, default="cifar10",
                    choices=["cifar10", "cifar100", "cinic", "imagenet"])
parser.add_argument("--dataset_path", type=str, default=None,
                    help="Override default dataset path")
parser.add_argument("--num_client", type=int, default=100)
parser.add_argument("--alpha", type=float, default=0.3,
                    help="Dirichlet distribution alpha")
parser.add_argument("--equal_size", action="store_true", default=False,
                    help="Each client gets exactly len(dataset)//num_client samples")
parser.add_argument("--num_class", type=int, default=10)
parser.add_argument("--dataset_seed", type=int, default=21)
parser.add_argument("--new_dataset_seed", type=int, default=30)

# ========== FL Training ==========
parser.add_argument("--join_ratio", type=float, default=0.05)
parser.add_argument("--global_rounds", type=int, default=500)
parser.add_argument("--local_epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--learning_rate", type=float, default=0.1)
parser.add_argument("--momentum", type=float, default=0.0)
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument("--num_workers", type=int, default=0,
                    help="DataLoader num_workers (0 = main process)")
parser.add_argument("--local_lr_decay", type=float, default=0.998,
                    help="Per-round local LR decay")
parser.add_argument("--checkpoint_path", type=str, default="default")
parser.add_argument("--save_freq", type=int, default=10)
parser.add_argument("--eval_freq", type=int, default=10)
parser.add_argument("--task_name", type=str, default="default")

# ========== Server Aggregation ==========
parser.add_argument("--server_momentum", type=float, default=0.85)
parser.add_argument("--server_lr", type=float, default=1.0)

# ========== Client Loss Weights ==========
parser.add_argument("--ce_weight", type=float, default=1.0)
parser.add_argument("--prox_weight", type=float, default=0.01,
                    help="FedProx proximal term weight")
parser.add_argument("--lc_weight", type=float, default=0.1,
                    help="FedLC label-calibrated loss weight")
parser.add_argument("--lc_tau", type=float, default=1.0)
parser.add_argument("--decorr_weight", type=float, default=0.01,
                    help="FedDecorr feature decorrelation weight")
parser.add_argument("--mlb_branch_cls_weight", type=float, default=1.0)
parser.add_argument("--mlb_branch_kl_weight", type=float, default=5.0)
parser.add_argument("--mlb_temp", type=float, default=4.0)
parser.add_argument("--ntd_weight", type=float, default=0.1,
                    help="FedNTD not-true distillation weight")
parser.add_argument("--ntd_temp", type=float, default=3.0)
parser.add_argument("--dyn_alpha", type=float, default=0.01,
                    help="FedDyn alpha parameter")

# ========== TTA (EATA-C) Testing ==========
parser.add_argument("--adapt_steps", type=int, default=3,
                    help="TTA optimization steps per batch")
parser.add_argument("--adapt_lr", type=float, default=0.005,
                    help="TTA learning rate")
parser.add_argument("--fisher_alpha", type=float, default=50.0,
                    help="Fisher regularization coefficient")
parser.add_argument("--e_margin", type=float, default=0.4,
                    help="Entropy margin for sample selection (x lnC)")
parser.add_argument("--epsilon", type=float, default=0.05,
                    help="Cosine similarity threshold for diversity filtering")
parser.add_argument("--alpha_ma", type=float, default=0.1,
                    help="EMA update factor for diversity filtering")
parser.add_argument("--consistency_weight", type=float, default=1.0)
parser.add_argument("--entropy_reg_weight", type=float, default=0.1)
parser.add_argument("--test_batch_size", type=int, default=64)
parser.add_argument("--episodic", action="store_true", default=False,
                    help="Episodic adaptation (reset model per domain)")
parser.add_argument("--model_path", type=str, default=None)
parser.add_argument("--fisher_path", type=str, default=None)
parser.add_argument("--num_fisher_samples", type=int, default=2000)
parser.add_argument("--fisher_data_split", type=str, default="train",
                    choices=["train", "test"],
                    help="Data split for Fisher computation. Default=train avoids test leakage.")
parser.add_argument("--severity", type=int, default=5)
parser.add_argument("--precomputed_c_path", type=str, default=None,
                    help="Path to pre-computed CIFAR-C/ImageNet-C dataset dir, e.g. ./data/CIFAR-10-C. "
                         "If not set, uses online corruption (slower but no pre-downloaded files needed).")

# ========== Corruption Types ==========
parser.add_argument("--corruption", type=str, default=None,
                    help="Single corruption type, or None for all 15")

# ========== FedTTA (Federated TTA) ==========
parser.add_argument("--clients_per_corruption", type=int, default=5,
                    help="Number of target clients per corruption type")
parser.add_argument("--samples_per_client", type=int, default=None,
                    help="Max samples per target client (None = use all)")
parser.add_argument("--corruption_list", type=str, nargs="+", default=None,
                    help="Subset of corruptions. None = all 15 CIFAR-C corruptions")
parser.add_argument("--tta_mode", type=str, default="domain_fedacg_eatac",
                    choices=["source_only", "local_tent", "local_eata",
                             "local_eata_c", "naive_fedavg_tta",
                             "naive_fedacg_tta", "domain_fedavg_eatac",
                             "domain_fedacg_eatac"],
                    help="FedTTA mode/baseline")
parser.add_argument("--tta_rounds", type=int, default=10,
                    help="Number of FedTTA communication rounds")
parser.add_argument("--lambda_tta", type=float, default=0.85,
                    help="FedACG momentum coefficient for TTA aggregation")
parser.add_argument("--gamma_prox", type=float, default=0.01,
                    help="Proximal regularization weight (local TTA stays near accelerated state)")
parser.add_argument("--local_tta_batches", type=int, default=None,
                    help="Number of batches for local TTA per round (None = full dataset)")
parser.add_argument("--eval_after_adapt", action="store_true", default=False,
                    help="Evaluate after adaptation (re-forward after TTA)")
parser.add_argument("--freeze_bn", action="store_true", default=True,
                    help="Freeze BN running statistics during TTA (do not update running mean/var)")
parser.add_argument("--update_bn_stats", action="store_true", default=False,
                    help="Allow BN running stats to update during TTA (overrides freeze_bn)")
parser.add_argument("--alpha_calib", type=float, default=0.1,
                    help="Calibration coefficient for entropy regularization (alias for entropy_reg_weight)")
parser.add_argument("--beta_fisher", type=float, default=50.0,
                    help="Fisher regularization coefficient (alias for fisher_alpha)")
parser.add_argument("--target_label_skew", action="store_true", default=False,
                    help="Apply Dirichlet label skew within each target corruption")
parser.add_argument("--target_alpha", type=float, default=0.3,
                    help="Dirichlet alpha for target label skew")
parser.add_argument("--target_participation_rate", type=float, default=1.0,
                    help="Fraction of target clients selected per round (overrides join_ratio)")
parser.add_argument("--weight_by_num_samples", action="store_true", default=False,
                    help="Weight client contribution by num_samples in FedTTA aggregation")
parser.add_argument("--reliability_tau", type=float, default=0.5,
                    help="Softmax temperature for per-domain reliability normalization (lower = sharper)")
