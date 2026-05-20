# FEDEATA: Federated Test-Time Adaptation with EATA-C

Domain-isolated federated test-time adaptation under distribution shifts.

**Pipeline**: FL source training (FedACG) → Fisher precomputation → FedTTA evaluation (8 modes)

## Installation

```bash
pip install -r requirements.txt
```

## Dataset Preparation

| Dataset | Type | Download |
|---------|------|----------|
| CIFAR-10-C | Synthetic corruption | Auto-downloaded via `torchvision`; CIFAR-10-C `*.npy` files to `./data/CIFAR-10-C/` |
| CIFAR-100-C | Synthetic corruption | Auto-downloaded via `torchvision`; CIFAR-100-C `*.npy` files to `./data/CIFAR-100-C/` |
| ImageNet-C | Synthetic corruption | Download ImageNet-C `*.npy` files to `./data/ImageNet-C/` |

Set environment variable for custom data root:
```bash
export FEDEATA_DATA_ROOT=/path/to/data
```

## Full Experiment Pipeline

### Step 1: Federated Source Training (FedACG)

```bash
# CIFAR-10
python train_fedacg.py --method fedacg --fedacg_mode fedacg \
    --dataset cifar10 --alpha 0.3 --num_client 100 \
    --backbone eatac_resnet18 --subnet_drop_rate 0.2 --subnet_train_prob 0.3 \
    --global_rounds 500 --local_epochs 5 --learning_rate 0.1 \
    --momentum 0.0 --weight_decay 5e-4 --join_ratio 0.05 --batch_size 64 \
    --task_name run1

# CIFAR-100
python train_fedacg.py --method fedacg --fedacg_mode fedacg \
    --dataset cifar100 --alpha 0.3 --num_client 100 \
    --backbone eatac_resnet18 --subnet_drop_rate 0.2 --subnet_train_prob 0.3 \
    --global_rounds 500 --local_epochs 5 --learning_rate 0.1 \
    --momentum 0.0 --weight_decay 5e-4 --join_ratio 0.05 --batch_size 64 \
    --task_name run1
```

Output: `results/{dataset}_{num_client}clients_alpha{alpha}/fedacg_eatac_resnet18/{task_name}/model_round{round}.pt`

### Step 2: Fisher Precomputation

```bash
python compute_fisher.py --dataset cifar10 --backbone eatac_resnet18 \
    --subnet_drop_rate 0.2 \
    --model_path results/cifar10_100clients_alpha0.3/fedacg_eatac_resnet18/run1/model_round499.pt \
    --fisher_path results/cifar10_100clients_alpha0.3/fedacg_eatac_resnet18/run1/ \
    --num_fisher_samples 2000
```

### Step 3: FedTTA Evaluation (8 modes)

```bash
# Single corruption debug
python run_fedtta.py --dataset cifar10 --tta_mode domain_fedacg_eatac \
    --backbone eatac_resnet18 --subnet_drop_rate 0.2 \
    --model_path <MODEL_PATH> --fisher_path <FISHER_PATH> \
    --precomputed_c_path ./data/CIFAR-10-C \
    --corruption snow --severity 5 \
    --clients_per_corruption 5 --samples_per_client 500 \
    --tta_rounds 3 --local_tta_batches 5 \
    --adapt_steps 1 --adapt_lr 0.005 \
    --e_margin 0.4 --epsilon 0.05 \
    --task_name debug

# Full evaluation (all 15 corruptions)
python run_fedtta.py --dataset cifar10 --tta_mode domain_fedacg_eatac \
    --backbone eatac_resnet18 --subnet_drop_rate 0.2 \
    --model_path <MODEL_PATH> --fisher_path <FISHER_PATH> \
    --precomputed_c_path ./data/CIFAR-10-C \
    --severity 5 --clients_per_corruption 5 --samples_per_client 500 \
    --tta_rounds 10 --local_tta_batches 5 \
    --adapt_steps 3 --adapt_lr 0.005 \
    --e_margin 0.4 --epsilon 0.05 \
    --task_name full_eval
```

## All TTA Modes

| Mode | Description | Type |
|------|-------------|------|
| `source_only` | No adaptation (lower bound) | Baseline |
| `local_tent` | Tent (no communication) | Baseline |
| `local_eata` | EATA (no communication) | Baseline |
| `local_eata_c` | EATA-C (no communication) | Baseline |
| `naive_fedavg_tta` | Single shared adapter, FedAvg (mixed domains) | Ablation |
| `naive_fedacg_tta` | Single shared adapter + momentum (mixed domains) | Ablation |
| `domain_fedavg_eatac` | Per-domain adapter, FedAvg aggregation | Ablation |
| `domain_fedacg_eatac` | Per-domain adapter + momentum (FedACG) | **Main** |

## Project Structure

```
FEDEATA/
├── run_fedtta.py            # FedTTA entry (8 modes)
├── train_fedacg.py          # FL training entry (FedACG)
├── compute_fisher.py        # Fisher precomputation
├── test_eatac.py            # Single-model EATA-C eval
├── dataset/
│   ├── dataset.py           # Data loading, Dirichlet partition, CIFAR-C
│   └── target_clients.py    # Target client builder
├── src/
│   ├── algorithm/
│   │   ├── FedTTA/          # Federated TTA logic
│   │   ├── EATA_C/          # Local TTA algorithm
│   │   ├── FedACG/          # FL training
│   │   ├── FedAvg/          # FL baseline
│   │   └── Base/            # FL base classes
│   └── model/
│       ├── eatac_resnet.py  # EATAC-ResNet with stochastic depth
│       └── resnet.py        # Standard ResNet
└── utils/
    ├── config.py            # Argparse parameters
    ├── metrics.py           # Shared evaluation metrics
    ├── logger.py            # Logging
    └── utils.py             # Seeds, paths, checkpoint loading
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset` | cifar10 | cifar10 / cifar100 / imagenet |
| `--tta_mode` | domain_fedacg_eatac | One of 8 modes |
| `--model_path` | None | Trained FL checkpoint |
| `--fisher_path` | None | Precomputed Fisher |
| `--adapt_steps` | 3 | TTA gradient steps per batch |
| `--adapt_lr` | 0.005 | TTA learning rate |
| `--e_margin` | 0.4 | Entropy filter margin |
| `--epsilon` | 0.05 | Diversity filter threshold |
| `--tta_rounds` | 10 | FedTTA communication rounds |

## Citation

If you use this code, please cite:

```
@misc{fedeata,
  title={FEDEATA: Federated Test-Time Adaptation with Domain Isolation},
  year={2026}
}
```
