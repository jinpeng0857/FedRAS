#!/bin/bash
# ============================================================
# FEDEATA: Full experiment pipeline — CIFAR-100
# Usage: bash scripts/run_cifar100.sh [gpu_id]
# ============================================================
set -e

GPU="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
BASE="results/cifar100_100clients_alpha0.3"

echo "========================================="
echo "FEDEATA CIFAR-100 Full Pipeline"
echo "GPU: $GPU"
echo "========================================="

# ---- Step 1: FedACG Source Training ----
echo "[1/4] FedACG training..."
python train_fedacg.py --method fedacg --fedacg_mode fedacg \
    --dataset cifar100 --alpha 0.3 --num_client 100 \
    --backbone eatac_resnet18 --subnet_drop_rate 0.2 \
    --subnet_train_prob 0.3 \
    --global_rounds 600 --local_epochs 2 --learning_rate 0.1 \
    --momentum 0.9 --weight_decay 5e-4 --join_ratio 0.2 \
    --batch_size 64 --task_name run1


python train_fedacg.py --method fedacg --fedacg_mode fedacg \
    --dataset cifar10 --alpha 0.3 --num_client 100 \
    --backbone eatac_resnet18 --subnet_drop_rate 0.2 \
    --subnet_train_prob 0.3 \
    --global_rounds 600 --local_epochs 2 --learning_rate 0.1 \
    --momentum 0.9 --weight_decay 5e-4 --join_ratio 0.15 \
    --batch_size 64 --task_name run1
# Find latest model
MODEL_PATH="$BASE/fedacg_eatac_resnet18/run1/model_round499.pt"
MODEL_DIR="$BASE/fedacg_eatac_resnet18/run1"

# ---- Step 2: Fisher Precomputation ----
echo "[2/4] Fisher computation..."
python compute_fisher.py --dataset cifar100 --backbone eatac_resnet18 \
    --subnet_drop_rate 0.2 \
    --model_path "results/cifar100_100clients_alpha0.3/fedacg_eatac_resnet18/run1/model_round599.pt" \
    --fisher_path "results/cifar100_100clients_alpha0.3/fedacg_eatac_resnet18/run1/fisher_cifar100_2000.pt" \
    --num_fisher_samples 2000

python compute_fisher.py --dataset cifar10 --backbone eatac_resnet18 \
    --subnet_drop_rate 0.2 \
    --model_path "results/cifar10_100clients_alpha0.3/fedacg_eatac_resnet18/run1/model_round599.pt" \
    --fisher_path "results/cifar10_100clients_alpha0.3/fedacg_eatac_resnet18/run1/fisher.pt" \
    --num_fisher_samples 2000

FISHER_PATH="$MODEL_DIR/fisher_cifar100_2000.pt"

# ---- Step 3: Single-model EATA-C Baseline ----
echo "[3/4] Single-model EATA-C evaluation..."
for SEED in 1 2 3; do
    python test_eatac.py --dataset cifar100 --backbone eatac_resnet18 \
        --subnet_drop_rate 0.2 --fisher_alpha 500.0 \
        --model_path "results/cifar100_100clients_alpha0.3/fedacg_eatac_resnet18/run1/model_round599.pt" \
        --fisher_path "results/cifar100_100clients_alpha0.3/fedacg_eatac_resnet18/run1/fisher_cifar100_2000.pt" \
        --adapt_steps 1 --adapt_lr 0.005 \
        --e_margin 0.4 --epsilon 0.5 --alpha_ma 0.05 \
        --test_batch_size 64 --seed 2 --precomputed_c_path ./data/CIFAR-100-C
done

# ---- Step 4: FedTTA (Domain FedACG-EATA-C) ----
echo "[4/4] FedTTA domain_fedacg_eatac..."
python run_fedtta.py --dataset cifar100 \
    --tta_mode domain_fedacg_eatac \
    --backbone eatac_resnet18 --subnet_drop_rate 0.2 \
    --model_path "results/cifar100_100clients_alpha0.3/fedacg_eatac_resnet18/run1/model_round599.pt" \
    --fisher_path "results/cifar100_100clients_alpha0.3/fedacg_eatac_resnet18/run1/fisher_cifar100_2000.pt" \
    --precomputed_c_path ./data/CIFAR-100-C \
    --severity 5 --clients_per_corruption 5 --samples_per_client 500 \
    --tta_rounds 10 --local_tta_batches 5 \
    --adapt_steps 3 --adapt_lr 0.01 \
    --e_margin 0.8 --epsilon 0.4 \
    --reliability_tau 0.5 --gamma_prox 0.01 \
    --lambda_tta 0.85 --freeze_bn \
    --task_name full_pipeline

echo "========================================="
echo "CIFAR-100 pipeline complete!"
echo "========================================="
python test_eatac.py --dataset cifar10 --backbone eatac_resnet18 \
        --subnet_drop_rate 0.2 --fisher_alpha 50.0 \
        --model_path "results/cifar10_100clients_alpha0.3/fedacg_eatac_resnet18/run1/model_round599.pt" \
        --fisher_path "results/cifar10_100clients_alpha0.3/fedacg_eatac_resnet18/run1/fisher.pt" \
        --adapt_steps 1 --adapt_lr 0.01 \
        --e_margin 0.6 --epsilon 0.4 --alpha_ma 0.1 \
        --test_batch_size 64 --seed 2 --precomputed_c_path ./data/CIFAR-10-C