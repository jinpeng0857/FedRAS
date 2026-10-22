#!/usr/bin/env bash
set -Eeuo pipefail

# Supplementary source-training algorithms for the fixed FedDR TTA protocol:
# FedAvg, FedProx, SCAFFOLD, FedDyn, and FedAdam.  This script only creates
# federated initialization checkpoints; it does not run TTA evaluation.
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:1}"

COMMON=(
  --dataset cifar100
  --backbone resnet18
  --alpha 0.3
  --num_client 100
  --equal_size
  --join_ratio 0.15
  --global_rounds 500
  --local_epochs 3
  --batch_size 64
  --num_workers 2
  --local_lr_decay 0.998
  --lr_scheduler cosine
  --label_smoothing 0.1
  --use_randaugment
  --randaugment_n 2
  --randaugment_m 9
  --ema_decay 0.0
  --num_groups 32
  --eval_clean_every 50
  --eval_freq 10
  --save_freq 50
  --amp
  --seed 42
  --dataset_seed 21
  --new_dataset_seed 30
  --task_name run1_match
  --device "$DEVICE"
  --use_bn_layer
)

# FedAvg: standard sample-count-weighted aggregation.
"$PYTHON" -m train.train_fedavg \
  --method fedavg \
  --learning_rate 0.05 \
  --momentum 0.9 \
  --weight_decay 0.0005 \
  --prox_weight 0.0 \
  "${COMMON[@]}"

# FedProx: FedAvg aggregation with the canonical proximal coefficient.
"$PYTHON" -m train.train_fedprox \
  --method fedprox \
  --learning_rate 0.05 \
  --momentum 0.9 \
  --weight_decay 0.0005 \
  --fedprox_mu 0.01 \
  "${COMMON[@]}"

# SCAFFOLD: server/client control variates.
"$PYTHON" -m train.run \
  --method scaffold \
  --learning_rate 0.1 \
  --momentum 0.0 \
  --weight_decay 0.0 \
  --scaffold_server_lr 1.0 \
  "${COMMON[@]}"

# FedDyn: dynamic local regularization and server history.
"$PYTHON" -m train.run \
  --method feddyn \
  --learning_rate 0.05 \
  --momentum 0.0 \
  --weight_decay 0.0005 \
  --feddyn_alpha 0.01 \
  "${COMMON[@]}"

# FedAdam: adaptive server moments over client deltas.
"$PYTHON" -m train.run \
  --method fedadam \
  --learning_rate 0.05 \
  --momentum 0.0 \
  --weight_decay 0.0005 \
  --fedadam_beta1 0.9 \
  --fedadam_beta2 0.999 \
  --fedadam_tau 0.001 \
  --fedopt_server_lr 1.0 \
  "${COMMON[@]}"
