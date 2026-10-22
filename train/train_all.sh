#!/usr/bin/env bash
set -Eeuo pipefail

# 以最佳 FedACG run1（2026-07-11 13:26 日志配置）为统一训练配方。
# 仅 seed、dataset_seed、method 和 task_name 按实验变化。
# 保存目录由代码中的 make_save_path() 根据 method、backbone 和 task_name 自动生成。
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=0,1

# ==================== FedACG matched run1 ====================
# 原始最佳权重：
# train/results/cifar100_100clients_alpha0.3_equal/fedacg_resnet18/cifar100_resnet18_run1/model_round499.pt
/home/ubuntu/anaconda3/envs/vllm/bin/python -m train.train_fedacg \
  --method fedacg \
  --fedacg_mode fedacg \
  --dataset cifar100 \
  --backbone resnet34 \
  --alpha 0.3 \
  --num_client 100 \
  --equal_size \
  --join_ratio 0.15 \
  --global_rounds 500 \
  --local_epochs 3 \
  --batch_size 64 \
  --num_workers 2 \
  --learning_rate 0.05 \
  --momentum 0.9 \
  --weight_decay 0.0005 \
  --local_lr_decay 0.998 \
  --lr_scheduler cosine \
  --label_smoothing 0.1 \
  --use_randaugment \
  --randaugment_n 2 \
  --randaugment_m 9 \
  --server_momentum 0.9 \
  --server_lr 1.0 \
  --ema_decay 0.999 \
  --ce_weight 1.0 \
  --prox_weight 0.01 \
  --num_groups 32 \
  --eval_clean_every 50 \
  --eval_freq 10 \
  --save_freq 50 \
  --amp \
  --seed 42 \
  --dataset_seed 21 \
  --new_dataset_seed 30 \
  --task_name run1_match \
  --device cuda:1 \
  --use_bn_layer

# ==================== FedACG matched seed52 ====================
/home/ubuntu/anaconda3/envs/vllm/bin/python -m train.train_fedacg \
  --method fedacg \
  --fedacg_mode fedacg \
  --dataset cifar100 \
  --backbone resnet18 \
  --alpha 0.3 \
  --num_client 100 \
  --equal_size \
  --join_ratio 0.15 \
  --global_rounds 500 \
  --local_epochs 3 \
  --batch_size 64 \
  --num_workers 2 \
  --learning_rate 0.05 \
  --momentum 0.9 \
  --weight_decay 0.0005 \
  --local_lr_decay 0.998 \
  --lr_scheduler cosine \
  --label_smoothing 0.1 \
  --use_randaugment \
  --randaugment_n 2 \
  --randaugment_m 9 \
  --server_momentum 0.9 \
  --server_lr 1.0 \
  --ema_decay 0.999 \
  --ce_weight 1.0 \
  --prox_weight 0.01 \
  --num_groups 32 \
  --eval_clean_every 50 \
  --eval_freq 10 \
  --save_freq 50 \
  --amp \
  --seed 52 \
  --dataset_seed 31 \
  --new_dataset_seed 30 \
  --task_name seed52_match \
  --device cuda:1 \
  --use_bn_layer



# ==================== FedAvg matched run1 ====================
# FedAvg 不使用 FedACG server momentum/EMA；其余客户端训练协议与最佳 run1 对齐。
/home/ubuntu/anaconda3/envs/vllm/bin/python -m train.train_fedavg \
  --method fedavg \
  --dataset cifar100 \
  --backbone resnet18 \
  --alpha 0.3 \
  --num_client 100 \
  --equal_size \
  --join_ratio 0.15 \
  --global_rounds 500 \
  --local_epochs 3 \
  --batch_size 64 \
  --num_workers 2 \
  --learning_rate 0.05 \
  --momentum 0.9 \
  --weight_decay 0.0005 \
  --local_lr_decay 0.998 \
  --lr_scheduler cosine \
  --label_smoothing 0.1 \
  --use_randaugment \
  --randaugment_n 2 \
  --randaugment_m 9 \
  --ema_decay 0.0 \
  --prox_weight 0.0 \
  --num_groups 32 \
  --eval_clean_every 50 \
  --eval_freq 10 \
  --save_freq 50 \
  --amp \
  --seed 42 \
  --dataset_seed 21 \
  --new_dataset_seed 30 \
  --task_name run1_match \
  --device cuda:1 \
  --use_bn_layer

# ==================== FedProx matched run1 ====================
# FedProx 仅额外保留方法特有的 mu=0.01；其余客户端训练协议与最佳 run1 对齐。
/home/ubuntu/anaconda3/envs/vllm/bin/python -m train.train_fedprox \
  --method fedprox \
  --dataset cifar100 \
  --backbone resnet18 \
  --alpha 0.3 \
  --num_client 100 \
  --equal_size \
  --join_ratio 0.15 \
  --global_rounds 500 \
  --local_epochs 3 \
  --batch_size 64 \
  --num_workers 2 \
  --learning_rate 0.05 \
  --momentum 0.9 \
  --weight_decay 0.0005 \
  --local_lr_decay 0.998 \
  --lr_scheduler cosine \
  --label_smoothing 0.1 \
  --use_randaugment \
  --randaugment_n 2 \
  --randaugment_m 9 \
  --ema_decay 0.0 \
  --fedprox_mu 0.01 \
  --num_groups 32 \
  --eval_clean_every 50 \
  --eval_freq 10 \
  --save_freq 50 \
  --amp \
  --seed 42 \
  --dataset_seed 21 \
  --new_dataset_seed 30 \
  --task_name run1_match \
  --device cuda:1 \
  --use_bn_layer


# ==================== FedDyn matched run1 ====================
# FedDyn：动态正则项，alpha=0.01；其余训练协议与最佳 run1 对齐。
/home/ubuntu/anaconda3/envs/vllm/bin/python -m train.run \
  --method feddyn \
  --dataset cifar100 \
  --backbone resnet18 \
  --alpha 0.3 \
  --num_client 100 \
  --equal_size \
  --join_ratio 0.15 \
  --global_rounds 500 \
  --local_epochs 3 \
  --batch_size 64 \
  --num_workers 2 \
  --learning_rate 0.05 \
  --momentum 0.0 \
  --weight_decay 0.0005 \
  --local_lr_decay 0.998 \
  --lr_scheduler cosine \
  --label_smoothing 0.1 \
  --use_randaugment \
  --randaugment_n 2 \
  --randaugment_m 9 \
  --ema_decay 0.0 \
  --feddyn_alpha 0.01 \
  --num_groups 32 \
  --eval_clean_every 50 \
  --eval_freq 10 \
  --save_freq 50 \
  --amp \
  --seed 42 \
  --dataset_seed 21 \
  --new_dataset_seed 30 \
  --task_name run1_match \
  --device cuda:1 \
  --use_bn_layer

# ==================== FedAdam matched run1 ====================
# FedAdam：服务器端自适应矩估计；其余训练协议与最佳 run1 对齐。
/home/ubuntu/anaconda3/envs/vllm/bin/python -m train.run \
  --method fedadam \
  --dataset cifar100 \
  --backbone resnet18 \
  --alpha 0.3 \
  --num_client 100 \
  --equal_size \
  --join_ratio 0.15 \
  --global_rounds 500 \
  --local_epochs 3 \
  --batch_size 64 \
  --num_workers 2 \
  --learning_rate 0.05 \
  --momentum 0.0 \
  --weight_decay 0.0005 \
  --local_lr_decay 0.998 \
  --lr_scheduler cosine \
  --label_smoothing 0.1 \
  --use_randaugment \
  --randaugment_n 2 \
  --randaugment_m 9 \
  --ema_decay 0.0 \
  --fedadam_beta1 0.9 \
  --fedadam_beta2 0.999 \
  --fedadam_tau 0.001 \
  --fedopt_server_lr 1.0 \
  --num_groups 32 \
  --eval_clean_every 50 \
  --eval_freq 10 \
  --save_freq 50 \
  --amp \
  --seed 42 \
  --dataset_seed 21 \
  --new_dataset_seed 30 \
  --task_name run1_match \
  --device cuda:1 \
  --use_bn_layer

# ==================== SCAFFOLD matched run1 ====================
# SCAFFOLD：服务器/客户端控制变量，server_lr=1.0；其余训练协议与最佳 run1 对齐。
/home/ubuntu/anaconda3/envs/vllm/bin/python -m train.run \
  --method scaffold \
  --dataset cifar100 \
  --backbone resnet18 \
  --alpha 0.3 \
  --num_client 100 \
  --equal_size \
  --join_ratio 0.15 \
  --global_rounds 500 \
  --local_epochs 3 \
  --batch_size 64 \
  --num_workers 2 \
  --learning_rate 0.1 \
  --momentum 0.0 \
  --weight_decay 0.0 \
  --local_lr_decay 0.998 \
  --lr_scheduler cosine \
  --label_smoothing 0.1 \
  --use_randaugment \
  --randaugment_n 2 \
  --randaugment_m 9 \
  --ema_decay 0.0 \
  --scaffold_server_lr 1.0 \
  --num_groups 32 \
  --eval_clean_every 50 \
  --eval_freq 10 \
  --save_freq 50 \
  --amp \
  --seed 42 \
  --dataset_seed 21 \
  --new_dataset_seed 30 \
  --task_name run1_match \
  --device cuda:1 \
  --use_bn_layer
