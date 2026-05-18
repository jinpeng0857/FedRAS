python train_fedacg.py --method fedacg --fedacg_mode fedacg --dataset cifar10 --alpha 0.3 --num_client 100 --backbone eatac_resnet18 --subnet_drop_rate 0.2 --subnet_train_prob 0.3 --global_rounds 500 --local_epochs 5 --learning_rate 0.1 --momentum 0.0 --weight_decay 5e-4 --join_ratio 0.05 --batch_size 64 --task_name run2 

python compute_fisher.py --dataset cifar10 --backbone eatac_resnet18 --subnet_drop_rate 0.2 --model_path results/cifar10_100clients_alpha0.3/fedacg_eatac_resnet18/run2/model_round499.pt --fisher_path results/cifar10_100clients_alpha0.3/fedacg_eatac_resnet18/run2/fisher.pt --num_fisher_samples 2000 

baseline: Source-Only Mean Acc: 45.42%   EATA-C  Mean Acc:73.82%    Mean ECE:10.53%    seed=1保证可复现性
python test_eatac.py --dataset cifar10 --backbone eatac_resnet18 --fisher_alpha 20.0 --subnet_drop_rate 0.2 --adapt_steps 2 --adapt_lr 0.01 --e_margin 0.4 --epsilon 0.4 --test_batch_size 64 --seed 1 --model_path results/cifar10_100clients_alpha0.3/fedacg_eatac_resnet18/run2/model_round499.pt --fisher_path results/cifar10_100clients_alpha0.3/fedacg_eatac_resnet18/run2/fisher.pt --precomputed_c_path ./data/CIFAR-10-C

seed=2   EATA-C  Mean Acc:73.79%       Mean ECE:10.63%    Total Forwards:7065    Total Backwards:4710
python test_eatac.py --dataset cifar10 --backbone eatac_resnet18 --fisher_alpha 20.0 --subnet_drop_rate 0.2 --adapt_steps 2 --adapt_lr 0.01 --e_margin 0.4 --epsilon 0.4 --test_batch_size 64 --seed 2 --model_path results/cifar10_100clients_alpha0.3/fedacg_eatac_resnet18/run2/model_round499.pt --fisher_path results/cifar10_100clients_alpha0.3/fedacg_eatac_resnet18/run2/fisher.pt --precomputed_c_path ./data/CIFAR-10-C

seed=3   EATA-C  Mean Acc:  73.68%               Mean ECE: 10.69%
python test_eatac.py --dataset cifar10 --backbone eatac_resnet18 --fisher_alpha 20.0 --subnet_drop_rate 0.2 --adapt_steps 2 --adapt_lr 0.01 --e_margin 0.4 --epsilon 0.4 --test_batch_size 64 --seed 3 --model_path results/cifar10_100clients_alpha0.3/fedacg_eatac_resnet18/run2/model_round499.pt --fisher_path results/cifar10_100clients_alpha0.3/fedacg_eatac_resnet18/run2/fisher.pt --precomputed_c_path ./data/CIFAR-10-C

step=3是加强版，但计算消耗高，适合做补充
python test_eatac.py --dataset cifar10 --backbone eatac_resnet18 --fisher_alpha 50.0 --subnet_drop_rate 0.2 --adapt_steps 3 --adapt_lr 0.01 --e_margin 0.4 --epsilon 0.4 --test_batch_size 64 --model_path results/cifar10_100clients_alpha0.3/fedacg_eatac_resnet18/run2/model_round499.pt --fisher_path results/cifar10_100clients_alpha0.3/fedacg_eatac_resnet18/run2/fisher.pt --precomputed_c_path ./data/CIFAR-10-C


CIFAR100数据集
python train_fedacg.py --method fedacg --fedacg_mode fedacg --dataset cifar100 --alpha 0.3 --num_client 100 --backbone eatac_resnet18 --subnet_drop_rate 0.2 --subnet_train_prob 0.3 --global_rounds 500 --local_epochs 5 --learning_rate 0.1 --momentum 0.0 --weight_decay 5e-4 --join_ratio 0.05 --batch_size 64 --task_name run3 

python compute_fisher.py --dataset cifar100 --backbone eatac_resnet18 --subnet_drop_rate 0.2 --model_path results/cifar100_100clients_alpha0.3/fedacg_eatac_resnet18/run3/model_round499.pt --fisher_path results/cifar100_100clients_alpha0.3/fedacg_eatac_resnet18/run3 --num_fisher_samples 2000

# CIFAR-100 test (precomputed corruption)
python test_eatac.py --dataset cifar100 --backbone eatac_resnet18 --fisher_alpha 50.0 --subnet_drop_rate 0.2 --adapt_steps 3 --adapt_lr 0.005 --e_margin 0.4 --epsilon 0.4 --test_batch_size 64 --model_path results/cifar100_100clients_alpha0.3/fedacg_eatac_resnet18/run3/model_round499.pt --fisher_path results/cifar100_100clients_alpha0.3/fedacg_eatac_resnet18/run3/fisher_cifar100_2000.pt --precomputed_c_path ./data/CIFAR-100-C 


## 4. 参数速查表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | cifar10 | cifar10 / cifar100 |
| `--tta_mode` | domain_fedacg_eatac | 共 8 种模式（见 3.1-3.8） |
| `--model_path` | None | 源域 FedACG 训练好的 `.pt` 文件 |
| `--fisher_path` | None | 预计算 Fisher `.pt` 文件 |
| `--severity` | 5 | CIFAR-C 损坏严重度 (1-5) |
| `--clients_per_corruption` | 5 | 每个 corruption 拆成几个 target client |
| `--samples_per_client` | None | 每个 target client 最多样本数 |
| `--tta_rounds` | 10 | FedTTA 通信轮数 |
| `--local_tta_batches` | None | 每轮每客户端适应多少 batch（None=全量） |
| `--target_participation_rate` | 1.0 | 每轮参与率（覆盖 `--join_ratio`） |
| `--lambda_tta` | 0.85 | FedACG 动量系数 |
| `--gamma_prox` | 0.01 | 局部近端正则强度 |
| `--adapt_steps` | 3 | 每 batch 内 TTA 梯度步数 |
| `--adapt_lr` | 0.005 | TTA 学习率 |
| `--beta_fisher` | 50.0 | Fisher 正则强度（即 `fisher_alpha`） |
| `--alpha_calib` | 0.1 | 熵正则强度（即 `entropy_reg_weight`） |
| `--e_margin` | **0.4** | 熵过滤阈值 = margin * ln(C) |
| `--epsilon` | **0.05** | 多样性过滤余弦阈值 |
| `--reliability_tau` | **0.5** | reliability softmax 温度（越小越选择性） |
| `--weight_by_num_samples` | False | 聚合时是否乘以样本数权重 |
| `--freeze_bn` | True | 冻结 BN running stats |
| `--update_bn_stats` | False | 允许 BN running stats 更新（覆盖 freeze_bn） |
| `--eval_after_adapt` | False | 适应后重新 forward 评估而不是 predict-before-adapt |
| `--seed` | 42 | 随机种子 |



---

# CIFAR-100 完整训练 → 测试流程

所有命令从项目根目录 `D:/dp1/FEDEATA` 执行。

## 1. 源域 FedACG 训练

```bash
python train_fedacg.py \
    --method fedacg --fedacg_mode fedacg \
    --dataset cifar100 --alpha 0.3 --num_client 100 \
    --backbone eatac_resnet18 --subnet_drop_rate 0.2 --subnet_train_prob 0.3 \
    --global_rounds 500 --local_epochs 5 --learning_rate 0.1 \
    --momentum 0.0 --weight_decay 5e-4 \
    --join_ratio 0.05 --batch_size 64 --server_momentum 0.85 --prox_weight 0.01 \
    --save_freq 10 --task_name run1
```

输出：`results/cifar100_100clients_alpha0.3/fedacg_eatac_resnet18/run3/model_round499.pt`

## 2. Fisher 信息计算

```bash
python compute_fisher.py \
    --dataset cifar100 --backbone eatac_resnet18 --subnet_drop_rate 0.2 \
    --model_path results/cifar100_100clients_alpha0.3/fedacg_eatac_resnet18/run3/model_round499.pt \
    --fisher_path results/cifar100_100clients_alpha0.3/fedacg_eatac_resnet18/run3 \
    --num_fisher_samples 2000 --test_batch_size 64
```

输出：`results/cifar100_100clients_alpha0.3/fedacg_eatac_resnet18/run3/fisher_cifar100_2000.pt`

## 3. 干净集测试

```bash
python test_clean.py \
    --model_path results/cifar100_100clients_alpha0.3/fedacg_eatac_resnet18/run3/model_round499.pt \
    --dataset cifar100 --backbone eatac_resnet18 --batch_size 128
```

## 4. 单模型 EATA-C 基线（15 corruption 全测）

```bash
python test_eatac.py \
    --dataset cifar100 --backbone eatac_resnet18 --subnet_drop_rate 0.2 \
    --model_path results/cifar100_100clients_alpha0.3/fedacg_eatac_resnet18/run3/model_round499.pt \
    --fisher_path results/cifar100_100clients_alpha0.3/fedacg_eatac_resnet18/run3/fisher_cifar100_2000.pt \
    --fisher_alpha 50.0 --adapt_steps 3 --adapt_lr 0.005 \
    --e_margin 0.4 --epsilon 0.05 --test_batch_size 64 --seed 1
```

## 5. 单 corruption 快速调试（snow）

```bash
python run_fedtta.py \
    --dataset cifar100 \
    --tta_mode domain_fedacg_eatac \
    --backbone eatac_resnet18 --subnet_drop_rate 0.2 \
    --model_path results/cifar100_100clients_alpha0.3/fedacg_eatac_resnet18/run3/model_round499.pt \
    --fisher_path results/cifar100_100clients_alpha0.3/fedacg_eatac_resnet18/run3/fisher_cifar100_2000.pt \
    --precomputed_c_path ./data/CIFAR-100-C \
    --corruption snow --severity 5 \
    --clients_per_corruption 5 --samples_per_client 500 \
    --target_label_skew --target_alpha 0.3 \
    --target_participation_rate 1.0 \
    --tta_rounds 3 --local_tta_batches 5 \
    --adapt_steps 1 --adapt_lr 0.005 \
    --e_margin 0.4 --epsilon 0.05 \
    --reliability_tau 0.5 --gamma_prox 0.0 \
    --task_name debug_snow_fedtta
```

确认主流程跑通后，试 gamma_prox=0.01：

```bash
python run_fedtta.py \
    --dataset cifar100 \
    --tta_mode domain_fedacg_eatac \
    --backbone eatac_resnet18 --subnet_drop_rate 0.2 \
    --model_path results/cifar100_100clients_alpha0.3/fedacg_eatac_resnet18/run3/model_round499.pt \
    --fisher_path results/cifar100_100clients_alpha0.3/fedacg_eatac_resnet18/run3/fisher_cifar100_2000.pt \
    --precomputed_c_path ./data/CIFAR-100-C \
    --corruption snow --severity 5 \
    --clients_per_corruption 5 --samples_per_client 500 \
    --target_label_skew --target_alpha 0.3 \
    --target_participation_rate 1.0 \
    --tta_rounds 3 --local_tta_batches 5 \
    --adapt_steps 1 --adapt_lr 0.005 \
    --e_margin 0.4 --epsilon 0.05 \
    --reliability_tau 0.5 --gamma_prox 0.01 \
    --task_name debug_snow_fedtta_prox
```

## 6. 多 corruption 逐个排查（依次替换 --corruption）

支持以下 15 个 corruption 名称：

```
gaussian_noise  shot_noise  impulse_noise
defocus_blur   glass_blur  motion_blur  zoom_blur
snow  frost  fog  brightness  contrast
elastic_transform  pixelate  jpeg_compression
```

例：换 `--corruption gaussian_noise` 即可测噪声类损坏。

## 7. 15 corruption 主实验

```bash
python run_fedtta.py \
    --dataset cifar100 \
    --tta_mode domain_fedacg_eatac \
    --backbone eatac_resnet18 --subnet_drop_rate 0.2 \
    --model_path results/cifar100_100clients_alpha0.3/fedacg_eatac_resnet18/run3/model_round499.pt \
    --fisher_path results/cifar100_100clients_alpha0.3/fedacg_eatac_resnet18/run3/fisher_cifar100_2000.pt \
    --precomputed_c_path ./data/CIFAR-100-C \
    --severity 5 \
    --clients_per_corruption 5 --samples_per_client 500 \
    --target_label_skew --target_alpha 0.3 \
    --target_participation_rate 1.0 \
    --tta_rounds 10 --local_tta_batches 5 \
    --adapt_steps 3 --adapt_lr 0.005 \
    --e_margin 0.4 --epsilon 0.05 \
    --reliability_tau 0.5 --gamma_prox 0.01 \
    --lambda_tta 0.85 \
    --task_name full_15corr_5cli_10rnd
```

## 8. Baseline 对比（均基于 CIFAR-100）

### 8.1 Source-Only

```bash
python run_fedtta.py --dataset cifar100 --tta_mode source_only \
    --backbone eatac_resnet18 --subnet_drop_rate 0.2 \
    --model_path results/cifar100_100clients_alpha0.3/fedacg_eatac_resnet18/run3/model_round499.pt \
    --clients_per_corruption 5 --samples_per_client 500 \
    --test_batch_size 64 --task_name baseline_source_only --seed 1
```

### 8.2 Local Tent

```bash
python run_fedtta.py --dataset cifar100 --tta_mode local_tent \
    --backbone eatac_resnet18 --subnet_drop_rate 0.2 \
    --model_path results/cifar100_100clients_alpha0.3/fedacg_eatac_resnet18/run3/model_round499.pt \
    --clients_per_corruption 5 --samples_per_client 500 \
    --adapt_lr 0.005 --test_batch_size 64 --task_name baseline_local_tent --seed 1
```

### 8.3 Local EATA

```bash
python run_fedtta.py --dataset cifar100 --tta_mode local_eata \
    --backbone eatac_resnet18 --subnet_drop_rate 0.2 \
    --model_path results/cifar100_100clients_alpha0.3/fedacg_eatac_resnet18/run3/model_round499.pt \
    --clients_per_corruption 5 --samples_per_client 500 \
    --adapt_steps 3 --adapt_lr 0.005 --e_margin 0.4 --epsilon 0.05 \
    --test_batch_size 64 --task_name baseline_local_eata --seed 1
```

### 8.4 Local EATA-C

```bash
python run_fedtta.py --dataset cifar100 --tta_mode local_eata_c \
    --backbone eatac_resnet18 --subnet_drop_rate 0.2 \
    --model_path results/cifar100_100clients_alpha0.3/fedacg_eatac_resnet18/run3/model_round499.pt \
    --fisher_path results/cifar100_100clients_alpha0.3/fedacg_eatac_resnet18/run3/fisher_cifar100_2000.pt \
    --clients_per_corruption 5 --samples_per_client 500 \
    --adapt_steps 3 --adapt_lr 0.005 --beta_fisher 50 \
    --e_margin 0.4 --epsilon 0.05 --test_batch_size 64 --task_name baseline_local_eatac --seed 1
```

### 8.5 Naive FedAvg-TTA（无域隔离，证明负迁移）

```bash
python run_fedtta.py --dataset cifar100 --tta_mode naive_fedavg_tta \
    --backbone eatac_resnet18 --subnet_drop_rate 0.2 \
    --model_path results/cifar100_100clients_alpha0.3/fedacg_eatac_resnet18/run3/model_round499.pt \
    --fisher_path results/cifar100_100clients_alpha0.3/fedacg_eatac_resnet18/run3/fisher_cifar100_2000.pt \
    --clients_per_corruption 5 --samples_per_client 500 \
    --tta_rounds 5 --local_tta_batches 5 --target_participation_rate 0.2 \
    --adapt_steps 3 --adapt_lr 0.005 --beta_fisher 50 \
    --e_margin 0.4 --epsilon 0.05 --test_batch_size 64 --task_name baseline_naive_fedavg --seed 1
```

### 8.6 Naive FedACG-TTA（无域隔离 + 动量）

```bash
python run_fedtta.py --dataset cifar100 --tta_mode naive_fedacg_tta \
    --backbone eatac_resnet18 --subnet_drop_rate 0.2 \
    --model_path results/cifar100_100clients_alpha0.3/fedacg_eatac_resnet18/run3/model_round499.pt \
    --fisher_path results/cifar100_100clients_alpha0.3/fedacg_eatac_resnet18/run3/fisher_cifar100_2000.pt \
    --clients_per_corruption 5 --samples_per_client 500 \
    --tta_rounds 5 --local_tta_batches 5 --target_participation_rate 0.2 \
    --lambda_tta 0.85 --gamma_prox 0.01 \
    --adapt_steps 3 --adapt_lr 0.005 --beta_fisher 50 \
    --e_margin 0.4 --epsilon 0.05 --test_batch_size 64 --task_name baseline_naive_fedacg --seed 1
```

### 8.7 Domain FedAvg-EATA-C（域隔离，无动量）

```bash
python run_fedtta.py --dataset cifar100 --tta_mode domain_fedavg_eatac \
    --backbone eatac_resnet18 --subnet_drop_rate 0.2 \
    --model_path results/cifar100_100clients_alpha0.3/fedacg_eatac_resnet18/run3/model_round499.pt \
    --fisher_path results/cifar100_100clients_alpha0.3/fedacg_eatac_resnet18/run3/fisher_cifar100_2000.pt \
    --clients_per_corruption 5 --samples_per_client 500 \
    --tta_rounds 10 --local_tta_batches 5 --target_participation_rate 1.0 \
    --adapt_steps 3 --adapt_lr 0.005 --beta_fisher 50 \
    --e_margin 0.4 --epsilon 0.05 --test_batch_size 64 \
    --freeze_bn --task_name baseline_domain_fedavg --seed 1
```

### 8.8 Domain FedACG-EATA-C（★ 主方法，域隔离 + FedACG 动量）

```bash
python run_fedtta.py --dataset cifar100 --tta_mode domain_fedacg_eatac \
    --backbone eatac_resnet18 --subnet_drop_rate 0.2 \
    --model_path results/cifar100_100clients_alpha0.3/fedacg_eatac_resnet18/run3/model_round499.pt \
    --fisher_path results/cifar100_100clients_alpha0.3/fedacg_eatac_resnet18/run3/fisher_cifar100_2000.pt \
    --clients_per_corruption 5 --samples_per_client 500 \
    --tta_rounds 10 --local_tta_batches 5 --target_participation_rate 1.0 \
    --lambda_tta 0.85 --gamma_prox 0.01 \
    --adapt_steps 3 --adapt_lr 0.005 --beta_fisher 50 \
    --e_margin 0.4 --epsilon 0.05 --test_batch_size 64 \
    --freeze_bn --reliability_tau 0.5 \
    --task_name main_domain_fedacg --seed 1
```

---