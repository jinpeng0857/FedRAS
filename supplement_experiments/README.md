# FEDEATA 六组补充实验脚本

本目录只提供实验编排和结果整理，复用仓库已有的 `main_test.py`（适配方法）与
`test.source_only_test`（冻结 Source）。脚本不会自动下载数据、训练 checkpoint，
也不会用缺失结果填 0；每个实验先写 `preflight.json`，再逐任务写日志和结果。

## 实验对应关系

| 文件 | 实验 | 入口/协议 | 主要输出 |
|---|---|---|---|
| `exp1_tiny_imagenet_c.py` | Tiny-ImageNet-C，ResNet18 | FedACG + FedRAS，15 域、3 cycle | Domain、C1/C3、Δ、Revisit、Clean |
| `exp2_resnet34_cifar100c.py` | CIFAR-100-C，ResNet34 | FedACG + FedRAS，15 域、3 cycle | Domain、C1/C3、Δ、Revisit、Clean |
| `exp3_fl_initialization.py` | FL 初始化 | FedACG/FedAvg（可选 FedProx）× Source/ASR/ReCAP/FedRAS | 每个初始化和方法一行 |
| `exp4_component_ablation.py` | RGA/DSR 组件 | Base、Base+RGA、Base+DSR、Full FedRAS | Domain、C1/C3、Δ、Revisit、Clean |
| `exp5_coordinate_ablation.py` | DSR 坐标选择 | Random、Drift-only、Fisher-only、Fisher×Drift | 同上；Random 默认 seed 0/1/2 |
| `exp6_continual_curve.py` | 45-block continual | CIFAR-100-C，15 corruption × 3 cycle | `continual_45_block.csv/json`，可选 PNG |

仓库代码的规范方法名是 **FedRAS = RGA + DSR**。表格中的 “FedRAS” 在脚本中
作为论文显示别名输出为 `FedRAS (FedRAS)`，底层仍传给 `main_test.py --tta_method fedras`。
同理，用户表中的 RGA/DSR 分别对应当前实现的 RGA/DSR；映射记录在实验 4 的
脚本和汇总标签中。

## 运行约定

在项目根目录执行，例如：

```bash
python supplement_experiments/exp1_tiny_imagenet_c.py \
  --model_path train/results/tiny-imagenet_100clients_alpha0.3_equal/fedacg_resnet18/<run>/model_round499.pt \
  --statistics_path outputs/<matching_statistics>.pt \
  --cifar_c_path /path/to/Tiny-ImageNet-C \
  --clean_data_root /path/to/tiny-imagenet-200 \
  --device cuda:0

python supplement_experiments/exp2_resnet34_cifar100c.py \
  --model_path <resnet34_checkpoint.pt> \
  --statistics_path <matching_statistics.pt>

python supplement_experiments/exp3_fl_initialization.py \
  --fedacg_model <fedacg_checkpoint.pt> \
  --fedacg_statistics <fedacg_statistics.pt> \
  --fedavg_model <fedavg_checkpoint.pt> \
  --fedavg_statistics <fedavg_statistics.pt> \
  --include_fedprox

python supplement_experiments/exp4_component_ablation.py \
  --model_path <fedacg_resnet18_checkpoint.pt> \
  --statistics_path <matching_statistics.pt>

python supplement_experiments/exp5_coordinate_ablation.py \
  --model_path <fedacg_resnet18_checkpoint.pt> \
  --statistics_path <matching_statistics.pt>

python supplement_experiments/exp6_continual_curve.py \
  --model_path <fedacg_resnet18_checkpoint.pt> \
  --statistics_path <matching_statistics.pt> --plot

# 若已有 main_test.py JSON，只提取 45-block 曲线，不重新跑模型：
python supplement_experiments/exp6_continual_curve.py --skip_run \
  --result_path <existing_continual.json> --output_dir outputs/supplement_experiments/exp6_existing
```

所有正式流默认 `--max_batches -1`、batch size 64、15 个固定 corruption、3 个
continual cycle 和 predict-then-update。开发检查可传 `--max_batches N`，但该结果
只能作为预检查。先看命令而不运行模型时，在任意入口加 `--dry_run`；已有完整 JSON
会自动复用，强制重跑使用 `--force_rerun`。`--allow_missing` 只会记录并跳过缺失
输入，不能生成可进入论文表格的结果。

每个实验目录包含：

* `preflight.json`：checkpoint、statistics 和数据目录的存在性清单；
* 每任务一个 JSON 和 `logs/*.log`；
* `metrics.json` 与同名 CSV 汇总；实验 6 额外导出 45 行 block 曲线。

实验 3 的 FedAvg/FedProx checkpoint 和 statistics 必须分别使用
`adaptation_anchor`，FedACG 使用 `lookahead_anchor`；statistics 必须由同一个
checkpoint 生成。脚本只负责评测，不会把联邦训练方法误当作 TTA 方法进行排名。
