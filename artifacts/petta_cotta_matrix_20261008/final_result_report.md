# PeTTA / CoTTA 评测结果（截至 2026-10-09）

## 实验协议

- 联邦初始化：FedACG checkpoint，适配锚点为 `lookahead_anchor`。
- 数据集：CIFAR-100-C、Tiny-ImageNet-C；严重度 5；每数据集预定 15 种 corruption。
- 每次完整评测包括 Domain-wise 15 个块，以及 Continual 3 轮共 45 个块；`max_batches=-1`，遍历完整数据。
- CIFAR-100-C 每块 10,000 张图；Tiny-ImageNet-C 每块 10,000 张图。
- 结果有效性：已完成结果按 15/15、45/45 块、样本总数及有限指标校验。

## 已完成结果

| 数据集 | checkpoint | 算法 | Domain-wise 准确率 | Continual 准确率 | 记录样本数（Domain / Continual） |
|---|---|---:|---:|---:|---:|
| CIFAR-100-C | FedACG run1 | PeTTA | 54.28% | 52.01% | 150,000 / 450,000 |
| CIFAR-100-C | FedACG run1 | CoTTA | 59.09% | 56.18% | 150,000 / 450,000 |
| CIFAR-100-C | FedACG run2 | CoTTA | 58.51% | 55.09% | 150,000 / 450,000 |

准确率来自各 JSON 的 `domain_wise.summary.mean_accuracy` 和 `continual.summary.mean_accuracy`。块样本记录数按块求和；Continual 含 3 轮重复遍历，因此不是去重后的数据量。

## 未完成项

没有通过最终校验的结果文件：CIFAR-100-C 的 run2 PeTTA、run3 PeTTA/CoTTA；Tiny-ImageNet-C 三个 FedACG checkpoint 的 PeTTA/CoTTA。最终状态日志记录这些任务 `FAIL_1`；对应运行日志中的 CUDA 错误为显存不足。最初的 Tiny 统计预处理另有 `STATISTICS_FAILED` 记录，之后重试仍未生成可用的 Tiny 评测结果。因此 Tiny-ImageNet-C 当前没有可报告的准确率。

早期 CIFAR run1 PeTTA 曾因校验脚本把每块样本数误设为 5,000 而记录 `INVALID_RESULT`；修正为 10,000 后，该 JSON 完整通过校验。这个早期状态行保留作审计记录，不代表最终结果无效。

## 文件

- `summary.csv`：每个已完成实验的 Domain-wise / Continual 汇总。
- `per_corruption.csv`：逐 corruption、逐 Continual cycle 的详细指标。
- `results/`：已完成实验的原始 JSON。
- `status.tsv`：服务器队列状态审计记录（含早期校验器错误和失败重试）。
