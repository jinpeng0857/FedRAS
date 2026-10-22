# FEDEATA 论文方法与实验框架（代码对应版）

> 本文件为主表实验版本的论文框架。正文只允许引用当前代码、保留的 JSON/日志和明确标注证据边界的结果。

## 0. 当前固定版本与可引用结果

论文正文和主表复现实验统一使用下面这一版 FedDR 配置。远端代码已恢复为主表 `s07_i10_r010_lri0_run1` 对应的 drift-tiered FSLR 版本；此前的 directional-risk/effective-risk 控制器不属于本主表结果。

| 项目 | 最终固定设置 |
|---|---|
| 初始化 | FedACG run1，`model_round499.pt`，`lookahead_anchor` |
| 适配范围 | 全部 BatchNorm2d affine 参数；卷积层、分类器及 running statistics 不更新 |
| 优化器 | SGD，learning rate `1×10⁻³`，momentum `0.9`，weight decay `0` |
| batch | `128` |
| DLRS | 全样本软可靠性加权；不使用样本级归因贡献；`α=0.7`，PLPD reference `0.2`，区域尺度 `[0.6, 1.2]`，`λ_RI=0.5` |
| FSLR | drift-tiered；检查间隔 `10`；restore strength `0.7`；漂移阈值 `(0.10, 0.20, 0.35)`；恢复比例 `(0, 0.02, 0.05, 0.10)`；可靠性阈值 `0.55`；EMA decay `0.90`；FSLR gradient clip `15.0`；`LAMBDA_RI=0` |
| 评测 | CIFAR-100-C severity 5，15 域，continual 3 cycles，predict-then-update，完整流 |

在该固定配置下，保留的 run1 JSON 为 `test/outputs/cifar100_fedacg_feddr_sgd1e3_bs128_fslr_s07_i10_r010_lri0_run1.json`：Domain-wise `61.7707%`、Continual `62.4333%`，三个 cycle 分别为 `62.4867% / 62.5387% / 62.2747%`，cycle3-cycle1 为 `-0.2120` 个百分点；初始 clean `76.970%`，流后冻结 clean `74.950%`，前四域 frozen revisit 均值 `56.7625%`。这些数值来自单个 `model_round499.pt`/`lookahead_anchor` checkpoint，不是多重复均值，也不支持显著性或 SOTA 结论。

## 1. 论文主线

论文研究联邦训练模型部署后的连续测试时自适应。主方法 FedDR 由三个代码模块组成：

~~~
FedACG federated checkpoint
        ↓
DLRS: sample reliability → sample-wise region scale → joint soft weight
        ↓
FSLR: periodic Fisher×Drift sparse restoration
        ↓
FedDR deployment adaptation
~~~

FedACG 是主初始化来源；DLRS 和 FSLR 是部署端方法。FedAvg、FedProx、SCAFFOLD、FedDyn、FedAdam 只提供补充初始化，用于测试初始化变化对固定 FedDR 的影响。

## 2. 问题定义

联邦训练产生部署锚点 θ₀。测试阶段连续到达无标签 batch Bₜ，分布在 corruption 或 cycle 之间变化。代码采用 predict-then-update：

$$
\hat y_t=f_{\theta_t}(x_t),\qquad
\theta_{t+1}=\operatorname{FedDR}(\theta_t,\mathcal B_t).
$$

当前 batch 的准确率用更新前原图预测计算；测试结束后的 clean/revisit 只做冻结评估。

## 3. FedDR 方法

### 3.1 适配参数

主路径冻结卷积层和分类器，仅训练全部 BatchNorm2d affine 参数。BN 使用当前 batch 统计，不更新 running statistics。主表 FedDR 使用 batch size 128、SGD、learning rate 1×10⁻³、momentum 0.9 和 zero weight decay；`LAMBDA_RI=0`，因此主表不是 directional-risk/effective-risk 版本。

### 3.2 DLRS

对每个样本计算预测熵 Hᵢ、patch-shuffle 后的 PLPD 和 Top-1/Top-2 margin mᵢ：

$$
r_H=\operatorname{clip}(1-H_i/\log C,0,1),\quad
r_P=\operatorname{clip}((PLPD_i-0.2)/0.8,0,1),
$$

$$
r_i=\operatorname{clip}\{0.7[(r_H+r_P)/2]+0.3m_i,0,1\}.
$$

样本可靠性控制 ReCAP 区域：

$$
\tau_i=0.6+0.6r_i.
$$

区域代理返回 regional entropy REᵢ 和 instability RIᵢ。区域软权重为：

$$
w_i^{reg}=2\sigma((0.7\log C-RE_i)/(0.1\log C)),
$$

最终优化权重为 wᵢ=rᵢwᵢᵣₑg，损失为：

$$
\mathcal L=\frac1B\sum_iw_i(RE_i+0.5RI_i).
$$

所有样本都保留在 batch 中，代码没有硬门控；阈值只写入诊断 counters。

可靠性权重只用于全样本软加权，不引入样本级归因贡献版本。

### 3.3 FSLR

每 10 个测试 batch 计算可训练 BN affine 与锚点的归一化漂移 dₜ。漂移分级选择恢复预算：

$$
d_t<0.10\Rightarrow0\%,\quad
0.10\le d_t<0.20\Rightarrow2\%,\quad
0.20\le d_t<0.35\Rightarrow5\%,\quad
d_t\ge0.35\Rightarrow10\%.
$$

元素按归一化 Fisher 与当前绝对漂移的乘积排序：

$$
s_k=(F_k/\operatorname{RMS}(F))
(|\theta_k-\theta_{0,k}|/\operatorname{RMS}(|\theta-\theta_0|)).
$$

选中元素执行 strength 0.7 的软恢复：

$$
\theta_k\leftarrow0.3\theta_k+0.7\theta_{0,k}.
$$

如果存在 SGD momentum buffer，选中元素的动量同步衰减。

## 4. 实验设计

### 4.1 主比较

主表方法名固定为：

$$
\text{Source / Tent / EATA / SAR / CoTTA / RoTTA / PeTTA / ASR / FedDR}.
$$

保留的 run1 证据已经包含 Source、Tent、EATA、SAR、CoTTA、RoTTA、PeTTA 和 FedDR；ASR 的原始结果文件当前不在保留目录中，因此主表必须把 ASR 标为“待恢复证据”，不能用估计值补齐。主表中的 SAR 应明确区分普通实现与 `sar_official`，并固定选择一个版本，不能混列。

当前可审计的 Continual 主表（15 个 corruption、3 cycles、pre-update，单次 run1）如下：

| Method | Continual mean (%) | Evidence |
|---|---:|---|
| Source | 40.6127 | `cifar100_source.json` |
| Tent | 61.8029 | `cifar100_fedacg_tent.json` |
| EATA | 60.2202 | `cifar100_fedacg_eata.json` |
| SAR (official) | 62.8640 | `cifar100_fedacg_sar_official.json` |
| CoTTA | 59.2464 | `cifar100_fedacg_cotta.json` |
| RoTTA | 35.51 | `test/logs/fedacg_run1_rotta.log` |
| PeTTA | 40.62 | `test/logs/fedacg_run1_petta.log` |
| ASR | **待恢复原始证据** | 当前保留目录无 ASR JSON/日志 |
| FedDR (ours) | 62.4333 | `cifar100_fedacg_feddr_sgd1e3_bs128_fslr_s07_i10_r010_lri0_run1.json` |

该表只登记已有证据；它不能作为多 seed 平均、显著性检验或“所有方法中最佳”的结论。SAR 普通实现的日志为 `62.53%`，与 official 版本不同，不能在同一主表中混用。

### 4.2 协议

- CIFAR-100-C，severity 5，ResNet-18；
- 15 个 corruption 按 main_test.py 固定顺序；
- domain-wise 每域重建适配状态；
- continual 连续 15 域、3 cycles；
- 流结束后冻结评估 clean 和前四域回访；
- FedDR 使用 FedACG lookahead_anchor；
- FedDR 使用固定 batch size 128 和第 0 节的 SGD/FSLR 配置；
- 当前主表 FedDR 是单个 run1 checkpoint，不得把 test seed 或 cycle 当作独立重复；外部基线按各自已声明的实现/官方配置单独登记；
- 独立重复单位是训练 checkpoint，而不是 corruption、cycle 或 batch。

### 4.3 机制验证

机制验证只改变方法开关，所有数值超参数仍保持第 0 节固定值：

1. DLRS（无 FSLR）；
2. DLRS + FSLR（FedDR）；
3. 去掉 DLRS 或 FSLR 的组件开关。

所有消融保持 checkpoint、统计文件、流顺序、batch size、优化器、FSLR 配置和随机种子不变。

### 4.4 联邦补充

对 FedAvg、FedProx、SCAFFOLD、FedDyn、FedAdam，各自训练匹配 checkpoint 后，固定 main_test.py --tta_method feddr 进行部署评测。该表不与 TTA 主表混合排名。

## 5. 结果字段与论证边界

每次运行输出 schema version 9 JSON。论文结果必须能追溯到：

- federated method、checkpoint、anchor key；
- feature/Fisher statistics 文件；
- domain-wise 和 continual 的域级记录；
- clean retention、revisit；
- DLRS reliability、region scale、joint weight；
- FSLR drift、budget、restore events、drift reduction。

论文只报告代码实际存在的指标。RoTTA/PeTTA 当前只有单次日志证据，ASR 需要恢复原始结果后才能填入；这些条目都不能被表述为多重复完整结论。

当前主结果表先登记本节可审计的单次 run1 结果；待测试 seeds 和独立 checkpoint 完成后，再以 `mean ± std` 替换单次值。除该固定证据外，不在正文增加未验证的性能数字。

## 6. 论文结果章节建议顺序

1. **主比较**：同一 FedACG anchor 下的 Source/Tent/EATA/SAR/CoTTA/RoTTA/PeTTA/ASR/FedDR，并在表注中逐项标明 artifact 和协议差异。
2. **长期连续性**：三 cycle accuracy、clean retention、四域回访和漂移曲线。
3. **机制验证**：仅报告在固定配置下的 DLRS/FSLR 组件开关结果。
4. **初始化补充**：FedAvg/FedProx/SCAFFOLD/FedDyn/FedAdam + 固定 FedDR。
5. **代价与审计**：时间、显存、统计生成开销、恢复比例和失败 batch。

正式多重复完成前，单次 run1 结果只能作为固定配置的已完成证据；不得预写性能提升、最佳排名或统计显著性。

## 7. 代码入口

~~~bash
python prepare_main_statistics.py ...
python -m test.source_only_test ...
python main_test.py --tta_method feddr ...
bash test/comparison_all.sh
bash train/train_federated_supplements.sh
~~~

主 FedDR 固定配置由远端恢复后的 `main_test.py`、`sparse_restore.py` 和 `s07_i10_r010_lri0_run1` JSON/日志共同锁定；当前没有重新启动长实验。恢复前的 directional/effective-risk 文件分别保存在 `main_test.py.before_restore_main_table_20260817` 和 `sparse_restore.py.before_restore_main_table_20260817`，可用于审计与回滚。

远端代码审计信息（`/mnt/sda/PythonProject/CJP_project/FEDEATA`）：恢复后 `main_test.py` SHA-256 为 `4ab273852b6f2dbce4683fdd15534c30c42d6ef4901cce454435d542de1b402c`，`sparse_restore.py` SHA-256 为 `57cc7e0c02f3fbe0ebd25f624a0dbbf07736411ac23c915be019bd8708763208`；恢复前的 directional/effective-risk 备份 SHA-256 分别为 `b27ddf1e119410feb03e894a980219a90d41eab878d658e27c04b0235375037f` 和 `e93e1d7d928e847fdacc45c5fc99e49c84698702a23ed93d5e2c08c6f23e8f4d`。恢复后 `python -m py_compile main_test.py sparse_restore.py` 已通过。

主表 run1 的适配命令应显式传入 `--batch_size 128 --feddr_sgd_lr 0.001 --recovery_interval 10 --restore_strength 0.7 --restore_budget_mode drift_tiered --restore_drift_thresholds 0.10 0.20 0.35 --restore_drift_ratios 0.0 0.02 0.05 0.10 --restore_reliability_threshold 0.55 --restore_reliability_ema_decay 0.90 --fslr_grad_clip_norm 15.0 --anchor_key lookahead_anchor`，并使用 `model_round499.pt` 与匹配的 `fedacg_run1_statistics.pt`。不要直接把旧版 `feddr_main.sh` 的 `5×10⁻⁴/interval=5/strength=0.9` 参数当作主表复现实验。
