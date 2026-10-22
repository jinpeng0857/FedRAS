可以。你现在这篇 FedRAS 如果按之前确定的 **6 个核心补实验** 来整理，我建议直接做成下面这种形式。核心原则是：**主实验看泛化，消融实验看模块贡献，continual 曲线看长期稳定性**。

你当前论文已经固定了 continual protocol：15 个 corruption 连续跑 3 个 cycle，共 45 blocks，并且使用 `Cycle 1 / Cycle 3 / Δ3−1 / Revisit / Clean` 作为长期评估指标。

---

# 实验 1：Tiny-ImageNet-C 泛化实验

这个实验回答：

> FedRAS 是否只在 CIFAR-100-C 上有效？

### Table III. Results on Tiny-ImageNet-C

我建议**纵轴写 Method，横轴写指标**：

| Method | Domain-wise Acc. ↑ | Cycle 1 ↑ | Cycle 3 ↑ | Δ3−1 ↑ | Revisit ↑ | Clean ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Source | 18.16 | 18.16 | 18.16 | 0.00 | 9.57 | 65.45 |
| Tent | 11.86 | 1.95 | 0.64 | −1.31 | 0.68 | 0.83 |
| EATA |  |  |  |  |  |  |
| SAR | 9.30 | 1.12 | 0.53 | −0.59 | 0.57 | 0.35 |
| ASR | 11.86 | 2.50 | 0.60 | −1.90 | 0.67 | 0.89 |
| ReCAP | 13.35 | 3.22 | 0.70 | −2.52 | 0.68 | 0.90 |
| **FedRAS** | **13.60** | **12.17** | **11.73** | **−0.44** | **9.81** | **56.67** |

### 横轴
实验指标：

\[
\text{Domain-wise}, A_1,A_3,\Delta_{3-1},\text{Revisit},\text{Clean}
\]

### 纵轴
不同方法：

\[
\text{Source/Tent/EATA/SAR/ASR/ReCAP/FedRAS}
\]

这个表最完整。

如果 Tiny-ImageNet-C 算力压力比较大，可以去掉 Revisit 和 Clean，只保留：

| Method | Domain-wise | Cycle 1 | Cycle 3 | Δ3−1 |
|---|---:|---:|---:|---:|

但完整版更好。

---

# 实验 2：Backbone Generalization

回答：

> FedRAS 是否依赖 ResNet18？

你论文当前已经明确计划比较 ResNet-18 和 ResNet-34。

这里我不建议把所有 baseline 全重新跑。

## Table IV. Generalization Across Backbones

推荐这种：

| Backbone | Method | Cycle 1 ↑ | Cycle 3 ↑ | Δ3−1 ↑ | Revisit ↑ | Clean ↑ |
|---|---|---:|---:|---:|---:|---:|
| ResNet-18 | Source | 40.62 | 40.62 | 0.00 | 26.10 | 76.97 |
|  | ASR | 63.61 | 63.29 | −0.32 | 58.76 | 73.73 |
|  | ReCAP | 62.55 | 60.49 | −2.06 | 55.41 | 69.69 |
|  | **FedRAS** | **63.62** | **63.70** | **+0.08** | **58.38** | **74.50** |
| ResNet-34 | Source |  |  |  |  |  |
|  | ASR |  |  |  |  |  |
|  | ReCAP |  |  |  |  |  |
|  | **FedRAS** |  |  |  |  |  |

### 横轴
continual 指标：

\[
A_1,\quad A_3,\quad \Delta_{3-1},\quad Revisit,\quad Clean
\]

### 纵轴
实际上有两级：

一级：

\[
\text{ResNet18 / ResNet34}
\]

二级：

\[
\text{Source / ASR / ReCAP / FedRAS}
\]

这个布局最好。

---

# 实验 3：Federated Initialization

这个回答：

> FedRAS 的效果是否依赖 FedACG？

你的论文现在主实验使用 FedACG，并且已经声明要测试 different federated initializations。

我建议使用：

- FedAvg
- FedProx
- FedACG

## Table V. Generalization Across Federated Initializations

有两种排法。

### 推荐版

| FL Initialization | Method | Cycle 1 ↑ | Cycle 3 ↑ | Δ3−1 ↑ | Clean ↑ |
|---|---|---:|---:|---:|---:|
| FedAvg | Source | 38.78 | 38.78 | 0.00 | 73.60 |
|  | ASR | 58.17 | 59.60 | +1.42 | 72.02 |
|  | ReCAP | 58.74 | 59.57 | +0.82 | 70.94 |
|  | **FedRAS** | **58.79** | **60.04** | **+1.26** | **72.55** |
| FedProx | Source |  |  |  |  |
|  | ASR |  |  |  |  |
|  | ReCAP |  |  |  |  |
|  | **FedRAS** |  |  |  |  |
| FedACG | Source | 40.62 | 40.62 | 0.00 | 76.97 |
|  | ASR | 63.61 | 63.29 | −0.32 | 73.73 |
|  | ReCAP | 62.55 | 60.49 | −2.06 | 69.69 |
|  | **FedRAS** | **63.62** | **63.70** | **+0.08** | **74.50** |

### 横轴

\[
Cycle1,\ Cycle3,\ \Delta_{3-1},\ Clean
\]

### 纵轴

\[
FedAvg/FedProx/FedACG
\]

下面再嵌套：

\[
Source/ASR/ReCAP/FedRAS
\]

---

如果版面不够，可以进一步压缩成：

| Initialization | Source C3 | ASR C3 | ReCAP C3 | FedRAS C3 |
|---|---:|---:|---:|---:|
| FedAvg | 38.78 | 59.60 | 59.57 | **60.04** |
| FedProx |  |  |  |  |
| FedACG | 40.62 | 63.29 | 60.49 | **63.70** |

但这个压缩版我只建议用于论文篇幅特别紧张的时候。

---

# 实验 4：RGA / DSR 核心组件消融

这是**整篇论文最重要的一张消融表**。

因为 FedRAS 就是：

\[
\boxed{
FedRAS = RGA + DSR
}
\]

## Table VI. Component Ablation of FedRAS

建议：

| Variant | RGA | DSR | Domain-wise ↑ | Cycle 1 ↑ | Cycle 3 ↑ | Δ3−1 ↑ | Revisit ↑ | Clean ↑ |
|---|:---:|:---:|---:|---:|---:|---:|---:|---:|
| Base | ✗ | ✗ | 61.73 | 61.24 | 24.11 | −37.13 | 4.36 | 5.85 |
| Base + RGA | ✓ | ✗ | 61.11 | 61.74 | 59.69 | −2.05 | 56.47 | 68.57 |
| Base + DSR | ✗ | ✓ | 61.73 | 62.01 | 61.31 | −0.69 | 56.23 | 74.92 |
| **FedRAS** | ✓ | ✓ | **60.86** | **62.97** | **63.43** | **+0.46** | **58.55** | **75.40** |

实验4中的 Full FedRAS 与表一中的 FedRAS 使用了相同的 FedACG ResNet-18 checkpoint、统计文件、anchor、seed、数据集和完整 15×3 continual stream，因此数据差异不是由权重或测试数据不同造成的。两次测试的协议配置不同：表一使用 batch size=128、adaptation learning rate=0.001、restore strength=0.7 和 drift-tiered restoration；实验4使用 batch size=64、adaptation learning rate=0.00025、restore strength=0.5，并启用 continuous-risk、hierarchical budget、Fisher×Drift selection 和 sparse ratio=0.01。因此表一的 FedRAS 结果（DW=62.23、C1=63.62、C3=63.70、Revisit=58.38、Clean=74.50）与实验4的 Full FedRAS 结果（DW=60.86、C1=62.97、C3=63.43、Revisit=58.55、Clean=75.40）不应视为同一协议下的重复结果。

### 横轴
模块开关 + 性能：

\[
RGA,\ DSR,\ DW,\ C1,\ C3,\Delta,\ Revisit,\ Clean
\]

### 纵轴
4 个 variant：

\[
Base
\]

\[
Base+RGA
\]

\[
Base+DSR
\]

\[
Full
\]

这张表一定要保留。

你当前正文已经有 DSR alone、RGA alone、Full 的文字分析，只是还缺完整表格。

---

# 实验 5：DSR Coordinate Selection 消融

这个实验回答：

> 为什么 DSR 要用 Fisher × Drift，而不是随机恢复？

你的 Eq.27 本质上定义的是 Fisher-drift score，因此需要直接验证这个设计。

## Table VII. Ablation of Sparse Coordinate Selection

建议：

| Selection Strategy | Cycle 1 ↑ | Cycle 3 ↑ | Δ3−1 ↑ | Revisit ↑ | Clean ↑ |
|---|---:|---:|---:|---:|---:|
| Random | 62.69±0.07 | 63.05±0.02 | +0.36±0.07 | 57.79±0.10 | 74.24±0.03 |
| Fisher-only | 61.74 | 60.18 | −1.57 | 56.90 | 69.16 |
| Drift-only | 60.48 | 60.82 | +0.34 | 55.93 | 74.83 |
| Low-Fisher + Drift |  |  |  |  |  |
| **Fisher × Drift** | **61.51** | **61.68** | **+0.18** | **56.69** | **75.35** |

如果你不想做 Low-Fisher，可以只留四种：

\[
Random
\]

\[
Fisher
\]

\[
Drift
\]

\[
Fisher\times Drift
\]

我其实更推荐四种，干净很多。

### 横轴

\[
Cycle1,\ Cycle3,\Delta_{3-1},Revisit,Clean
\]

### 纵轴

不同 coordinate selection 策略。

这里 **Domain-wise 可以不放**。

因为 DSR 的目标不是短期 domain-wise adaptation，而是：

\[
\text{long-term stability}
\]

所以重点看 C3、Δ、Revisit、Clean。

---

# 实验 6：45-block Continual Stability

这个严格来说，**最好不要做表格作为主结果，而是做 Figure**。

因为它的目的不是看最终一个数字，而是看：

> 模型是否随着测试流逐渐 collapse。

---

## Figure 3. Continual adaptation over 45 corruption blocks

### 横轴 X-axis

\[
\boxed{\text{Corruption Block}}
\]

范围：

\[
1\rightarrow45
\]

你可以在下面标：

- Cycle 1：Block 1–15
- Cycle 2：Block 16–30
- Cycle 3：Block 31–45

甚至加两条竖虚线：

\[
x=15,\quad x=30
\]

表示 cycle boundary。

---

### 纵轴 Y-axis

\[
\boxed{\text{Accuracy (\%)}}
\]

---

### 曲线

建议画：

- Tent
- SAR
- ASR
- ReCAP
- FedRAS

Source 可以画一条水平虚线，也可以不画。

不要把 8 个方法全部塞进去，否则会很乱。

---

## 它对应的原始数据表

你程序最好保存：

| Block | Corruption | Cycle | Tent | SAR | ASR | ReCAP | FedRAS |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Gaussian | 1 |  |  |  |  | 50.30 |
| 2 | Shot | 1 |  |  |  |  | 57.84 |
| … | … | … |  |  |  |  |  |
| 15 | JPEG | 1 |  |  |  |  | 58.48 |
| 16 | Gaussian | 2 |  |  |  |  | 56.13 |
| … | … | … |  |  |  |  |  |
| 45 | JPEG | 3 |  |  |  |  | 58.57 |

但是**这个不用直接放论文主文**。

拿它画 Figure 即可。

---

# 最后六个实验整理成一个总览

| 实验 | 纵轴 / Rows | 横轴 / Columns |
|---|---|---|
| **Tiny-ImageNet-C** | Source, Tent, EATA, SAR, ASR, ReCAP, FedRAS | DW, C1, C3, Δ, Revisit, Clean |
| **Backbone** | ResNet18/34 × Source, ASR, ReCAP, FedRAS | C1, C3, Δ, Revisit, Clean |
| **FL Initialization** | FedAvg/FedProx/FedACG × Source, ASR, ReCAP, FedRAS | C1, C3, Δ, Clean |
| **RGA/DSR Ablation** | Base, +RGA, +DSR, Full | RGA, DSR, DW, C1, C3, Δ, Revisit, Clean |
| **DSR Selection** | Random, Fisher, Drift, Fisher×Drift | C1, C3, Δ, Revisit, Clean |
| **45-block Stability** | 曲线 = Tent/SAR/ASR/ReCAP/FedRAS | X=Block 1–45；Y=Accuracy |

我尤其建议你不要每张表都把所有指标、所有 baseline 塞满。**Table III 是大而全的泛化表；Table IV/V 是小型 generalization 表；Table VI/VII 专门做机制消融；Figure 3 专门负责 long-term story。**

这样整个实验章节的逻辑就会非常顺：

\[
\boxed{\text{Does it work?}}
\]

CIFAR-100-C 主实验

↓

\[
\boxed{\text{Does it generalize?}}
\]

Tiny-ImageNet-C + Backbone + FL initialization

↓

\[
\boxed{\text{Why does it work?}}
\]

RGA/DSR + Fisher-drift 消融

↓

\[
\boxed{\text{Does it remain stable over time?}}
\]

45-block continual curve

这个结构对你现在的 FedRAS 会比较合适。
