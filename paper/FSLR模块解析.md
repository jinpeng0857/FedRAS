# FSLR 模块解析（论文实验视角）

> 本文档基于当前仓库代码（`sparse_restore.py`、`main_test.py`、`prepare_main_statistics.py`）、论文稿件（`论文初稿.md`、`FEDEATA_论文框架_待补主实验.md`）以及现有实验输出 JSON（`test/outputs/fedacg_run1/`、`outputs/feddr_run1/`、`tmp/tta_remote/outputs/`）梳理 FSLR（Fisher-guided Sparse Local Restoration）在论文实验中的定义、实现、证据与不一致之处。

## 1. 模块定位

FedDR = FedACG checkpoint（初始化）+ DLRS（短期可靠度控制）+ FSLR（长期漂移控制）。

- FSLR 是**部署端**模块，只作用在可训练 BatchNorm affine 参数（γ、β）上，不修改联邦训练/聚合。
- 与 DLRS 的时间尺度互补：DLRS 在每个优化步之前压低不可靠样本的更新贡献；FSLR 在更新累积之后周期性测量全局漂移并**稀疏拉回**少量"重要且漂移大"的坐标到部署锚点 θ₀。
- FSLR 是 source-statistics-assisted：对角 Fisher 重要性离线从源数据计算，在线阶段不访问源样本。

## 2. 算法定义（论文 3.3）

锚点：θ₀,A = 部署初始化时 A 中 BN affine 坐标的克隆。

每个坐标 j 的绝对漂移：d_j = |θ_j − θ₀,j|。

**触发统计量（归一化全局漂移）**：

```
D_t = sqrt(Σ_j (θ_j − θ₀,j)²) / (sqrt(Σ_j θ₀,j²) + 1e-12)
```

**Fisher 重要性（离线）**：以源 batch 损失 L_b、batch size n_b 计算

```
F_j = (1/Σ_b n_b) Σ_b n_b (∂L_b/∂θ_j)²
```

**漂移分级预算**（默认阈值 (0.10, 0.20, 0.35)，比例 (0, 0.02, 0.05, 0.10) 或最终版 (0, 0.02, 0.05, 0.30)）：

| D_t | 恢复比例 ρ_t |
|---|---|
| D_t < 0.10 | 0（不恢复） |
| 0.10 ≤ D_t < 0.20 | 0.02 |
| 0.20 ≤ D_t < 0.35 | 0.05 |
| D_t ≥ 0.35 | 0.10（最终版 0.30） |

**坐标排序分数**（只对正分数坐标排序，做全局 RMS 归一化）：

```
s_j = (F_j / RMS(F)) · (d_j / RMS(d))
```

恢复 top-K_t 个坐标，K_t = min(|A|, max(1, ⌈ρ_t|A|⌉))；ρ_t = 0 时不恢复。

**软恢复 + 动量衰减**（λ = restore strength）：

```
θ_j ← (1−λ)θ_j + λθ₀,j
v_j ← (1−λ)v_j          # SGD momentum buffer，衰减而非清零
```

`fixed` 模式为 legacy 消融：不看漂移，总是恢复全局 Fisher 最高的 1% 坐标（`sparse_ratio=0.01`）。

梯度裁剪（κ=15.0）不属于 FSLR：它既不决定漂移、不选坐标、也不执行恢复，是外部优化安全阀。

## 3. 代码实现映射

### 3.1 `sparse_restore.py` — `PeriodicSparseRestorer`

| 代码 | 对应公式/语义 |
|---|---|
| `_collect_trainable_bn_affine` | 收集所有 `BatchNorm2d` 中 `requires_grad` 的 `weight`/`bias`（FedDR 路径为全部 BN affine，含 layer4） |
| `anchors = param.detach().clone()` | 部署初始化时的恢复锚点 |
| `drift_stats()` | 归一化漂移，分 total / gamma / beta 三组统计 |
| `_ratio_for_drift()` | 漂移 → 比例 ρ_t（fixed 模式恒返回 sparse_ratio） |
| `_select_fisher_drift_indices()` | RMS 归一化 Fisher × 当前绝对漂移排序；**只选正分数元素**（drift=0 或 fisher=0 的元素不参与）；含可选的 sample-contribution 加分项（FedDR 构建时权重为 0，禁用） |
| `step()` | 每 `interval` 个 batch 检查一次；`reliability` 参数非 None 时：当前批可靠度 < 阈值 → 跳过恢复（`skipped_low_reliability`）；否则 budget 比例被当前可靠度缩放（`selected_ratio = base_ratio × reliability`） |
| 恢复循环 | `flat_param[local] = (1−λ)·flat_param[local] + λ·flat_anchor[local]`；对存在 `momentum_buffer` 的坐标执行 `(1−λ)` 衰减 |
| `stats` | 输出完整诊断：scheduled/restore/skipped 事件数、`budget_event_counts`、`restored_assignments_total`、`restored_unique_parameters`、`repeat_restore_mean`、`last_selected_drift_energy_ratio`、`last_selection_jaccard`、`drift_before/after_last_restore`、`last_drift_reduction`、per-layer 明细等 |

### 3.2 `prepare_main_statistics.py` — Fisher 制备

- `compute_fishers(..., mode="main")`：`model.eval()`、`requires_grad_(False)`，仅对 BN affine 开梯度；BN 使用**源 running statistics**；损失用**源真标签**的 CrossEntropy；每 batch 后累加 `(grad² · batch_size)`，最后除以总样本数。
- 即 `F_j = Σ_b n_b·(∂L_b/∂θ_j)² / Σ_b n_b`（batch-mean gradient square × 样本数加权），与论文公式一致。
- 默认用完整 CIFAR-100 训练集（50000 样本、782 个 batch，batch=64；feddr_main.sh 中 batch=128 → 391 batch）；同一锚点、同一种子。
- 保存的 artifact 还含 ReCAP feature variance（500 个样本）与 meta（fisher_estimator/label_source/bn_mode/denominator/scope 等）。

### 3.3 `main_test.py` — 实验接线

- `apply_feddr_config`：当 `tta_method` 以 `_fslr` 结尾（`feddr` → 解析为 `dlrs_fslr`）时，锁定最终固定配置：interval=5、strength=0.9、drift-tiered、阈值 (0.10,0.20,0.35)、比例 (0,0.02,0.05,0.30)、可靠性阈值 0.55、EMA decay 0.90、batch 128、lr 5e-4、SGD。
- `build_main_adapter`：`use_fslr = tta_method.endswith("_fslr")` → 构建 `PeriodicSparseRestorer`；否则 `DisabledSparseRestorer`（只记录漂移不动参数）。
  - `dlrs`/`dlrs_fslr`（FedDR）→ `DLRSAdapter`，`parameter_scope = "bn_affine_all"`（含 layer4）。
  - `recap_fslr`/`eata_recap_fslr`（legacy 消融）→ `DualReliabilityAdapter`。
- `DLRSAdapter.step`（FedDR 主路径）：`recovery_credit = mean(sample_reliability)`，`restorer.step(reliability=recovery_credit)` → **当前批平均可靠度同时决定是否授权恢复与预算缩放**。
- `DualReliabilityAdapter.step`（recap 路径）：`restorer.step()` 不带 reliability → 无可靠度门控。
- 梯度裁剪：`fslr_grad_clip_norm=15.0` 默认只对 `*_fslr` 方法生效（legacy 策略）；组件消融可用 `--grad_clip_norm` 统一所有行（0 表示全部关闭）。
- 输出 JSON 的 `restoration` 块由 args 生成（interval/strength/阈值/比例/参数与动量更新式），运行级诊断来自 `restorer.stats`。

## 4. 实验协议中 FSLR 的证据链

每个运行输出 schema 9 JSON，与 FSLR 相关的可追溯字段：

- 配置块：`restoration.{enabled,budget_mode,selection,interval,restore_strength,drift_thresholds,drift_ratios,reliability_threshold,reliability_policy,parameter_update,momentum_update}`
- domain-wise 每域：`restore_events / scheduled_restore_events / skipped_low_drift_events / restore_budget_event_counts / restored_assignments_total / final_bn_affine_drift`
- continual 每 cycle 与每 corruption 块：同上 + `newly_restored_unique_parameters`
- 流末：`final_bn_affine_drift`（total/gamma/beta）、`clean_retention`、`post_stream.revisit`

**Domain-wise 协议**：每域重建模型/优化器/restorer → 单域内漂移通常不足 0.10 → FSLR 基本不触发（见 §5）。

**Continual 协议**：3 cycles 连续不重置 → 漂移跨域累积 → FSLR 在 cycle 1 中段开始持续触发，是 FSLR 的主要观测场景。

## 5. 实际实验结果证据（现有 JSON）

### 5.1 drift-tiered 模式（`recap_fslr`，旧配置 interval=10 / λ=0.5 / 比例 (0,0.02,0.05,0.10)，seed 0/1/2）

**Domain-wise（3 seeds）**：60.99% / 61.04% / 60.98%。所有域 `scheduled_restore_events=15`、`restore_events=0`、`skipped_low_drift_events=15`，`budget_event_counts={"0":15,...}`；域末归一化漂移 ≈ 0.064–0.072（< 0.10 阈值）。→ **domain-wise 下 FSLR 完全不执行恢复**，只做漂移监控。

**Continual（seed0，15 域 × 3 cycles）**：

| Cycle | 平均精度 | restore_events | scheduled | restored_assignments | 新增唯一坐标 | 域末漂移 total |
|---|---|---|---|---|---|---|
| 1 | 62.47% | 199 | 235 | 17,910 | 2,062 | 0.162 |
| 2 | 62.46% | 236 | 236 | 21,240 | 516 | 0.187 |
| 3 | 62.41% | 235 | 235 | 21,150 | 222 | 0.199 |

- Δ₃₋₁ = −0.05pp（几乎平稳）；budget 全部落在 0.02 档（drift 始终处于 [0.10, 0.20)）。
- 触发过程：第 1 域 gaussian_noise 15 次检查全部跳过（漂移 < 0.10）→ shot_noise 触发 1 次 → impulse_noise 10 次 → 之后每个检查（15–16 次/域）都触发。
- 每次恢复 90 个元素 = ⌈0.02 × 4480⌉（4480 = 30 个 BN 张量的可训练标量数，bn_affine_excluding_layer4 旧范围）。
- cycle 1 新增唯一坐标 2062 个 → 排序随漂移移动（Fisher×drift 动态选择），cycle 2/3 新增急剧减少（516/222）→ 逐渐收敛到同一批"重要且漂移大"的坐标（repeat restore）。

### 5.2 fixed 1% 模式（`recap_fixed_fslr`，legacy 消融，seed0）

- Continual 均值 61.88%，Δ₃₋₁ = −1.24pp（持续下滑），流末漂移 0.340（**高于** drift-tiered 的 0.199）。
- cycle 1：`restore_events=235/235`（无漂移门槛，每检查必恢复），每次恢复 45 个元素（⌈0.01×4480⌉），`restored_unique_parameters=45`、`repeat_restore_mean=235` → **固定 1% 反复恢复同一批全局最高 Fisher 坐标，不看当前漂移**。
- 对比结论（单 run、旧配置、诊断性质）：drift-tiered 预算比 fixed 1% 产生更低漂移、更稳的长期曲线 —— 支持"漂移感知预算"设计动机，但未达多重复统计强度。

### 5.3 DLRS-only vs DLRS+FSLR（`outputs/feddr_run1/`，旧配置 dlrs_adaptive 系列，seed0）

| 方法 | Domain-wise | Continual | Clean retention | 遗忘（initial−final） |
|---|---|---|---|---|
| dlrs（无 FSLR） | 62.29% | 62.21% | 71.44% | 5.53pp |
| dlrs_fslr（FedDR） | 62.15% | 62.82% | 75.40% | 1.57pp |

→ FSLR 对 **clean retention** 的改善显著（遗忘 5.53pp → 1.57pp），符合"防止残留漂移损害源域能力"的设计。

### 5.4 旧配置 FedDR 完整 run（`tmp/tta_remote/outputs/`，interval=10/λ=0.5 版本）

- `feddr_run1.json`：domain 60.01%、continual 62.29%、clean 75.96%、revisit 57.45%、drift 0.166。
- `feddr_repeat_seed52.json`：domain 60.08%、continual 62.11%、clean 75.15%、revisit 56.66%、drift 0.172。
- 注意：这两个 JSON 的 `restoration` 块仍记录 interval=10、strength=0.5、比例含 0.10（旧配置），与当前代码的最终固定配置不同。

## 6. 配置版本差异（重要，写论文前必须统一）

| 版本 | interval | strength | 恢复比例 | 可靠性门控 | 出处 |
|---|---|---|---|---|---|
| 论文初稿 §3.3 / §4.3（旧） | 10 | 0.5 | (0, 0.02, 0.05, 0.10) | 无（recap 路径） | `论文初稿.md` L156-185、L263；`test/outputs/`、`tmp/tta_remote/` 全部 JSON（schema 8） |
| 框架文档第 0 节 / 当前代码（最终固定） | 5 | 0.9 | (0, 0.02, 0.05, 0.30) | 阈值 0.55 + EMA 0.90，当前批可靠度门控并缩放预算 | `FEDEATA_论文框架_待补主实验.md` L16；`main_test.py` L108-121、L528-539；`feddr_main.sh` L58-64 |

要点：

1. **`论文初稿.md` 的 FSLR 配置与最终固定配置不一致**（interval 10→5、λ 0.5→0.9、0.10→0.30 档），且旧 JSON 全部按旧配置生成。论文 3.3 与实现细节两处必须统一到框架文档版本（schema 9）。
2. **框架文档声称的 run1 结果（domain 60.80% / continual 62.11% / clean 75.51% / revisit 56.19% / drift 0.3153）在本地没有任何 JSON 支撑**——本地所有 FedDR JSON 均为旧配置（interval=10/λ=0.5，drift 0.16–0.20，clean 75.15–75.96%）。该数字只能来自远端按新配置运行的结果或待补记录，引用前必须补上对应 JSON 与配置自检。
3. `feddr_repeat_seed52` 的 continual 62.11% 与框架文档数字巧合一致，但 domain/clean/drift 均不同，不能混用。

## 7. 已知问题与实验注意事项

1. **梯度裁剪 confound**：legacy 策略只对 `*_fslr` 方法裁剪（κ=15.0），非 `_fslr` 行不裁剪。任何 "base vs base+FSLR" / "DLRS vs DLRS+FSLR" 的组件归因都必须用 `--grad_clip_norm` 统一（论文初稿 4.5.1 已指出）。旧 JSON 之间的 FSLR 与非 FSLR 对比不能直接引用。
2. **reliability 门控只在 FedDR/DLRS 路径生效**：`DLRSAdapter` 传 `reliability=recovery_credit`；`DualReliabilityAdapter`（recap_fslr 等 legacy 行）不传 → 无门控。`sparse_restore.py` 顶部 docstring 写"paper controller calls step() without it"与当前 `main_test.py` 行为矛盾，属过期注释，应更新。
3. **reliability 缩放预算会产生新 ratio key**：`selected_ratio = base_ratio × reliability`（如 0.02×0.9=0.018）不在初始 `budget_event_counts` 键集合 {0, 0.02, 0.05, 0.30} 中，会动态新增键，统计口径需注意。
4. **正分数过滤**：drift-tiered 排序只选 `s_j > 0` 的坐标（drift=0 或 Fisher=0 的元素永不入选）；这与 fixed 模式的"全局最高 Fisher 前缀"语义不同，论文需明确。
5. **`_count_for_ratio` 的 ceil 行为**：ρ=0.02、|A|=4480 → ⌈89.6⌉=90 个元素/事件；论文若写"2%"需说明取整规则。
6. **fixed 1% 模式**只保留为 legacy 消融（全局最高 Fisher 1%），不作为主方法定义。

## 8. 对论文写作的建议

- 将 §3.3 与 §4.3 的 FSLR 配置统一为最终固定版（interval 5、λ 0.9、比例含 0.30、可靠度门控 0.55）。
- 用最终配置重新生成全部主表/消融 JSON（schema 9），确保每个数字可追溯到具体文件；框架文档的 run1 数字补上出处或改为占位。
- 机制证据建议按 §5 组织：domain-wise（FSLR 不触发，仅监控）vs continual（触发与预算分布）分开报告；配 Fig. 4 的 drift 轨迹 + restore event 标记。
- 报告 `D_t`、scheduled vs executed 事件、selected 坐标数/比例、`selected drift-energy ratio`、`selection Jaccard`、`last_drift_reduction` 作为机制诊断（字段均已存在于 `restorer.stats`）。
- 组件消融（base / DLRS / FSLR / DLRS+FSLR）四行必须共享同一裁剪策略与同一 FSLR 配置；旧配置下的跨行对比只能标注为诊断结果。
