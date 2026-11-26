# FedRAS 配置与评估协议：本地最新已完成工件核验版

编写日期：2026-10-28（Asia/Shanghai）。本文件是结果协议清单，不是新的实验计划，也不是官方基线复现声明。

## 1. 取证规则与适用范围

以原始结果 JSON 的 protocol、adaptation、restoration、config、revision 及机制审计为依据；报告日期、目录名或文件复制时间不能单独确定实验身份。CIFAR-100-C 主结果采用 final_revision_20261022/simplified_corrected/run1..3；Tiny-ImageNet-C 采用后续样本顺序修复后的 tiny_cache_replay_orderfix_retry1 工件。旧配置、匹配重放、reasonable 基线和成本实验分别列出，不拼成同一默认协议。

全部实验使用已存储的联邦训练 checkpoint，在单客户端执行测试时适应。本轮未进行联邦训练，未验证测试时多客户端通信、聚合或异构客户端协作。路径中的 100clients 表示上游训练配置；primary_comparison_roster 等字段仅为名单，不能证明对应方法已经执行。

## 2. CIFAR-100-C 默认配置的评估协议

| 项目 | 实际记录 |
|---|---|
| 数据集 / 主干 | CIFAR-100-C / ResNet-18 |
| 初始化 | 三个存储的 FedACG checkpoint；anchor_key=lookahead_anchor |
| severity / batch size / eval seed | 5 / 128 / 0 |
| 预测顺序 | pre_update：先预测计分，再对该批次适应 |
| Domain-wise | 每个 corruption 重置模型、优化器与 restorer；15 个 block，每个 10,000 样本，共 150,000 次样本呈现 |
| Continual | 同一模型及优化器状态跨域持续；15 corruption × 3 cycle=45 block，共 450,000 次样本呈现 |
| 批次数 | 每个 block 79 批，continual 共 3,555 批；最后一批不足 128，未丢弃 |
| 截断 / workers | max_batches=-1；num_workers=2 |
| Clean | untouched anchor 的 initial_clean 单列；最终模型冻结后评估 10,000 clean 样本 |
| Revisit | 冻结最终模型评估 gaussian_noise、shot_noise、impulse_noise、defocus_blur，每域 10,000 样本；对四域准确率取均值 |
| 冻结检查 | 三个结果均记录 parameters_unchanged、adapter_counters_unchanged、restorer_counters_unchanged=true |
| 重复单元 | stored_checkpoint，n=3；training_seed_provenance=unknown。eval seed=0 不是上游训练种子 |

Canonical corruption 顺序为：

`gaussian_noise → shot_noise → impulse_noise → defocus_blur → glass_blur → motion_blur → zoom_blur → snow → frost → fog → brightness → contrast → elastic_transform → pixelate → jpeg_compression`

CIFAR 主结果 JSON 未单独保存 within_domain_order 或样本清单哈希，因此本清单不将 Tiny 的样本排列规则直接外推为 CIFAR 的已核验规则。

## 3. CIFAR 默认适应与恢复配置

| 项目 / 字段 | 值与含义 |
|---|---|
| JSON 方法名 / 入口 | method=FedDR；tta_method=dlrs_fslr；本文称 FedRAS |
| 更新参数 | 结果计数为 4,480 个 BN affine 标量；stem 与 residual stages 1–3，layer4 不更新 |
| 优化器 | SGD；lr=0.001；momentum=0.9；weight_decay=0 |
| RGA 变体 | dlrs_ablation_variant=anchor_only |
| 权重 | dlrs_anchor_rectify_beta=0.3；w_i=1−0.3·1[current prediction ≠ anchor prediction] |
| 样本参与 | sample_filter=soft_reliability_all_samples；dlrs_hybrid_threshold=0.0；dlrs_weight_mode=sample_only |
| 其他权重字段 | dlrs_sample_reliability_mode=entropy；dlrs_cue_ablation=full；不能因字段名 entropy 就把 anchor_only 分支称为纯熵加权 |
| 权重归一化 | dlrs_normalize_by_weight=false；匹配源码中加权项取 mean，统一缩放权重不会被除以权重总和抵消 |
| 视图一致性 | dlrs_view_consistency=false；dual/fast perturbation=false；early_exit=false |
| 梯度裁剪 | fslr_grad_clip_norm=15.0；是否真的裁剪须看逐运行计数 |
| Controller | revision.controller=disagreement；审计中三条 continual 轨迹所有批次 controller_ess=1.0 |
| 风险系数 | gamma_anchor=0.5，gamma_drift=0，gamma_ess=0.5；ESS 输入被 controller 固定为 1，不能将残留 gamma_ess 字段解释为实际启用 ESS 耦合 |
| 有效预算模式 | restoration.budget_mode=continuous_risk；continuous_risk=true；config.restore_budget_mode 残留 drift_tiered 不应覆盖有效模式 |
| 恢复间隔 | recovery_interval=10；每条 continual 轨迹记录 355 次恢复事件 |
| 风险预算 | rho_min=0，rho_max=0.10，tau_drift=0.35 |
| 层配额 | hierarchical_budget=true；layer_quota_cap=0.35 |
| 分配 | revision.ranking=drift_only；restoration.selection_criterion=drift_only |
| 恢复强度 | restore_strength=0.7 是基准系数；continuous_risk 分支使用 lambda_t=0.7·(0.5+0.5·risk_score)，不能写成每次固定 lambda=0.7 |
| 事件触发 | config.fslr_event_trigger=true，但三条结果 num_event_triggers=0；实际事件为周期恢复 |
| 门控 / 在线 Fisher | disable_fslr_reliability_gate=true；fslr_online_fisher_beta=0 |
| Momentum 字段 | fslr_momentum_discounting=false；不据此推断所有恢复后的优化器状态操作均关闭 |
| Replay | schedule_record=null，schedule_replay=null：完整自由运行 |

分配语义由与结果哈希相同的 sparse_restore.py 核验：drift_only 的跨 tensor 质量由各 tensor 归一化绝对位移求和得到，层内按绝对位移排序；这两处在线选择均不使用 Fisher。源协方差仍可参与区域熵计算，因此“去掉 Fisher 分配”不等于“不需要任何离线源统计”。

JSON 没有单独序列化论文符号 eta_0、s_0，故本清单不把论文文字值冒充结果字段；完整 config 保存在伴随证据 JSON。region_scale_min/max 是入口字段，anchor_only 的固定区域尺度分支不能只根据这两个字段反推。

### 默认轨迹及结果身份

| checkpoint | DW | C1 | C2 | C3 | Clean | Revisit | 恢复数量 | config_hash |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| run1 | 62.7667 | 63.9947 | 64.3140 | 64.2860 | 74.5000 | 59.2300 | 13,363 | `5f710d967fbb19c1` |
| run2 | 62.5627 | 63.4007 | 63.8047 | 63.8180 | 73.7700 | 57.9250 | 14,316 | `8503bb3fa86ccf46` |
| run3 | 62.9140 | 63.8367 | 64.1687 | 64.1840 | 74.8600 | 58.5375 | 13,741 | `e7c880d63889cc7e` |
| mean ± sample SD | 62.7478 ± 0.1764 | 63.7440 ± 0.3077 | 64.0958 ± 0.2624 | 64.0960 ± 0.2461 | 74.3767 ± 0.5554 | 58.5642 ± 0.6529 | — | — |

所有准确率单位为 %；差值单位为百分点。主表 C1=63.744、C2=64.096、C3=64.096 可由此重算。三个 checkpoint 的模型/统计路径逐条见伴随 evidence JSON；run1 的目录为 cifar100_resnet18_run1，run2/run3 为 run2/run3，不能统一简写成 run{1,2,3} 后声称精确路径一致。

- [result.json](C:/Users/33502/Desktop/FEDRAS_Overleaf/output/final_revision_20261022/simplified_corrected/run1/result.json)
- [result.json](C:/Users/33502/Desktop/FEDRAS_Overleaf/output/final_revision_20261022/simplified_corrected/run2/result.json)
- [result.json](C:/Users/33502/Desktop/FEDRAS_Overleaf/output/final_revision_20261022/simplified_corrected/run3/result.json)

## 4. 旧配置 factorial、自由分配比较与匹配 replay

| 工件组 | Controller | 分配 / 约束 | 可回答的问题 |
|---|---|---|---|
| full/factorial/L0-C0 | disagreement | uniform loss；Fisher 分配 | 与同 controller 的 L1 对比权重配置效应 |
| full/factorial/L0-C1 | rga_ess | uniform loss；Fisher 分配 | 与 C0 对比 ESS controller 配置效应 |
| full/factorial/L1-C0 | disagreement | RGA loss；Fisher 分配 | Table XI 的 Fisher 参照 |
| full/factorial/L1-C1 | rga_ess | RGA loss；Fisher 分配 | 旧 Full 配置；与最初 P1 Full 为不同执行 |
| simplified_corrected | disagreement | RGA loss；纯位移分配；自由预算 | 默认完整系统效果 |
| full/matched | rga_ess / 固定参考计划 | 每步事件、lambda、数量、tensor 配额与候选资格固定 | 旧计划下的层内排序效应 |

Table XI 的 Fisher 与默认位移恢复数量均值分别为 16,529.333 和 13,806.667；因此该表是自由运行分配策略比较，不是 matched-dose 的整体分配机制隔离。旧 matched replay 也不是最终默认 controller 下重新生成的无 Fisher 全分配对照。

| replay checkpoint | 两臂事件数 | 两臂恢复数量 | 两臂 dose | 逐事件一致 |
|---|---:|---:|---:|---|
| run1 | 355 | 17,093 | 6689.965121 | 是；同 schedule_hash，均 consumed_all |
| run2 | 355 | 18,124 | 7140.588575 | 是；同 schedule_hash，均 consumed_all |
| run3 | 355 | 17,165 | 6733.486655 | 是；同 schedule_hash，均 consumed_all |

## 5. 最新可核验 baseline 包：reasonable，不是统一学习率包

以下为 full/baseline/run1..3 的完整原始结果；revision.baseline_track=reasonable，prediction=pre_update，batch=128，severity=5，seed=0，45 个 continual block。它们只执行 continual，不能据此填写 DW。该包不同于主表中缺少完整原始来源的历史转录基线。

| 方法 | 实际优化器 | lr | 参数范围 | 样本规则 | C3 均值 | Clean 均值 |
|---|---|---:|---|---|---:|---:|
| EATA | sgd | 0.00025 | bn_affine_all | eata | 61.2562 | 73.8600 |
| SAR | sam_sgd | 0.00025 | norm_affine_excluding_layer4 | margin | 63.0396 | 74.7100 |
| ASR | sgd | 0.00025 | bn_affine_all | asr_entropy | 62.5311 | 74.2467 |
| ReCAP | sgd | 0.00025 | norm_affine_excluding_layer4 | margin | 62.8393 | 73.7833 |

四方法记录 momentum=0.9、weight_decay=0。SAR 的 config.optimizer=sgd 是入口底层字段，adaptation.optimizer=sam_sgd 才描述实际包装后的优化器。EATA 记录 e_margin_factor=0.4、d_margin=0.05、fisher_alpha=2000；SAR 记录 margin_factor=0.4、reset_threshold=0.2、sam_rho=0.05；ReCAP 记录 confidence_gate=none、margin_threshold=0.15、soft_weight_temperature_factor=0.1；ASR 记录 asr_anchor_fisher_alpha=0、asr_anchor_l2_alpha=0。这些字段的存在不证明所有通用配置分支均实际参与执行。

默认 FedRAS 的 lr=0.001，而此包 lr=0.00025，参数范围也因方法而异。因此可以报告该 reasonable 包的结果，但不得将它描述为与默认 FedRAS 严格统一超参数的实验，也不得未经来源对齐直接替换主表历史数字。

## 6. Tiny-ImageNet-C：采用修复后的样本顺序，另列配置差异

采用 tiny_cache_replay_orderfix_retry1 中 scheme_a/b_rn18/rn34 的原始 JSON。每域先建立排序后的 class/file 清单，再应用固定 numpy.RandomState(0) permutation；15 corruption、severity=5、batch=128、SGD lr=0.001、momentum=0.9、seed=0，每域 10,000 样本，15 DW + 45 continual block，并执行冻结最终模型的 Clean/Revisit。

| 配置项 | Scheme A（历史 primary 复跑） | Scheme B（简化控制配置） | CIFAR 默认 |
|---|---|---|---|
| ablation_variant | full | anchor_only | anchor_only |
| sample_reliability_mode | full | full | entropy |
| weight_mode | product | sample_only | sample_only |
| hybrid_threshold | 0.2 | 0.0 | 0.0 |
| normalize_by_weight | true | true | false |
| view_consistency | true | false | false |
| event_trigger 字段 | true | false | true（实际触发0次） |
| selection_criterion | fisher_drift | drift_only | drift_only |
| controller | 无 revision 包装记录，不推断等于 disagreement | disagreement | disagreement |

| 结果 | DW | C1 | C2 | C3 | Clean | Revisit |
|---|---:|---:|---:|---:|---:|---:|
| Scheme A RN18 | 34.8967 | 36.1180 | 36.5427 | 36.4687 | 58.7500 | 30.3825 |
| Scheme A RN34 | 37.0067 | 37.6687 | 37.7227 | 37.7093 | 59.9000 | 32.0425 |
| Scheme B RN18 | 35.7520 | 35.2753 | 35.1080 | 34.8367 | 57.4700 | 30.4300 |
| Scheme B RN34 | 37.5900 | 37.1713 | 37.0033 | 36.8493 | 58.8600 | 31.9500 |

Scheme B 的两个原始结果记录 sparse_restore.py SHA256=22475daeb234f675787b8fe93611357ae1cd94b53325f204b748f4649e7f90b9；本地同哈希源码的 drift_only 跨层质量及层内排序均走位移分支。因此旧 replay_report 中“Scheme B 仍用 Fisher×drift 层配额”的说法与该哈希源码不一致，本清单优先采用结果哈希及匹配源码，不沿用该旧说明。

即使 Scheme B 使用纯位移分配，它的权重归一化和 sample_reliability_mode 仍与 CIFAR 默认不同，因此不能称所有数据集严格使用相同配置。未修复目录 final_revision_20261022/tiny_rn18_default 与 tiny_rn34_default 的低准确率结果保留为历史尝试，不与修复后轨迹混合汇总。

## 7. 顺序、初始化与敏感性扩展

multiclient/order1..3 实际为同一 run1 anchor 的 corruption-order seed 1、2、3；不是多客户端，也不是同顺序噪声重复。init_fedavg_default、init_fedprox_default、init_fedacg_default 是不同联邦初始化的固定 checkpoint 结果，不构成新的训练种子重复；具体 anchor_key 与路径以各 JSON 为准。r34_default 是 CIFAR-100-C ResNet-34 的单 checkpoint 扩展，不与 RN18 三 checkpoint pooled。

β 扫描采用 sweep_beta_guardfix/beta0.1、0.2、0.3、0.5；cap 扫描采用 sweep_cap/cap0.20、0.35、0.50。均为 run1，continual only，固定 disagreement controller 与 drift_only 分配。β 增大同时改变损失权重和自由运行恢复轨迹，不能解释为同 dose 的可靠性排序对照。

## 8. 成本及重复执行协议

成本包位于 focused_revision_20261019/cost_measurement。一个 run1 checkpoint；每方法一次舍弃 warm-up 后三次正式进程重复；45 continual block，batch=128。方法为 full、drift_only、asr、recap、tent。stream wall 排除后续 Clean/Revisit；adaptation 时间在 adapter 更新前后 CUDA 同步；显存为 peak allocated MiB。schedule replay、trajectory audit、review diagnostics 不计入该成本测量。

该包 full/drift_only 是单独运行的历史配置，不是默认 simplified_corrected 三 checkpoint 的成本重复，也不是 matched replay 两臂。三次固定 seed 执行的 C3/Clean/Revisit 各自相同；这证明该配置包在这些端点上的重复一致性，不证明不同 seed 或其他配置也没有波动。

## 9. 统计汇总与比较边界

对主结果按 checkpoint 先求端点，再报告三 checkpoint 均值及样本标准差（ddof=1）。DW 是15域准确率均值；Cycle 是该轮15域准确率均值；各域等样本数使域均值与对应 micro accuracy 一致。Revisit 为四个冻结评估域的算术均值。配对差必须使用同一 checkpoint、同一评估协议和明确配置对。45 block、3,555 batch 和450,000次样本呈现不能当作独立训练重复。

上游训练种子未知，n=3 仅为存储 checkpoint 的重复。历史 t/CI 如保留，应标为探索性、限定到所列 checkpoint，并避免把未检出差异写成等效。共享协议、reasonable 包、旧主表转录、batch64 completion、batch128 packet、Tiny order-fix 必须分别标记。

## 10. 排除项及未完成证据

simplified/run1..3 初次运行实际为 fisher_drift，应排除于纯位移默认均值；不能因目录名 simplified 就视作默认结果。sweep_beta 的失败尝试不替代 guardfix 四点结果。batch64 的 CoTTA/PeTTA completion 不填 batch128 行。CoTTA/PeTTA 的 batch128 补齐文件可单独使用，但不是 EATA+ASR 组合基线，也不是 Tiny batch128 全套基线。

本清单没有发现能替代 A1 权重匹配随机对照、A2 替代信号同权重对照或 A3 EATA+ASR 的结果证据；未将名单字段或参数默认值算作已完成实验。多客户端异构、独立训练 seeds、batch sweep、severity sweep、源协方差及 anchor质量消融不由本协议清单补证。

## 11. 可直接用于实验设置段落的文本

我们在存储的 FedACG ResNet-18 checkpoint 上执行单客户端测试时适应，不进行额外联邦训练或测试时通信。CIFAR-100-C 主实验使用 severity 5、batch size 128 和固定评估 seed 0，并采用先预测后更新。Domain-wise 评估对15个腐蚀域分别重置模型、优化器和恢复状态；continual 评估依固定域顺序连续运行三轮，共45个 block 和450,000次样本呈现。默认配置只更新4,480个 BatchNorm affine 参数，使用 SGD，学习率0.001、momentum 0.9、weight decay 0。RGA 使用 beta_a=0.3 的 anchor disagreement 权重，保留全部样本，不按权重总和归一化；恢复控制器使用 disagreement-only 输入，每10批调度恢复，采用分层纯位移分配、0.35层配额上限和0.10最大风险预算。恢复强度由风险调制，0.7为基准系数。适应结束后冻结模型进行 Clean 和四域 Revisit 评估。主结果为三个存储 checkpoint 的均值及样本标准差；上游训练种子不可得，因此这些重复用于描述 checkpoint 敏感性。Tiny-ImageNet-C 与 reasonable 基线的具体配置差异另表报告，不宣称全部实验采用统一超参数。

## 12. 伴随证据

同目录 FedRAS_protocol_evidence_20261028.json 保存逐工件绝对路径、当前文件 SHA256、完整协议与配置，以及结果记录的源代码哈希。结果哈希是本次读取时计算；源代码哈希是原始结果保存的记录，两者不得混淆。工件文件未重新执行，原始 checkpoint/统计资产的服务器路径也未在本次远程重新校验。
