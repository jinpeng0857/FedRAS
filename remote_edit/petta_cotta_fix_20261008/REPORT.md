# PeTTA / CoTTA 修复与验收记录

日期：2026-10-08。修改位置：服务器 `/mnt/sda/PythonProject/CJP_project/FEDEATA`，Conda `vllm`。

## 修复内容

- PeTTA：散度改为“更新前的滚动类别原型相对固定源域原型的偏移”，先计算散度、再更新原型。
- PeTTA：用 `alpha_0 * exp(-d)` 计算教师更新系数，避免浮点减法抵消；没有人为增加系数下限。
- PeTTA：训练阶段更新学生、教师、锚模型的 RobustBN 统计量；返回预测前的输出，更新后恢复评估模式。
- PeTTA：接通 `petta_regularizer` 和 `petta_loss_func`；L2 按官方实现对每个参数张量求和，再对张量数平均。
- PeTTA：增强使用所选数据集的归一化参数。
- PeTTA / CoTTA：增强保留输入空间尺寸，覆盖 32 和 64 像素场景。
- CoTTA：补全 `test/comparison_all.sh` 对三个 FedACG 检查点的调用。
- 更新统一入口和根目录兼容入口；未修改其他算法的优化器和实验配置。

官方参考：
- https://github.com/hthieu166/petta/blob/main/src/adapter/petta.py
- https://github.com/hthieu166/petta/blob/main/src/utils/petta_memory.py
- https://github.com/qinenergy/cotta/blob/main/cifar/cotta.py

## 验收证据

| 检查层级 | 结果 | 范围 |
| --- | --- | --- |
| 故障复现 | 新增 12 项中修复前有 8 项失败 | 原型散度、BN、参数传递、L2、归一化、尺寸、脚本调用 |
| 回归测试 | 102 / 102 通过 | `python -m unittest discover -s tests -v` |
| 真实检查点核心功能 | 6 / 6 组通过，共 12 个批次 | FedACG run1–run3 × 两种方法，每组 gaussian_noise / fog 各 64 张 severity=5 样本 |
| 真实评测入口 | 两种方法各 15 个 Domain-wise + 45 个 Continual 块通过 | batch_size=8，max_batches=1；PeTTA memory_size/update_frequency=8 |
| 数值与更新 | 两种入口各 60 个批次均更新，nonfinite_batches=0 | 核心检查还验证学生参数变化、教师状态变化、同一输入的预测变化 |
| CoTTA 机制 | 通过 | 默认 32 次增强；测试覆盖低置信度增强平均、EMA、概率为 1 的恢复、固定锚模型 |
| 脚本与补丁 | 通过 | `bash -n test/comparison_all.sh`、`git diff --check` |

核心检查使用 CPU，PeTTA 保留默认 memory_size/update_frequency=64，CoTTA 保留默认 augmentation_count=32。
三个检查点的 PeTTA 首次更新均为 divergence=0、teacher_alpha≈0.001，教师状态和后续预测均发生变化。
这是功能验收，未重新执行完整数据量的论文实验，也不据此声称准确率提升。
高散度下更新系数仍可按算法减小；验收没有通过强制更新下限改变算法。
保留本项目已有的统一记忆更新周期和 FedDR 匹配评测协议，不宣称与官方整套实验协议逐项等同。

## 修改文件

- `test/tta_baselines.py`
- `evaluator/engine.py`
- `engine.py`
- `test/comparison_all.sh`
- `tests/test_petta_cotta_regression.py`（新增）

以上相对路径均以服务器项目目录为根。备份在本目录 `before/`，补丁在 `changes.patch`。
具体结果见 `acceptance_summary.json`、`core_acceptance.json`、`petta_entrypoint_smoke.json`、`cotta_entrypoint_smoke.json`。
历史 `test/outputs/` 结果保留；修复前的 PeTTA 结果不代表修复后实现，正式实验应写入新结果目录重新运行。

## 复测命令

在项目根目录并激活 vllm 后执行：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -v
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python artifacts/petta_cotta_fix_20261008/check_core.py
bash -n test/comparison_all.sh
git diff --check
```

入口 smoke 的完整参数已保存在两份 JSON 的 `config` 字段中，均限制每种损坏为一个批次。
