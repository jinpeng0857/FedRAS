# 四项审稿问题的实验配置

日期：2026-11-02。根据用户提供的解决方案整理，参数核对对象为本机
`D:/dp1/FEDEATA-revision-20261018` 的修订入口与 preset；本目录位于
`D:/dp1/FEDEATA`，没有修改两个工作区的算法实现，也没有启动实验。

## 配置文件

| 文件 | 内容 |
|---|---|
| `00_environment.json` | 三个 checkpoint/源统计相对路径、共同评估协议、CIFAR FedRAS 完整机制参数及需核对的隐式默认值 |
| `01_baseline_lr.json` | FedRAS 默认参考、FedRAS@2.5e-4、四种基线@2.5e-4 的补跑配置，以及有条件的四基线@1e-3 |
| `02_rga.json` | uniform-scaled 三 checkpoint；entropy-only 的待实现定义；判定阈值 |
| `03_dsr.json` | 关闭事件触发的 adaptive 参考；固定均值/固定剂量两臂；逐 checkpoint 校准公式与日志字段 |
| `04_analysis_rules.json` | C3/Clean/Revisit 字段、配对 CI、方向计数、覆盖校验和报告表字段 |
| `05_tiny_and_order_optional.json` | Tiny 坍塌诊断与 3 order × 3 checkpoint × 2 method 的可选设计 |

这些 JSON 是本实验包的配置格式，不是旧 `matrix.py` 可直接接收的配置。
同一实验先合并 `evaluation_cli`，FedRAS 臂再合并 `fedras_cli`，最后由该臂
`cli` 覆盖。布尔 true 表示独立开关，false 表示不发出该开关；null 表示
未解决的校准值，必须阻止执行。

## 第一批建议准备的命令

先核对 `00_environment.json` 中的资产相对路径。`runtime-root` 指包含修订
入口的代码目录，`asset-root` 指包含训练 checkpoint、统计和数据的目录，
两者可以不同。在服务器上使用服务器实际路径；配置不假定 Windows 本机
已经具有完整数据。生成器会检查模型、统计和数据根目录存在；数据完整性、
有效配置与结果覆盖仍需在运行时验收。

```powershell
python D:/dp1/FEDEATA/scripts/revision_20261102/prepare_commands.py `
  --runtime-root D:/dp1/FEDEATA-revision-20261018 `
  --asset-root D:/dp1/FEDEATA `
  --output-root D:/dp1/FEDEATA/output/revision_20261102 `
  --arms fedras_default fedras_lr025 rga_uniform_scaled `
  --output D:/dp1/FEDEATA/output/revision_20261102_commands.json
```

这会准备 9 条 continual 命令，每条含冻结后的 Clean/Revisit。若已有与当前
参考严格对应的 `simplified_corrected` 完整三 checkpoint 结果，完成源文件、
checkpoint、统计及流身份核对后，可省略 `fedras_default`，只准备 6 条。
命令逐条保存在输出 JSON 的 `commands[].argv`，目录与运行 ID 各自独立。
生成器本身不执行命令，不复用已有 case 目录，也不覆盖已有命令清单。

确认资源和预先约定的实验臂后，可在 PowerShell 中逐条执行生成的清单：

```powershell
$plan = Get-Content -Raw D:/dp1/FEDEATA/output/revision_20261102_commands.json | ConvertFrom-Json
foreach ($job in $plan.commands) {
  New-Item -ItemType Directory -Path $job.output_dir -ErrorAction Stop | Out-Null
  Push-Location -LiteralPath $job.cwd
  try {
    $program = $job.argv[0]
    $arguments = @($job.argv | Select-Object -Skip 1)
    $job | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath (Join-Path $job.output_dir 'command.json') -Encoding utf8
    & $program @arguments *> (Join-Path $job.output_dir 'run.log')
    if ($LASTEXITCODE -ne 0) { throw "Experiment failed: $($job.arm)/$($job.checkpoint)" }
  } finally { Pop-Location }
}
```

退出码为零只表示进程完成。结果验收按 `04_analysis_rules.json`，不能将
生成命令、smoke 通过或进程完成写成完整实验通过。默认不启动 GPU 作业。

## 条件实验与需要补齐的内容

1. 四基线已有的 reasonable@2.5e-4 工件优先核对后复用；不可核实时显式选择
   `eata_lr025 sar_lr025 asr_lr025 recap_lr025` 补跑，共 12 条 continual。
   EATA 使用单独的 `_eata_statistics.pt`，不能替换为普通源统计。
2. `*_lr100` 四基线被阻止：当前 `configure_reasonable` 会无条件把 CLI 的
   学习率覆盖成 2.5e-4。必须实现 preset 后的受控 override 并验证结果中的
   实际学习率，再启用这 12 条 DW+continual 命令。这里的 DW 和 continual 是
   两套独立状态评估；12 次进程分别产生 12 条 DW 与 12 条 continual 轨迹。
   同时准备 `fedras_lr100_full` 三条 DW+continual 参考，不拿 continual-only
   结果填 DW。不得因条件分支开启而隐藏不利的低学习率比较。
3. `rga_entropy_only` 被阻止：现有 entropy reliability 选项不等于二值熵掩码。
   配置规定按批 top-k、稳定 tie-break、beta=0.3，并保留 disagreement DSR
   传感器。当前 runtime 的变体白名单也需要扩展。先 run1 的结果只能作为 pilot。
4. DSR 两个固定臂的 rho/lambda 为 null。先完成 `dsr_adaptive_periodic` 三条
   参考，再逐 checkpoint 校准、展开单 checkpoint 配置并解除阻止。不要把
   run1 的均值用于全部 checkpoint；历史 1.1% 不能直接作为匹配结果。
   所有 DSR 隔离臂均关闭 emergency trigger，防止固定预算被紧急分支改变。
   adaptive 参考因此是一个明确的新协议，与原默认单列比较。
5. 现有机制审计可提供事件 ratio/strength/count/dose；kappa 和 c 的恢复前
   位移范数仍需增加日志。缺这些日志不应伪装成已有 contraction 证据。
6. Tiny 和 order 扩展当前不能由命令生成器执行。Tiny 必须绑定资产并使用
   已有 order-fix wrapper；顺序扩展需要显式置换与哈希 hook。

生成器要求用户显式选择臂，但不会自动评估科学上的条件分支是否成立。
例如高学习率补跑是否启动，需要先按低学习率比较规则判断。

## 与原方案相比的明确约定

- 等尺度 lr 使用精确公式值 0.0008995，而不是四舍五入的 0.0009。
  0.335 是原方案的参考分歧率，不是三个 checkpoint 的实测共同值。
  非常数样本权重无法通过一个 lr 复现梯度方向；自由运行的 DSR 轨迹也可能
  改变。因此此臂称预设近似尺度对照，不能宣称总梯度范数或总剂量严格相同。
- entropy-only 采用每批约 33.5% 的 top-k 规则，避免在测试准确率上寻找阈值。
  |差值|<0.1 pp 只作描述性接近，不能据此证明统计等效。
- 同 lr 保留各方法的参数范围、过滤规则和 SAM 差异，不能称全部超参数统一。
  ASR 无 reset 也不足以证明等于 Tent，仍要核对目标函数、记忆与优化器。
- checkpoint 是重复单元；训练 seed 未知。df=2 的配对 t 区间只作探索性、
  所列 checkpoint 条件下的描述。批次和域不能当作独立重复。
- 历史顺序极差用于原方案的报告门槛，不当作标准误或正式噪声分布。
  规则对阈值中间地带和方向不一致情况明确返回未定结论。
- 这些阈值来源于已有历史结果之后提出的解决方案。可以在新实验运行前锁定，
  不将其描述成早于全部历史数据的预注册。保存配置 SHA256、代码版本与时间。
