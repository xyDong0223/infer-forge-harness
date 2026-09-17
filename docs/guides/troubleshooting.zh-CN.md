# 排障与执行边界

先记录 run_id、失败 task_id / attempt、实际命令、代码版本和原始 stderr，再查看状态文件、validator 错误及证据。用 `cli/adaptation.py ... status --run-id ... --events` 查询持久状态；不直接改 SQLite，不覆盖失败 attempt。

先用 `status --run-id ... --format text` 查看统一解释。`Reason` 给出具体原因，
`Next` 指明责任人和动作，`Command` 是可选的下一步命令，`Evidence` 指向原始证据。
JSON 的同一信息位于 `progress`；并行算子各自的下一步位于 `task_progress`。

| reason_code | 下一步 |
| --- | --- |
| `START_GRAPH` | 提供 `model_path`、`user_id`，由 harness 生成契约，用 `--execute` 首次执行当前 run 的 Graph |
| `ENVIRONMENT_COMMAND_FAILED` | 环境证明已被接受，但节点执行失败；查看退出码和日志，修复后按提示不带 `--resume` 重跑环境节点，再恢复完整 Graph |
| `WORKER_UNCLAIMED` / `DIAGNOSIS_PENDING` | 启动对应 stage 的 worker 并领取任务，核对返回 run；不代表已经证实没有 worker |
| `WORKER_RUNNING` | 看当前 attempt 的日志并按需续租；有效租约不是存活证明 |
| `LEASE_EXPIRED` | 重新领取，Scheduler 会分配新 token/attempt；查询本身不会恢复租约 |
| `DIAGNOSIS_NOT_APPLIED` | Main Agent 阅读已验证的结论，再执行提示中的 `apply-diagnosis` |
| `EXTERNAL_REPAIR_REQUIRED` | 按结论修改实现或重新发现契约，不能把它改成 RETRY 绕过修复 |
| `DISPATCH_BLOCKED` / `SHIM_DISPATCH_BLOCKED` | 根据原始错误补充实测字段或 shim 登记，再恢复 Graph |
| `WAITING_FOR_DECISION` | 查看 request/response 路径；未配置 decider 时由外部 Agent 写入回复 |
| `RESUME_GRAPH` | 算子队列已无待执行工作，恢复 Graph 做最终验证 |

自动恢复等待决策时，Graph 会在开始等待之前输出请求文件、回复路径、等待上限和
是否配置 `--decide-command`。默认文件等待仍保持原有行为，不会替用户自动生成决策。
`--until-node` 完成会明确输出 `UNTIL_NODE_REACHED`；这只是部分执行完成。

## 常见现象

| 现象 | 检查与处理 |
| --- | --- |
| 缺少 yaml / pytest，或子进程 import 失败 | 激活虚拟环境并安装 `.[test]`；内部 `python3` 也需使用同一环境 |
| 测试收集时报缺少 Torch | 基础 test extra 不包含 Torch；选择不依赖它的文件或本地 E2E，见[测试指南](../../tests/README.md) |
| runtime path overlaps the source repository | 将数据库、Journal 和产物移到外部运行目录，不把 checkout 当 workspace |
| output already owned / attempt already has a formal result | 保留旧 attempt；重新领取或为独立 CLI 选择新的外部输出位置 |
| target 为 planned / unknown | 对照[兼容性矩阵](../../compatibility/matrix.yaml)；增加配置不等于实现支持，不能改状态绕过门禁 |
| `INPUT_REQUIRED: execution.user_id` | 向使用者获取其 ID，以 `--user-id <使用者ID>` 传给 environment/proof 命令；Graph 用 `--set user_id=<使用者ID>`。也可配置契约 `execution.user_id`，兼容已有 `USER_ID` 环境变量；不要从主机用户名或示例资源名猜测 |
| ownership / prefix 拒绝 | 核对使用者传入的 `--user-id`、契约 `execution.user_id`（或兼容的 `USER_ID`）、资源名和实际加载的[集群配置](../../config/README.md) |
| discovery 要求 environment proof | 先执行或导入完整环境证明；重新证明已有 Pod 时显式使用 `--attach-pod` |
| Graph 为 WAITING_FOR_OPERATORS / exit 3 | 由外部 worker 完成对应任务，再恢复同一个 Graph/run |
| 提交因 lease、hash 或 identity 被拒绝 | 核对当前 claim、证据字节与 output 所有权；旧 token 和旧 attempt 不能用于新任务提交 |
| capacity analyzer not found | 按 memory budget 的 `--analyzer` 或 `AI_INFRA_SKILLS_DIR` 配置外部分析器，不把缺少数据当成容量通过 |

路径规则见[运行时写入策略](../migration/runtime-write-policy.zh-CN.md)，worker 结果字段见[证据协议](../migration/worker-results.md)。

## 不要混淆几种恢复

| 情况 | 对应机制 |
| --- | --- |
| running 任务的租约过期 | Scheduler 恢复过期领取，新的 claim 使用新 token 和 attempt |
| 算子任务显式失败 | diagnosis 完成后，由 Main Agent 显式 `apply-diagnosis` 消费 RETRY / BLOCKED |
| 诊断要求修改实现或重新发现 | Main Agent 处理外部修复或新的实测契约；不是自动重试 |
| 历史状态缺失后续任务边 | `reconcile` 根据现有证据修复缺失边，不是失败任务的 retry 命令 |
| Graph 节点失败 | `--auto-recover` 按节点恢复机制处理，不替代算子队列的诊断协议 |

`resolve-diagnosis` 完成不等于源任务已恢复。操作顺序见 [CLI 总览](../../cli/README.md#诊断后的恢复)，实际边界见 [Scheduler](../../engine/scheduler.py)。

## CPU 参考与精度拒绝

[CPU reference probe](../../tools/probe/cpu_reference_logits_probe.py) 按 checkpoint 的量化配置选择路径：非量化权重不要求 INT8 scale；手动反量化仅支持当前实现识别的 compressed-tensors、int-quantized、对称 per-channel INT8 权重。其他量化格式会拒绝，不能将其解释为 XPU 精度失败，也不能伪造 scale 或 PASS 继续。

该 probe 要求加速器对参考进程不可见。CPU reference 的格式检查通过也不保证特定模型能在 Transformers 中加载和执行；参考本身仍需独立验证。

精度任务将 top-1、top-5 等验收结果写入最终报告。端到端包装还核对报告、`accuracy_status.json` 和子命令退出码的一致性；缺失或矛盾的结果不能提升为 PASS。排障时同时保留报告、`acceptance_errors`、`validator.errors` 和退出码，而不只截取一个状态字符串。源码见 [accuracy differential](../../operations/validation/accuracy_differential.py) 与 [correctness executor](../../runners/correctness_executor.py)。

## 提交问题时附带什么

提供脱敏后的命令、代码 commit、Python/依赖版本、平台目标、run/task/attempt 身份、原始错误及证据路径与哈希，并明确是否为 simulation。硬件问题还需 Pod、镜像 digest、模型/插件 revision 和设备身份。不要附带凭据、模型权重或原始生产流量。
