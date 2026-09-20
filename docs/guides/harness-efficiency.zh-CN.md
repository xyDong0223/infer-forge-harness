# Harness 执行效率评估

第一版评估时间与重复工作。连接 Scheduler 的 Graph 在每次执行调用结束后自动
生成评估，包括交付完成、失败、等待 Worker 和 `--until-node` 分段退出。
等待与分段退出只生成过程快照；只有当前 Graph 的 `DELIVERY_RECORDED` 才标为
`completed`。模拟运行保留 `simulation` 标识，不证明真实硬件效率。

任务契约为 [hev-001](../../tasks/hev-001-harness-efficiency/task.yaml)，报告契约为
[harness_efficiency.schema.yaml](../../contracts/harness_efficiency.schema.yaml)。
生产入口的完成、拒绝与进程重启场景纳入必跑的本地 E2E。

## 使用

原有 Graph 命令无需增加参数。末尾带 `assessment: harness_efficiency` 的
`EFFICIENCY_RECORDED` JSON 消息返回
`report_path`、`markdown_path`、`manifest_path` 和 `artifact_root`。
每份报告使用本 run 内一个全新的 `harness-efficiency` attempt，不覆盖历史报告。
其中保存输入快照、JSON 报告、Markdown 报告、独立算术校验结果以及文件哈希清单。

也可以在任意时刻对已有 run 手动生成快照，例如 Worker 提交结果之后：

```bash
python3 cli/validation/harness_efficiency.py \
  --state /external/state.sqlite --run-id my-adaptation
```

只读取 Scheduler，不创建新 run、不领取任务、不恢复租约、不改变功能验收结果。
报告中的输入快照有哈希绑定，阶段与审视线索引用具体事件 ID 或 Task Memory block ID。
重算同一批输入的时间指标不会随报告生成时刻增长。

当前自动入口只覆盖连接 Scheduler 的 Graph。单独运行 deployment/performance CLI、
Graph 规划、未连接 Scheduler 的旧模式，以及被 SIGKILL 强制终止的进程不触发收尾；
可在恢复之后使用上述命令评估已持久化的记录。

## 指标定义

| 指标 | 定义与边界 |
| --- | --- |
| `elapsed_seconds` | run 创建至最后一条已持久化事件；包含调用之间的等待，不包含报告生成之后的空闲 |
| `recorded_wall_seconds` | 已完成 Graph block 与已闭合 Worker claim-to-result 区间的并集；重叠只计一次 |
| `unobserved_wall_seconds` | 总耗时减去上述并集；不能认定为空闲、低效或浪费 |
| 阶段 `closed_interval_seconds` | 同一阶段已闭合区间的时长总和；含等待和工具开销，阶段之间可能重叠 |
| `worker_retries` | 同一任务第一遍领取之后的领取次数，包括租约恢复；不是再次执行的浪费判决 |
| `failed_worker_attempts` | Scheduler 记录的失败次数，包括证据验证拒绝 |
| `unclosed_worker_attempts` | 有领取但没有相应完成/失败记录的尝试；时长为 null，不用下次领取时间猜测结束 |
| `diagnosis_attempts` | diagnosis 阶段实际领取次数；未领取的诊断不计为已经执行 |
| `repeated_graph_executions` | 同一节点超过第一次的执行次数；必要服务/精度回归也可能出现在这里 |
| `graph_reuses` | 明确通过 Journal 复用的 block；与重复执行分开统计 |
| `graph_bookkeeping_blocks` | Scheduler 交接、交付和 recovery 回执等记账 block，不计为节点再次执行 |
| `known_queue_wait_seconds` | 有排队起点证据的等待时长之和；并行任务的等待可能重叠，不能作为总等待时间 |
| `unknown_queue_wait_count` | 缺少排队起点记录的领取次数，例如部分旧租约恢复路径 |

缺失的 Task Memory 或未闭合尝试会在报告里提示。旧记录若有不合法时间戳或
缺少领取的终止事件，将保留输入与 REWORK 校验结果，不用估计值填补。
Token、模型调用次数、费用、设备利用率保留为 null。
自动恢复的新执行会逐次记录 `recovery_execution` block（包括失败的重跑），
计入时间与重复工作；历史 run 若只有恢复回执，无法补算其重跑次数与耗时。
恢复决策等待、修复 action 和尚未闭合的执行仍可能落在未归因时间内。

## 评估结论与后续迭代

报告给出耗时最多的已记录阶段和重复执行的证据线索，暂不设置总分、效率阈值，
也不自动修改 harness。必要验证不能因为次数多就被认定为浪费。
这份报告可以作为后续流程审视与同类任务基线比较的输入。

`RECORDED` 只代表效率报告已生成并经过算术校验。功能交付仍由原有 Scheduler、
validators 与 delivery receipt 判定。自动评估失败输出
`EFFICIENCY_EVALUATION_FAILED`，保留原工作流退出码及功能结论；手动评估失败退出 2。
CI 应检查评估消息和报告，不能仅凭 Graph 退出 0 推断效率评估已完成。
