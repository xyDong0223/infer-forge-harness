# CLI：选择入口与接入 worker

命令从仓库根目录、激活的虚拟环境运行。首次使用见[快速上手](../docs/guides/quickstart.zh-CN.md)；真实执行先按[根 README](../README.md#运行真实模型适配)创建 run 并绑定环境。

## 选择入口

| 目标 | 入口 |
| --- | --- |
| 管理 run、环境、算子队列与租约 | [adaptation.py](adaptation.py) |
| 执行、计划或恢复完整任务图 | [workflow/graph.py](workflow/graph.py) |
| 登记模型、扫描能力和缺口 | [intake/](intake/)、[discovery/](discovery/) |
| 环境证明、部署计划和轻量启动 | [deployment/](deployment/) |
| 诊断、补丁放置和候选交接 | [operators/](operators/) |
| 数值、服务正确性、API 与支持矩阵 | [validation/](validation/) |
| 查询 Journal 和 Task Memory | [state/](state/) |
| 检查源码引用与目录归属 | [maintenance/check_repo_references.py](maintenance/check_repo_references.py) |

用各入口的 `--help` 查看实际参数。`scheduler.py` 是较低层接口，新集成优先使用 `adaptation.py`。文件级算子交接请求不等于 Scheduler 中的任务已经完成。

`deployment/proof.py` 默认 `--phase environment`：只证明 MiniMax-M2.5 基线，跳过 toy，返回 `ENVIRONMENT_READY`。即使传入目标模型示例，也不会启动目标服务。目标适配完成前置检查后，由 Graph 的 `--phase service --attach-pod ...` 执行目标 toy 与服务证明。`--phase all` 保留显式的旧版单次部署入口，不能代替模型适配工作流。契约的 `environment_proof` / `service_proof` 类型仍优先于 `--phase`；MiniMax 示例明确属于环境证明。

## Worker 的领取与提交

`STATE` 必须指向已有外部数据库。Agent 使用 `--run-id` 定向领取，可进一步指定 `--task-id`；选择和过期恢复都受同一 scope 约束。核对返回的 run、stage 和环境身份。

```bash
STATE="/absolute/path/to/state.sqlite"
python cli/adaptation.py --state "$STATE" claim \
  --run-id "$RUN_ID" --worker torch-agent-01 --stage torch --limit 1 --lease-seconds 300
```

从返回任务读取 `task_id`、`lease_token`、`attempt` 和 `input.workspace`，不要自行生成。调查文件放 `scratch/`，正式证据放当前 `output/`；每阶段证据角色、哈希和独立验证要求以 [worker 协议](../docs/migration/worker-results.md)为准。

以下变量来自实际 claim 和当前 attempt 的结果文件。长任务需在租约到期前续租：

```bash
python cli/adaptation.py --state "$STATE" renew-lease \
  --task-id "$TASK_ID" --worker torch-agent-01 \
  --lease-token="$LEASE_TOKEN" --lease-seconds 300
python cli/adaptation.py --state "$STATE" complete \
  --task-id "$TASK_ID" --worker torch-agent-01 \
  --lease-token="$LEASE_TOKEN" --result "$RESULT_JSON"
```

`--lease-token=...` 的等号避免把以 `-` 开头的 token 误当成选项。失败用 `fail` 保存原始错误，不提交空 PASS；证据校验失败也会进入持久化诊断。

显式 `managed-v2` 新 run 另用 `freeze-candidate`、`validate-worker` 产生验证凭据，
`execution-status`、`reconcile-execution` 和 `reconcile-validation` 处理本地执行恢复。
完整参数及尚未支持的真实 Pod/device 边界见 [受管 worker 协议](../docs/migration/managed-worker.zh-CN.md)。

## 诊断后的恢复

Diagnosis worker 领取 `--stage diagnosis`，按协议提交诊断报告：

```bash
python cli/adaptation.py --state "$STATE" resolve-diagnosis \
  --task-id "$DIAGNOSIS_TASK_ID" --worker diagnosis-agent-01 \
  --lease-token="$DIAGNOSIS_LEASE_TOKEN" --result "$DIAGNOSIS_RESULT_JSON"
```

这些变量来自诊断任务自己的 claim，不复用失败源任务的 token。诊断完成只表示结论已保存；Main Agent 阅读结论后，对已验证的 `RETRY` 或 `BLOCKED` 显式执行：

```bash
python cli/adaptation.py --state "$STATE" apply-diagnosis \
  --task-id "$DIAGNOSIS_TASK_ID"
```

`RETRY` 将失败源任务重新入队，下次 claim 分配新 attempt；`BLOCKED` 记录结论已消费，源任务仍保持失败。`DISPATCH_TORCH_FIX`、`DISPATCH_XPU_FIX`、`REDISCOVER_OPERATOR` 不能直接用该命令执行，需要 Main Agent 先处理外部修复或重新发现，不能改写成 `RETRY` 来跳过工作。

## 状态与执行边界

使用 `adaptation.py --state "$STATE" status --run-id "$RUN_ID" --events` 查看实际状态。恢复继续用原数据库、run_id、版本和 artifact root；不要重用旧 attempt 作为输出。

需要直接查看停在哪里、为什么停、由谁处理时：

```bash
python cli/adaptation.py --state "$STATE" status --run-id "$RUN_ID" --format text
```

默认 JSON 保留 `run`、`tasks`、`events`，新增统一的 `progress` 和逐任务
`task_progress`。Graph 的 JSON summary 也带相同结构。每条说明包含
`state`、`location`、`reason_code`、`summary`、`next_action`（责任人、动作、
说明、可选命令参数数组）、`evidence` 和 `observed_at`；协议见
[`progress.schema.yaml`](../contracts/progress.schema.yaml)。这些字段只解释进度，
不参与验收，也不会降低 Validator 的要求。

| 统一状态 | 含义 |
| --- | --- |
| `RUNNING` | 最近观察到节点启动或有效的 worker 租约；需通过日志/心跳确认进程仍存活 |
| `WAITING` | 等待领取、上游阶段、诊断或外部决策 |
| `BLOCKED` | 输入、环境、证据或实现问题阻止继续 |
| `ACTION_REQUIRED` | 需要明确操作，例如重新领取过期任务、应用诊断或打破失败循环 |
| `READY` | 可以执行/恢复下一步；不代表模型功能就绪 |
| `COMPLETED` | Graph 已记录最终功能/模拟交付，或某个任务已完成；查看 location 和原始状态 |

`WORKER_UNCLAIMED` 只表示任务未被领取。当前没有 worker 可用性注册表，不能
据此断言没有配置 worker。提示里的 `claim` 命令带 run/task scope，
执行者仍须核对返回身份。命令只作为建议显示，不会自动执行。

Scheduler 模式下，Graph 将最近一条说明存入 run 的 `metadata.graph_progress`
和事件历史。新进程查询时会结合当前队列重新计算算子等待、租约和诊断状态；
例如 worker 全部完成后，提示恢复 Graph，不沿用过时的等待文案。Graph 运行状态是
带时间戳的最后观察，不能用于断言进程仍活着。graph-only 模式输出说明但不持久化到 Scheduler。

`status` 以只读事务读取已有数据库，不创建数据库、不恢复租约、不应用诊断。
历史交付说明不会重新验证证据文件；实际继续运行/交付仍由原有验收门禁检查。
`details.raw_status` 区分 `FUNCTIONAL_READY` 与 `SIMULATION_PASS`。

连接 Scheduler 的 Graph 返回 exit 3 表示等待 worker，exit 2 表示阻塞或返工。省略 `--scheduler-state` 的 graph-only 模式不生成 scheduler-backed 功能交付凭据；`--until-node` 的部分执行和供应商交接的 `DELIVERED` 都不是模型功能就绪。

Graph 和 deployment proof 不带 `--execute` 时为计划模式，其他独立入口可能直接访问集群。更详细的错误定位与不同恢复机制见[排障指南](../docs/guides/troubleshooting.zh-CN.md)。新增命令的源码归属见[贡献指南](../CONTRIBUTING.md)。
