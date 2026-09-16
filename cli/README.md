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

## Worker 的领取与提交

`STATE` 必须指向已有外部数据库。claim 按阶段领取，当前没有 `--run-id` 过滤参数；worker 必须能处理该数据库对应阶段的任务，并核对返回的 run 和环境身份。

```bash
STATE="/absolute/path/to/state.sqlite"
python cli/adaptation.py --state "$STATE" claim \
  --worker torch-agent-01 --stage torch --limit 1 --lease-seconds 300
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

连接 Scheduler 的 Graph 返回 exit 3 表示等待 worker，exit 2 表示阻塞或返工。省略 `--scheduler-state` 的 graph-only 模式不生成 scheduler-backed 功能交付凭据；`--until-node` 的部分执行和供应商交接的 `DELIVERED` 都不是模型功能就绪。

Graph 和 deployment proof 不带 `--execute` 时为计划模式，其他独立入口可能直接访问集群。更详细的错误定位与不同恢复机制见[排障指南](../docs/guides/troubleshooting.zh-CN.md)。新增命令的源码归属见[贡献指南](../CONTRIBUTING.md)。
