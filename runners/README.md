# Runners：执行序列与证据编排

`runners/` 负责把多个原子动作按顺序执行，并确保每次执行都有独立 attempt、日志、
正式输出和 Validator 结果。Runner 回答“这次执行按什么顺序进行、失败时留下什么”，
具体平台命令来自 Adapter 或 Runtime。

## 与相邻层的区别

| 层 | 负责内容 |
| --- | --- |
| `cli/` | 参数解析和用户入口 |
| `operations/` | 单个领域任务的业务实现 |
| `runners/` | 跨动作或跨任务的执行顺序、重试、日志和证据收集 |
| `engine/` | 持久化调度、租约、诊断和恢复决策 |
| `validators/` | 独立验收输出是否满足契约 |

## 文件地图

| 文件 | 主要职责 |
| --- | --- |
| `graph_runner.py` | 解析 Workflow、准备节点命令、复用事实、执行失败边、衔接 Scheduler 和最终交付 |
| `task_runner.py` | 执行一个 Task contract，分配 attempt，并把部署类任务交给对应 Runner |
| `deployment_proof.py` | 准备或 attach Pod，检查/安装软件栈，证明环境和服务路径 |
| `correctness_executor.py` | 执行 kernel、端到端精度和长上下文正确性序列 |
| `triage_executor.py` | 复现失败并采集真实调用参数和错误证据 |
| `patch_executor.py` | 放置、检查和记录可重放 patch 的执行过程 |
| `performance_runner.py` | 执行基准、trace 和性能指标比较 |
| `evidence.py` | 运行子进程、流式保存日志、崩溃时创建不可覆盖快照 |
| `watch.py` | 对长时间静默节点记录心跳和最近活动，不把心跳当成功证明 |

## 一次执行的基本形状

```text
CLI 解析参数
  -> Runner 分配 fresh attempt
  -> Operation / Adapter / Runtime 执行动作
  -> logs/ 保存过程和崩溃快照
  -> output/ 保存候选结果
  -> Validator 检查结果
  -> manifest 记录文件和哈希
  -> Graph 或 Scheduler 决定后续状态
```

## 约束

- Runner 不应硬编码厂商设备命令或框架启动参数，应通过 Adapter/Runtime 获取。
- 每次重试必须分配新 attempt，不能覆盖先前失败证据。
- Runner 可以调用 Validator，但不能自行把“命令成功”解释成任务 PASS。
- `graph_runner.py` 是当前最大的组合模块；新增功能优先放到明确的 Operation、
  Engine 服务或小型 Runner 中，不继续堆入无关分支。
- 长任务必须持续保存日志和心跳；租约有效不等于子进程仍然存活。

修改执行顺序后，应运行对应单元/集成测试；新增完整能力还必须更新
`tests/e2e/scenarios/` 中的生产流程场景。
