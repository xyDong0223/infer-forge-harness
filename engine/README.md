# Engine：持久化调度与恢复决策

`engine/` 管理一个适配 run 如何持续推进。它保存任务状态、租约、事件和诊断
结论，并把 Graph 与异步算子任务连接起来。它不直接操作 Pod、安装 Runtime 或
实现算子。

## 主要对象

```text
AdaptationRun
  -> OperatorTask(torch)
  -> OperatorTask(xpu)
  -> OperatorTask(integration)

任一阶段失败
  -> DiagnosticTask
  -> RETRY / BLOCKED / 外部修复或重新发现
```

Scheduler 只根据经过验证的状态转换推进任务。退出码 0、一个 PASS 字符串或
manifest 存在，都不能单独完成任务。

## 文件地图

| 文件 | 主要职责 |
| --- | --- |
| `contracts.py` | `AdaptationRun`、`OperatorSpec`、任务、事件和 `BugReport` 的持久化结构 |
| `scheduler.py` | SQLite `EventStore`、任务状态机、租约、诊断和阶段推进 |
| `discovery.py` | 将实测 gap report 转换成严格 `OperatorSpec`，拒绝缺失的 shape/dtype/layout/semantics |
| `result_validation.py` | 校验 worker 结果、身份、证据角色、哈希和独立验证关系 |
| `graph_bridge.py` | 将 Graph 节点证据与 Scheduler 中的算子任务、最终交付门禁连接起来 |
| `brain.py` | 恢复决策请求/响应协议，以及外部 Agent 和规则决策器 |
| `recovery.py` | 在预算内执行“决策 → 动作 → 重新验证”循环 |
| `progress.py` | 把原始 Graph/Scheduler 状态投影成位置、原因、责任人和下一步说明；不改变状态 |
| `skill_registry.py` | 将 `task_type` 解析到版本化 Skill 方法与工具契约 |
| `state/journal.py` | 追加式事实记录，供 Graph 判断证据来源和复用条件 |
| `state/task_memory.py` | Agent Task Loop 的当前 block、历史执行和下一 block |
| `fake_agents.py` | 本地 E2E 的模拟 worker；产物只能证明 simulation 路径 |

## 状态所有权

- `scheduler.py` 是 run、task、lease 和 event 的唯一状态转换入口。
- `Journal` 保存执行事实及其 provenance，不替代 Scheduler 状态。
- `Task Memory` 保存 Agent 的工作上下文，不替代证据和 Validator。
- `progress.py` 只解释已有状态，不执行建议命令，也不给出验收 verdict。

## 依赖边界

Engine 可以使用 `core/` 和独立 Validator，但不能拼 kubectl 命令、访问具体设备、
选择 vLLM/SGLang 参数或写厂商实现。平台动作由 Runner/Operation 通过 Adapter
执行。需要修改代码或重新采集契约的诊断结论由 Main Agent 消费，Scheduler 不会
猜测修复内容。

## 修改和测试

改变状态机时至少检查：正常完成、结果拒绝、租约过期、新 attempt、诊断消费和
进程重启恢复。不要直接修改 SQLite 来构造成功场景；通过公开状态转换建立测试
数据。Scheduler 相关入口统一通过 `cli/adaptation.py` 使用。
