# Workflows：声明式流程拓扑

`workflows/` 描述 Task 节点如何连接：先执行什么、成功后去哪、失败后进入哪个诊断
或修复节点。Workflow 不实现命令，也不包含平台专属 shell。

## 当前流程

| 文件 | 用途 | 当前成熟度 |
| --- | --- | --- |
| `model_adaptation.yaml` | 模型 intake、环境证明、扫描、算子任务和最终回归 | 主要生产流程，本地 E2E 必选 |
| `performance_analysis.yaml` | 功能就绪后的基准和 profiler 分析 | 流程骨架（占位引用，不可执行） |
| `performance_optimization.yaml` | 优化候选、正确性回归和性能回归 | 流程骨架（占位引用，不可执行） |
| `test_release.yaml` | 发布相关测试流程 | 流程骨架（占位引用，不可执行） |

## Schema 约定

所有 Workflow 文件统一使用 `infer.kunlun/v1alpha1` 和同一套结构，新增流程不得
另起 schema 或 API 组名：

```yaml
api_version: infer.kunlun/v1alpha1
kind: Workflow
metadata:
  name: <workflow-name>
spec:
  description: <一句话说明>
  entry_task: <首个节点 id>
  nodes:
    - id: <节点 id,全文件唯一>
      task: tasks/<task-id>/task.yaml   # 或 validator: validators/<file>.py
      on_success: <下一个节点 id>
      on_failure: <失败边节点 id 或 NEEDS_HUMAN>
```

- 拓扑的唯一权威位置是 `spec.nodes`;`stages:` 等历史写法已废弃。
- 骨架流程可以占位引用已有 task,但必须在 `spec.description` 上方用注释声明
  “流程骨架,不可执行”,并在上表标注成熟度。
- 节点 `task` 引用的契约文件必须真实存在;Graph Runner 只保证模型适配主流程
  的节点与 `runners/graph_runner.py` 的执行绑定一一对应。

## 节点如何落到代码

```text
Workflow node.task
  -> tasks/<task-id>/task.yaml
  -> metadata.task_type
  -> catalog/skill_catalog.yaml
  -> Skill execution contract
  -> CLI / Operation / Runner
  -> Validator
```

Graph Runner 负责解析这些声明、解析输入 artifact、分配 attempt、记录 Journal，
并跟随成功或失败边。连接 Scheduler 时，算子节点会映射到持久化的
`torch -> xpu -> integration` 阶段。

## 修改原则

- Workflow 只表达拓扑、输入输出关系和 terminal state。
- 命令行放 `cli/`，业务行为放 `operations/`，多步执行序列放 `runners/`。
- 所有引用的 Task、Skill、Tool 和输入 artifact 必须存在。
- 新增能力必须有本地 E2E，覆盖完成、拒绝和重启恢复。
- 改变失败边时检查是否产生循环；重复失败不能靠无限 RETRY 掩盖。

使用 `cli/workflow/graph.py` 先计划，再在明确的外部 artifact root 上执行。Graph-only
完成不等于 scheduler-backed 功能交付。
