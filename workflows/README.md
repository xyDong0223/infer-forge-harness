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
  # 可执行流程必须登记必选本地回归场景;没有这一行的文件是骨架,不可执行。
  regression_scenario: tests/e2e/scenarios/<capability>.yaml
spec:
  description: <一句话说明>
  entry_task: <首个节点 id>
  nodes:
    - id: <节点 id,全文件唯一>
      task: tasks/<task-id>/task.yaml   # 骨架节点写 PLANNED
      on_success: <下一个节点 id>
      on_failure: <失败边节点 id 或 NEEDS_HUMAN>
```

- 拓扑的唯一权威位置是 `spec.nodes`;`stages:` 等历史写法已废弃。
- 可执行流程必须在 `metadata.regression_scenario` 登记
  `tests/e2e/scenarios/` 下已存在的必选本地场景;注册守卫
  (`tests/e2e/test_scenario_contracts.py`)会校验链接双向存在。
- 骨架流程的每个节点必须写 `task: PLANNED`(或仅声明 `validator`),并在注释
  里记录转正时的目标契约。Graph Runner 对 PLANNED/无 task 节点立即以
  `NO_CONTRACT` 停止,所以骨架即使被误传给 `--execute` 也不会触碰外部系统;
  `tests/unit/test_workflow_skeletons.py` 强制这条规则。
- 可执行流程的节点 `task` 引用必须真实存在,且节点与
  `runners/graph_runner.py` 的执行绑定一一对应;转正一个骨架 = 建立真实 task
  契约 + 接入执行映射 + 登记 `regression_scenario`。

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
