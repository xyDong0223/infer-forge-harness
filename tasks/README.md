# Tasks：可验证任务契约

`tasks/` 中每个目录描述一个有边界的工程目标，包括输入、动作、产物、验收标准和
终止状态。Task 是“必须证明什么”的声明，不是 Python 业务实现。

## 目录结构

```text
tasks/<task-id>/
  task.yaml              # 版本化任务契约
  instances/*.yaml       # 特定模型/环境的参数实例，可选
  manifests/*.yaml       # 任务引用的部署模板，可选
```

`metadata.name` 是稳定 task id，`metadata.task_type` 用来解析 Skill 方法。Workflow
通过 task id 引用契约，CLI/Operation 执行动作，Validator 按契约验收。

## 任务族

| 前缀 | 范围 |
| --- | --- |
| `kdp-*` | 环境、服务和 accuracy smoke 等部署证明 |
| `mat-*` | 模型适配、发现、正确性和算子生命周期 |
| `mem-*` | 显存预算和容量相关任务 |

编号用于稳定引用，不代表每个编号都连续存在，也不表示执行顺序；实际顺序由
Workflow 决定。

## 契约应该表达什么

- 明确的输入 artifact 与来源；
- 可执行动作或受管执行入口；
- 必须产出的机器可读报告和证据文件；
- Validator 能独立检查的 acceptance 条件；
- PASS、REWORK、BLOCKED 等状态的含义；
- 需要真实环境时的身份和 evidence mode 要求。

## 不应放在 Task 中

- kubectl、设备 API 或推理框架实现细节；
- Scheduler 状态修改；
- 仅对一次调查有效的临时命令；
- 无法由证据验证的宽泛目标；
- 模型权重、凭据、私有端点或大型 trace。

修改 Task 字段时，同步检查对应 Operation、Validator、Workflow、Skill catalog、
示例 instance 和 E2E。不要通过放宽契约让既有失败“变绿”；先确认行为变化是否
有新的独立证据支持。
