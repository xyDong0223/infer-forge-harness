# Skills

Skill 是一类工程工作的可复用方法单元，描述前置条件、判断规则、验证方式和退出
条件。它告诉 Agent “怎样完成这类任务”，但不持有 Workflow 状态，也不替代可执行
Tool 或 Validator。

每个 Skill package 通常包含：

```text
skills/<package>/
  SKILL.md     # Agent 使用的权威方法说明
  skill.yaml   # 机器可读元数据、版本和资源声明
```

[`../catalog/skill_catalog.yaml`](../catalog/skill_catalog.yaml) 把 Workflow 的
`task_type` 映射到方法单元。使用 package 的条目通过 `method_package` 指定目录。
Graph Runner 会把解析后的 catalog 契约和 `SKILL.md` 快照到每个 attempt 的
`input/skill.json`，通过 `INFER_FORGE_SKILL_CONTRACT` 传给执行进程，并在 Task
Memory 和恢复决策请求中记录同一份方法身份。Workflow 执行前会检查 package 和
工具引用。

新 Skill 只有在 Golden Task 和独立 Validator 通过后才能进入正式流程。方法失败
时先分类原因，再根据证据修订，不能因为第一个看似合理的解释就改写通用规则。

## 当前方法包

| Package | 解决的问题 |
| --- | --- |
| `model_scanner/` | 从 checkpoint 与实际注册路径识别模型能力需求 |
| `model_bringup_loop/` | 在固定环境中逐步完成 toy、服务和真实请求 bring-up |
| `runtime_diagnosis/` | 区分模型、插件、kernel、runtime state 和 API 故障 |
| `torch_fallback/` | 建立可验证、可撤销的 Torch fallback |
| `fallback_validation/` | 证明 fallback 的真实 dispatch、正确性和边界 |
| `kernel_correctness/` / `kernel_grade/` | 独立 reference、负控和设备数值验收 |
| `memory_budget/` | 根据模型与设备事实形成容量结论 |

## 边界

- 确定性命令登记为 Tool，不把大段可执行脚本复制进 `SKILL.md`。
- 多任务顺序属于 Workflow，不让单个 Skill 隐式完成整条交付链。
- 验收规则属于 Validator，Skill 可以引用但不能降低门禁。
- 实践经验必须注明触发条件和适用版本，避免把一次修复推广为通用规则。
- 修改方法后用失败复现、Golden Task 和独立 Validator 证明它确实改善了结果。
