# Catalog

`catalog/` 是版本化事实与能力索引。它保存 harness 当前承认哪些工具、软件栈、
Skill、设备事实和支持结论，以及这些结论依赖什么证据。

| 文件 | 保存内容 |
| --- | --- |
| `tool_catalog.yaml` | 可确定执行的工具命令、接口和引用 |
| `runtime_catalog.yaml` | 已声明的推理软件栈及支持硬件；还需要 registry 实现才能执行 |
| `skill_catalog.yaml` | Workflow `task_type` 对应的工程方法和执行契约 |
| `xpu_specs.yaml` | 从设备环境观察到的硬件/API surface 事实 |
| `support_matrix.yaml` | 模型和平台组合的支持状态及证据等级 |

## 与 Config 的区别

```text
config/   = 这次准备使用什么参数和环境
catalog/  = 仓库根据已有证据承认什么能力和事实
```

配置中写了一个 Runtime 名称，不代表它已经实现；catalog 中声明 Runtime，也不代表
registry 已接线或真实环境已验证。Loader、兼容矩阵和证据门禁仍会分别检查。

## 更新规则

- 新事实必须附带来源、适用版本和证据；不要根据名字或预期填入支持结论。
- `planned`、`declared`、`supported` 和真实硬件 PASS 不能混用。
- 修改 Tool/Skill 路径时同步更新实际文件和引用检查。
- 支持状态的提升必须由对应 Workflow 和 Validator 的持久化结果支撑。
- 过时事实应被显式替代或降级，不能留下两个无适用范围的矛盾结论。

Catalog 是系统的知识索引，不是运行状态数据库。单次 run、lease、attempt 和日志
分别属于 Scheduler 与外部 artifact root。
