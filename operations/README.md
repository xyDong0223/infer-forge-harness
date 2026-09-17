# Operations：领域任务实现

`operations/` 保存 Task 的实际业务行为。CLI 只负责读取参数并调用这里的
`execute(args)`；Validator 独立判断结果；Runner 在需要多个动作时负责组织顺序。

## 领域目录

| 目录 | 任务范围 |
| --- | --- |
| `intake/` | 模型身份、revision 和 checkpoint 信息采集 |
| `discovery/` | 模型扫描、能力匹配、runtime drift、shim 和 gap 分类 |
| `deployment/` | 显存预算、部署计划和 toy bring-up |
| `operators/` | 算子任务生命周期与供应商交接材料 |
| `validation/` | API、精度、tensor diff 和支持矩阵更新 |

## 一个 Operation 应该包含什么

- 加载该任务所需的输入和 Task contract；
- 调用注入或解析出的 Adapter/Runtime/Tool；
- 生成结构化报告及人类可读材料；
- 调用对应 Validator，并如实记录拒绝原因；
- 将正式结果写入调用方提供的当前 attempt `output/`。

它不应解析 `sys.argv`、创建调度数据库、决定完整 Workflow 路由，或绕过
Validator 直接提升状态。

## 调用关系

```text
cli/<domain>/<command>.py
  -> operations/<domain>/<command>.py:execute(args)
  -> adapter / runtime / portable tool
  -> validators/<task>_validator.py
  -> 当前 attempt 的 output/
```

同名 CLI 与 Operation 不是重复实现：CLI 是薄入口，Operation 才持有任务行为。
如果同一业务规则同时出现在两边，应保留 Operation 中的一份并让 CLI 委托调用。

## 当前注意事项

多数 Operation 仍接收 `argparse.Namespace` 形状的 `args`，这是历史迁移后的接口，
调用者需要查看 CLI 参数才能知道完整字段。新增复杂任务优先使用显式 dataclass 或
带类型的函数参数，再由 `execute(args)` 做薄转换。

新增 Operation 时同步更新 Task contract、CLI、Validator、Workflow/Skill catalog
引用和相应测试。不要在 `tools/` 新建宿主机任务入口。
