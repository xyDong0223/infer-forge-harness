# Core：跨流程共享基础能力

`core/` 保存不属于某个具体任务、平台或工作流的稳定基础能力。这里的代码回答
“系统各层共同使用什么数据和规则”，不负责执行完整任务。

## 职责边界

这一层负责：

- 描述目标、指标、产物和 Adapter 接口的数据类型；
- 解析 hardware / engine / backend / plugin 组合并检查兼容性；
- 管理外部 run、attempt 和 artifact 的路径与写入规则；
- 提供仓库资源根目录和通用错误类型；
- 比较性能指标等跨工作流纯逻辑。

这一层不应包含模型适配步骤、Pod 操作、CLI 参数解析、任务调度或具体验收规则。

## 文件地图

| 文件 | 主要职责 |
| --- | --- |
| `contracts.py` | `TargetContext`、`Metric`、`Artifact` 以及 Adapter Protocol 等跨层数据契约 |
| `target.py` | 目标规范化、任务目标绑定、兼容矩阵查询和 supported 门禁 |
| `storage.py` | 外部状态根、run/attempt 目录、写入所有权、manifest 与原子写入 |
| `paths.py` | 版本化仓库资源的统一根路径，避免各模块自行猜测仓库位置 |
| `facade.py` | 根据 `TargetContext` 组合 Hardware Adapter 和 Runtime Adapter |
| `performance.py` | 与具体 profiler 无关的指标比较逻辑 |
| `errors.py` | 跨任务共用的结构化失败基类 |

## 常见调用路径

```text
Task / Workflow 输入
  -> core.target 解析 TargetContext
  -> core.facade 解析 adapter bundle
  -> operation 或 runner 执行

任意受管执行入口
  -> core.storage 分配 attempt
  -> output/ 写正式证据
  -> ArtifactStore 写 manifest
```

`core/facade.py` 是刻意设置的组合边界：它会访问 `adapters` 和 `runtimes` 的
registry，但不会直接实现厂商行为。除此之外，通用 core 模块不应依赖具体平台。

## 修改原则

- 只有被多个领域复用、语义稳定的内容才进入 `core/`。
- 新增字段时考虑已有持久化 JSON 和旧 run 的兼容性。
- 路径和写入规则统一修改 `storage.py`，不要在调用方复制检查。
- 平台分支放进 Adapter，验收分支放进 Validator，执行顺序放进 Runner。
