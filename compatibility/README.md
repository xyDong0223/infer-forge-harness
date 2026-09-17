# Compatibility：平台组合支持边界

`matrix.yaml` 明确声明 hardware、engine、backend 和 plugin 的组合是否允许进入执行
流程。它是执行前门禁，不是硬件验证报告。

| 状态 | 含义 |
| --- | --- |
| `supported` | harness 已接入该组合，可以继续执行；最终结果仍需 run 证据证明 |
| `planned` | 已进入路线图或预留配置，但当前拒绝执行 |
| `unsupported` | 明确不在当前支持范围，并应记录原因 |
| 未列出 | 未知组合，默认拒绝 |

当前唯一标记为 `supported` 的组合是 Kunlun P800 + vLLM + Kunlun backend/plugin。
这不表示任意模型已经适配，也不替代环境证明、算子验证或服务回归。

## 与其他声明的关系

- `config/` 选择本次希望使用的目标；
- `compatibility/matrix.yaml` 判断该目标是否进入受支持流程；
- `catalog/runtime_catalog.yaml` 判断软件栈是否声明并接入 loader；
- Adapter/Runtime 实现决定是否真正可执行；
- 当前 run 的 Validator 证据决定本次交付是否通过。

提升状态前必须同时具备实现、生产流程场景和相应证据。只增加配置文件或 registry
名称不能把 `planned` 改成 `supported`。
