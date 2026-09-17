# Contracts：机器可读协议

`contracts/` 保存跨进程、跨 Agent 和持久化文件之间的结构协议。Python dataclass
描述进程内对象，Schema 描述落盘或交换的 JSON/YAML；两者变更时必须保持一致。

| Schema | 用途 |
| --- | --- |
| `task_contract.schema.yaml` | Task YAML 的目标、输入、动作、产物和验收结构 |
| `worker_result.schema.yaml` | worker 提交的 verdict、身份、证据角色和哈希 |
| `artifact_manifest.schema.yaml` | attempt 文件清单、哈希和 workspace 身份 |
| `status.schema.yaml` | 独立任务执行后的状态报告 |
| `progress.schema.yaml` | 位置、原因、责任人、下一步和证据的只读进度说明 |
| `deployment_manifest.schema.yaml` | 部署请求和环境参数结构 |
| `platform.schema.yaml` | hardware/engine/backend/plugin 目标描述 |
| `performance.schema.yaml` | workload、benchmark 和性能指标请求 |

## 修改规则

- 新字段应优先保持向后兼容，并明确 optional/default 语义。
- 改名或删除字段前检查历史 run、Task instance、worker result 和 E2E fixture。
- Schema 只定义结构；PASS 的业务含义仍由 Validator 判定。
- 文档示例和 Python contract 不能比 Schema 更宽松。
- 不要用任意 `metadata` 字段绕过已经存在的正式字段。

新增或修改协议后，运行引用检查、对应 schema/validator 测试和至少一个生产入口
场景，确保生产者与消费者同时更新。
