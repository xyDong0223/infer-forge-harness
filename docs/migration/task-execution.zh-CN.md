# P3：执行元数据与恢复状态收敛

本次只减少重复的执行信息和状态维护，不新增 Agent 框架，不改变模型适配的领域门禁。P2 的真实 Pod/设备驱动仍未接通，managed-v2 对相应执行继续阻断；本地测试不能证明硬件可用。

## Task 是执行元数据来源

24 个 Graph Task 在 `spec.execution_descriptor` 中声明：

- `schema_version: 1` 和 argv 数组；只允许 Python CLI 入口及已知简单占位符，不解释 shell。
- 按 CLI flag 定义输入的 fact kind、相对 artifact 路径和可选性。
- `status_file` 和 `success_exit_keys`；成功状态引用本 Task 的 `exit_states`，不借用其他 Task 的成功状态。

`produces`、`consumes`、`validator` 继续来自原 Task 字段。`consumes` 包含语义依赖，不等于每一项都要新增 CLI 参数。Graph 的节点表、Journal 的 kind 映射、delivery 的状态文件名均从描述派生。工具目录对这些命令引用 Task，保留工具 ID、副作用与输出说明，不再复制入口。

fan-out 的枚举/聚合、准备 Pod、生成服务 contract 等领域逻辑仍在 Python；Workflow 只管节点和路由。只读加载器不会执行命令，也不会将 `runs_with` 的说明性 shell 字符串 eval 成程序。

[源 Task schema](../../contracts/task_definition.schema.yaml) 覆盖现有 26 份定义；两份非 Graph 定义没有执行描述，不自动变成可执行节点。[部署实例 schema](../../contracts/task_contract.schema.yaml) 仍单独使用，未因源码形状不同而放宽。

新增 attempt 的 `input/task_execution.json` 保存解析后的描述与 Task 源文件哈希。新 Journal fact 带有 Task binding，定义改变后不复用旧结果；历史未绑定 fact 继续按原证据规则检查，不伪造补签。执行过程中源定义变更会拒绝，需重新加载 Graph。

## Task Memory 是投影，不是独立真相

SQLite 继续管理任务/lease/控制事件；Journal 继续管理证据索引，并追加带 run/workflow/subject 身份的 `TaskMemoryProjection` 事件。block、claims、observations、下一步和环境变化先追加并 fsync，再原子替换 JSON 视图。事件有哈希链和并发游标，重复保存不会重复追加历史，冲突写入要求重新加载。

查询从 Journal 重建内存视图，不写文件；删除或损坏已迁移的缓存不会删除历史。需要显式恢复文件时：

```bash
python3 cli/state/task_memory.py \
  --path /external/run/task_memory.json \
  --task-id model_adaptation --subject <model-id> \
  --journal /external/run/journal.jsonl --run-id <run-id> --rebuild
```

`task-id` 为实际 workflow 文件名（去掉后缀）；使用 run 返回的实际路径与身份。没有匹配的事件时 rebuild 拒绝；损坏或不完整的 Journal 也拒绝，不自动清空/截断。重建摘要不会执行 Task、改变 lease、重新验证结果或提升 verdict。

旧 run 兼容策略：

- 没有投影事件时，读取原 v1 Memory，查询不迁移。**此时旧文件不可丢弃**。
- 第一次真实写入时，把旧文件原文、SHA-256 和 `revalidated: false` 与本次增量一次性追加；历史 claims/observations 不丢失，也不被升级为新证据。
- 已存在的 pending handoff 在迁移前返回，保持原 source version 和文件校验；不要主动重写其绑定的旧 Memory。
- 旧格式 handoff 没有 Task 描述绑定时，首次接受重试前将原 `commands.json` 与当前描述的只读解析结果比较。命令变化、缺少绑定或无法只读证明的旧 fan-out 会拒绝，不消耗预算、不执行；需先检查并显式报告阻断。已受理 decision ID 的重放不受新 preflight 影响。
- 新 handoff 绑定 attempt 内的 `input/task_memory_snapshot.json` 和 Task 定义，不绑定可变缓存。恢复该缓存不影响决策凭据，改动真实证据仍会拒绝决策。

## 唯一公开 scheduler 命令入口

使用 `python3 cli/adaptation.py --state /external/adaptation.db ...`；旧顶层 scheduler 入口已删除，无兼容别名。

新增 `list [--run-id <run-id>]` 为只读任务查询，未知数据库或 run 拒绝。其他命令沿用 adaptation 的参数与 JSON envelope：创建返回 `run`，发现返回 `task`；创建显式指定 backend。不要仅机械替换旧脚本名后假设输出完全同形。

旧协议 run 继续可查询，不自动改为 managed-v2，不新建替代 run，不改写旧 revision。升级或重新适配需要独立明确操作。SQLite 的只读连接可能产生 WAL/SHM 协调文件，但不会通过查询恢复 lease 或修改业务状态。

## 验证边界

单测覆盖定义/schema、错误 argv/路径、按 Task 接受状态、源哈希漂移、旧事实读取、投影幂等与失败恢复。生产 CLI 本地 E2E 覆盖完成、拒绝、丢失缓存后的新进程恢复，以及旧协议只读查询。测试使用外部 run root，替换的仅是外部集群/runtime/Agent；实际 scheduler、validators 和持久证据仍参与。
