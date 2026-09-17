# Codex 交互式模型适配（P1）

P1 的调用方向是 **Codex → 现有 Harness CLI**。没有新增模型 API、常驻
worker、MCP 服务或第二套 Task。Codex 负责调查与决策；Graph、scheduler
和既有 validators 仍决定什么可以执行、什么已经通过。

项目入口是 `.agents/skills/infer-forge-adaptation/SKILL.md`，三类子 Agent
位于 `.codex/agents/`：实现者、诊断者、验证者。它们不覆盖用户的模型、
推理强度或权限配置；不支持自定义角色时可用同样的普通子 Agent 指令。
目录和字段遵循 [Codex Skills](https://learn.chatgpt.com/docs/build-skills) 与
[Subagents](https://learn.chatgpt.com/docs/agent-configuration/subagents) 官方文档。
现有 `skills/` / catalog 继续提供领域方法，不在入口复制算子或部署知识。

## 首次配置与继续

以下命令仅用于已获授权的实际适配。开发、评审或运行本地模拟测试不授权
访问集群。实际 resource owner ID 必须由用户提供，不能从路径/用户名猜测。

先创建一次 run，或者直接读取已有 run。保持外部状态库和 artifact root：

```bash
python3 cli/adaptation.py --state /external/adaptation.db create-run \
  --run-id RUN --model MODEL --model-revision PINNED_MODEL_REVISION \
  --plugin-revision PINNED_PLUGIN_REVISION --backend BACKEND \
  --artifact-root /external/runs/RUN
```

首次调用现有 Graph 配置完整输入，并显式选择交互模式。环境契约仍由
harness 生成，服务契约仍来自 MAT-005；不要传手写 `contract_instance`。

```bash
python3 cli/workflow/graph.py \
  --scheduler-state /external/adaptation.db --run-id RUN \
  --artifact-root /external/runs/RUN --subject MODEL \
  --env hardware=P800 --env stack_commit=PINNED_PLUGIN_REVISION \
  --set model_path=/mounted/model --set user_id=USER_SUPPLIED_ID \
  --interaction-mode codex --execute --resume --json
```

按任务实际需要提供 `--target`、`--operator-report`、`--shim-registry` 等既有
选项。粗粒度缺口不能替代实测 OperatorSpec。Graph 在执行前保存版本化
输入；缺少配置的历史 run 需要明确补齐首次调用，不能从聊天记录推测。

之后换一个 Codex 会话也可以从持久状态继续：

```bash
python3 cli/adaptation.py --state /external/adaptation.db context --run-id RUN
python3 cli/adaptation.py --state /external/adaptation.db advance --run-id RUN
```

`advance` 从保存的结构化配置重建参数，检查 run、model、state/artifact
路径及 workflow 哈希，不执行历史 shell 字符串。workflow 有变更时先审阅，
再显式重新配置 Graph；不自动接受漂移。保存的自动恢复/外部 decider 配置
不能与 Codex 控制者并用。

没有未消费的交接时，可用 `advance --run-id RUN --set KEY=VALUE` 显式补充
缺失输入，例如用户提供的 `user_id`。既有 Graph 身份与路径检查仍生效；
有 pending/executing/blocked handoff 时不能用新设置绕过它。

## 返回边界

| 退出码 | 含义 | Codex 下一步 |
| --- | --- | --- |
| `0` | 本次推进/已提交恢复成功，或已完成 run 的状态回读 | 检查 progress 与最终 delivery receipt；单节点成功不等于交付 |
| `3` | operator worker/诊断任务尚待完成 | 定向 claim，提交实测证据，再 advance |
| `4` | Graph 决策交接已持久化 | 读取 handoff，调查后 submit-decision；进程不等待答案文件 |
| `2` | 缺输入、拒绝、预算耗尽、明确阻塞或执行状态不确定 | 停止自动推进，处理返回原因 |

`advance` / `submit-decision` 的 stdout 是一个 JSON 文档。Graph 原始输出保留
在返回 `artifact_root` 的受管 attempt 日志中。重复读取已有交接不会新建
失败 attempt 或重置预算。已完成 run 的回读明确不重新验收历史证据。

默认 Graph `headless` 路径保持兼容；`codex` 模式拒绝同时指定
`--auto-recover` / `--decide-command`。同一状态库/run 的受管 Graph 与交互命令
使用本地进程锁；存在未消费交接时，即使从 headless 入口调用也不能绕过。
这不是跨主机控制或跨 run 的 Pod 资源锁。

## 提交一个 Graph 决策

`context.handoff` 包含 `handoff_id`、`source_version`、失败节点/attempt、
原错误和证据文件哈希、方法快照、允许动作、历史及剩余预算。它与 operator
诊断是不同来源：operator 继续走 `fail → diagnosis → resolve/apply-diagnosis`，
不要把 operator task ID 当成 Graph handoff ID。

首版只开放已有单次重跑能完整执行的三个动作：

- `RETRY`：根据证据重跑原节点。
- `RETRY_WITH_PARAMS`：首版只允许 environment/service proof 修改请求中已声明的
  `proof_health_interval`（有限、非负数，实际传入 health-interval-seconds）。
  不允许换模型、Pod、端口/服务身份、owner ID、方法或输入文件；尚未被节点
  消费的 `max_model_len` 等值不能伪装成有效调参。
- `BLOCKED`：记录不能继续的结论，不执行运行时操作。

旧的 triage/patch/rollback 等动作不在这个接口开放列表中。需要源码修复或
算子实现时，先在用户授权范围内执行对应协议并留证；不能把修复伪装成
无条件重试。未知动作、空证据、非法参数在受理前拒绝。

将现有 `Decision` JSON 放在外部工作目录。不要修改已冻结的失败 attempt：

```json
{
  "next_action": "RETRY",
  "diagnosis": "基于所引用日志和调查结果说明本次重试的理由",
  "facts": ["已观察到的事实"],
  "hypotheses": ["仍待验证的假设"],
  "params": {},
  "confidence": 0.8,
  "evidence_refs": ["从 handoff.source.source_files 选择的真实绝对文件路径"]
}
```

示例中的路径和内容是占位，不是可接受的证据。提交时使用查询返回的
`source_version` 和由调用者生成、保持稳定的逻辑决策 ID：

```bash
python3 cli/adaptation.py --state /external/adaptation.db submit-decision \
  --run-id RUN --handoff-id HANDOFF --decision-id DECISION_ID \
  --expected-version SOURCE_VERSION --decision /external/decisions/decision.json
```

受理与运行分开：先用事务记录决策 ID、payload 哈希和受理凭据，再执行
一次恢复，最后记录执行凭据。节点仍由原 validator 验收；成功后还需要
`advance` 完成其余 Graph 和最终服务/精度门禁，不因决策 JSON 宣称通过。

若重试仍失败，原决定的完成凭据和新 handoff 在同一事务记录，预算和历史
延续到新 attempt。查询、无效提交及相同决策重放不消费预算，也不重置预算。

## 重放、拒绝与不确定执行

- 同一 ID、同一 handoff/version、同一 payload 返回原凭据，不再执行。
- 同 ID 的不同 payload、已消费交接的新 ID、错误 run/version，以及文件、
  Journal、方法或环境发生变化的旧交接均拒绝。
- 已受理但没有执行完成凭据时，返回 `GRAPH_EXECUTION_UNCERTAIN`。重放只查
  原凭据；即使进程锁已释放，也不重新执行。先检查原本地/远端执行与日志。
- 明确 BLOCKED 或预算耗尽时停止。P1 没有强制重置/忽略陈旧检查的命令；
  不通过另建 run、直接改库或换决策 ID 越过阻塞。必要的证据修复/状态恢复
  需要单独诊断与明确处理，不能自动推测。

## Worker 与 P2 边界

定向 claim、lease、冻结任务包和证据格式见
[worker-results.md](worker-results.md)。执行者身份不能因为主 Agent 代提交而
被改成主 Agent；独立 validator 必须实际运行测试，不能只是换一个名字。

P1 不宣称已实现独立数值 Runner、候选身份认证、跨 run Pod 锁或远端进程
取消。这些仍属于 P2。当前只有一个主控制者，源码/CPU 任务可在不重叠路径
并行，共享 Pod 的修改、设备检查与服务回归必须串行。lease 到期不能证明
旧进程已退出。模拟交付只能称 `SIMULATION_PASS`，不是硬件就绪。
