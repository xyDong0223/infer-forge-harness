# 运行期写入整理

本次调整不搬迁 `tools/`、`engine/` 等源码目录，而是先明确运行过程中
“谁拥有目录、什么可以写进去、重试是否覆盖、产物如何找到”。

## 三类内容的边界

| 内容 | 位置与含义 |
| --- | --- |
| 版本化源码 | 仓库内的实现、测试、契约、文档和可重放补丁；通过正常代码变更交付 |
| 运行状态与证据 | 仓库外的 run 目录；保存调度状态、执行记录、日志与结果 |
| 临时探索 | 当前 attempt 的 `scratch/`；不自动作为正式证据提交 |

源码改动仍然允许，但必须是有任务上下文的明确变更。不要把调试脚本、
模型输出、Agent 对话、临时状态库和每次执行日志随手写进仓库。
本次没有实现候选源码 worktree 的自动创建或提升机制。

## 外部根目录

根目录选择顺序：

1. `INFER_FORGE_STATE_ROOT`。
2. `$XDG_STATE_HOME/infer-forge`。
3. `~/.local/state/infer-forge`。

例如：

```bash
export INFER_FORGE_STATE_ROOT="$HOME/.local/state/infer-forge"
python3 tools/run_adaptation.py create-run \
  --run-id example-001 --model <model-id> --backend <backend> \
  --model-revision <revision> --plugin-revision <revision>
```

默认调度数据库为根目录下的 `state.sqlite`，允许包含多个 run。
显式指定 `--state /external/path/state.sqlite` 时，默认 run 目录位于
该数据库同级的 `runs/<run-id>/`。`create-run --artifact-root` 可以覆盖
该 run 的目录；它会写入 run metadata，后续 worker 从调度器接收，不自行猜测。

输出路径必须位于当前源码仓库之外，也不能是其祖先目录、整个 home 或
文件系统根目录。路径检查会解析符号链接；给仓库内路径换一个外部链接
并不能让它变成合规的运行目录。

## Run 与 attempt

```text
<external-state-root>/
  state.sqlite
  runs/<run-id>/
    run.json
    journal.jsonl
    task_memory.json
    tasks/<task-id>/attempts/000001/
      .attempt.json
      input/
      scratch/
      output/
      logs/
      manifest.json
    tasks/<task-id>/attempts/000002/
      ...
```

`run.json` 固定目录所属的 run；同一目录不能被另一 run 静默接管。
编号由独占创建目录分配，并发领取不会获得同一个 attempt。
含斜杠、冒号等字符的 task ID 会编码为带摘要的安全目录名，因此不要
用原始 task ID 自行拼接路径。

| 子目录 | 写入约定 |
| --- | --- |
| `input/` | 固定本次执行的输入和任务快照，不用于修改上游证据 |
| `scratch/` | 探索脚本、中间结果、临时排查；不会进入正式 manifest |
| `output/` | 当前候选结果和提交给 validator 的正式证据 |
| `logs/` | 命令、Agent 和执行日志 |

重试创建新目录，不覆盖旧结果。仅构造路径对象不会创建目录；
graph 的 plan 模式不会分配执行 attempt。

## 各入口如何使用

**持久化 scheduler**：`claim` 返回的 `input.workspace` 包含四个子目录和
本次 attempt 的身份。它同时保留 `attempt` 和 `lease_token`，前者仍然是
调度器执行次数，不应通过目录名推断。任务快照写入 `input/task.json`。
Worker 应在 `output/` 下生成本次证据；提交指向其他目录的证据会被拒绝，
即使文件存在且摘要正确。`complete` / `fail` 登记结果与 manifest。
无法登记的失败会显式记录 `artifact_registration_error`，不被当成成功。

**Graph runner**：使用外部 `--artifact-root`，或通过 `--run-id` 选择默认
run 目录。Journal 和 Task Memory 默认属于当前 run。每个实际执行的节点
及恢复重试分配独立 attempt，恢复成功后使用新的输出路径。

**Deployment task runner**：独立调用时分配新 attempt；如果 graph 已经
传入某次 attempt 的输出目录，则复用该目录，不再套一层 attempt。
读取返回状态中的实际 `artifact_root`，不要再假设结果直接落在
`--artifact-dir` 指定目录下。性能执行也为每次 invocation 分配独立目录。

**普通 `tools/* --out`**：为兼容直接脚本调用，显式的外部 `--out` 仍然是
实际输出目录，但独立调用必须使用不存在或为空的目录；重复使用已有产物的
目录会报错。Graph 管理的输出必须位于当前 attempt 的 `output/` 内。
只读查询或列表模式不需要凭空分配输出目录。
`tensor_diff --out` 的文件型输出保留原文件位置，但必须在仓库外且不能
覆盖已有报告；旁边的 `<报告名>.manifest.json` 只登记该报告，不扫描同目录输入。
诊断、补丁放置和 correctness executor 的 CLI 也使用相同的目录型输出规则。

**Agent 决策交接**：每次请求及格式错误后的重问都有自己的
`tasks/decision/attempts/<id>/`，请求位于 `input/decision_request.json`，
响应写入 `output/decision.json`。command 模式会把这两个实际路径传给
decider；file 模式的外部 Agent 必须查找本次请求并写入对应响应路径，
不能继续使用旧的固定 `<brain-dir>/decision.json`。

## Manifest 的作用与限制

`core.storage.ArtifactStore` 提供受约束的相对路径写入和产物登记。
正式清单记录身份、执行 outcome，以及每个文件的相对路径、大小、
SHA-256 和基本类型；`scratch/` 不登记。符号链接、非普通文件、
越界路径和缺失的必需产物不会被悄悄接受。JSON/文本的受管写入以及
manifest 替换使用原子发布；默认不覆盖已有文件。

Manifest 解决的是定位和完整性，不是语义正确性。清单存在、命令退出码
为 0 或某个文件声称 `PASS`，都不能替代独立验证和调度器证据门禁。
输入目录只读、日志不可手动改写等约定，也不是 OS 权限强制隔离。

## 旧运行迁移

不要一边执行一边移动旧证据。先停止该 run 的写入，再把运行数据备份到
外部目录，并检查数据库、Journal、Task Memory 和结果报告中的绝对路径。
仅复制目录不会自动修复这些引用；本次没有提供自动数据库重写工具。
旧记录仍可读取，但新执行遵循外部目录和新 attempt 规则。

仓库内旧的 `artifacts/` 示例不再是可直接使用的运行输出位置。
如果已有一个准备好的 Pod，目录调整并不意味着应该创建新 Pod；
重新证明环境时仍须使用 `--attach-pod` 并保留原 run 身份。

## 没有承诺的能力

这不是容器或文件系统沙箱。拥有任意 shell 权限的 Agent 仍然可以绕过
Python API 写入其他位置；直接调用底层函数、第三方程序的缓存及外部
Agent 的行为，也不等同于已经被全局拦截。
本阶段约束 harness 的受管入口和任务交接，给正常执行提供一致的目录归属。
仓库只读挂载、进程写权限白名单、候选源码 worktree 和自动清理保留策略，
都不在本次实现范围内。
