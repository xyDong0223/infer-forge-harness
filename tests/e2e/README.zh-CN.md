# 能力场景

> 本文档是 `tests/e2e/README.md` 的中文译本,内容以英文原版为准。

模型适配是第一个可执行的能力场景。它的注册条目是
[scenarios/model_adaptation.yaml](scenarios/model_adaptation.yaml),由生产
workflow 引用。性能优化和上下文/显存调优尚未声明为可用场景。

## 必选本地层级

```bash
python -m pip install -e '.[test]'
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -m local_e2e tests/e2e \
  --basetemp /external/fresh-e2e-directory \
  --junitxml /external/model-adaptation-e2e.xml
```

使用全新的外部 `--basetemp`:pytest 拥有并可能清空该目录。永远不要把它指向已有
的适配 run。不需要 Torch、集群、凭据和模型权重。GitHub Actions 的
local-scenarios 任务在 PR 和 push 上运行这一层,保留进程日志、SQLite 数据库、
Journal、Task Memory、attempt manifest 和 JUnit 报告。把该任务设为分支保护的必
选检查是单独的仓库设置。

该场景运行的是**未经裁剪的生产 graph 拓扑**,不是缩短的测试专用流程:

```text
create-run CLI
  -> Graph CLI:MiniMax 环境(无 toy)、intake、扫描、分类
  -> 持久化算子分派
  -> evaluation、plan、toy bring-up、shim 交接、service、accuracy、baseline
  -> WAITING_FOR_OPERATORS
  -> 独立的模拟 Agent 进程:claim -> compute -> validate -> complete
  -> 新 Graph 进程带 --resume
  -> 全新的合并模型 service/accuracy 回归,实时集成门禁
  -> 内存/API 检查,支持矩阵
  -> 持久化的 SIMULATION_PASS 交付凭据
```

只有外部集群/运行时观察、远端 revision 解析、Agent 实现和外部容量规划 provider
使用替身。CLI、operation、任务验证器、graph 边、调度器租约/状态转换、结果验证
和产物存储都真实运行。子进程引导是测试专用的,对意外的外部动作失败关闭(fail
closed)。Worker 数值计算和独立证据门禁在本地运行;没有任何验证器被替换成成功
函数。

必选用例覆盖:完成、worker 证据缺失与持久化诊断、claim 被遗弃后的租约过期与重
启、旧租约被拒绝、所有 worker 成功后算子前的服务证据仍被拒绝、不完整的算子契
约、环境证据损坏、新服务证明后的过期精度结果,以及拒绝把模拟证明导入真实 run。
每个用例都检查场景没有修改仓库源码文件。

环境输入用例从无种子 YAML 开始,检查生成的契约及其来源元数据,移除旧的
`USER_ID`,在集群访问之前对照共享 status schema 检查持久化的缺失输入拒绝,然后
提供 `--user-id`,并在新进程中用其记录的 ID 和 Pod 恢复同一个 run。另一个用例在
集群访问之前拒绝手工的 Graph 契约覆盖。完整交付断言服务执行消费的是真实的
MAT-005 生成的契约。

所有持久化的部署证明状态都对照共享 status schema 检查,包括成功和失败。plan 之
前的 intake 失败会走无契约的 triage 路径,检查原始控制台日志的保留、从 Journal
恢复 triage,以及在同一个 Pod 中重试 intake。由此产生的 UNKNOWN 诊断保持
NEEDS_HUMAN;它不能进入补丁放置或厂商交接。在对外操作之前,改变已建立 run 的
所有者会被拒绝,其生成的计划和已准备的环境保持不变。环境 CLI 的重试和导入在成
功与失败的交接中都强制执行该所有者,即使环境中的 USER_ID 发生变化。对生成计划
的篡改会在对外操作之前阻断 service、triage 和补丁放置;完好的计划在重启后仍可
复用。

失败环境交接用例在外部命令边界注入运行时导入错误,检查持久化的 INSTALL_FAILED
证明和导入日志,拒绝所有者变更,并用相同所有者和 Pod 重试。环境失败止步于诊断
终态;模型 triage 要求环境成功。workflow 的 intake 要求 EnvironmentProof,且永远
不会创建它独立的一次性探测 Pod。

完成用例检查持久化事实顺序和实际 adapter 命令:MiniMax 基线启动先于 intake;目
标 toy 在发现之后、目标服务启动之前。边界用例检查省略 `--phase` 和显式
`--phase environment` 都只启动 MiniMax,且永远不会调用 toy 探测。

**SIMULATION_PASS 证明的是流程衔接,不是设备正确性、真实模型精度或性能。** 阅读
凭据的证据模式和 Journal 上下文;中间节点的 `*_READY` 状态本身不是硬件证据。

同一组生产场景还跨进程边界检查进度解释:未 claim 的任务、未做读侧恢复的过期租
约、排队中的诊断、被阻塞的实测契约、可恢复的 graph 和最终的模拟交付。状态查询
保留原始状态并报告下一个责任方,而不把解释当作证据。

## 生产 Graph/调度器交接

用 `cli/adaptation.py` 创建 run,然后显式连接 graph:

```bash
python cli/workflow/graph.py \
  --scheduler-state /external/state.sqlite \
  --run-id my-adaptation \
  --artifact-root /external/runs/my-adaptation \
  --subject MyModel \
  --operator-report /external/measured-operators.json \
  --shim-registry /external/shim-registry.json \
  --env hardware=P800 \
  --env stack_commit=PINNED_PLUGIN_COMMIT \
  --set model_path=/mounted/model \
  --set user_id=<user-supplied-id> \
  --execute --resume --json
```

使用 run 的确切模型、revision、环境和 artifact root。只有当分类结果已经携带完
整算子契约,或确实报告了没有可执行的缺口时,才可以省略 `--operator-report`。
粗略的能力名称不是编造张量契约的许可。`--shim-registry` 提供 MAT-029 使用的实
测声明;旧的仅文件请求不满足调度器任务。

不带 `--scheduler-state` 时保留旧的 graph-only 模式,不会签发新的调度器背书的
功能交付凭据。连接模式下,退出码 `3` 表示 worker 仍在等待:继续通过
`cli/adaptation.py` claim/complete 任务,然后用同一个 graph 和 `run_id` 恢复。
构造 CLI 参数时把不透明租约写成 `--lease-token=TOKEN`;合法的 URL-safe token 可
能以 `-` 开头。退出码 `2` 表示 blocked/rework。厂商工单或等待中的候选不是功能
就绪。`--until-node` 是有意停止部分遍历;该命令退出码 `0` 不是交付声明。

bridge 会重新验证环境交接、执行幂等发现,并在交付前重新检查持久化的 worker 证
据。分派和集成不会仅仅因为旧的 Journal 事实说它们成功过就被跳过。worker 就绪
后,服务和精度会重新运行:冻结的候选前基线不是合并后最终模型的证据。在服务和
精度执行之前记录的调度器快照,把两项回归绑定到已完成的任务;精度还绑定最终的
服务证明及其哈希。恢复尝试使用相同的捕获钩子。最终比较和凭据发布共享一个调度
器事务,因此并发的发现不能越过交付。被接受的调度器证明可以导入 Graph
Journal。已知的就绪 Pod 会被 attach,而不是被新 deployment 替换。

## 可选真实层级

硬件测试默认跳过,除非其各自的授权变量恰好等于 `1`。它们永远不会激活本地引
导,并要求一个已存在的**真实** run 及其已证明的 Pod。

把 `INFER_FORGE_HARDWARE_SCENARIO` 设为一个外部 JSON 配置,包含:

| 字段 | 要求 |
| --- | --- |
| `state`、`run_id`、`artifact_root`、`subject` | 已有的调度器 run 及其记录的 root/模型 |
| `pod`、`namespace` | 已证明的 Pod,位于 adapter 配置的 namespace 中 |
| `image_digest`、`hardware` | 记录的 run 环境身份 |
| `model_revision`、`plugin_revision` | 与 run 匹配的固定 revision |
| `environment` | run 使用的确切 Graph `--env` 映射 |
| `context` | Graph 节点参数,如模型路径、user_id、端口和 served-model 名 |
| `cleanup_policy` | `retain_prepared_pod`;这些测试永远不删除已准备的环境 |

`KUBECONFIG` 和任何运行时凭据保留在外部执行环境中,永远不在场景文件或仓库
里。调度器 run 的环境必须已经记录 `hardware` 和 `image_digest`。

```bash
# 重新证明已有 Pod,包括基线模型 prefill/decode。
INFER_FORGE_RUN_DEVICE_SMOKE=1 \
  python -m pytest -q -m device_smoke tests/e2e/test_model_adaptation_hardware.py

# 要求已完成的算子阶段和上游 Graph 事实。
# 从不带 --resume 的服务证明开始,强制全新的服务和精度工作。
INFER_FORGE_RUN_REAL_MODEL=1 \
  python -m pytest -q -m real_model tests/e2e/test_model_adaptation_hardware.py
```

上下文缺失会让已选择的测试失败;不会静默跳过或创建替代 run。真实模型运行只有
以调度器背书的 `FUNCTIONAL_READY` 凭据结束才算完成。本地 CI 不执行这两个可选层
级。

## 新增能力

添加一个从其实际生产 workflow 链接的场景注册条目,以及一个调用其真实入口的
`local_e2e` 测试。包含成功、拒绝和进程重启用例,断言持久化结果和证据,只对外
部依赖使用替身。运行时夹具放在检出目录之外,并保留进程日志供诊断。注册守卫会
检查必选用例名是否存在,以及可选真实层级是否保持显式 opt-in。

未来的性能场景还必须演示固定工作负载对比、正确性保持和 promote/rollback 决策。
未来的上下文/显存场景必须覆盖其实测边界和失败用例。不要通过复制 PASS 夹具把任
何一个注册为已实现。
