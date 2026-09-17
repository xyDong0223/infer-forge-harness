# Infer-Forge Agent 执行协议

> 本文档是 `AGENTS.md` 的中文译本,内容以英文原版为准;两者不一致时以英文版为准。

本文件是任何 Agent 在本仓库中工作的执行协议。修改代码、执行模型适配或汇报结果
之前必须阅读。本仓库是一个持久化编排 harness:Agent 必须使用它的任务契约和调度
器,而不能把项目当作一堆互不相关的脚本来用。

## 使命与事实来源

目标是通过一个可复现、有证据背书的循环,把推理模型适配到 vLLM 后端和目标 XPU:

```text
部署环境证明(已就绪 Pod + 代码 + XPU)
  -> 模型身份 intake
  -> 在该 Pod 内进行运行时/toy bring-up 与扫描
  -> 缺失算子发现
  -> OperatorSpec
  -> PyTorch 参考任务
  -> 独立参考验证
  -> XPU 实现任务
  -> 设备正确性与 dispatch 验证
  -> 集成与服务回归
  -> functional-ready
  -> 可选的 benchmark 与 profiling 优化
```

任务契约、验证器和实际执行证据是权威。`openwiki/` 材料是工程参考资料,可以解释
为什么某种做法合适,但它不能覆盖任务契约,也不能证明某个结果通过。

## 角色

### Main Agent

Main Agent 拥有一个 `AdaptationRun` 及其决策。它必须:

1. 在调度器中创建或恢复 run。
2. 在任何运行时调查之前运行部署环境证明任务。在该证明的状态绑定到 run 之前,
   不得开始模型扫描、能力评估、shim 扫描或算子适配。
3. 选择实现路径之前,阅读相关的 `openwiki/vllm-core/`、`openwiki/vllm-kunlun/`
   和 `openwiki/harness/` 材料。
4. 在声明模型可用之前,完成 intake、静态/运行时检查和 toy bring-up。
5. 把每一个确认的缺口转换成 `OperatorSpec` 和持久化任务。
6. 让相互独立的算子分支并行推进;不要因为一个算子失败就暂停调查其他缺口。
7. 把工作分派给合适的子 Agent,只消费其持久化的结果和证据。
8. 根据诊断结论决定重新发现、修复、重试、显式回退,还是把分支标记为 blocked。
9. 只有在要求的算子完成集成与服务回归之后,才能声明功能完成。

Main Agent 可以在任务需要时修改仓库代码,但仍必须记录受影响的任务、证据和验证
结果。

### PyTorch Agent

PyTorch Agent 接收 `OperatorSpec` 以及产生它的失败/调用证据。它产出参考实现、
测试和候选 manifest。它不得编造证据中不存在的张量 shape、dtype、layout 或语义;
不确定性的正确出口是诊断或 blocked。

### XPU Agent

XPU Agent 接收同一份 `OperatorSpec`,外加一份已通过独立验证的 PyTorch 参考。它
在目标环境中实现、构建、注册并测试目标 XPU 路径。仅有编译成功永远不够。

### Diagnosis Agent

Diagnosis Agent 接收结构化的 `BugReport`、源任务的输入输出以及所有被引用的产物。
它返回根因分析、修复结论、证据引用、置信度,以及具体的 `next_action`,例如
`REDISCOVER_OPERATOR`、`DISPATCH_TORCH_FIX`、`DISPATCH_XPU_FIX`、`RETRY` 或
`BLOCKED`。它必须区分观察到的事实和假设。

### Validator 与 Integration Agent

验证者独立于产出实现的 Agent。它们重新运行相关检查并撰写报告。集成工作证明的是
真实模型/服务路径,而不只是孤立的单元测试。

## 强制执行协议

### 源码归属

host 侧参数解析和命令入口放在 `cli/`。任务行为实现在对应的 `operations/` 域:
`intake`、`discovery`、`deployment`、`validation` 或 `operators`。CLI 只委派执
行;不重复实现任务逻辑或验收规则。

`engine/` 负责调度与恢复;`engine/state/` 负责 Journal 和 Task Memory。
`runners/` 负责可执行的 workflow/任务序列,不含命令行解析。共享契约、错误、版本
化资源路径和运行时存储属于 `core/`;从 `core.paths` 导入仓库资源根。库不得导入
`cli`,也不得解析 `sys.argv`。

`tools/` 只保留 `probe/`、可重放的 `patches/` 和可移植的 `torch/` 参考。不要在
其中新增 host 任务命令或协调状态模块。旧的 host 脚本路径已被删除,不是兼容别名。
条目移动时,同步更新 catalog、契约、workflow、测试和文档。运行
`python3 cli/maintenance/check_repo_references.py` 检查引用和依赖方向。见
[源码归类说明](docs/architecture/source-layout.zh-CN.md)。

### 能力回归场景

每一项新支持的能力都必须有一个必选的本地 E2E 场景,使用其生产 CLI/workflow、真
实的调度器和验证器以及持久化证据。只替换外部集群/运行时/Agent 依赖。必须覆盖完
成、拒绝和重启;绝不允许预置成功报告或修改调度器状态来让场景通过。模型适配是
`tests/e2e/` 下的第一个模板。真实设备 smoke 和真实模型回归是可选的、需显式授权
的层级。本地 `SIMULATION_PASS` 不等于硬件就绪。见
[场景协议](tests/e2e/README.md)。

### 运行时写入所有权

仓库文件是版本化源码,不是运行工作区。受管入口会拒绝把运行时输出、Journal、
Task Memory 和 SQLite 路径放在源码检出目录内。用 `INFER_FORGE_STATE_ROOT` 选择
外部目录;默认是 `$XDG_STATE_HOME/infer-forge`,或 `~/.local/state/infer-forge`。

每个 run 拥有一个由 `run.json` 标识的目录。每次执行或重试拥有一个新的
`tasks/<task-id>/attempts/<attempt-id>/` 目录。调度器的 claim 载荷在
`input.workspace` 中暴露这些路径:

- `input/`:分派的任务快照和拷贝来的执行输入。
- `scratch/`:临时调查;不进入正式清单。
- `output/`:当前 attempt 的候选结果和正式证据。
- `logs/`:执行日志。

永远不要把上一个 attempt 当作可写输出目录复用。Worker 证据必须位于所 claim
attempt 的 `output/` 内,而不是之前的 attempt 或任意外部目录。调度器记录
`result.json` 和 `manifest.json`;manifest 盘点文件和哈希,但不证明正确性,也
不授权晋级。保留历史 attempt 供诊断使用。

Graph/deployment/performance runner 自动分配受管 attempt。`cli/` 下的普通独立任
务命令保留其显式 `--out` 目录语义,但要求该目录是外部且全新的。读取返回的
`artifact_root` 或任务 workspace,不要凭猜测预测输出路径。

这些是协作工具约束,不是 OS 沙箱。任意 shell 命令仍然可以写到别处。Agent 编写的
实验属于 `scratch/`;仓库改动必须是有意的源码改动,并带有任务/证据上下文。不要
为每次运行时调查发明新的仓库目录。见
[运行时写入所有权](docs/migration/runtime-write-policy.zh-CN.md)。

### 1. 启动或恢复适配 run

使用位于临时源码文件之外的持久化状态数据库。永远不要因之前的命令中断就创建第二
个 run;恢复已有的 `run_id`。

```bash
python3 cli/adaptation.py \
  --state /path/to/adaptation.db \
  create-run \
  --run-id <run-id> \
  --model <model-id> \
  --model-revision <revision> \
  --plugin-revision <revision> \
  --backend <backend>
```

在做昂贵的工作之前,把模型 revision、插件 revision、目标硬件、运行时版本和
artifact root 记录到 run 上下文中。

### 2. 发现之前先证明部署环境

发现之前,Main Agent 必须运行部署环境证明任务并把结果绑定到 run。在证明记录了
就绪 Pod、可导入运行时、就绪的 vLLM-Kunlun 代码工作树和可见 XPU 设备之前,发现
会被拒绝。之后所有调查和子任务必须使用这次交接记录的 Pod 和代码上下文;新开一
个一次性 Pod 不等价。

```bash
python3 cli/adaptation.py \
  --state /path/to/adaptation.db \
  environment \
  --run-id <run-id> \
  --user-id <user-supplied-id>
```

`user_id` 是资源所有者 ID,由使用者提供。如果未知,执行前询问使用者;永远不要
从 host 登录名、仓库路径、示例契约或已有资源名猜测。通过 `--user-id`(Graph:
`--set user_id=<user-supplied-id>`)或契约中的 `execution.user_id` 传入。旧的显
式配置 `USER_ID` 仍然支持。环境 attempt 会记录该 ID,调度器重试会复用记录值。

harness 根据集群 profile 和环境任务生成环境契约,并持久化在 attempt 中。不要复
制模型专属的 YAML 示例,也不要手工编写启动配置。目标服务契约来自 Journal 中已验
证的 MAT-005 DeploymentPlan;Graph 拒绝手工提供的 `contract_instance`。直接回放
证明可以使用之前生成的外部契约。生成的 YAML 和运行时输出永远不属于版本化源码。

环境重试自动复用 run 中记录的 Pod(包括失败的证明),直接执行证明时会先 attach
到已有 deployment 再应用任何 manifest。用 `--attach-pod` 显式选择已准备好的
Pod。未就绪的 Pod 会被保留用于诊断,而不是被替换。

```bash
python3 cli/adaptation.py \
  --state /path/to/adaptation.db \
  environment \
  --run-id <run-id> \
  --attach-pod <prepared-pod>
```

如果部署证明已由独立的 workflow 步骤执行,可以改为导入其已验证的
`status.json`:

```bash
python3 cli/adaptation.py \
  --state /path/to/adaptation.db \
  environment \
  --run-id <run-id> \
  --status /path/to/environment/status.json
```

证明必须留下一个已准备好的 Pod:可导入的运行时、固定版本的 vLLM-Kunlun 代码工
作树、可见的目标 XPU 设备。所有运行时调查必须在那个 Pod 中执行,并引用其代码/
环境指纹。

**保留一个调试环境。** 导入失败、算子错误、超时和服务崩溃,都是去检查日志并在同
一个 Pod 内修复/重启受影响进程的理由。不要为了重试一个模型而删除/重建 Pod、滚
动 deployment 或重装一套可用的栈。替换 Pod 需要已诊断的 Pod/节点故障或显式的环
境变更,并保留证据。

**MiniMax-M2.5 是环境门禁。** 每一次 `--phase environment` 调用,包括回放外部契
约,都从 `config/clusters/p800-cluster.yaml` 推导基线模型和 smoke 命令。目标权重
不是替代品。在模型调查之前,必须要求基线模型身份、health、prefill/decode 和后端
证据。直接启动这个已知可用的基线,不做 MAT-028/toy bring-up。环境证明结束就停;
只有在此之后才运行目标 intake、扫描、能力匹配、缺口发现和目标 toy bring-up。永
远不要在证明环境的过程中启动目标服务。部署 CLI 默认就是 `environment`;
`--phase all` 是显式的旧式独立部署,不属于本流程。

**先诊断,再修复。** 部署不再自动发现或执行 `tools/patches/patch_*.py`,包括针对
特定 commit 的 Kunlun 漂移修复。只修复由运行时导入、toy bring-up 或真实服务路径
针对所安装 revision 观察到的不兼容;记录 diff 和证据,并把修复以可重放的形式保
留在版本化源码中。不需要一刀切的漂移补丁或静态漂移预检步骤。

**先 toy,再上目标权重。** 用 dummy 权重运行 MAT-028,要求引擎构造、prefill 和
至少两个 decode token。任何修复之后,重复 toy bring-up,然后 shim 交接,然后目
标服务证明。部署执行器在新的目标服务启动前也会运行一次新的 toy 探测,包括直接
CLI 调用;失败会保留 Pod 并阻止完整 checkpoint 加载。已知可用的 MiniMax 环境
smoke 是刻意保留的真实权重基线。

**引擎已有模型网络时必须复用(硬规则)。** 考虑任何树外(OOT)模型实现之前,先
检查能力匹配:如果架构能通过 vLLM 的 ModelRegistry 或插件的注册模型解析
(`REGISTRY` / `MODULE` 判定),网络就已经存在——不要实现模型层。此时每个剩余
缺口都是算子层缺口,直接进入 OperatorSpec 路径。OOT 模型只用于没有任何注册的字
架构(`ABSENT`)——而且插件自身也把它的 OOT 模型视为临时状态(上游
`models/__init__.py` 带着 "Remove all of models registration" 的 TODO;Gemma4 已
经不带 Kunlun 专属模型文件交付)。run `glm52-int-w8a8-p800-001` 端到端走了复用
路径:`GlmMoeDsaForCausalLM` 解析到了 deepseek_v2 网络,所有工作都发生在算子层。

**算子集成阶梯(优先靠上)。** 替换或新增算子时:

1. **装饰器注册** —— `@register_oot("LayerName")` 或
   `direct_register_custom_op` 注册进 torch dispatcher。在 vLLM 构建层时解析,对
   导入顺序不敏感,无需改文件。
2. **导入后包装** —— 在运行时包装已有符号(apply_torch_decode_patch 模式)。可
   逆,但对导入顺序敏感,源码检查不可见。
3. **文本补丁** —— 最后手段:`tools/patches/` 下带精确锚点的文件编辑,承担上
   述全部可重放约束。

其机制、四个算子命名空间(`torch.ops._C` / `torch.ops.xspeedgate_ops` /
`kunlun_ops` pybind 门面 / `torch.ops.vllm::*`)和已知陷阱(PluggableLayer 没有
`forward_oot` dispatch——不重写 `forward()` 就永远不会执行;dispatcher 注册无法
回滚)都在 `openwiki/vllm-kunlun/architecture.md` 中带上游行号记录。

报告必须包含足以识别张量输入、输出、shape/rank、dtype、layout、语义、调用点和
失败上下文的证据。通过编排入口转换报告:

```bash
python3 cli/adaptation.py \
  --state /path/to/adaptation.db \
  discover \
  --run-id <run-id> \
  --report /path/to/gap-report.json \
  --model <model-id> \
  --model-revision <revision> \
  --plugin-revision <revision> \
  --backend <backend>
```

发现是严格的。如果必填字段未知,保留不确定性,创建诊断或 blocked 任务,而不是
猜测。

### 3. 分派并完成子任务

Worker 只领取自己有资格执行的阶段:

```bash
python3 cli/adaptation.py \
  --state /path/to/adaptation.db \
  claim --worker <worker-id> --stage torch --limit 1
```

Worker 必须阅读完整的任务载荷,把产物写到 run 的 artifact root 下,运行该阶段要
求的独立检查,并提交 JSON 结果:

```bash
python3 cli/adaptation.py \
  --state /path/to/adaptation.db \
  complete \
  --task-id <task-id> \
  --worker <worker-id> \
  --lease-token <token-from-claim> \
  --result /path/to/result.json
```

完成要求当前未过期的 claim token。重用的 worker 名字不能授权旧的 attempt。长时
间运行的工作要在过期前续租:

```bash
python3 cli/adaptation.py \
  --state /path/to/adaptation.db \
  renew-lease --task-id <task-id> --worker <worker-id> \
  --lease-token <token-from-claim> --lease-seconds 300
```

结果遵循 `contracts/worker_result.schema.yaml` 和
`docs/migration/worker-results.md` 中的证据绑定规则。空结果、裸 PASS 报告、缺失
产物、哈希或任务/环境身份不匹配,以及缺少独立验证,都会被拒绝进入诊断。本地
fake-agent 运行必须显式声明 `metadata.evidence_mode: simulation`;它们不能满足由
环境背书的真实 run。

调度器只在前置任务通过后才创建下一个算子阶段。正常链路是
`torch -> xpu -> integration`。

### 4. 每个失败都走诊断

不要隐藏异常、直接改任务状态,或静默重试失败的实现。报告失败,让调度器创建持久
化诊断任务:

```bash
python3 cli/adaptation.py \
  --state /path/to/adaptation.db \
  fail \
  --task-id <task-id> \
  --worker <worker-id> \
  --lease-token <token-from-claim> \
  --error '<structured error or concise original message>'
```

Diagnosis Agent 领取 `--stage diagnosis`,检查 `BugReport` 和产物,然后用
`resolve-diagnosis` 提交结构化结论。该命令同样要求当前的 `--lease-token`。
Main Agent 必须先消费该结论,再选择下一步动作。

### 5. 交付前证明集成

功能交付至少要求:

- PyTorch 参考正确性和独立验证。
- XPU 构建/注册证据,以及相关 shape、dtype、layout 上的设备正确性。
- 真实模型路径 dispatch 到预期 XPU 算子的证据。
- 服务健康和真实模型请求/回归检查。
- 在声明 XPU-ready 的路径上没有未报告的 CPU/Torch shim 回退。
- 持久化产物和最终 run/任务总结。

## 证据门禁

Agent 不得仅因为命令返回退出码 0 就报告 `PASS`、`READY` 或 `PROMOTED`。结果必须
指向持久化证据文件。

| 阶段 | 必需证据 |
| --- | --- |
| 发现 | 可复现的调用点/失败、完整的 `OperatorSpec`、输入输出契约和源码引用。 |
| PyTorch | 参考源码、含边界条件的聚焦测试,以及独立的数值验证报告。 |
| XPU | 目标环境构建/注册记录、设备测试、dispatch 证明,以及独立的设备验证报告。 |
| 集成 | 真实模型/服务请求、输出或精度回归、健康证明和回退检查。 |
| 诊断 | 原始错误/traceback、复现证据、根因、修复结论、置信度和 `next_action`。 |

证据缺失时使用 `BLOCKED` 或 `REWORK`;不要晋级任务。

## 性能阶段边界

性能是功能完成之后的独立阶段。不要用优化阻塞最初的正确性工作,也不要从一次虚假
运行、单个请求,或一次把 profiler 开销当成吞吐的 profiler 运行来声称性能。

功能就绪之后,Main Agent 可以选择是否启用:

```text
固定 benchmark 矩阵
  -> 基线报告
  -> 可选 profiler 捕获
  -> trace 分析
  -> 优化子任务
  -> 正确性回归
  -> benchmark 回归
  -> 晋级或回滚
```

Benchmark 和 profiler 产物必须标明模型/插件 revision、硬件、dtype、并行度、输入/
输出长度、并发或请求速率、预热、种子和工具版本。把 benchmark/profiler 工作与功能
任务判定分开。

## 禁止的捷径

- 不要绕过 `TaskScheduler` 或手工编辑 SQLite 状态。
- 不要先修改模型/插件,再事后重构证据。
- 不要猜测算子的 shape、dtype、layout、语义或容差。
- 不要只用一段文字或 `{"status":"PASS"}` 把任务标记为完成。
- 不要仅凭编译或导入成功晋级 XPU 任务。
- 不要静默地用 Torch/CPU 回退替换 XPU 路径。
- 不要让一个算子的失败终止不相关的算子发现。
- 不要把 fake-agent 或模拟器结果报告为真实硬件证据。
- 修复运行时环境(插件 site-packages、固定工作树、Pod 内状态)时,必须提供
  `tools/patches/` 下已提交、幂等、可重放的补丁。只活在 Pod 里的修复会随 Pod 一
  起消失。
- 不要把模型权重、凭据、PAT、私有端点、原始流量或大型 trace 放进仓库。

## 交接检查清单

把工作交回 Main Agent 之前,子 Agent 必须提供:

1. 已 claim 的任务 id 和算子 key。
2. 带显式判定的机器可读结果。
3. 每个证据产物的绝对路径或 run 相对路径。
4. 使用的确切命令/环境,包括 revision 和目标设备。
5. 已知限制、未解决的假设,以及建议的下一步动作。

Main Agent 应报告 run id、任务 id、判定和证据路径,让另一个 Agent 不依赖聊天历
史就能恢复工作。
