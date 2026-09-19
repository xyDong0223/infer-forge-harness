# 自我迭代系统：运行 → 复盘 → 提炼 → 进化

> 状态：设计提案（2026-09-18）。依据 run `step35-flash-p800-003` 的完整证据。
> 证据包是运行时产物，按仓库协议**不进入版本化源码**；其不可变标识为
> tarball sha256 `8fd95917cc1e05664b5f7c6889cbb23549cd810f6d04a4c6a2eea3bc94fb7146`
> （包内 `MANIFEST.sha256` 覆盖全部 826 个文件），由运行所有者保存，
> 供第 7 节基准回放与第 10 节分期实施引用。

## 1. 问题

harness 已经有经验的**目的地**（`openwiki/harness/experiences/` 能力轴页面 +
`openwiki/harness/.claims/` 页面级 sidecar 证据绑定——目前仅有
`practice-experience.json`，能力轴页面尚无对应 sidecar）和经验的**原料**
（Journal 事实账本、Task Memory 循环块、attempt 产物），但中间缺少**明确的机制**：

- 一次 run 结束后，"总结经验"目前是 Agent 的手工动作。step35-flash-p800-003
  对 `experiences/moe.md` 的增补（sigmoid 无分组路由一节）只发生在运行侧的
  工作副本（见证据包 `harness-repo/uncommitted.diff`），**从未进入版本化源码**
  ——本 checkout 的 moe.md 不含该增补，对应的 `.claims/moe.json` 也不存在。
  没有任何任务契约要求提炼发生，也没有验证器检查它的证据绑定。
- 失败的重试热点（mat-006 分诊 11 次、mat-013 精度差分 8 次、mat-009 API
  一致性 5 次全败）只留在账本里，没有被转化为对 harness 自身的改进项。
- 能力沉淀与 harness 进化混在一起：经验页回答"下一个模型怎么做"，
  而探针修复、契约收紧、补丁归档回答"harness 下次怎么做得更好"。
  两者需要不同的产物与不同的门禁。

目标：把"执行一次 → 总结 → 分析 → 进化"定义为和适配主流程同级的、
有契约、有验证器、有证据的闭环，而不是依赖 Agent 自觉。

这个闭环在 RSI（递归自我改进）阶梯上的位置是自觉的：evo-002 是 L1
（攒经验技能），evo-003 是 L2（改工具代码）；L3（改"怎么改"）只在人工
签署下触及，L4（研发飞轮）不是目标（见第 11 节非目标）。白天是 run
执行，evo 闭环是 harness 的"睡眠"——把当天的经历蒸馏成下一个 run
可继承的形式。但**经验复利不等于能力复利**：笔记本变厚不算变强，
必须有一个可重算的度量锚点（见第 7 节能力基准）。

## 2. 闭环总览

```text
适配 run 到达图终态（DELIVERED / NEEDS_HUMAN；见 workflows/model_adaptation.yaml）
  -> evo-001 run-retrospective      机械复盘：从账本生成结构化复盘，禁止自由发挥
  -> evo-002 experience-distillation 能力提炼：可泛化结论 -> 能力轴页面 + claims
  -> evo-003 harness-evolution       自身进化：harness 缺陷 -> 改进提案任务
  -> adoption gate                   采纳门禁：测试 + E2E + 基准回放 + 提交，下一个 run 继承
```

注意：`FUNCTIONAL_READY` 是交付回执的 verdict，不是图终态；图的终态边是
`DELIVERED` 与 `NEEDS_HUMAN`（mat-016 / mat-020 的 `on_success` 直接指向
`DELIVERED`，终态之后没有可挂接的下游节点）。因此 evo-001 **不能**作为
图内下游节点触发，而是由独立的复盘入口在 run 到达终态后显式触发
（CLI 命令或外部调度），复用 run 的状态目录；重跑语义与普通任务一致：
每次执行创建新的 attempt，不回写历史 attempt。

三个 evo 节点是**持久任务**，走 TaskScheduler，与适配节点同样的
input/output/logs/manifest 结构与证据门禁。它们不改变已冻结基线，
不回写已完成的 attempt。

## 3. evo-001 run-retrospective（机械复盘）

**输入**：run 根目录（journal.jsonl、task_memory.json、各 attempt 的
manifest.json 与 status）、run-index。

**行为**：纯机械聚合，不调用模型、不做归因判断。产出：

- `retrospective.json`（结构化）+ `retrospective.md`（人读）：
  - 节点 × 尝试矩阵、最终状态、总尝试数；
  - **重试热点**：尝试数 > 1 的节点，按次数排序，附每次失败
    的 `failure_record.json` 及其 sha256 清单（按 attempt-id 排序。
    Journal 目前只为每次失败记录独立的 `failure_sha256`，没有链字段；
    若复盘需要链式不可篡改性，链的生成与校验格式必须作为 evo-001
    契约的一部分另行定义，不能假定账本已有）；
  - 未关闭项：终态非成功且无后续成功 attempt 的节点
    （本 run 即 mat-006 TRIAGE_FAILED、mat-009 UNKNOWN）；
  - 证据完整性：manifest 缺失、output 为空、哈希断链的节点清单
    （本 run 即 graph-shim-discovery、mat-008 最新 attempt 无 manifest）；
  - 时间线：各节点起止、墙钟耗时占比。
- 写入 run 的 `tasks/evo-001-run-retrospective/attempts/<attempt-id>/output/`，
  记 Journal kind `RunRetrospective`。

**验证器**（规划：`validators/` 下新增 retrospective validator）：复盘中的
每个数字必须能从账本重算（抽查式 recompute），任何与账本不符的聚合拒绝通过。
同时抽查账本条目与 attempt 产物的**哈希一致性**：journal 由产出 Agent
写入，对"乐观记录"乃至伪造日志的防线是证据绑定本身可复算，而不是信任
写入者。这一步是后两步的事实底座，所以它本身不允许"编"。

## 4. evo-002 experience-distillation（能力提炼）

**输入**：evo-001 的复盘 + 全部 attempt 产物。

**行为**：把可泛化结论提炼到能力轴页面，沿用
`openwiki/harness/experiences/index.md` 已定词汇（与 mat-003/mat-008 的
capability axes 同源）：

- 模型名只作 `first_seen` 引用，不作经验归属单位；
- 每条新增/修改的段落必须在页面 frontmatter 的 `sources` 和对应
  `.claims/<page>.json` 中绑定证据（`repo://` 路径或 run 产物哈希）；
- 被新证据推翻的旧结论不删除，按 Task Memory 的 `supersedes` 模式标注取代
  关系（如"后置包装可用"被 step35 的循环导入证据推翻后保留为反模式）；
- **未覆盖项必须显式书写**（沿用 moe.md"覆盖边界"的诚实惯例）。

**产出**：对 `openwiki/harness/experiences/*.md` 与 `.claims/*.json` 的
版本化源码改动 + `distillation_report.json`（每条结论 → 目标页面 → 证据列表）。
记 Journal kind `ExperienceDistillation`。

**验证器**：证据引用按类型分支校验——`repo://` 引用检查仓库路径真实存在；
run 产物引用（attempt 产物路径或 sha256）检查其在账本中的 artifacts/哈希
绑定存在，而不是当作文件路径检查。`first_seen` 引用的 run/attempt 在账本中
存在；页面 diff 只触及声明的段落。无证据绑定的段落不许落地。

本 run 的 moe.md 增补（sigmoid 路由两处 glue 缺陷 + 反模式 + accuracy 门禁
语义）是这一步的**人工实例**：它证明了提炼的内容形态可行，但它没有走契约、
没有 sidecar claims、也未进入版本化源码——evo-002 要补的正是这三件事。

## 5. evo-003 harness-evolution（自身进化）

**输入**：evo-001 的重试热点与未关闭项 + evo-002 的结论。

**行为**：把"harness 自身的问题"与"模型的问题"分离，每类改进生成一个
独立提案任务（一次一个，沿用算子接入的单一变更原则）：

| 来源（本 run 实例） | 进化方向 |
|---|---|
| mat-006 分诊 11 次仍未捕获 `speculative_attention` 调用参数 | 探针增强提案：失败调用参数捕获能力本身补全 |
| mat-009 api-conformance 5 次全 ERROR | 任务契约/实现修复提案 |
| mat-013 阈值与被压制尾部重排不兼容，靠人工签署收口 | 门禁度量提案：质量集中 + 尾部 reshuffle 的分类指标（而非放宽阈值） |
| graph-shim-discovery / mat-008 manifest 缺失 | 验证器收紧提案：缺 manifest 即拒绝 |
| 运行时漂移 6 处已记录未修 | 漂移地图更新（`vllm-0251-drift-map.md`），逐个转为补丁候选或 VENDOR_HANDOFF |

**产出**：每个提案是 `tasks/evo-003-*/instances/<proposal-id>.yaml` +
结构化 `proposal.json`（问题、证据、建议改动面、回滚路径、预期收益）。
记 Journal kind `HarnessEvolution`。

**提案分级**（对应 RSI 阶梯，门槛逐级升高）：

| 级 | 内容 | 门禁 |
|---|---|---|
| L1 | 经验/文档类：能力轴页面、漂移地图、claims | 普通 adoption gate |
| L2 | 工具代码类：探针、契约、任务、补丁 | 普通 adoption gate + 能力基准不回退（第 7 节） |
| L3 | 改"怎么改"：evo 流程自身、验证器与门禁的松紧 | 必须人工签署（偏差接受通道，见第 5 节硬约束 2），与放宽门禁同级 |

**硬约束**（防止自我进化退化为自我放水）：

1. 提案是**任务**，不是直接改代码。落地走正常源码改动流程，
   带任务上下文与证据。
2. 任何**放宽既有门禁**的提案（阈值、验证器、契约）必须走人工签署通道，
   且目录中保留原始失败记录——机器无权批准自己变宽容。本 run 首次使用了
   这样的通道（偏差接受任务 mat-030 + 验收记录 CLI），但其任务定义与实现
   只存在于运行侧未提交状态（证据包 `harness-repo/untracked/`），**本仓库
   尚无此机制**；把它正式入仓是 evo 一期的前置工作，在此之前 L3 与门禁
   放宽提案只能停留在提案态。
3. 增强类提案（新探针、新检查）只需证明它在本 run 证据上会做出正确判断
   （回放式验证）。

## 6. adoption gate（采纳门禁）

进化改动进入版本化源码前：

```text
proposal accepted
  -> 源码改动（cli/operations/runners/validators/...，按 source ownership）
  -> 单元测试
  -> tests/e2e/ 场景（真实 scheduler + 验证器，替换外部依赖）
  -> 能力基准回放（第 7 节，指标不回退）
  -> 提交（不允许像本 run 这样 27 改动 + 7 新文件游离在 HEAD 之外）
  -> 下一个 run 通过版本化源码继承
```

E2E 场景按 `tests/e2e/README.md` 的既有协议新增一个"运行后复盘"模板：
模拟一个含失败与重试的 run，断言 evo-001 产出的复盘数字与注入的账本一致、
evo-002 的每条 claim 都有证据绑定、evo-003 的提案不静默改门禁。
**不得预置成功报告**——与本仓库所有场景同一规则。

## 7. 能力基准：进化的度量锚点

autoresearch 式循环能转起来，靠的是每圈改动都有一个几分钟可重跑的指标
裁判。evo-003 的回放式验证只证明"提案在旧证据上判断正确"，并不回答
"采纳之后 harness 做下一个模型是否真的变强"。缺了这个锚点，闭环产出的
只是越来越厚的笔记本（经验复利），而不是能力复利。

**固定考卷**：已归档的证据包（step35-flash-p800-003 是第一份）固化为
基准。度量分两层：

- **便宜层（每次提案必跑，分钟级）**：在隔离状态目录中以仿真模式回放
  归档 run 的账本与任务序列（外部集群/运行时依赖按 e2e 协议替换），
  比较可重算指标——总尝试数、重试热点次数、未关闭项数量、证据完整性
  缺陷数（manifest 缺失、哈希断链）。要求：不高于基线。
- **昂贵层（周期性，真实 run）**：下一个真实适配 run 按同样口径度量，
  与历史 run 对比墙钟时间与尝试数。真实层回答"变强了没有"，便宜层只
  保证"没有变蠢"。

**判定**：L2/L3 提案进 adoption gate 前必须跑出"不比基线差"的便宜层
数字；指标不进不退的提案降级为可选美化，不与正确性/健壮性类提案同
优先级。考卷只增不改：新的已签署 run 归档后追加为新考卷，旧考卷保留
以测回归。

## 8. 源码归属（按 source-layout 协议）

| 内容 | 位置（均为规划新增） |
|---|---|
| CLI 入口（仅参数解析） | `cli/evolution/` |
| 任务行为 | `operations/evolution/`（retrospect / distill / evolve） |
| 工作流编排 | `runners/` 下新增 evolution runner；**不挂入** `workflows/model_adaptation.yaml` 图内（终态 `DELIVERED` 之后无下游可挂），由独立 CLI 入口在 run 到达终态后显式触发 |
| 任务契约 | `tasks/evo-001-run-retrospective/`、`evo-002-experience-distillation/`、`evo-003-harness-evolution/` |
| 验证器 | `validators/` 下新增 retrospective / distillation / evolution 三个验证器 |
| Journal 新 kind | `RunRetrospective` / `ExperienceDistillation` / `HarnessEvolution` |
| 目录登记 | `catalog/skill_catalog.yaml`、`tool_catalog.yaml` 同步更新 |
| E2E | `tests/e2e/` 新增复盘场景 |

`tools/` 不承载任何进化协调逻辑（协议红线）。

## 9. 与既有机制的关系

- **Journal / Task Memory** 是原料，不改动其 schema；evo 节点只是新的
  writer/reader。
- **能力轴经验页**是 evo-002 的目的地，词汇与结构沿用既有约定。
- **偏差接受的人工签署**在本 run 中首次以任务形态使用（mat-030），但其
  定义未入仓；evo-003 的门禁放宽与 L3 提案复用同一签署模式，正式契约
  随第一期落地。
- **openwiki 不覆盖任务契约**的原则不变：经验页指导 Agent 选择路径，
  但每个 evo 节点的通过与否只由契约与验证器判定。

## 10. 分期实施

1. **evo-001 + 验证器 + E2E**：机械复盘，纯聚合，风险最低，先落地。
   用 step35-flash-p800-003 的证据包做回放验证（预期产出：重试热点
   mat-006×11 / mat-013×8 / mat-009×5，未关闭项 2 个，manifest 缺失 2 处）。
   同期把偏差接受通道（mat-030 任务定义 + 验收记录 CLI，目前在证据包
   `harness-repo/untracked/`）正式入仓，作为人工签署先例。
2. **evo-003 + 能力基准便宜层**：提案生成与人工签署通道，把本 run 的 5 个
   进化方向转为首批提案实例；同时把 step35-flash-p800-003 证据包固化为
   第一份基准考卷。
3. **evo-002**：提炼与 claims 校验，把本 run 已手工完成的 moe.md 增补
   走一遍机器验证，补齐 `.claims/moe.json`。

## 11. 非目标

- 不做跨 run 的自动权重/阈值调参；门禁松紧永远由人签署。
- 不把一次实验的日志、trace、tensor dump 搬进经验页（沿用
  practice-experience.md §7：记忆存方法，证据留 artifact）。
- 不让 evo 节点修改它所属 run 的既有 attempt 或冻结基线。
- 不追求 L4 研发飞轮：目标、指标与合并按钮始终握在人手里；
  本系统只做到"有门禁的 L1/L2、人工签署的 L3"。
