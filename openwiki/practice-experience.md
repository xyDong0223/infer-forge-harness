---
type: practice
title: 模型适配实践经历
summary: >-
  在 Kunlun3 P800 上适配 Qwen3、MiniMax-M2.5 和 MiniMax-M3 的可复用经验：
  先建立服务基线，再用独立证据定位 dispatch、算子、运行时和模型组合问题。
generated:
  by: hand-authored
  at: 2026-09-09T00:00:00Z
evidence_version:
  repo: https://github.com/xyDong0223/infer-forge-harness
  ref: main
  commit: f3cea1e0c1b09bbabb87df13ed5e2ea602e3004d
sources:
- repo://README.md#L17-L78
- repo://skills/model_bringup_loop/SKILL.md
- repo://skills/kernel_grade/SKILL.md
- repo://tasks/mat-008-capability-evaluation/task.yaml
- repo://tasks/mat-021-platform-kernel-correctness/task.yaml
- repo://tasks/mat-022-end-to-end-accuracy/task.yaml
- repo://tasks/mat-023-long-context-sparse-correctness/task.yaml
- repo://tasks/mat-024-operator-task-dispatch/task.yaml
- repo://tasks/mat-025-baseline-freeze/task.yaml
- repo://tasks/mat-026-operator-candidate-integration/task.yaml
- repo://tools/tensor_diff.py
- repo://tools/operator_lifecycle.py
- repo://validators/correctness_validator.py
claims: .claims/practice-experience.json
---

# 模型适配实践经历

本页记录在 Kunlun3 P800 上进行 vLLM-Kunlun 模型适配时，已经被实际排查和验证过的工程方法。它不是某个模型的支持声明，也不把一次实验结果直接推广到所有版本、芯片或 vendor build。

## 1. 先建立可工作的基线

模型适配的第一目标是建立一个可重复的 serving baseline，而不是一开始就追求所有 vendor kernel 都被使用。

推荐顺序：

```text
model intake
  -> environment proof
  -> model scan
  -> capability match
  -> gap classification
  -> service bring-up
  -> independent accuracy
  -> baseline freeze
```

如果某个 vendor kernel 不能工作，但存在可验证、可恢复的 Torch fallback，应先用 fallback 继续完成服务 bring-up。只有在没有可用 fallback、模型完全无法启动时，算子实现才应阻塞主流程。

baseline 至少绑定：

- 模型 revision 和启动配置；
- vLLM-Kunlun、PyTorch、XPU runtime 和 kernel build；
- 服务状态和启动日志；
- 独立精度结果；
- pod、端口、可见卡和资源归属；
- 可恢复的 patch 或启动方式。

没有这些信息，后续候选算子回归无法回答“到底是哪一个变化导致了结果变化”。

## 2. 先确认真实路径，再判断缺口归属

静态扫描只能说明代码或 symbol 存在，不能证明模型会走到它，更不能证明它数值正确。

遇到服务失败时，先确认：

1. 实际 import 到哪一份模型或插件实现；
2. 实际调用了哪个 symbol；
3. 输入的 shape、dtype、stride、layout 和 metadata；
4. vendor kernel 是否真的被 dispatch；
5. 失败发生在模型实现、Kunlun 插件、vendor kernel、runtime state 还是 API parser。

常见误判包括：

- 模块存在，所以认为能力可用；
- symbol 注册成功，所以认为 dispatch 正确；
- 算子离线调用成功，所以认为服务内调用也正确；
- HTTP 200，所以认为模型输出正确；
- 输出有文本，所以认为 hidden state 没有被破坏。

MiniMax-M3 的实践说明，服务可以成功启动并返回 HTTP 200，但 dense attention 的数值错误仍会在几层之后把 hidden state 破坏，最终表现为与 prompt 无关的乱码。因此服务可用性和数值正确性必须分开验收。

## 3. 正确性门禁必须使用独立证据

算子或模型组合的验证至少需要三路数据：

```text
candidate output
reference output
negative control output
```

reference 必须独立于 candidate，优先使用 CPU/PyTorch float32 实现。candidate 和 reference 不能共享同一份可能错误的 layout、scale、mask 或 indexing 逻辑。

默认比较项：

- shape；
- dtype；
- max absolute error；
- relative L2；
- NaN/Inf；
- elementwise mismatch；
- reference provenance；
- 测试 geometry 是否能够区分不同 convention。

cosine 和 norm 只能作为辅助指标。量化 scale 错误可能只改变幅值，cosine 仍然接近 1；dense attention 的 norm 也可能只差几个百分点，但逐元素 relative L2 已经完全不通过。

negative control 不是装饰。它必须故意引入一个已知错误，并且确实打掉当前 gate；否则只能得到 `AMBIGUOUS`，不能得到 `PASS`。

## 4. P800 上优先采用可验证的 fallback

在 P800 适配中，Torch fallback 的价值不只是“临时绕过错误 kernel”，还可以作为：

- 服务 bring-up 的 unblocker；
- 独立 reference；
- vendor kernel 的行为对照；
- runtime-state 问题的隔离手段；
- 后续生成算子的回归基线。

遇到 Triton kernel 不可用时，优先检查是否能用 Torch 实现复现正确语义。不要在没有明确 launch boundary 和 correctness evidence 的情况下反复调 Triton block size。

但 fallback 也必须经过：

- shape 和 layout 验证；
- 空输入、warmup 和 steady-state 验证；
- 服务内真实路径验证；
- 独立精度验证；
- 显存和性能记录。

fallback 通过不代表 vendor kernel 正确，只代表当前服务有一条可工作的路径。

## 5. cache、量化、稀疏和 MoE 要分别验证

这些维度容易在服务层混合成一个“输出不对”，但排查方式不同。

### cache layout

不能只凭 tensor shape 判断 cache 语义。TP=8 且 KV head 为 1 时，不同布局可能因为 size-1 轴而同构。

需要记录：

- pair axis；
- block axis；
- head axis；
- block size；
- key/value 顺序；
- 写入后 read-back；
- index cache 与主 K/V cache 是否使用不同布局。

### quantization

量化 scale 错误必须用 relative L2 检查。cosine 可能完全看不出逐通道缩放错误。

同时记录：

- dynamic quantization 返回的是 max 还是 scale；
- scale 是 per-token、per-channel 还是 per-tensor；
- 权重存储轴；
- dequant reference；
- saturation 和 clamp；
- candidate/control 的相对 L2。

### sparse attention

验证 geometry 必须跨过实际边界，例如 `context_len > block_size * topk`，并记录 selected blocks。只测短上下文无法证明 sparse path 真正参与了计算。

### MoE

MoE probe 要固定并记录：

- router dtype；
- top-k tie-break；
- scoring function；
- correction bias；
- expert activation；
- scaling factor；
- TP/EP 路径；
- 小 batch 和大 batch 分支。

bf16 router logits 可能制造精确 tie，使 candidate 和 reference 选择不同 expert；这不一定表示 kernel 数学错误，需要用 fp32 router 或明确的 tie-breaking geometry 重新测试。

## 6. 算子生成采用异步候选模式

算子生成 Agent 不应默认阻塞主模型适配流程。

推荐编排：

```text
gap classification
  -> operator task dispatch
  -> main Agent continues with fallback
  -> service + accuracy pass
  -> baseline freeze
  -> integrate one candidate
  -> kernel / dispatch / service / accuracy regression
  -> keep or rollback
```

算子 Agent 的职责是生成候选实现，包括 reference_op、`.xpu` Kernel、Wrapper、注册信息、UT 和 build artifact。它不能自行给生成结果判 PASS。

候选进入服务前必须有：

- independent reference；
- kernel grade；
- build report；
- actual dispatch report；
- service regression；
- end-to-end accuracy regression；
- rollback path。

一次只接入一个候选算子。多个候选同时接入会破坏责任归因，也会让 rollback 失去边界。

## 7. 证据与状态如何沉淀

每个结论至少包含：

```text
claim
status
evidence paths
environment fingerprint
source task
superseded claim（如有）
```

建议区分以下状态：

- `OBSERVED`：观察到现象，但尚未完成归因；
- `TRIAGE_READY`：已经确认归属和下一步实验；
- `KERNEL_PASS`：独立算子门禁通过；
- `DEPLOYMENT_READY`：服务健康并可请求；
- `ACCURACY_PASS`：独立组合精度通过；
- `BASELINE_FROZEN`：可作为候选回归基线；
- `AMBIGUOUS`：实验不能区分 convention，禁止继续宣称通过；
- `VENDOR_HANDOFF`：本地没有可验证修复路径。

记忆适合保存稳定方法和重要实测约束；一次实验的日志、trace、tensor dump、启动脚本和中间结论应保存在 Task artifact 或 handoff 中，并由本页引用其结论。

## 相关页面

- [模型支持](model-support.md)
- [已知缺口、死代码与文档冲突](known-gaps.md)
- [量化](quantization.md)
- [MoE 与专家并行](moe-and-ep.md)
- [测试与 CI 的真实覆盖度](testing-and-ci.md)
