# 快速上手：本地演练与产物导览

第一次使用不必准备 P800、模型权重或 Agent API。本地 E2E 使用生产 Workflow、CLI、Scheduler 和 Validator，只替换外部集群、运行时观察和 Agent 等依赖。它展示的是编排与证据链，不是真实模型适配结果。

## 1. 安装

需要 Python 3.10+、Git，以及可安装 PyYAML/pytest 的环境：

```bash
git clone https://github.com/xyDong0223/infer-forge-harness.git
cd infer-forge-harness
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python cli/adaptation.py --help
```

已有 checkout 时直接从创建虚拟环境开始，不需要切换到某个开发分支。后续命令均在仓库根目录、激活的虚拟环境中执行，保证内部 `python3` 子进程也能找到依赖。保留 checkout：任务 YAML、catalog 和 probe 都从这里加载。

安装可能访问网络；下面的本地演练不会连接真实集群，也不需要 Torch。更多测试依赖见[测试指南](../../tests/README.md)。

## 2. 跑完整模型适配场景

```bash
DEMO_ROOT="$(mktemp -d)"
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  -m local_e2e tests/e2e \
  --basetemp "$DEMO_ROOT/pytest" \
  --junitxml "$DEMO_ROOT/model-adaptation-e2e.xml"
printf '演练产物：%s\n' "$DEMO_ROOT"
```

`pytest --basetemp` 可能清空指定目录，必须使用全新的外部位置，不能指向已有 run 或需要保留的资料。真实运行应使用持久状态目录，不使用这里的临时演练目录。

场景覆盖创建 run、环境与模型调查、持久化算子派发、跨进程 worker、租约恢复，以及最终服务和精度回归。成功场景生成 `SIMULATION_PASS`；拒绝场景会有意产生错误和 diagnosis，不能把日志中的一次失败直接等同于 pytest 失败。

`SIMULATION_PASS` 只证明内部流程能衔接，不证明设备数值、模型精度或性能。完整场景和可选真实设备层级见 [E2E 协议](../../tests/e2e/README.md)。

## 3. 找到并读懂产物

保持同一个 shell，列出本次演练生成的关键文件，不必猜 pytest 的目录名：

```bash
find "$DEMO_ROOT" -type f \( \
  -name run.json -o -name state.sqlite -o -name journal.jsonl -o \
  -name task_memory.json -o -name manifest.json \
\) -print
```

| 文件或目录 | 用途 |
| --- | --- |
| `model-adaptation-e2e.xml` | 整体测试结果，先看 failures 和 errors |
| `driver/*.log` | 每步实际命令、退出码、stdout/stderr |
| `run.json` | 当前运行的身份 |
| `state.sqlite` | Scheduler 的任务、租约和事件，通过 CLI 查询 |
| `journal.jsonl` | Graph 的事实索引、环境与证据引用 |
| `task_memory.json` | Graph 的执行位置与历史记录 |
| `tasks/*/attempts/*/output/` | 某次执行的报告和正式证据 |
| `manifest.json` | 产物清单与哈希，不替代独立验收 |

每个场景可能有独立数据库。从对应的 `driver/*.log` 读取实际 `--state` 和 `--run-id`，再查询同一场景：

```bash
python cli/adaptation.py --state /absolute/path/to/state.sqlite \
  status --run-id local-model-adaptation --events
```

最终结论应结合交付凭据的 evidence mode 和证据引用阅读；不要把中间节点的 `*_READY` 当成真实模型交付。状态解释见[结果与证据](../../README.md#结果与证据)。

## 4. 可选：体验一个数值比较工具

这个独立例子只比较本地 JSON 数组，展示候选、参考和用于识别错误的负控制，不使用模型或设备，也不创建适配 run：

```bash
DIFF_ROOT="$(mktemp -d)"
printf '[1.0, 2.0]\n' > "$DIFF_ROOT/candidate.json"
printf '[1.0, 2.0]\n' > "$DIFF_ROOT/reference.json"
printf '[9.0, 9.0]\n' > "$DIFF_ROOT/control.json"
python cli/validation/tensor_diff.py \
  --candidate "$DIFF_ROOT/candidate.json" \
  --reference "$DIFF_ROOT/reference.json" \
  --control "$DIFF_ROOT/control.json" \
  --max-relative-l2 0.02 \
  --out "$DIFF_ROOT/grade.json"
```

预期输出 `pass=true`、`relative_l2=0`、`control_discriminates=true`，同时生成 `grade.json` 和 `grade.json.manifest.json`。再次执行时换新的输出路径，不覆盖正式结果。这里的阈值只用于演示，真实算子阈值必须来自任务契约。

## 5. 转入真实执行

按[配置接入说明](../../config/README.md)准备资源、凭据、模型版本、runtime 和持久状态目录，再执行[真实模型适配步骤](../../README.md#运行真实模型适配)。

Graph 可以派发算子任务，但不会自动提供实现算子的 LLM；需要外部 worker 领取、实现、独立验证并提交证据，见 [CLI 总览](../../cli/README.md)。真实执行目前面向 P800 + vLLM-Kunlun，其他组合以[兼容性矩阵](../../compatibility/matrix.yaml)为准。

不带 `--execute` 的 Graph 是计划模式；独立 intake、probe 或验证命令并不都有 dry-run 开关，可能直接访问集群。遇到拒绝时按[排障指南](troubleshooting.zh-CN.md)保留现场，不更换 run 身份或绕过门禁。
