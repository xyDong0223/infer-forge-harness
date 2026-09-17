# 源码归类与依赖边界

本次整理采用完整迁移，不保留旧宿主机脚本入口或 Python import 兼容别名。
任务行为、证据门禁、退出码和外部运行目录规则保持不变；改变的是源码归属、
命令位置和模块调用方式。

## 目录职责

| 目录 | 应放什么 | 不应放什么 |
| --- | --- | --- |
| `cli/` | 宿主机参数解析、命令入口、输出路径保护、仓库维护命令 | 重复的任务算法和验收规则 |
| `operations/` | 可被调用和替换依赖的任务实现 | `parse_args()`、宿主机入口、调度数据库 |
| `engine/` | 调度器、任务状态机、诊断恢复、Skill registry | 设备算子实现、CLI |
| `engine/state/` | Journal 和 Task Memory 的读写、检索与状态操作 | 临时实验脚本 |
| `runners/` | Graph、部署、性能、诊断、补丁和 correctness 执行序列 | 参数解析与入口包装 |
| `core/` | 共享契约、Target、错误类型、源码资源定位、运行存储策略 | 厂商补丁、某个模型的任务逻辑 |
| `adapters/` | 硬件和集群操作边界 | 任务路由 |
| `runtimes/` | 推理运行时和插件的装配、命令与适配 | 验收规则 |
| `validators/` | 独立的契约与证据验收逻辑 | 生成候选实现或主动修复环境 |
| `tools/probe/` | 可传入 Pod 独立执行的探针 | 宿主机编排和状态管理 |
| `tools/patches/` | 可重放、幂等的运行时补丁及其安装工具 | 一次性临时修复 |
| `tools/torch/` | 可独立传输的 Torch 参考实现 | 通用宿主机 CLI |

`tasks/` 仍然保存任务契约、实例与验收材料，不是 Python 业务实现目录。
`workflows/`、`catalog/`、`config/` 与 `skills/` 仍然是版本化声明和工程方法。
这些目录中的命令引用已经同步到新的入口位置。
运行时安装脚本归属 `runtimes/scripts/`，由 Runtime 的 `installer_path()`
统一定位；不再与宿主机任务入口混在 `tools/` 顶层。

## 任务实现按领域分类

`cli/` 的任务命令与 `operations/` 的实现使用相同领域目录：

| 领域 | 任务范围 |
| --- | --- |
| `intake/` | 模型身份、revision、checkpoint intake |
| `discovery/` | 模型扫描、runtime drift、shim、能力匹配与评估、缺口分类 |
| `deployment/` | 内存预算、部署计划、toy bring-up |
| `validation/` | API conformance、accuracy differential、tensor diff、支持矩阵更新 |
| `operators/` | 算子生命周期与厂商交接 |

例如 `cli/discovery/scan_model_support.py` 解析参数并调用
`operations/discovery/scan_model_support.py` 的 `execute(args)`。
该实现保留原来的扫描、产物生成和 validator 调用；Python 使用者直接
导入实现模块，不再导入 CLI。

当前 `execute(args)` 保留原参数对象的字段形状，以避免搬迁时改变任务语义。
它不解析 `sys.argv`。这不是已经把所有任务输入改成统一 dataclass 的承诺。

Runner 也按相同边界拆分：CLI 解析参数，执行序列留在 `runners/`，
通过 `run(args)` 或既有的 `execute(...)` / Runner 类供程序调用。

## 主要命令入口

| 用途 | 当前入口 |
| --- | --- |
| 创建/恢复 AdaptationRun、环境绑定、worker 领取和完成 | `cli/adaptation.py` |
| Graph 规划与执行 | `cli/workflow/graph.py` |
| 环境/服务部署证明 | `cli/deployment/proof.py` |
| 诊断、补丁放置 | `cli/operators/triage.py`、`cli/operators/place_patch.py` |
| Correctness 执行序列 | `cli/validation/correctness.py` |
| Journal、Task Memory 命令 | `cli/state/journal.py`、`cli/state/task_memory.py` |
| 引用与目录边界检查 | `cli/maintenance/check_repo_references.py` |
| Scaffold 检查 | `cli/maintenance/validate_scaffold.py` |

可直接运行脚本，也可从仓库根目录使用模块形式：

```bash
python3 -m cli.adaptation --help
python3 -m cli.workflow.graph --help
python3 -m cli.discovery.scan_model_support --help
python3 cli/maintenance/check_repo_references.py
```

安装后的 `run-adaptation` 命令仍然保留，但其入口已经改为
`cli.adaptation:main`。已有 editable installation 需要重新安装，以刷新
入口和子包发现配置：

```bash
python3 -m pip install -e '.[test]'
```

run、worker 队列与租约只有 `cli/adaptation.py` 一个公开管理入口。原简化
scheduler 入口已移除；全库和按 run 的任务查询使用其只读 `list` 子命令。
原数据库、run ID、attempt 和原始证据不迁移、不改写；CLI 参数及 JSON 输出
迁移差异见 [CLI 总览](../../cli/README.md#状态与执行边界)。

外部自动化若记录了旧脚本路径或 import 路径，必须显式改用新入口；
本次不创建兼容跳板，也不重写旧运行记录中的命令和证据。
历史运行证据仍属于其原始源码 revision，重现时应使用对应版本。

## 源码路径与运行路径分开

`core.paths.REPO_ROOT` 定位版本化资源，例如任务契约、catalog 和 probe。
业务模块不再各自根据所在目录猜测仓库根目录。CLI 为支持直接脚本调用保留
最小的 import bootstrap；它不是运行输出根目录。

运行输出继续由 `core.storage.RunPaths` 和 `ArtifactStore` 管理。
不要把 `operations/` 当成新的实验输出目录，也不要因为代码搬迁重新创建
AdaptationRun 或已经准备好的 Pod。详见
[运行期写入规则](../migration/runtime-write-policy.zh-CN.md)。

## 自动检查约束

仓库引用检查负责检查当前源码引用、CLI 调用和关键依赖方向：

- 宿主机命令不得重新出现在 `tools/` 顶层。
- `tools/` 的子目录只能是约定的三类，不再任意新增杂项目录。
- 核心库、任务实现、状态模块和执行器不能导入 `cli`，不能解析命令行，
  也不能重新加入可执行入口。
- 当前命令、配置和文档引用的源码文件必须存在。

检查跳过虚拟环境、缓存、运行产物、上游参考资料和历史 `*-findings.yaml`
证据，避免把外部历史路径当作当前可执行入口。测试同时覆盖每个 CLI 的
独立启动、业务模块的调用以及原有的设备/运行时依赖边界。
