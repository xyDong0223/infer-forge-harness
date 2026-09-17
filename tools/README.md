# Tools：可传输工具、可重放补丁与参考实现

`tools/` 中的内容可以脱离宿主机编排层，被复制到 Pod 或独立环境执行。这里只允许
三类明确资产：

| 目录 | 内容 | 典型调用者 |
| --- | --- | --- |
| `probe/` | 只观察并输出结构化结果的便携探针 | Operation / Runner |
| `patches/` | 幂等、可重放、可检查的源码或安装修复 | Patch Runner / 部署诊断 |
| `torch/` | 可独立传输的 PyTorch reference | 正确性与算子任务 |

宿主机命令入口放 `cli/`，任务实现放 `operations/`，执行序列放 `runners/`，Runtime
安装器放 `runtimes/scripts/`。不要新增顶层任务脚本或 `utils/`、`misc/` 一类无法
说明所有权的目录。

## Probe 要求

- 输入通过参数或明确文件传入，不读取 Scheduler 内部状态；
- stdout 输出稳定的机器可读结果，诊断细节写 stderr 或报告字段；
- 不修改目标环境，除非工具契约明确说明并由上层授权；
- 输出记录 shape、dtype、版本、调用路径等复现所需身份。

## Patch 要求

- 必须幂等：已应用时明确报告 SKIP；
- 必须使用精确 anchor，源码漂移时失败而不是猜测修改；
- 保存修改前后差异、适用 revision 和验证证据；
- 只在观察到匹配的不兼容后执行，不能作为无条件安装步骤。

## Torch reference 要求

Reference 应独立表达算子语义，不复用候选实现中可能错误的布局、scale 或索引逻辑。
它必须能被独立 Validator 使用，并覆盖实测 OperatorSpec 中的相关边界。

完整源码归属规则见[源码布局](../docs/architecture/source-layout.zh-CN.md)。
