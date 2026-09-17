# 推理软件栈适配（Runtimes）

这里的 **Runtime** 不是“正在运行的进程”，也不是 run、task 或 scheduler 的
运行状态。它表示模型服务依赖的一整套推理软件栈，例如：

```text
vLLM + vLLM-Kunlun + 对应 Python 环境和安装方式
```

`runtimes/` 把这套软件栈特有的知识集中起来，使部署流程不需要到处判断
“当前是 vLLM 还是 SGLang”“插件叫什么”“服务命令应该怎样拼”。

## 这一层负责什么

当前 `VllmKunlunRuntime` 负责：

- 选择 Python 虚拟环境和 site-packages；
- 定位该软件栈的安装脚本；
- 生成 import、版本和代码 revision 检查命令；
- 根据统一的 server 配置生成 vLLM 服务启动命令；
- 定义能够说明服务退回 CPU/Torch 路径的日志标记；
- 生成软件栈环境指纹所需的命令。

它回答的是：**“在已经选定的执行环境中，这套推理软件应该怎样安装、检查和
启动？”**

## 它不负责什么

| 问题 | 负责位置 |
| --- | --- |
| 命令在哪个 Pod 中执行、怎样复制文件、怎样访问 XPU | `adapters/` |
| 安装、检查、启动、探测按什么顺序执行 | `runners/` |
| 任务何时调度、重试或进入诊断 | `engine/` |
| 输出和证据是否满足验收条件 | `validators/` |
| 模型、集群和部署参数取什么值 | `config/` |
| 可重放的 site-packages 或源码修复 | `tools/patches/` |

例如，Runtime 生成 import 检查命令，Hardware Adapter 把命令送进 Pod，Runner
决定何时执行，Validator 再判断留下的证据是否通过：

```python
command = runtime.import_check_command()
result = hardware.exec(pod, command)
```

## 当前文件

| 文件 | 职责 |
| --- | --- |
| `registry.py` | 把 catalog 中的规范名称解析成具体实现；声明但未实现的 Runtime 会明确失败 |
| `vllm_kunlun.py` | `vllm-kunlun` 软件栈的环境、检查和启动命令 |
| `scripts/install_vllm_kunlun.sh` | 可复制到目标环境执行的安装脚本 |
| `../config/profiles/p800-vllm-kunlun.yaml` | 环境路径等可配置身份；不是行为实现 |
| `../catalog/runtime_catalog.yaml` | 已声明的软件栈及其支持硬件；声明本身不代表实现完成 |

目前只有 `vllm-kunlun` 真正接入。配置中出现 SGLang 或其他 engine，不表示相应
Runtime 已经实现。

## 调用路径

```text
TargetContext
  -> core.facade.resolve_adapters()
  -> runtimes.registry.get_runtime("vllm-kunlun")
  -> VllmKunlunRuntime
  -> Runner 调用其安装、检查或启动命令
  -> Hardware Adapter 在 Pod 中执行
```

新增软件栈时，应新增独立实现并在 `registry.py` 显式登记。不要把新框架的命令
继续塞进 `VllmKunlunRuntime`，也不要在通用 Runner 中增加散落的框架名称判断。

## 关于目录名称

更直白的名称可以是 `inference_stacks/`。目前仍保留 `runtimes/`，因为 Runtime
已经是 catalog、协议和配置中的领域名称，直接迁移会触及大量持久化契约和引用。
在决定全仓改名以前，以“推理软件栈适配”理解这个目录。
