# Adapters：硬件与集群操作边界

`adapters/` 隔离外部执行环境的差异。它负责把“在目标环境执行命令、复制文件、
查询设备、创建或删除受管资源”等请求转换成具体平台操作。

## 与 Runtime 的区别

```text
Runtime Adapter：要执行什么软件栈命令
Hardware Adapter：在哪里、通过什么平台机制执行
```

例如，`VllmKunlunRuntime` 生成 vLLM 的启动命令，`KunlunP800Adapter` 使用
kubectl 在指定 Pod 中执行它。

## 当前入口

调用方通过 `adapters.get_hardware(canonical_name)` 获取实现，不直接 import
厂商子包。当前接通的实现是 `kunlun-p800`；名称解析与兼容性检查由
`core.target` / `core.facade` 完成。

| 位置 | 职责 |
| --- | --- |
| `__init__.py` | Hardware Adapter registry、公共导出和规范名称到实现的映射 |
| `kunlun_p800/adapter.py` | 集群配置、kubectl 操作、Pod exec/file push、XPU 查询和写操作安全门禁 |
| `kunlun_p800/README.md` | P800 实现的具体能力与当前边界 |

## 安全与所有权

写操作必须核对 namespace、资源名前缀和受管对象所有权。读取操作可以用于诊断，
创建、修改和删除资源必须经过 Adapter 的安全门禁。凭据、KUBECONFIG、模型权重
和私有地址不写入仓库。

## 不应放在这里

- vLLM/SGLang 安装参数和服务启动语义：放 `runtimes/`；
- Task 的动作顺序与重试：放 `runners/`；
- 输出是否通过：放 `validators/`；
- 调度状态：放 `engine/`；
- 一次性 shell 拼接：提炼为 Adapter 方法或便携 probe。

新增硬件实现时，先登记 compatibility/catalog，再实现公共 Protocol 所需能力，
最后用契约测试证明通用层没有直接依赖厂商模块。
