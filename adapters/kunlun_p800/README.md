# Kunlun P800 adapter

这是 Kunlun P800 的 Hardware Adapter 实现，当前同时封装 Kubernetes 集群访问和
P800 设备观察。

## 提供的能力

- 从版本化 cluster config 加载 namespace、资源前缀和 kubectl 上下文；
- 查询 Pod、等待 readiness、在 Pod 中执行命令；
- 把 probe、patch 或安装脚本复制进目标 Pod；
- 读取并解析 `xpu_smi` 设备信息；
- 从 Pod 内执行 HTTP 健康检查；
- apply、delete 和 rollback 受 harness 管理的 Kubernetes 资源；
- 在所有写操作前校验资源所有权和允许的名称前缀。

调用方应通过 `adapters.get_hardware("kunlun/p800")` 或
`core.facade.resolve_adapters()` 获取实现，不直接 import 这个子包。Registry 的
规范名称以 `catalog` 和 `core.target.canonical_hardware()` 为准。

## 执行示例

```python
hardware_type = get_hardware("kunlun/p800")
hardware = hardware_type()
result = hardware.exec(pod, "xpu_smi -L")
```

实际调用通常由 Operation 或 Runner 发起。Runtime 负责生成 vLLM-Kunlun 命令，
本 Adapter 只负责把命令可靠地送入指定 Pod。

## 安全边界

读操作可用于环境证明和诊断。创建、修改、删除资源必须满足配置中的 namespace、
owner prefix 和资源类型约束；不得通过直接拼接 kubectl 绕过这些检查。删除操作只
针对当前 run 明确拥有的临时资源。

## 当前技术债务

集群传输与设备 API 目前融合在同一个实现里。未来如果多个硬件共享 Kubernetes
传输，或同一硬件支持不同集群，应拆成 Cluster Adapter 与 Device Adapter；在拆分
完成前，不要把更多推理框架语义加入这里。
