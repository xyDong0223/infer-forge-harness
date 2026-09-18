# 配置接入

> 本文档是 `config/README.md` 的中文译本,内容以英文原版为准。

这些输入用于接入你自己的环境,而不是用来证明资源存在或某个模型被支持。

| 位置 | 配置内容 |
| --- | --- |
| [examples/](examples/README.md) | 硬件、引擎、后端、插件和工作负载 |
| [clusters/](clusters/README.md) | namespace、context、镜像、队列、节点池、PVC 和设备资源 |
| [profiles/](profiles/README.md) | 运行时 venv、site-packages 和引擎模块 |
| [manifests/](manifests/README.md) | 部署资源模板 |
| [任务实例](../tasks/) | 模型专属的部署输入和固定 revision |

## 接入环境

从 [compatibility/matrix.yaml](../compatibility/matrix.yaml) 中受支持的组合开始。
该矩阵决定平台能否执行;[catalog/support_matrix.yaml](../catalog/support_matrix.yaml)
记录模型级的验证结果。两者都不能替代当前 run 的证明。

按[真实模型适配步骤](../README.md#运行真实模型适配)操作之前:

1. 把开发集群的镜像、队列、节点池、PVC 和挂载假设替换为已确认的资源。
2. 设置外部 `KUBECONFIG`。向使用者询问其资源所有者 ID,并向 environment/proof/
   planner CLI 传 `--user-id <user-supplied-id>`(Graph:`--set
   user_id=<user-supplied-id>`);契约可以记录 `execution.user_id`。显式输入优先于
   契约和旧的 `USER_ID` 环境变量。永远不要从 host 登录名或示例资源名推断该 ID。
   核对 namespace、context、容器、deployment 类型和资源所有权前缀。
3. 对照实际镜像和网络,核对运行时路径、setup 命令和代理设置。
4. 固定模型/插件 revision,并核对集群 profile 的验证基线模型。harness 从该
   profile 生成环境契约;MAT-005 从 intake 和规划证据生成目标契约。不要复制模型
   专属的 YAML。
5. 选择持久的外部状态目录,先检查计划,并在发现之前证明环境。

`ClusterConfig.load(path)` 接受显式 profile,但一些生产调用方使用默认的 P800
profile。新增一个 YAML 不会切换所有调用方,也不会注册 adapter;不要假设存在通用
的 `--cluster-config` 选项。

## 路径与证据边界

模型路径通常指向 **Pod 内**的文件。Artifact root 指向**运行 Harness 的主机上**
的外部路径。两者保持分离;使用返回的 artifact root 或所 claim 的 workspace,不
要凭猜测预测 attempt 路径。

凭据、权重和运行证据保存在仓库之外。用显式 `--attach-pod` 重新证明已有 Pod;改
变环境身份后不要静默复用旧证据。见
[运行时写入所有权](../docs/migration/runtime-write-policy.zh-CN.md)和
[排障指南](../docs/guides/troubleshooting.zh-CN.md)。
