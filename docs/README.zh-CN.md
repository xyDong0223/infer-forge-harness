# 文档导航

> 本文档是 `docs/README.md` 的中文译本,内容以英文原版为准。

- [本地快速上手](guides/quickstart.zh-CN.md):安装、完整本地场景、产物检查,以
  及一个可选的张量对比示例。
- [CLI 指南](../cli/README.md)和[配置接入](../config/README.md):命令选择、
  worker 交接和接入你自己的环境。
- [排障指南](guides/troubleshooting.zh-CN.md):常见拒绝、诊断恢复和 CPU 参考边
  界。
- [`architecture/source-layout.zh-CN.md`](architecture/source-layout.zh-CN.md):
  当前源码归属、命令/实现分离和迁移规则。
- [`architecture/implementation-layers.zh-CN.md`](architecture/implementation-layers.zh-CN.md):
  当前中文技术实现指南,解释每个抽象层、目标解析、执行流程和实现边界。架构从这
  里开始;第一次运行用上面的快速上手。
- [`architecture.md`](architecture.md):更早的分层契约、目标轴和组件归属图,含
  历史迁移说明。
- [`architecture/`](architecture/):平台中立的核心模型、adapter/契约规则和目标
  解析。
- [`guides/`](guides/):如何新增平台、性能分析如何组织,包括中文的
  [新增 workflow 接入指南](guides/add-workflow.zh-CN.md)和
  [旧 Skill 迁移指南](guides/migrate-legacy-skill.zh-CN.md)。
- [`migration/`](migration/):把已有的脚本化流程迁入 harness。
- [`migration/runtime-write-policy.zh-CN.md`](migration/runtime-write-policy.zh-CN.md):
  外部运行目录、按 attempt 的写入所有权和产物清单。
- [`assets/`](assets/):图示,含可编辑源文件。

仓库规则以根目录 `AGENTS.md` 为准(中文译本见
[AGENTS.zh-CN.md](../AGENTS.zh-CN.md));本目录解释这些规则为什么是这样设计的。
