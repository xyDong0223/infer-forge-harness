# 贡献指南

> 本文档是 `CONTRIBUTING.md` 的中文译本,内容以英文原版为准。

## 变更边界

每项变更必须标明它修改的是 Contract、Workflow、Task、Skill、Tool、Runner、
Adapter、Validator、Catalog 还是文档。不要在一次变更中混合不相关的层级。

契约变更需要迁移说明和回归夹具(fixture)。Skill 变更需要 Golden Task,或明确说
明为什么无法提供夹具。工具变更需要 fake adapter 或确定性契约测试。Adapter 变更
需要环境范围内的集成证据。Validator 变更必须保持对已知坏状态的拒绝行为。

## 源码归属

host 命令入口属于 `cli/`,任务实现放在对应的 `operations/` 域。调度属于
`engine/`,协调状态属于 `engine/state/`,可执行序列属于 `runners/`。库不得导入
`cli`,也不得解析命令行参数。

`tools/` 只保留可移植 probe、可重放补丁和 Torch 参考。版本化资源位置使用
`core.paths`,外部运行时写入使用 `core.storage`。不要在已删除的路径上新增兼容
脚本。目录映射和要求的调用方更新见
[源码归类说明](docs/architecture/source-layout.zh-CN.md)。

## 运行时与机密

永远不要提交模型权重、token、私有端点、原始生产流量或大型 trace。使用外部
artifact root,只提交脱敏后的摘要和校验和。

## Pull Request

使用聚焦的 PR 标题前缀,如 `[Contract]`、`[Workflow]`、`[Task]`、`[Skill]`、
`[Tool]`、`[Adapter]`、`[Validator]`、`[Test]` 或 `[Docs]`。正文中包含任务
ID、变更层级、已运行的测试、已知限制和可复现命令。

## 本地检查

每一项新支持的能力必须包含一个可执行的本地 E2E 场景,并从其生产 workflow 链接。
使用真实的 CLI、workflow、调度器、验证器和持久化产物;只对集群/运行时/Agent 等
外部边界使用替身。覆盖成功交付、拒绝和进程重启。真实设备 smoke 和真实模型回归
是单独的可选层级,不是本地场景的前置条件。见
[能力场景](tests/e2e/README.md)。

```bash
python -m pytest -q tests
python cli/maintenance/check_repo_references.py
python -m pytest -q -m local_e2e tests/e2e
```

P800 集成测试为显式 opt-in,执行前必须标明 namespace、镜像 digest、模型
revision、硬件和清理策略。
