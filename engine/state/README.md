# Engine State：Journal 与 Task Memory

这里保存两类辅助状态。它们都持久化到 run 的外部目录，但用途不同，也不能互相
替代。

| 组件 | 回答的问题 | 不负责什么 |
| --- | --- | --- |
| `journal.py` | 哪个任务在什么环境产生了什么事实和证据？ | 不决定任务是否进入下一阶段 |
| `task_memory.py` | Agent 当前做到哪个 loop block，下一步准备做什么？ | 不证明实现正确，也不持有 worker 租约 |

Scheduler 的 SQLite 数据仍然是 run、task、lease 和 diagnosis 状态的权威来源；
Validator 与正式 evidence 是正确性结论的权威来源。

## Journal

Journal 是追加式事实记录。Graph 用它查找可复用的成功事实，但只有在输入、环境
指纹和证据仍然匹配时才能复用。不要删除或改写旧记录来掩盖失败。

## Task Memory

Task Memory 把长时间 Agent 工作拆成 loop block，记录局部目标、退出条件、执行
结果、观察到的问题和下一 block。它的作用是让中断后的 Agent 恢复工作上下文，
不是另一个工作流引擎。

## 写入规则

- 文件必须位于源码仓库之外的 run 目录。
- 使用模块提供的原子保存函数，避免进程中断留下半份 JSON。
- 证据只保存路径引用；正式文件仍写入当前 attempt 的 `output/`。
- 已完成 block 和历史事实应保留，新结论通过追加或 supersedes 表达。
