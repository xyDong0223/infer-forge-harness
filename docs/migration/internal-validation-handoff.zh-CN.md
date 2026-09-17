# 内网验证与反馈交接

本地开发者不需要访问企业集群。内网 Agent 使用同一份源码、既有 run 和准备好的 Pod 执行已授权的验证；外部开发者只核对最小反馈并继续修复代码。没有新的远程调度器，也没有外部 PASS 导入接口。

## 当前边界

- `validation-plan` 只生成计划，不连接集群、不打开 scheduler 数据库、不执行命令。
- `feedback-export` 读取指定 JUnit，按字段白名单生成分享包，不复制原始日志。
- `feedback-check` 离线核对请求、版本、文件完整性与报告一致性，**不证明另一台机器真的执行过、不完成 task、不 promote run**。
- `managed-v2` 的真实 Graph/environment、XPU dispatch/fallback 和模型服务受管驱动仍未接通，继续 `BLOCKED`。不能换成 legacy、手填 PASS 或修改数据库来通过。
- 新增的进程 supervisor 是待接入的基础组件。它不构成可信 Pod 身份证明，不持有 scheduler lease，不负责跨 run 资源占用；不能单独用它执行正式适配任务。

## 交给内网 Agent 的工作清单

1. 核对源码版本和工作树差异，先跑必选本地回归。不要以真实设备测试被 skip、pytest exit 0 作为设备通过。
2. 按 [真实测试配置](../../tests/e2e/README.md) 准备外部 scenario JSON。使用已有 `run_id`、数据库、artifact root、Pod 和固定 revision；`user_id` 必须由用户提供，不能从机器登录名推断。
3. 生成一个 tier 的计划。计划中的私有路径、环境和命令只留在内网。开始执行前确认用户确实授权该 tier；`real_model` 不随 `device_smoke` 自动启动。
4. 使用计划固定的测试入口，设置 `INFER_FORGE_VALIDATION_REQUEST` 指向生成的 request，以及 `INFER_FORGE_HARDWARE_SCENARIO`。只在授权后设置对应的 `INFER_FORGE_RUN_DEVICE_SMOKE=1` 或 `INFER_FORGE_RUN_REAL_MODEL=1`。不设置测试替身、成功预填或本地 E2E bootstrap。
5. 保留原始 JUnit、stdout/stderr、attempt 和 scheduler 证据在内网，记录 pytest 的实际退出码。失败先诊断；不替换 Pod、不自动修复环境、不降级协议。
6. 导出反馈，把公开 request 与分享目录的两个 JSON 文件交回。详细 traceback 如需分享，另行审阅脱敏，不直接上传数据库、原始日志或模型输入输出。

```bash
python3 cli/adaptation.py validation-plan \
  --scenario /external/private/hardware-scenario.json \
  --tier device_smoke --out /external/fresh-validation-plan

# 内网 Agent 检查 private plan，取得授权后执行其中固定测试命令。
# 不要把整个 private plan 发送到外部。

python3 cli/adaptation.py feedback-export \
  --request /external/fresh-validation-plan/public/request.json \
  --junit /external/internal-results.xml --exit-code 0 \
  --out /external/fresh-share-packet

python3 cli/adaptation.py feedback-check \
  --request /external/fresh-validation-plan/public/request.json \
  --feedback-dir /external/fresh-share-packet
```

示例的 `--exit-code 0` 必须替换成实际退出码；不要为了导出结果修改它。输出目录必须在源码外且未使用过。源码或 scenario 改动后需要新计划，不复用旧请求。

真实硬件 fixture 在执行前后校验 request、scenario 与实际源码 tree，并将绑定属性写入 JUnit。只检查 commit SHA 不够：未提交的代码也参与 tree 哈希。目标 case 未执行、skip、缺少绑定或执行中版本变化，均不能生成“测试已通过”的观测。即使存在完整绑定与通过观测，仍只是待核实的外部反馈；哈希不是签名或设备认证。

分享包默认只含 `feedback.json` 和 `manifest.json`。原始 JUnit 只保留大小与哈希；不包含错误文本、绝对路径、用户/Pod 名称、端点、环境变量或模型请求内容。未知字段不透传。这是最小化导出，不是对任意原始材料的自动脱敏保证。

## 进程协议组件的内部检查

`cli/deployment/managed_process.py` 提供 `start / inspect / cancel`，供下一步远端驱动对接和独立基础测试使用。三个操作都需要同一个外部 `--root` 与不可变 `--request` JSON：

```json
{
  "schema_version": 1,
  "execution_id": "process-protocol-smoke",
  "nonce": "unique-request-nonce-0001",
  "argv": ["python3", "-c", "print('process protocol only')"],
  "cwd": "/external/dedicated-process-test",
  "timeout_seconds": 30
}
```

首次 start 持久化请求后启动独立监控进程；重复同一请求不会再启动。取消标记阻止迟到的 start，只有确认对应进程组结束才返回终态。inspect 在新调用进程中读取已有结果；缺失状态、身份不明或残留子进程保留 UNKNOWN，不伪造成功。`EXITED`、`returncode: 0` 只表示进程结果，不表示算子或模型通过验收。

监控进程死亡后，inspect 不自动修复或发信号。显式重复 cancel 在取消请求满两秒后可升级为 SIGKILL，但仍须匹配原 PID、启动身份和进程组；身份不明时不能强行清除状态。内网 Linux 可先运行无模型依赖的组件回归：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider \
  tests/unit/test_managed_process.py tests/e2e/test_process_protocol.py \
  --basetemp /external/fresh-process-pytest \
  --junitxml /external/process-protocol.xml
```

组件报告不属于 hardware feedback 的固定 case，不能喂给 `feedback-export` 冒充设备测试。先反馈通过/失败数量、源码版本及经审阅的失败摘要；原始 JUnit/日志留内网，按需要进一步提供脱敏证据。

不要将该组件包装在 `execute-worker` 中后把本地 kubectl 的退出当作远端已终止。仍需完成可信 Pod UID/容器 incarnation 采集、启动前账本预留、传输与恢复、lease 和资源锁联动，才能接入正式设备任务。此基础检查不需要模型权重或 XPU，不能替代真实设备 tier。

## 反馈后的下一步

外部开发者根据固定 reason code 和版本绑定先定位代码问题；需要更多证据时只请求指定的脱敏片段。修复后生成新源码版本与新验证请求，由内网原 scheduler 按既有协议重试。不得将外部反馈直接转换为 `complete`、`FUNCTIONAL_READY` 或设备 readiness。
