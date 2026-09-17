# 受管 worker 协议（P2 本地核心）

复用既有 run、task、stage、attempt、lease、metadata/events，不另建 Agent runtime 或 validator stage。新 run 显式设置 `create-run --worker-protocol managed-v2`；旧 run 保留 schema-1 旧语义，不能追认为受管验证，不能用 create-run 更换协议。

## 支持范围

支持候选目录冻结、candidate/reference/negative-control 独立进程实测、数值重新判定、三 stage 本地模拟、执行凭据和最终候选快照绑定。支持一个共享 SQLite 库内按 `cluster/namespace/pod_uid` 预留串行资源，以及本地进程组身份、续租、超时取消和保守恢复。

**尚不支持真实 Pod 的安装/修复/设备测试/toy/服务组合受管驱动、真实 XPU dispatch 和远端终止观测。** managed-v2 真实 Graph/environment 执行和真实 XPU/integration 验证明确阻断；不能退回旧 shell 或自报 PASS。本地 HTTP/数值模拟不是硬件 readiness；旧协议也不因此获得新保证。

这是 P2 的可本地验收增量，不是完整 P2。下一增量接设备侧驱动及模拟故障注入；真实设备 smoke 另需授权。资源控制只约束同库合作入口，不是 OS 沙箱，也不支持跨数据库/多主机控制器。

## 冻结候选

实现者写入当前 claim 的 `output/candidate/`，所有普通文件进入 inventory，拒绝 symlink。绑定基线 revision、整个 tree、OperatorSpec hash、环境基线 fingerprint 和 run/task/stage/attempt。候选身份与环境基线分开；不能改 fingerprint 绕过同一 Pod 的锁。

```bash
python3 cli/adaptation.py --state /external/adaptation.db freeze-candidate \
  --task-id TASK --worker CONTROLLER --lease-token TOKEN \
  --producer IMPLEMENTER --candidate-root /external/current-attempt/output/candidate \
  --base-revision PINNED_REVISION
```

controller 是实际 lease 持有者，producer 是实际实现者，validator 是独立验收者，分别记录。同一 attempt 冻结后不能改内容再重新固定；失败经原 diagnosis/retry 创建新 attempt。

## 测量契约

必须在 discovery 接受的 `OperatorSpec.semantics.validation` 内声明 `schema_version: 1`：

- `candidate_entry`：候选内 Python 相对路径，导出 `run_case(inputs) -> outputs`，按 OperatorSpec tensor 名映射。
- `reference`：独立实现的绝对 `entry`、源码 `files: {absolute_path: sha256}`、`provenance`。不能在 candidate tree 内或用无关文件 hash 冒充独立入口。
- `cases`：非空、唯一 ID；输入有 values/shape/dtype/layout，输出有 shape/dtype/layout；符号维度显式提供 bindings。
- `thresholds`：每个输出声明有限非负 `max_relative_l2`，不可通过验证 CLI 临时放宽。
- `negative_control`：`zeros` 或带非零 `value` 的 `offset`；实测必须击穿同一阈值。
- XPU/integration 另需 `expected_dispatch`、`fallback.allowed_devices`。builtin 模拟支持 symbol `run_case`、device `simulation-cpu`、ranks `[0]`；integration 另需 `service.require_http_status: 200`。

模拟浮点 list 自省为 float64，整数为 int64，不冒充低精度/XPU tensor。真实 CPU 路径读取实际 tensor 信息。不支持的 geometry/driver 明确阻断；输入标签和候选自报字段不是设备观测。

## 执行与提交

```bash
python3 cli/adaptation.py --state /external/adaptation.db validate-worker \
  --task-id TASK --worker CONTROLLER --lease-token TOKEN \
  --validator VALIDATOR --candidate-id CANDIDATE_ID --validation-id VALIDATION_ID \
  --evidence /external/current-attempt/output/evidence-input.json
```

evidence-input 是既有 stage evidence key 到当前 attempt 文件的映射，不提供自报 independent_validation。Runner 固定 recipe hash，分别执行 candidate/reference/control，记录日志和原始观测，重新计算数值与 stage 门禁。

Runner 写独立报告及附 `managed_validation_id` 的 schema-1 envelope；**不自动 complete**。控制者仍用 `complete --result ...`。接收、reconcile、Graph delivery 重新核验当前候选/参考/recipe、attempt/环境身份、证据与结果 hash、数据库凭据和三个 execution 的可信成功终态。换 validator 名、伪造 JSON 或退出码零不够。

相同 validation/execution ID 和请求只读回原凭据，不重复启动；验证完成而提交前中断时提交同一结果。执行未知不能换 ID 重跑。失败走原 diagnosis，旧 attempt 保留。

## 进程与资源恢复

`execute-worker` 接 argv，不 eval 字符串；要求当前 lease 和当前 attempt 内 output/scratch/logs。不能包装 kubectl/ssh 的退出码就宣称远端终止。
先持久化 STARTING，再记录 host/boot/PID/birth identity、日志、心跳；无 stdout 时仍续租。只取消确认身份一致的本次子进程组，残留未知则保留 UNKNOWN。

```bash
python3 cli/adaptation.py --state /external/adaptation.db execution-status --run-id RUN --active-only
python3 cli/adaptation.py --state /external/adaptation.db reconcile-execution --run-id RUN --execution-id EXECUTION_ID
python3 cli/adaptation.py --state /external/adaptation.db reconcile-validation --run-id RUN --validation-id VALIDATION_ID
```

查询只读。reconcile 只接受同 host/boot、已知 PID 且整个组已消失的实际观测；无 PID、不同主机或远端不明继续阻断。没有 force unlock 或手写 terminal 报告。lease 过期不释放进程资源、不恢复 task；先解决旧执行，再按原协议重领或诊断。

若中断发生在验证协调者内，先确认所有 child execution 已终止，再用 reconcile-validation 观察协调者的 host/PID/birth identity。只有协调者确定消失、所有子执行无未决记录、凭据未并发变化，才把未完成验证收尾为 FAIL；不重跑测量、不补写 PASS。尚活着或身份未知的协调者继续阻断。

资源 key 必须等于已绑定环境 proof 的 `resource_identity: {cluster, namespace, pod_uid}`，同库跨 run 可见。真实驱动须从可信集群观测产生 UID，不能用 Pod 名或 fingerprint 代替。
