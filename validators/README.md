# Validators：独立验收门禁

`validators/` 只根据任务契约和持久化证据判断结果是否完整、一致、可接受。
Validator 不生成候选实现、不主动修复环境，也不因为命令退出码为 0 就判定 PASS。

## 返回约定

当前 Validator 通常返回 `list[str]`：空列表表示报告满足结构和验收规则；每个字符串
描述一个具体拒绝原因。业务状态为 FAIL 的报告仍可能是结构合法的报告，因此
“报告格式有效”和“任务通过”必须分别判断。

## 文件分组

| 类别 | 文件示例 | 检查内容 |
| --- | --- | --- |
| Task/输入契约 | `contract_validator.py`、`intake_validator.py` | 必填字段、placeholder、输入身份 |
| 环境与部署 | `deployment_validator.py`、`bringup_validator.py`、`plan_validator.py` | Pod/runtime/device 证明、toy token、部署约束 |
| Discovery | `scan_validator.py`、`capability_validator.py`、`gap_validator.py`、`drift_validator.py` | 实际路径、能力结论和 gap 证据是否充分 |
| 正确性 | `accuracy_validator.py`、`correctness_validator.py`、`conformance_validator.py` | 独立 reference、误差阈值、case 证据和 API 行为 |
| 算子生命周期 | `operator_lifecycle_validator.py`、`shim_validator.py`、`handoff_validator.py` | dispatch、阶段证据、集成和交接完整性 |
| 资源与性能 | `memory_validator.py`、`performance_validator.py` | 预算输入、指标方向和回归阈值 |

## 独立性的含义

- Candidate 与 reference 不能共享同一份可能错误的实现。
- 生产者声明的 PASS 必须由 Validator 根据原始字段重新计算或核对。
- manifest 只证明文件清单和哈希，不证明数值或服务正确。
- HTTP 200、import 成功、编译成功都只是局部事实。

个别 Validator 为核对持久化调度身份会只读访问 Engine，例如算子生命周期验证。
这类访问不得修改 Scheduler 状态。

## 新增或修改门禁

错误信息要指出缺少或矛盾的具体字段，使 Agent 能采取行动。测试至少覆盖通过、
缺字段、伪造 PASS、身份不匹配和关键边界值；避免写一份逐行复制实现逻辑、却无法
发现真实错误的测试。
