# 由 harness 生成部署契约

KDP-001 和 KDP-001a 的模型示例已移除。环境阶段不再需要输入模型 YAML：

```bash
python cli/adaptation.py --state /external/state.sqlite environment \
  --run-id my-run --user-id <使用者ID>
```

环境 Task 和 `config/clusters/p800-cluster.yaml` 是生成源。harness 生成 MiniMax
基线契约，在 attempt 的 `input/task_contract.json` 保存输入，在 `output/task_contract.yaml`
保存实际执行契约。生成源路径及哈希记录在契约元数据中。目标模型配置不会覆盖环境基线。

Graph 删除 `--set contract_instance=...`，改为传入 `--set user_id=...`。
MAT-005 根据本次 intake 和分类证据生成 `kdp_instance.yaml`；服务、诊断与修复节点从
Journal 的有效 DeploymentPlan 取得它。服务和修复缺少规划产物时阻塞。
规划前的失败由 triage 保存原始状态及命令日志，输出 `NEEDS_HUMAN/UNKNOWN` 供诊断，
不会启动服务或把未知失败归因给 vendor。

直接重放已生成的服务契约：

```bash
python cli/deployment/proof.py /external/generated-service.yaml \
  --execute --phase service --attach-pod <原有Pod> --user-id <原使用者ID>
```

同一环境的恢复必须沿用已记录的 user ID；切换所有者会被拒绝，应为新所有者建立新 run。
生成文件属于 run，不写回仓库；测试夹具使用合成数据，
不作为 Agent 的部署模板。

本变更覆盖 KDP-001a 环境契约、MAT-005 服务契约和 KDP-001b 的 Graph 输入选择。
本地 E2E 验证缺参拒绝、生成、同一 run 恢复、完整交付和手工契约拒绝；模拟结果不代表硬件就绪。
