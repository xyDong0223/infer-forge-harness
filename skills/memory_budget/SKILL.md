---
name: memory-budget
description: Explain where P800 HBM went for a running vLLM-Kunlun deployment, reconciled against xpu_smi, and decide whether the budget is acceptable.
---

# Memory Budget Skill

## Purpose

Turn a running deployment into an evidence-backed HBM breakdown: weights, KV
pool, graph capture, and the unattributed remainder. The upstream skill
[`llm-serving-capacity-planner`](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS)
does the log decomposition; this Skill supplies the two things it cannot know on
Kunlun and decides the outcome.

## Why an upstream skill is not enough

| Upstream assumption | P800 reality | Where it is handled here |
| --- | --- | --- |
| `references/gpu-specs.json` has the device | No P800 entry, so total HBM falls back to a guess | `catalog/xpu_specs.yaml`, populated from `xpu_smi` with the observing pod and timestamp recorded |
| `nvidia-smi` is available | Absent; the container ships `xpu_smi` with positional columns | `KunlunP800Adapter.parse_xpu_smi` reads columns 2/17/18 and `as_nvidia_smi_csv` renders them in the shape the analyzer parses |
| `--mem-fraction-static` governs the KV pool | vLLM's `--gpu-memory-utilization` does, and the SGLang-only fields stay unknown | Report the vLLM evidence section; never translate a missing field into an SGLang one |

## Required inputs

1. A pod owned by the operator prefix, still running the server under analysis.
2. The server log **of the process that is currently serving**. A log from an
   earlier attempt reconciles to a wrong remainder — that is the failure the
   validator's unattributed-memory floor exists to catch.
3. A clone of the upstream skills repository, located by `--analyzer` or
   `AI_INFRA_SKILLS_DIR`. Nothing hardcodes a clone path.

## Procedure

```bash
export KUBECONFIG=/path/to/kubeconfig
export AI_INFRA_SKILLS_DIR=/path/to/AI-Infra-Auto-Driven-SKILLS
python3 tools/memory_budget.py \
  --pod <owned-pod> \
  --server-log /workspace/server_minimax.log \
  --out <artifact-dir>/memory
```

The Tool writes four artifacts: `xpu_smi.csv` (the device snapshot),
`capacity_planner.json` (the upstream report, unmodified),
`memory_budget.json`, and `memory_budget.md`. Then decide with
`validators/memory_validator.py`, passing the contract's thresholds
(`min_free_mib`, `max_utilization_pct`, `min_kv_cache_tokens`,
`max_card_spread_mib`).

## Rules

Read-only against the cluster: `xpu_smi` and `cat` on a log, nothing else. Do
not restart a server to obtain a cleaner log — a restart invalidates the
snapshot it was meant to explain. Report unknown fields as unknown rather than
deriving them from a datasheet.

## Acceptance

The Skill is complete when the breakdown reconciles with the device counters
(`reconciled: true`), the remainder is attributed to driver/runtime rather than
silently dropped, and every number in the report names the log line or command
it came from.

## Reference result — MiniMax-M2.5-W8A8 on 8x P800, 2026-09-07

| Category | MiB | % of 98304 |
| --- | --- | --- |
| model weights | 27832 | 28.3% |
| KV pool | 59412 | 60.4% |
| graph capture | 61 | 0.1% |
| unattributed (driver/runtime/allocator) | 3134 | 3.2% |
| free | 7864 | 8.0% |

KV pool held 1,962,496 tokens (9.98x concurrency at the full 196,608-token
context), per-card spread 0 MiB across all 8 ranks. `framework` reads 0 because
current vLLM does not log an initial-free-memory line; that memory is inside
the unattributed remainder, not missing.
