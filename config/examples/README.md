# Target examples

User-facing target configurations. Each file pins the same four dimensions —
hardware, engine, backend, plugin — and is loaded through `core.target`, which
normalizes the hardware name and applies the compatibility gate before any
cluster action.

- `p800-vllm-kunlun.yaml`: the one combination that is `supported` today.
- `p800-sglang-kunlun.yaml`: declared `planned`; the runtime is not implemented
  yet, so execution is blocked rather than silently redirected.
- `b200-sglang.yaml`: declared `planned`; no B200 adapter is wired yet.
- `p800-vllm-kunlun-performance.yaml`, `b200-sglang-performance.yaml`: the same
  targets expressed for the performance workflow, which adds a workload.

Editing a file here does not grant support. Support status is decided by
`compatibility/matrix.yaml`, and `planned` combinations stay blocked until a
real adapter exists and evidence backs promotion.
