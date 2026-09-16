# Kunlun P800 adapter

The Hardware-axis implementation for the Kunlun P800 cluster: kubectl
primitives, the write-safety gates that limit mutations to owned prefixes,
pod exec and file push, and `xpu_smi` device probes.

Reach it only through `adapters.get_hardware("kunlun/p800")`; a test enforces
that no caller imports this package directly. Cluster and hardware concerns
are fused here today — splitting them is tracked in
`docs/architecture.md`.
