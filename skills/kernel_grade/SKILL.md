---
name: kernel-grade
description: Grade a P800 kernel with an independent reference and a discriminating control.
---

# Kernel Grade

Grade the smallest failing operator call. Dump candidate and reference tensors,
compare elementwise and by relative L2, count NaN/Inf, and record reference
provenance. The reference must not be produced by the candidate implementation.
Use a negative control that would miss the gate if the suspected error were
absent. If the geometry cannot distinguish two conventions, return
`AMBIGUOUS` and create another Loop Block.
