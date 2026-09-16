# Adapter and contract model

Use configuration first, then compose existing adapters, then add an adapter,
then add a generic task, and modify Core only when the capability is genuinely
platform-independent.

The stable objects are `TargetContext`, `Capability`, `ResourceSnapshot`,
`Workload`, `Metric`, `Artifact`, and `GateResult` in `core/contracts.py`.
Adapters should return these objects or serializable equivalents and must not
write platform-specific decisions into Core.

The three platform seams are:

- Hardware: devices, resources, hardware capabilities, profiler collection.
- Engine: installation, service lifecycle, generic serving semantics.
- Backend/plugin: engine-hardware integration, vendor operators, patches.

The executable interfaces are declared as `RuntimeAdapter`,
`HardwareAdapter`, and `PerformanceAdapter` in `core/contracts.py`.
Performance workflows use the same target, artifact, journal, evidence, and
gate context as adaptation workflows; only workload, profiling, and metric
operations are specialized.

Every adapter needs identity, capability declaration, evidence, unit tests,
and a clean-room replay test.
