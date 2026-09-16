# Performance analysis

Performance analysis shares target resolution, environment proof, artifacts,
journal, evidence, and gates with model adaptation. Device-specific benchmark
and profiler behavior belongs in `PerformanceAdapter`.

Every report must identify hardware, engine/backend, revisions, dtype,
parallelism, input/output lengths, warmup, concurrency, request rate, and tool
versions. A profiler trace is evidence, not a performance result until metrics
are extracted and compared with a baseline.
