# Performance analysis

Performance analysis shares target resolution, environment proof, artifacts,
journal, evidence, and gates with model adaptation. Device-specific benchmark
and profiler behavior belongs in `PerformanceAdapter`.

Every report must identify hardware, engine/backend, revisions, dtype,
parallelism, input/output lengths, warmup, concurrency, request rate, and tool
versions. A profiler trace is evidence, not a performance result until metrics
are extracted and compared with a baseline.

## Comparison and report semantics

`Metric.direction` may explicitly select `higher_is_better` or
`lower_is_better`. Known legacy throughput, latency, TPOT and TTFT names retain
direction inference; unknown or ambiguous names require explicit direction.
The comparison matches metric names and labels, requires matching units, and
rejects duplicate keys, nonfinite values, conflicting directions and undefined
zero-baseline comparisons as `INCOMPARABLE`.

Baseline entries define required measurements. A missing candidate is `FAIL`;
a missing baseline is `UNKNOWN`. Only a nonempty set of entirely passing gates
produces `PASS`. Empty results and uncomparable data cannot imply improvement.

`PerformanceRunner` compares the unprofiled benchmark measurements with the
baseline. It preserves those under `benchmark_metrics` (also exposed through
the existing `metrics` key), and keeps extracted profiler measurements
separately under `trace_metrics`. Profiler output never replaces the benchmark
measurements used to make the performance decision.

The runner requires an external run root and allocates a fresh attempt on every
invocation. It atomically persists `output/performance_report.json` inside that
attempt, including workload, baseline, measurements, artifact references and
gates. Use the returned `report_path`, `artifact_root`, and `manifest_path` rather
than joining a fixed filename to the constructor's root. Local adapter artifacts
are copied into the attempt; remote references remain references.
This report does not by itself establish target hardware execution or
functional readiness. Platform adapters and the surrounding workflow must
still provide the environment and execution evidence required above.
