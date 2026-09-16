# Migrating an existing workflow

Map existing assets into the harness incrementally:

| Existing asset | Harness representation |
| --- | --- |
| Shell script | Task or adapter method |
| Kubernetes YAML | Environment and deployment contract |
| Launch command | Engine/backend command builder |
| Patch | Replayable patch specification |
| Health check | Service-proof gate |
| Benchmark script | Workload adapter |
| Profiler | Performance adapter |
| Human decision | Explicit gate or diagnosis |
| Logs/results | Evidence artifacts |

Start with a configuration-only target. Keep the original workflow as a
golden reference, run both paths, and migrate one task at a time. Do not move
weights, credentials, private endpoints, or large traces into the repository.
