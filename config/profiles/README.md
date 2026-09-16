# Runtime profiles

Static environment identity for one runtime install: venv path,
site-packages, and engine module. These values come from configuration, never
from code — 19 files used to hardcode `/opt/vllm_kunlun` and drifted freely.

- `p800-vllm-kunlun.yaml`: the vLLM-Kunlun stack on P800.

Behavioural runtime differences (launch command, readiness, fallback markers,
fingerprints) live in `runtimes/`, not here.
