# OpenWiki Source Registry

`openwiki/` contains separately sourced reference modules. The directory structure is part of the documentation contract: source-specific evidence, licenses, and revision baselines stay within the module that owns them.

| Module | Scope | Source and evidence baseline | Detailed notice |
|---|---|---|---|
| [`vllm-core/`](vllm-core/index.md) | vLLM hardware backend and platform integration | Supplied engineer-authored Wiki; vLLM main `94848eda600a07c28675f5753a11b2c212c146ed` | [vllm-core/SOURCE.md](vllm-core/SOURCE.md) |
| [`vllm-kunlun/`](vllm-kunlun/index.md) | Kunlun3 P800 out-of-tree vLLM plugin | `baidu/vLLM-Kunlun` `v0.25.1-dev`, source commit `3ced109af2510479e1b2eb846a8aca1fbdcbdf62` | [vllm-kunlun/SOURCE.md](vllm-kunlun/SOURCE.md) |
| [`harness/`](harness/index.md) | Infer-Forge project engineering practice | `xyDong0223/infer-forge-harness` task, tool, and validator evidence | [harness/index.md](harness/index.md) |

Imported material remains reference material. Platform contracts, executable task definitions, and validation results in this repository remain authoritative for the Agent platform.
