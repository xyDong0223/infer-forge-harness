# OpenWiki

> 中文译本见 [README.zh-CN.md](README.zh-CN.md);如有不一致,以本英文版为准。

Layered engineering reference material, split by provenance so the layers are
never confused with each other:

- `vllm-core/`: upstream vLLM contracts and hardware-backend integration.
- `vllm-kunlun/`: the Kunlun P800 plugin implementation, operator namespaces,
  and known traps with upstream line numbers.
- `harness/`: this project's own practice and filed experiences
  (`harness/experiences/`), organized by the capability that hit them rather
  than by model.

`SOURCE.md` registers each layer's provenance and evidence baseline; `index.md`
is the reading path.

Reference material is never proof. A task contract or validator verdict
overrides anything here, and no claim in this directory may be used to mark a
result PASS.
