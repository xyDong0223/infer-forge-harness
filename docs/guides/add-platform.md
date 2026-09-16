# Adding a platform

1. Add a target entry to `compatibility/matrix.yaml`.
2. Add a hardware adapter implementing `HardwareAdapter`.
3. Add an engine/backend runtime implementing `RuntimeAdapter`.
4. Add a profile and deployment contract; do not put credentials in Git.
5. Add workload and profiler adapters if performance is required.
6. Add unit, contract, and clean-room replay tests.
7. Change `planned` to `supported` only after real evidence exists.

The current target scope is:

- Kunlun P800: vLLM-Kunlun and SGLang-Kunlun.
- NVIDIA B200: SGLang only.

The matrix is intentionally not an inventory of every possible combination.
