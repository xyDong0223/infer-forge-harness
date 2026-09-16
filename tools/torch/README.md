# Torch helpers

Reusable host-side torch implementations used by executors and probes when a
vendor kernel is unavailable or suspect.

- `paged_decode.py`: pure-torch paged decode attention for the Kunlun attention
  backend, used because both vendor decode kernels fail inside the server at
  Qwen3-8B's decode geometry while accepting the same arguments in isolation.

A helper here is a shim, not a resolution. Anything on a declared XPU-ready
path that routes through this directory must be reported as a fallback and
handed to the operator path — silent substitution is prohibited.
