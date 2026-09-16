# Catalog

The fact registry. It stores what the harness believes and, critically, the
evidence each belief rests on:

- `tool_catalog.yaml`: deterministic tool commands.
- `runtime_catalog.yaml`: declared runtimes and the hardware they support.
  A declaration without a wired loader fails loudly instead of pretending to
  exist.
- `skill_catalog.yaml`: the method unit behind every workflow `task_type`.
- `xpu_specs.yaml`: device facts observed from the cluster, including the
  P800 `api_surface` fact that the vendor device is reached through the
  CUDA API surface.
- `support_matrix.yaml`: model support status, graded by attached evidence.

`catalog/` holds discovered facts; `config/` holds declared inputs. A new fact
enters only with the evidence that produced it.
