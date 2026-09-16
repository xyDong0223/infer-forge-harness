# Unit tests

Deterministic tests. They must not touch a cluster, a real model, or the
network, and they must not depend on a specific hardware environment.

Several files here are standing guards rather than feature tests: they pin
invariants such as concrete adapters staying behind the factory, runtime
commands coming from the profile instead of hardcoded strings, and unsupported
target combinations being refused before any work starts. When a guard fails,
fix the code or change the invariant deliberately — do not delete the guard.
