# Integration tests

Cross-module tests, such as the scheduler end-to-end walk. They exercise
several layers together but still run locally.

Environment-scoped tests are opt-in. They must never run against a shared
cluster without explicit configuration identifying the namespace, image digest,
model revision, hardware, and cleanup policy — and they must not leave pods
behind on failure.
