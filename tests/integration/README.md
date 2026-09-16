# Integration tests

Cross-module tests, such as the scheduler end-to-end walk. They exercise
several layers together but still run locally.

Complete user-facing capability scenarios live in [`../e2e/`](../e2e/).
The mandatory model-adaptation scenario uses the production Graph CLI and
persistent scheduler together; the scheduler tests here remain focused
cross-module coverage rather than a substitute for that scenario.

Environment-scoped tests are opt-in. They must never run against a shared
cluster without explicit configuration identifying the namespace, image digest,
model revision, hardware, and cleanup policy — and they must not leave pods
behind on failure.
