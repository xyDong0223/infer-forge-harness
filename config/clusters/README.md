# Cluster profiles

One profile per target cluster. A profile describes resources the harness may
use: namespace, image, model PVC, XPU count, queue, and node pool.

- `p800-cluster.yaml`: the Kunlun P800 cluster (the wired target).
- `b200-cluster.yaml`: a second-cluster template. No B200 adapter is wired
  yet, so this file describes intent, not a capability.

Kubeconfig contents and credentials stay outside the repository.
