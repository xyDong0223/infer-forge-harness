# Configuration

Use these inputs to connect your own environment, not as proof that resources
exist or that a model is supported.

| Location | Configure |
| --- | --- |
| [examples/](examples/README.md) | Hardware, engine, backend, plugin and workload |
| [clusters/](clusters/README.md) | Namespace, context, image, queue, node pool, PVC and device resources |
| [profiles/](profiles/README.md) | Runtime venv, site-packages and engine module |
| [manifests/](manifests/README.md) | Deployment resource templates |
| [Task instances](../tasks/) | Model-specific deployment inputs and pinned revisions |

## Connecting an environment

Start with a supported combination in [compatibility/matrix.yaml](../compatibility/matrix.yaml).
That matrix permits platform execution; [catalog/support_matrix.yaml](../catalog/support_matrix.yaml)
records model-level validation. Neither replaces proof for the current run.

Before following the [real adaptation steps](../README.md#运行真实模型适配):

1. Replace development-cluster image, queue, node-pool, PVC and mount assumptions with confirmed resources.
2. Set external `KUBECONFIG`. Ask the user for their resource-owner ID and pass `--user-id <user-supplied-id>` to the environment/proof/planner CLI (Graph: `--set user_id=<user-supplied-id>`); contracts can record `execution.user_id`. Explicit input takes precedence over the contract and legacy `USER_ID` environment variable. Never infer the ID from the host login or example resource names. Check namespace, context, container, deployment kind and resource ownership prefix.
3. Check runtime paths, setup commands and proxy settings against the actual image and network.
4. Pin model/plugin revisions and the matching deployment contract. Check the cluster profile's validation base model as well as the target model.
5. Choose a persistent external state directory, inspect the plan and prove the environment before discovery.

`ClusterConfig.load(path)` accepts an explicit profile, but some production
callers use the default P800 profile. Adding another YAML does not switch every
caller or register an adapter; do not assume a universal `--cluster-config`
option exists.

## Path and evidence boundaries

Model paths normally refer to files **inside the Pod**. Artifact roots refer to
external paths **on the host running Harness**. Keep these separate; use the
returned artifact root or claimed workspace rather than predicting attempt paths.

Credentials, weights and run evidence stay outside the repository. Re-prove an
existing Pod with explicit `--attach-pod`; do not silently reuse old evidence
after changing environment identity. See [runtime ownership](../docs/migration/runtime-write-policy.zh-CN.md)
and [troubleshooting](../docs/guides/troubleshooting.zh-CN.md).
