# Deployment manifest templates

Pod and deployment templates rendered by the task runners. A template holds
placeholders that the harness substitutes from the contract; it must not carry
credentials, private endpoints, or cluster-specific secrets.

`p800-vllm-kunlun.yaml` is the P800 deployment template. Per-model pod
templates live next to the tasks that own them (`tasks/*/manifests/`).
