#!/usr/bin/env bash
# Launch MiniMax-M3 on P800 with the torch stand-ins loaded.
#
# Kept in the repo rather than only in the pod's /tmp: the dev pod disappeared on
# 2026-09-08 and this script went with it, while everything it references survived here.
# Assemble the stand-ins first with tools/deploy_m3_shims.sh.
#
# Diagnostics are all opt-in; set one at a time, since several of them wrap the same
# forward and the last one patched wins:
#   M3_TRACE=1 [M3_TRACE_TOKENS=<n>] [M3_TRACE_DUMP=<dir>]  per-layer norms / tensor dump
#   M3_ROPE_CHECK=1        qk-norm+RoPE vs vLLM's own RotaryEmbedding
#   M3_LINEAR_CHECK=1      int8 gate_up / clamped SwiGLU / down_proj
#   M3_ATTEND_CHECK=1 [M3_ATTEND_TOKENS=<n>]   sparse attend vs cache-free full attention
#   M3_DENSE_SELFTEST=1    the torch dense attention stand-in against the same reference
#   M3_TORCH_DENSE_ATTN=0  go back to the platform's dense attention kernel
set -x
export VIRTUAL_ENV=/opt/vllm_kunlun PATH=/opt/vllm_kunlun/bin:$PATH
source /workspace/vLLM-Kunlun/setup_env.sh
export XPU_USE_MOE_SORTED_THRES=1 XFT_USE_FAST_SWIGLU=1 XPU_USE_FAST_SWIGLU=1
export XMLIR_CUDNN_ENABLED=1 XPU_USE_DEFAULT_CTX=1 XMLIR_FORCE_USE_XPU_GRAPH=1
export XMLIR_ENABLE_MOCK_TORCH_COMPILE=false XMLIR_DYNAMO_WORKAROUND=1
export VLLM_HOST_IP=$(hostname -i)
export PYTHONPATH=${M3_SHIM_DIR:-/tmp/m3_shim}
exec python3 -m vllm.entrypoints.openai.api_server \
  --model /mnt/cluster/MiniMax-M3-W8A8-INT8-Dynamic --served-model-name MiniMax-M3 \
  --host 0.0.0.0 --port "${M3_PORT:-8390}" --tensor-parallel-size 8 --dtype bfloat16 \
  --block-size 128 --max-model-len 32768 --max-num-batched-tokens 8192 --max-num-seqs 32 \
  --gpu-memory-utilization 0.92 --trust-remote-code --enforce-eager \
  --limit-mm-per-prompt '{"image":0,"video":0}'
