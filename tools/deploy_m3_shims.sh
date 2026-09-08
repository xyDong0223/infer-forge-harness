#!/usr/bin/env bash
# Assemble the MiniMax-M3 stand-in tree inside a pod, from patches/ in this repo.
#
# Why this exists: the tree used to be hand-assembled in /tmp with kubectl cp and
# appended-to bootstrap lines, and when the dev pod disappeared on 2026-09-08 all of it
# went with it. The shims themselves were safe in the repo; only the wiring was lost. This
# script is that wiring, so a rebuild is one command instead of a reconstruction.
#
# The bootstrap deliberately piggybacks on flashinfer/norm.py: MiniMax-M3 imports it at
# the first layer's norm, before the first attention call, and nothing else in the process
# would import a shim module in time. Everything lives under PYTHONPATH, never in
# site-packages, so removing the path removes all of it.
#
# Usage, from inside the pod (patches/ copied alongside):
#   bash deploy_m3_shims.sh /tmp/m3_patches /tmp/m3_shim
set -euo pipefail

PATCHES="${1:-/tmp/m3_patches}"
TARGET="${2:-/tmp/m3_shim}"

if [ ! -d "$PATCHES" ]; then
  echo "no patches directory at $PATCHES" >&2
  exit 1
fi

rm -rf "$TARGET"
mkdir -p "$TARGET/flashinfer"

# The stand-in is imported as m3_probe_ops in the pod; keep that name so the bootstrap and
# patches/m3_sparse_attend_selfcheck.py (which looks for either name) both resolve.
install -m 644 "$PATCHES/m3_fused_qknorm_rope_probe.py" "$TARGET/m3_probe_ops.py"
for module in \
  m3_bind_kv_cache_probe \
  m3_torch_index_topk \
  m3_torch_sparse_attn \
  m3_torch_dense_attn \
  m3_moe_routing_activation \
  m3_layer_trace \
  m3_rope_selfcheck \
  m3_int8_linear_selfcheck \
  m3_sparse_attend_selfcheck \
  m3_dense_attn_selfcheck
do
  install -m 644 "$PATCHES/$module.py" "$TARGET/$module.py"
done

: > "$TARGET/flashinfer/__init__.py"
install -m 644 "$PATCHES/m3_flashinfer_norm_probe.py" "$TARGET/flashinfer/norm.py"

cat >> "$TARGET/flashinfer/norm.py" <<'BOOTSTRAP'

# --- bootstrap, appended by tools/deploy_m3_shims.sh ---------------------------------
# Piggyback, on purpose: flashinfer.norm is imported by the first layer's norm, before the
# first attention call, so registering here is guaranteed to be in time.

# The fused qk-norm/RoPE/dual-cache-insert op upstream calls but nothing registers here.
import m3_probe_ops as _m3_probe_ops
_m3_probe_ops.register()

# bind_kv_cache rejects out-of-tree platforms upstream (M3-03, filed upstream separately).
import m3_bind_kv_cache_probe as _m3_bind_probe
_m3_bind_probe.patch()

# The triton index and attend kernels blow the P800 shared-memory budget (50180 and 69636
# against a 49152 limit), so both are replaced with torch rather than retuned.
import m3_torch_index_topk as _m3_torch_index
_m3_torch_index.patch()
import m3_torch_sparse_attn as _m3_torch_attn
_m3_torch_attn.patch()

# The platform's dense attention is off by relL2 0.326/0.438/0.076 on layers 0/1/2 against
# cache-free causal attention over its own q/k/v; the torch stand-in measures 0.0015.
# Set M3_TORCH_DENSE_ATTN=0 to go back to the kernel.
import m3_torch_dense_attn as _m3_dense
_m3_dense.patch()

# M3 routes with sigmoid+bias and activates with SwiGLU-OAI, but the plugin's monolithic
# MoE never receives the routing arguments and hard-codes plain SwiGLU. Neither raises.
import m3_moe_routing_activation as _m3_moe
_m3_moe.patch()

# Diagnostics, each off unless its own env var is set.
import m3_layer_trace as _m3_trace                      # M3_TRACE (+ M3_TRACE_DUMP)
_m3_trace.patch()
import m3_rope_selfcheck as _m3_rope_check              # M3_ROPE_CHECK
_m3_rope_check.patch()
import m3_int8_linear_selfcheck as _m3_linear_check     # M3_LINEAR_CHECK
_m3_linear_check.patch()
import m3_sparse_attend_selfcheck as _m3_attend_check   # M3_ATTEND_CHECK
_m3_attend_check.patch()
import m3_dense_attn_selfcheck as _m3_dense_check       # M3_DENSE_ATTN_CHECK
_m3_dense_check.patch()
BOOTSTRAP

python3 -c "import ast,sys; ast.parse(open('$TARGET/flashinfer/norm.py').read())"
echo "assembled $TARGET:"
ls "$TARGET" "$TARGET/flashinfer"
