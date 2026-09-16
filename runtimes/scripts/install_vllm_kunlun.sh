#!/usr/bin/env bash
# Install vllm-kunlun inside a vLLM-Kunlun-Base container.
#
# Single source of truth:
#   https://github.com/baidu/vLLM-Kunlun/blob/v0.25.1-dev/docs/source/installation.md
# Placeholders substituted from docs/source/conf.py (pip_vllm_version = 0.25.1).
#
# Step numbering below maps 1:1 onto the document's sections. Any line that is
# NOT in the document is marked `# DEVIATION:` with the evidence for it.
set -euo pipefail

VK_REF="${VK_REF:-v0.25.1-dev}"
VLLM_VERSION="${VLLM_VERSION:-0.25.1}"
TORCH_VERSION="${TORCH_VERSION:-2.9.0}"
WORKDIR="${WORKDIR:-/workspace}"

XPYTORCH_RUN=xpytorch-cp310-torch290-ubuntu2004-x64.run
XPYTORCH_URL="https://klx-sdk-release-public.su.bcebos.com/kunlun2jituan/20260806/${XPYTORCH_RUN}"
KUNLUN_OPS_URL="https://klx-sdk-release-public.su.bcebos.com/kunlun2jituan/20260806/kunlun_ops-0.1.227%2B2b100f96-cp310-cp310-linux_x86_64.whl"
XSPEEDGATE_URL="https://vllm-ai-models.bj.bcebos.com/aiak_share/20260827/torch29/xspeedgate_ops-1.5.1%2B87067b3.torch29-cp310-cp310-linux_x86_64.whl"

# DEVIATION: the cluster has no direct egress; every download needs this proxy.
export http_proxy="${http_proxy:-http://agent.baidu.com:8891}"
export https_proxy="${https_proxy:-http://agent.baidu.com:8891}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,.baidu-int.com,.bcebos.com,.baidubce.com}"

# DEVIATION: the base image ships a venv at /opt/vllm_kunlun but a login shell
# leaves VIRTUAL_ENV empty, so `uv pip` would install outside the venv.
export VIRTUAL_ENV="${VIRTUAL_ENV:-/opt/vllm_kunlun}"
export PATH="$VIRTUAL_ENV/bin:/root/.local/bin:$PATH"

step() { echo; echo "########## [$(date +%H:%M:%S)] $* ##########"; }

# DEVIATION: fetching through the proxy fails intermittently with
# "gnutls_handshake() failed", so network steps are retried.
retry() {
  local attempts="$1"; shift
  local n=1
  until "$@"; do
    if [ "$n" -ge "$attempts" ]; then
      echo "FAILED after ${attempts} attempts: $*" >&2
      return 1
    fi
    n=$((n + 1))
    echo "retry ${n}/${attempts}: $*" >&2
    sleep 5
  done
}


step "0. Environment"
python3 -V
uv --version
echo "VIRTUAL_ENV=$VIRTUAL_ENV"

# DEVIATION: PyTorch >= 2.9 makes torch.utils.cpp_extension compile with
# `-std=c++20`, but the Ubuntu 20.04 base image only has g++ 9.4.0, which
# rejects that option ("did you mean '-std=c++2a'?") and aborts the _kunlun
# native build. g++-10 (focal-updates/universe) is the smallest fix.
step "0b. Ensure a C++20-capable compiler"
if ! command -v g++-10 >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y -qq g++-10 gcc-10
fi
export CC=gcc-10 CXX=g++-10
g++-10 --version | head -1

# DEVIATION: the intranet PyPI mirror is much faster than pypi.org through the
# proxy; pypi.org stays as a fallback for wheels the mirror lacks.
#
# DEVIATION: but `--index-strategy unsafe-best-match` queries *every* index for *every*
# package, so leaving pypi.org in the list makes step 3 reach through the proxy even for
# packages the mirror already has -- and that times out. Measured 2026-09-08 on a fresh
# pod: `Failed to fetch https://pypi.org/simple/anthropic/ ... operation timed out` after
# uv's own 5 retries, while `curl https://pip.baidu-int.com/simple/anthropic/` returned
# 200. So the extra index defaults to the mirror too; set PIP_EXTRA_INDEX_URL back to
# pypi.org if a wheel genuinely only exists upstream.
INDEX_ARGS=(--index-url "${PIP_INDEX_URL:-https://pip.baidu-int.com/simple/}"
            --extra-index-url "${PIP_EXTRA_INDEX_URL:-https://pip.baidu-int.com/simple/}"
            --index-strategy unsafe-best-match)

step "1. Install PyTorch (torch==$TORCH_VERSION)"
uv pip install "${INDEX_ARGS[@]}" "torch==${TORCH_VERSION}" torchvision torchaudio

step "2. Install vLLM (vllm==$VLLM_VERSION)"
uv pip install "${INDEX_ARGS[@]}" "vllm==${VLLM_VERSION}" --no-build-isolation --no-deps

step "3. Build and install vllm-kunlun ($VK_REF)"
cd "$WORKDIR"
# DEVIATION: the document substitutes |vllm_kunlun_version| to "main", but
# conf.py documents that a vX.Y.Z line should track its release branch, and
# main lags v0.25.1-dev while pairing with the same vllm 0.25.1.
if [ -d vLLM-Kunlun/.git ]; then
  cd vLLM-Kunlun && retry 5 git fetch origin --prune
else
  retry 5 git clone https://github.com/baidu/vLLM-Kunlun && cd vLLM-Kunlun
fi
# DEVIATION: `git checkout <branch>` on an existing clone leaves the local
# branch behind origin, so the recorded commit would not match the ref. A
# detached checkout of the remote ref pins the exact tree.
git checkout --detach "origin/${VK_REF}"
git rev-parse HEAD

# DEVIATION: resolving requirements.txt transitively upgrades torch 2.9.0 ->
# 2.14.0, because requirements.txt pins compressed-tensors==0.17.0 which asks
# for torch>=2.10. The resulting libtorch_cuda.so fails to load with "undefined
# symbol: ncclCommResume" and the _kunlun build aborts; worse, step 4 later
# overwrites torch with the KL3 2.9.0 build, leaving the extension compiled
# against a different ABI.
# `--no-deps` avoids that upgrade but breaks binary pairs: pydantic 2.12.0 then
# sits next to the image's pydantic-core 2.23.4 and the API server dies at
# import with "installed pydantic-core version is incompatible".
# So resolve dependencies normally, pin torch, and override only the single
# package that forces the upgrade.
printf 'torch==%s\n' "$TORCH_VERSION" > /tmp/kdp-constraints.txt
printf 'compressed-tensors<0.17\n' > /tmp/kdp-overrides.txt
retry 3 uv pip install "${INDEX_ARGS[@]}" -r requirements.txt \
  --constraint /tmp/kdp-constraints.txt --override /tmp/kdp-overrides.txt
uv pip install --no-build-isolation --no-deps .
python3 -c "import torch; assert torch.__version__.startswith('${TORCH_VERSION}'), torch.__version__"
python3 -c "import pydantic, fastapi; print('pydantic', pydantic.VERSION)"

step "4. Install the KL3-customized build of PyTorch"
cd "$WORKDIR"
[ -f "$XPYTORCH_RUN" ] || retry 3 wget -q --show-progress -O "$XPYTORCH_RUN" "$XPYTORCH_URL"
rm -rf xpytorch_unpack
bash "$XPYTORCH_RUN" --noexec --target xpytorch_unpack
cd xpytorch_unpack
sed -i 's/pip/uv pip/g; s/CONDA_PREFIX/VIRTUAL_ENV/g' setup.sh
bash setup.sh

step "5. Install Kunlun-related packages"
uv pip install "${INDEX_ARGS[@]}" "$KUNLUN_OPS_URL"
uv pip install "${INDEX_ARGS[@]}" "$XSPEEDGATE_URL"

step "6. Verify"
uv pip list 2>/dev/null | grep -iE "^(vllm|vllm-kunlun|torch|kunlun-ops|xspeedgate-ops) " || true
cd "$WORKDIR/vLLM-Kunlun"
python3 -c "
import vllm, vllm_kunlun, torch
print('vllm        =', vllm.__version__)
print('torch       =', torch.__version__)
print('vllm_kunlun =', vllm_kunlun.__file__)
"

step "DONE. Next: source $WORKDIR/vLLM-Kunlun/setup_env.sh, then start api_server"
