#!/usr/bin/env bash
# Create (or update) the isolated vLLM environment used for multi-GPU / multi-node
# models. Separate from $OFA_ROOT/env so the Ollama-based install is never touched.
#
#   tools/vllm-env.sh            # create/update at $OFA_VLLM_ENV
#   OFA_VLLM_VERSION=0.32.0 tools/vllm-env.sh
#
# Run on a login node with the same OS as the GPU nodes (Kestrel: kl5).
set -euo pipefail
umask 022

OFA_ROOT="${OFA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OFA_VLLM_ENV="${OFA_VLLM_ENV:-$OFA_ROOT/env/envs/vllm}"
OFA_VLLM_VERSION="${OFA_VLLM_VERSION:-0.32.0}"
OFA_VLLM_PYTHON="${OFA_VLLM_PYTHON:-3.12}"
MM="$OFA_ROOT/env/micromamba"

if [[ ! -x "$OFA_VLLM_ENV/bin/python" ]]; then
    echo "Creating Python $OFA_VLLM_PYTHON env at $OFA_VLLM_ENV"
    MAMBA_ROOT_PREFIX="$OFA_ROOT/env" "$MM" create -y -p "$OFA_VLLM_ENV" \
        -c conda-forge "python=$OFA_VLLM_PYTHON" pip
fi

# PIP_CACHE_DIR off the home quota; the vLLM wheel set (torch + CUDA libs) is ~10 GB.
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/scratch/$USER/.cache/pip}"
"$OFA_VLLM_ENV/bin/python" -m pip install --upgrade pip
"$OFA_VLLM_ENV/bin/python" -m pip install "vllm==$OFA_VLLM_VERSION" "huggingface_hub[cli]"

# pip pulls the newest CUDA compiler wheels, but FlashInfer's runtime JIT needs
# them to match torch's CUDA headers ("CUDA compiler and toolkit headers are incompatible").
TORCH_CUDA="$("$OFA_VLLM_ENV/bin/python" -c 'import torch; print(torch.version.cuda)')"
"$OFA_VLLM_ENV/bin/python" -m pip install "nvidia-cuda-nvcc==$TORCH_CUDA.*" \
    "nvidia-cuda-crt==$TORCH_CUDA.*" "nvidia-nvvm==$TORCH_CUDA.*" "nvidia-cuda-cccl==$TORCH_CUDA.*"
# The runtime wheel ships only libcudart.so.<major> under lib/; the JIT links
# with -L$CUDA_HOME/lib64 -lcudart.
CUDA_LIB="$("$OFA_VLLM_ENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/nvidia/cu${TORCH_CUDA%%.*}/lib"
ln -sfn "libcudart.so.${TORCH_CUDA%%.*}" "$CUDA_LIB/libcudart.so"
[[ -e "$CUDA_LIB/../lib64" ]] || ln -s lib "$CUDA_LIB/../lib64"

"$OFA_VLLM_ENV/bin/python" - <<'PY'
import torch, vllm
print(f"vllm {vllm.__version__}  torch {torch.__version__}  cuda {torch.version.cuda}")
PY
echo "vLLM env ready: $OFA_VLLM_ENV"
