#!/bin/bash
#SBATCH --job-name=ofa-vllm
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=ofa-vllm-%j.log
#
# Serve a model too large for one GPU with vLLM across one or more whole GPU
# nodes, for use with `ofa --vllm`. Tensor parallel within a node, pipeline
# parallel across nodes; no Ray (vLLM's --nnodes/--headless multiprocessing).
#
#   OFA_VLLM_MODEL=/path/to/model sbatch -A <account> -p <gpu partition> \
#       --nodes=4 $OFA_ROOT/tools/vllm-serve.sbatch
#
# Settings (environment, exported to the job by sbatch):
#   OFA_VLLM_MODEL       local model directory (pre-download with `hf download`)
#   OFA_VLLM_NAME        served model name (default: basename of OFA_VLLM_MODEL)
#   OFA_VLLM_MAX_LEN     max context length (default 131072)
#   OFA_VLLM_ARGS        extra `vllm serve` arguments
#   OFA_VLLM_ENV         vLLM environment (default $OFA_ROOT/env/envs/vllm; see tools/vllm-env.sh)
#   OFA_VLLM_STARTUP_TIMEOUT  seconds to wait for /health before giving up (default 5400)
#
# When /health is up, the endpoint and a per-job API key are written (mode
# 0600) to $OFA_SCRATCH/.ofa_vllm_endpoint.json; `ofa --vllm` reads it.
set -euo pipefail

OFA_ROOT="${OFA_ROOT:-/nopt/nrel/apps/cpu_stack/software/openfoam/assistant}"
OFA_VLLM_ENV="${OFA_VLLM_ENV:-$OFA_ROOT/env/envs/vllm}"
OFA_SCRATCH="${OFA_SCRATCH:-/scratch/$USER}"
: "${OFA_VLLM_MODEL:?set OFA_VLLM_MODEL to a local model directory}"
[[ -d "$OFA_VLLM_MODEL" ]] || { echo "OFA_VLLM_MODEL=$OFA_VLLM_MODEL is not a directory"; exit 1; }
NAME="${OFA_VLLM_NAME:-$(basename "$OFA_VLLM_MODEL")}"
NNODES="$SLURM_JOB_NUM_NODES"
GPUS_PER_NODE="$(nvidia-smi -L | wc -l)"
HEAD="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
API_PORT=$((40000 + RANDOM % 10000))
MASTER_PORT=$((30000 + RANDOM % 10000))
API_KEY="$("$OFA_VLLM_ENV/bin/python" -c 'import secrets; print(secrets.token_urlsafe(32))')"
ENDPOINT="$OFA_SCRATCH/.ofa_vllm_endpoint.json"
trap 'rm -f "$ENDPOINT"' EXIT

echo "vLLM: $NAME on $NNODES node(s) x $GPUS_PER_NODE GPUs (tp=$GPUS_PER_NODE pp=$NNODES), head $HEAD:$API_PORT"

# Weights are on a shared filesystem; never reach out to the Hub from compute nodes.
export HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
# FlashInfer JIT-compiles kernels at startup: use the nvcc shipped in the env, and
# keep compile caches off the home quota.
export CUDA_HOME="$("$OFA_VLLM_ENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/nvidia/cu13"
[[ -x "$CUDA_HOME/bin/nvcc" ]] || { echo "no nvcc at $CUDA_HOME/bin (rerun tools/vllm-env.sh)"; exit 1; }
export PATH="$OFA_VLLM_ENV/bin:$CUDA_HOME/bin:$PATH"
# NCCL OFI (Slingshot RDMA) net plugin; without it inter-node traffic falls back to
# TCP sockets. Defaults to Kestrel's build if present; set OFA_VLLM_NCCL_PLUGIN= to disable.
OFA_VLLM_NCCL_PLUGIN="${OFA_VLLM_NCCL_PLUGIN-/nopt/nlr/apps/kestrel-gpu/software/nccl/v2.27.7-1_cray_mpich/plugin}"
if [[ -n "$OFA_VLLM_NCCL_PLUGIN" && -d "$OFA_VLLM_NCCL_PLUGIN/lib" ]]; then
    export LD_LIBRARY_PATH="$OFA_VLLM_NCCL_PLUGIN/lib:${OFA_VLLM_LIBFABRIC:-/opt/cray/libfabric/1.22.0/lib64}:${LD_LIBRARY_PATH:-}"
    export NCCL_NET_PLUGIN=ofi FI_PROVIDER="${FI_PROVIDER:-cxi}"
fi
export XDG_CACHE_HOME="$OFA_SCRATCH/.cache"
export FLASHINFER_WORKSPACE_BASE="$OFA_SCRATCH" VLLM_CACHE_ROOT="$XDG_CACHE_HOME/vllm"
export OFA_VLLM_ENV OFA_VLLM_MODEL NAME NNODES GPUS_PER_NODE HEAD API_PORT MASTER_PORT API_KEY
export OFA_VLLM_MAX_LEN="${OFA_VLLM_MAX_LEN:-131072}" OFA_VLLM_ARGS="${OFA_VLLM_ARGS:-}"

srun --nodes="$NNODES" --ntasks="$NNODES" --ntasks-per-node=1 --kill-on-bad-exit=1 bash -c '
    common=(--tensor-parallel-size "$GPUS_PER_NODE" --pipeline-parallel-size "$NNODES"
            --nnodes "$NNODES" --node-rank "$SLURM_NODEID"
            --master-addr "$HEAD" --master-port "$MASTER_PORT"
            --max-model-len "$OFA_VLLM_MAX_LEN")
    if [[ "$SLURM_NODEID" == 0 ]]; then
        # The key travels via env, not argv, so it never shows up in ps.
        VLLM_API_KEY="$API_KEY" exec "$OFA_VLLM_ENV/bin/vllm" serve "$OFA_VLLM_MODEL" "${common[@]}" \
            --served-model-name "$NAME" --host 0.0.0.0 --port "$API_PORT" $OFA_VLLM_ARGS
    else
        exec "$OFA_VLLM_ENV/bin/vllm" serve "$OFA_VLLM_MODEL" "${common[@]}" --headless $OFA_VLLM_ARGS
    fi
' &
SRUN_PID=$!

deadline=$((SECONDS + ${OFA_VLLM_STARTUP_TIMEOUT:-5400}))
until curl -sf "http://$HEAD:$API_PORT/health" >/dev/null; do
    kill -0 "$SRUN_PID" 2>/dev/null || { echo "vLLM exited before becoming healthy"; wait "$SRUN_PID"; exit 1; }
    # A failed worker can leave the others waiting forever; free the nodes instead.
    (( SECONDS < deadline )) || { echo "vLLM not healthy after ${OFA_VLLM_STARTUP_TIMEOUT:-5400}s; stopping"; kill "$SRUN_PID"; exit 1; }
    sleep 15
done
( umask 077
  printf '{"base_url": "http://%s:%s/v1", "model": "%s", "api_key": "%s", "job_id": "%s"}\n' \
      "$HEAD" "$API_PORT" "$NAME" "$API_KEY" "$SLURM_JOB_ID" > "$ENDPOINT" )
echo "vLLM ready: http://$HEAD:$API_PORT/v1 (model '$NAME'). Run: ofa --vllm"
wait "$SRUN_PID"
