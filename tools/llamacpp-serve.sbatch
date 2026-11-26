#!/bin/bash
#SBATCH --job-name=ofa-llamacpp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=ofa-llamacpp-%j.log
#
# Serve a GGUF model with llama.cpp's llama-server on one whole GPU node. Same contract as
# tools/vllm-serve.sbatch: when /health is up it writes the endpoint and a per-job API key
# (mode 0600) to the endpoint file that `ofa --vllm` reads; src/ofa_vllm.py submits it for
# recipes with "engine": "llamacpp".
#
# Settings (environment, exported to the job by sbatch):
#   OFA_VLLM_MODEL       first .gguf shard of the model
#   OFA_VLLM_NAME        served model name (default: basename of OFA_VLLM_MODEL)
#   OFA_VLLM_ARGS        extra llama-server arguments
#   OFA_VLLM_ENDPOINT    endpoint file to write (default $OFA_SCRATCH/.ofa_vllm_endpoint.json)
#   OFA_VLLM_STARTUP_TIMEOUT  seconds to wait for /health before giving up (default 2400)
#   OFA_VLLM_IDLE_MINUTES     stop after this many minutes without requests (default 60; 0 = never)
#   OFA_LLAMACPP_BIN     llama-server binary (default $OFA_ROOT/lib/llama.cpp/bin/llama-server)
#   OFA_LLAMACPP_LIBS    CUDA runtime library dir it was built against (default: Kestrel's CUDA 12.8)
set -euo pipefail

OFA_ROOT="${OFA_ROOT:-/nopt/nrel/apps/cpu_stack/software/openfoam/assistant}"
OFA_SCRATCH="${OFA_SCRATCH:-/scratch/$USER}"
: "${OFA_VLLM_MODEL:?set OFA_VLLM_MODEL to the first .gguf shard}"
[[ -f "$OFA_VLLM_MODEL" ]] || { echo "OFA_VLLM_MODEL=$OFA_VLLM_MODEL is not a file"; exit 1; }
NAME="${OFA_VLLM_NAME:-$(basename "$OFA_VLLM_MODEL" .gguf)}"
BIN="${OFA_LLAMACPP_BIN:-$OFA_ROOT/lib/llama.cpp/bin/llama-server}"
export LD_LIBRARY_PATH="${OFA_LLAMACPP_LIBS:-/nopt/nlr/apps/kestrel-gpu/cuda/12.8.1/lib64}:${LD_LIBRARY_PATH:-}"
HOST="$(hostname)"
PORT=$((40000 + RANDOM % 10000))
ENDPOINT="${OFA_VLLM_ENDPOINT:-$OFA_SCRATCH/.ofa_vllm_endpoint.json}"
KEYFILE="$(umask 077; mktemp "$OFA_SCRATCH/.ofa_llamacpp_key.XXXXXX")"
trap 'rm -f "$ENDPOINT" "$KEYFILE"' EXIT
"$OFA_ROOT/env/bin/python3" -c 'import secrets; print(secrets.token_urlsafe(32))' > "$KEYFILE"
API_KEY="$(cat "$KEYFILE")"

echo "llama.cpp: $NAME on $HOST:$PORT ($(cat "$OFA_ROOT/lib/llama.cpp/VERSION" 2>/dev/null))"
# A GPU that is listed but unusable makes llama-server fail late and obscurely; check up front.
if ! nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >/dev/null 2>&1; then
    echo "GPU preflight failed on $HOST"; exit 1
fi

# --api-key-file keeps the key out of `ps`; the web UI and its MCP proxy are not needed.
"$BIN" -m "$OFA_VLLM_MODEL" -a "$NAME" --host 0.0.0.0 --port "$PORT" --api-key-file "$KEYFILE" \
    --metrics --no-webui --no-webui-mcp-proxy -ngl 999 $OFA_VLLM_ARGS &
SRV=$!

deadline=$((SECONDS + ${OFA_VLLM_STARTUP_TIMEOUT:-2400}))
until curl -sf "http://$HOST:$PORT/health" >/dev/null; do
    kill -0 "$SRV" 2>/dev/null || { echo "llama-server exited before becoming healthy"; wait "$SRV"; exit 1; }
    (( SECONDS < deadline )) || { echo "llama-server not healthy after ${OFA_VLLM_STARTUP_TIMEOUT:-2400}s; stopping"; kill "$SRV"; exit 1; }
    sleep 10
done
( umask 077
  printf '{"base_url": "http://%s:%s/v1", "model": "%s", "api_key": "%s", "job_id": "%s"}\n' \
      "$HOST" "$PORT" "$NAME" "$API_KEY" "$SLURM_JOB_ID" > "$ENDPOINT" )
echo "llama.cpp ready: http://$HOST:$PORT/v1 (model '$NAME'). Run: ofa --vllm"

IDLE_MIN="${OFA_VLLM_IDLE_MINUTES:-60}"
if (( IDLE_MIN > 0 )); then
    activity() {
        curl -sf -H "Authorization: Bearer $API_KEY" "http://$HOST:$PORT/metrics" \
            | awk '/^llamacpp:(prompt_tokens_total|tokens_predicted_total|requests_processing|requests_deferred) / {s += $NF} END {print s}'
    }
    last="$(activity || true)"; idle_since=$SECONDS
    while kill -0 "$SRV" 2>/dev/null; do
        sleep 60
        now="$(activity || true)"
        if [[ "$now" != "$last" ]]; then last="$now"; idle_since=$SECONDS; fi
        if (( SECONDS - idle_since >= IDLE_MIN * 60 )); then
            echo "No requests for $IDLE_MIN min; stopping llama-server at $(date +%T)"
            rm -f "$ENDPOINT"
            kill "$SRV"; wait "$SRV" || true
            exit 0
        fi
    done
fi
wait "$SRV"
