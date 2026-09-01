#!/bin/bash
#SBATCH --job-name=ofa-rag-update
#SBATCH --account=hpcapps
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#
# Self-resubmitting recurring RAG updater.
#
# Runs tools/update-rag.sh, then submits its own successor to run again
# after $OFA_RAG_UPDATE_INTERVAL (default 1 day), forming a chain — the
# HPC-native substitute for cron, which is disabled by PAM on the login
# nodes.
#
# Allocation mirrors what `ofa` itself requests (debug partition, 32
# cores, 80G RAM, 1 GPU, 30 min, account hpcapps) so the embedding step
# runs on the GPU. Only changed files are re-embedded (mtime cache), so a
# routine run is quick.
#
# Start the chain once with:
#   sbatch tools/update-rag.sbatch [collections...]
# e.g.
#   sbatch tools/update-rag.sbatch all
#   sbatch tools/update-rag.sbatch hpc_docs vasp_src
#
# Tunables (pass via --export or env):
#   OFA_RAG_UPDATE_INTERVAL   when to run next (default "now+1day";
#                             any `sbatch --begin=` value, e.g. "now+6hours")
#   OFA_RAG_UPDATE_COLLECTIONS the collection list, if not given as args
#
# Stop the chain: `scancel` the pending ofa-rag-update job (squeue -n
# ofa-rag-update) — with no running job left to resubmit, the chain ends.

set -uo pipefail

# Resolve OFA_ROOT from this script's own path (works whether launched by
# sbatch from anywhere, since SLURM copies the script but sets SLURM_SUBMIT_DIR).
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/tools/update-rag.sbatch" ]; then
    OFA_ROOT="$SLURM_SUBMIT_DIR"
else
    # Fallback: the Kestrel deploy path.
    OFA_ROOT="/nopt/nrel/apps/cpu_stack/software/openfoam/assistant"
fi
export OFA_ROOT

INTERVAL="${OFA_RAG_UPDATE_INTERVAL:-now+1day}"

# Collections: prefer args, then env, then the script default (hpc_docs).
if [ "$#" -gt 0 ]; then
    COLLECTIONS=("$@")
elif [ -n "${OFA_RAG_UPDATE_COLLECTIONS:-}" ]; then
    read -r -a COLLECTIONS <<< "$OFA_RAG_UPDATE_COLLECTIONS"
else
    COLLECTIONS=(hpc_docs)
fi

echo "[$(date '+%F %T')] ofa-rag-update job ${SLURM_JOB_ID:-<local>} starting: collections=${COLLECTIONS[*]}"

# --- resubmit the successor FIRST -----------------------------------------
# Submit before running the work so a failure in the update itself doesn't
# break the recurring chain (we'd rather keep trying tomorrow than stop).
# Guard against accidental runaway chains if the partition rejects --begin.
_acct_arg=""
[ -n "${SLURM_JOB_ACCOUNT:-}" ] && _acct_arg="--account=$SLURM_JOB_ACCOUNT"
if command -v sbatch >/dev/null 2>&1; then
    if sbatch --begin="$INTERVAL" $_acct_arg \
              --export=ALL,OFA_RAG_UPDATE_INTERVAL="$INTERVAL" \
              "$OFA_ROOT/tools/update-rag.sbatch" "${COLLECTIONS[@]}"; then
        echo "[$(date '+%F %T')] queued next run at: $INTERVAL"
    else
        echo "[$(date '+%F %T')] WARNING: could not queue successor; chain will stop after this run" >&2
    fi
fi

# --- do the actual update -------------------------------------------------
bash "$OFA_ROOT/tools/update-rag.sh" "${COLLECTIONS[@]}"
_rc=$?

echo "[$(date '+%F %T')] update finished with status $_rc"
exit "$_rc"
