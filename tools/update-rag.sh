#!/bin/bash
# update-rag.sh — pull git-backed RAG source repos and rebuild their
# ChromaDB collections. Safe to run on a login node (embedding falls back
# to CPU) and idempotent (rebuild_indices.py mtime-caches, so unchanged
# files are not re-embedded).
#
# Usage:
#   tools/update-rag.sh                 # default: hpc_docs only
#   tools/update-rag.sh hpc_docs        # one collection
#   tools/update-rag.sh hpc_docs vasp_src reframe_src
#   tools/update-rag.sh all             # every git-backed collection below
#   tools/update-rag.sh --list          # show the collection -> repo map
#
# Exit status is non-zero if any requested collection fails, so cron can
# alert on it.

set -uo pipefail

# Resolve OFA_ROOT from this script's location (tools/ is directly under it).
_SELF="$(readlink -f "${BASH_SOURCE[0]}")"
OFA_ROOT="$(cd "$(dirname "$_SELF")/.." && pwd)"
export OFA_ROOT

PYTHON="$OFA_ROOT/env/bin/python3"
REBUILD="$OFA_ROOT/src/rebuild_indices.py"

# Declarative map: collection -> "repo_subdir<TAB>branch". Only collections
# whose sources are git clones we control belong here. Keep in sync with
# collections.toml. A collection with multiple git sources (e.g. marbles,
# quantum_computing) lists them space-separated in the repos field, still
# paired with a branch each via the "dir:branch" form.
declare -A REPO_MAP=(
    [hpc_docs]="HPC:gh-pages"
    [reframe_src]="reframe-universal:rh9"
    [marbles_src]="marblesThermal:moving-body marbles-papers:main"
    [quantum_computing]="quantum-code:main quantum-papers:main"
    [vasp_src]="vasp:main"
)

_ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[update-rag $(_ts)] $*" >&2; }

list_map() {
    echo "Configured git-backed collections:" >&2
    for c in "${!REPO_MAP[@]}"; do
        printf '  %-18s <- %s\n' "$c" "${REPO_MAP[$c]}" >&2
    done
}

if [ "${1:-}" = "--list" ]; then
    list_map
    exit 0
fi

# Resolve the requested collection set.
if [ "$#" -eq 0 ]; then
    COLLECTIONS=(hpc_docs)               # default
elif [ "$1" = "all" ]; then
    COLLECTIONS=("${!REPO_MAP[@]}")
else
    COLLECTIONS=("$@")
fi

# Pull each git source backing the requested collections. Dedupe repos so a
# repo shared by two collections is only pulled once per run.
declare -A _pulled=()
pull_repo() {
    local spec="$1" dir branch
    dir="${spec%%:*}"; branch="${spec##*:}"
    local path="$OFA_ROOT/repos/$dir"
    if [ -n "${_pulled[$dir]:-}" ]; then return 0; fi
    _pulled[$dir]=1
    if [ ! -d "$path/.git" ]; then
        log "WARN: $path is not a git clone; skipping pull (will still rebuild)"
        return 0
    fi
    log "pull $dir ($branch)"
    # Fetch + hard-align to the tracked branch. These are read-only mirror
    # clones for RAG, so a hard reset is safe and avoids merge prompts if
    # someone touched the working tree.
    git -C "$path" fetch --quiet origin "$branch" \
        && git -C "$path" checkout --quiet "$branch" \
        && git -C "$path" reset --hard --quiet "origin/$branch" \
        || { log "ERROR: git pull failed for $dir"; return 1; }
}

rc=0
for coll in "${COLLECTIONS[@]}"; do
    spec="${REPO_MAP[$coll]:-}"
    if [ -z "$spec" ]; then
        log "ERROR: '$coll' is not a git-backed collection. Try --list."
        rc=1
        continue
    fi
    # Pull every git source for this collection.
    for src in $spec; do
        pull_repo "$src" || rc=1
    done
    # Rebuild just this collection (mtime-cached; only changed files re-embed).
    log "rebuild collection $coll"
    if ! "$PYTHON" "$REBUILD" --collection "$coll"; then
        log "ERROR: rebuild failed for $coll"
        rc=1
    fi
done

if [ "$rc" -eq 0 ]; then
    log "done (all requested collections updated)"
else
    log "done WITH ERRORS (see above)"
fi
exit "$rc"
