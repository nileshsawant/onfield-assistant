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

# Declarative map: collection -> space-separated "dir:branch" source specs.
# Keep in sync with collections.toml. Each source dir may be:
#   * a git clone            -> pulled on the given branch
#   * a container of clones  -> each nested clone pulled on its OWN branch
#     (the :branch here is then ignored; e.g. quantum-code holds several
#     independent project repos)
#   * a non-git snapshot     -> nothing to pull, rebuild only (e.g. the
#     vendored *-papers PDF drops, vasp)
# pull_repo() auto-detects which shape each dir is, so a source can change
# shape over time without editing this map.
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

# Never let git block this unattended job on an interactive prompt: SSH in
# batch mode (fail instead of asking for a key/password) with a short
# connect timeout, and disable git's credential prompt.
export GIT_SSH_COMMAND="ssh -o BatchMode=yes -o ConnectTimeout=10"
export GIT_TERMINAL_PROMPT=0

# Fetch + hard-align one clone to a branch. These are read-only mirror
# clones for RAG, so a hard reset is safe and avoids merge prompts if
# someone touched the working tree. Branch defaults to the checked-out one.
_pull_one() {
    local path="$1" branch="${2:-}"
    [ -z "$branch" ] && branch="$(git -C "$path" branch --show-current 2>/dev/null)"
    [ -z "$branch" ] && { log "ERROR: no branch for $path (detached HEAD?)"; return 1; }
    log "pull $(basename "$path") ($branch)"
    git -C "$path" fetch --quiet origin "$branch" \
        && git -C "$path" checkout --quiet "$branch" \
        && git -C "$path" reset --hard --quiet "origin/$branch" \
        || { log "ERROR: git pull failed for $path"; return 1; }
}

# Pull the git source backing a collection. Three shapes are handled:
#   1. the dir is itself a clone            -> pull it on the mapped branch
#   2. the dir contains nested clones       -> pull each on its own branch
#      (e.g. repos/quantum-code holds several independent project repos)
#   3. the dir is a non-git snapshot        -> nothing to pull (rebuild only)
# Dedupe by dir so a source shared by two collections is pulled once.
declare -A _pulled=()
pull_repo() {
    local spec="$1" dir branch
    dir="${spec%%:*}"; branch="${spec##*:}"
    local path="$OFA_ROOT/repos/$dir"
    if [ -n "${_pulled[$dir]:-}" ]; then return 0; fi
    _pulled[$dir]=1

    if [ -d "$path/.git" ]; then
        _pull_one "$path" "$branch"
        return
    fi

    # Not a clone itself — look for nested clones one level down.
    local rc=0 found=0 sub
    for sub in "$path"/*/; do
        [ -d "$sub/.git" ] || continue
        found=1
        _pull_one "${sub%/}" "" || rc=1   # nested repos use their own branch
    done
    if [ "$found" -eq 0 ]; then
        log "note: $dir is a non-git snapshot; rebuild only (nothing to pull)"
    fi
    return "$rc"
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
