#!/usr/bin/env bash
#
# ofa installer — bootstraps a self-contained ofa install onto any Linux
# HPC login node. Idempotent; can be re-run to update individual pieces.
#
# What it does, in order (each step is opt-out via a flag):
#   1. Detects arch (x86_64 / aarch64).
#   2. Downloads Miniforge and installs to $OFA_ROOT/env (self-contained
#      Python distribution, no dependency on the host's system Python).
#   3. pip installs requirements.txt into that env.
#   4. Downloads the Ollama static binary into $OFA_ROOT/bin/ollama.
#   5. Downloads the BAAI/bge-small-en-v1.5 embedding model into
#      $OFA_ROOT/embedding_model via huggingface-cli.
#   6. Optionally pulls the default LLM (gemma4:31b, ~154 GB).
#   7. Optionally runs the interactive site.toml wizard.
#   8. Optionally rebuilds the RAG indices from collections.toml.
#   9. Writes $OFA_ROOT/env.sh (sourceable) and
#      $OFA_ROOT/tools/modulefile.lua.template (Lmod).
#
# Prereqs:
#   * bash 4+, curl, tar, coreutils (readlink -f), find. Standard on any
#     modern HPC login node.
#   * ~250 GB free disk if you pull the default LLM. ~500 MB for the
#     Python env + Ollama binary + embedding model alone.
#   * Outbound HTTPS to github.com / huggingface.co / ollama.com from
#     the login node.
#
# See ./README.md 'Install on a new HPC' and site.example.toml for the
# porting checklist before running this.

set -euo pipefail

# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------
SKIP_MODEL_PULL=0
SKIP_WIZARD=0
SKIP_INDICES=0
NON_INTERACTIVE=0
FORCE=0
VERBOSE=0
INSTALL_PREFIX=""

usage() {
    cat <<'EOF'
ofa installer.

Usage: ./install.sh [OPTIONS]

Options:
  --skip-model-pull      Do not pull the default LLM. Default: prompt to pull
                         gemma4:31b (~154 GB on disk).
  --skip-wizard          Do not run the interactive site.toml wizard. If
                         no site.toml exists, ofa falls back to the Kestrel
                         defaults hard-coded in src/ofa_site.py.
  --skip-indices         Do not rebuild the RAG indices. You can run
                         src/rebuild_indices.py manually later once you've
                         populated repos/ with your source dirs.
  --non-interactive      Do not prompt for anything. Opt-in steps default
                         to skipped; existing artifacts are kept as-is.
  --force                Redo work even if artifacts already exist
                         (re-downloads Miniforge, Ollama, embedding model,
                         overwrites site.toml).
  --verbose              Enable set -x tracing.
  --prefix PATH          Install target. Default: the directory install.sh
                         lives in (i.e. the repo root you cloned).
  -h, --help             This message.

Environment overrides (all optional):
  OFA_INSTALL_PYTHON_VERSION   Miniforge base python version (default: unpinned).
  OFA_INSTALL_OLLAMA_VERSION   Pin an Ollama release, e.g. v0.5.4 (default: latest).
  OFA_INSTALL_MODEL_ID         Model to pull (default: gemma4:31b).
  OFA_INSTALL_EMBEDDING_MODEL  HuggingFace repo id (default: BAAI/bge-small-en-v1.5).
  OFA_INSTALL_KEEP_INSTALLER   Set to 1 to keep the downloaded miniforge/ollama
                               tarballs after install for debugging.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-model-pull) SKIP_MODEL_PULL=1; shift ;;
        --skip-wizard)     SKIP_WIZARD=1; shift ;;
        --skip-indices)    SKIP_INDICES=1; shift ;;
        --non-interactive) NON_INTERACTIVE=1; shift ;;
        --force)           FORCE=1; shift ;;
        --verbose)         VERBOSE=1; shift ;;
        --prefix)          INSTALL_PREFIX="${2:-}"; shift 2 ;;
        -h|--help)         usage; exit 0 ;;
        *) echo "Unknown flag: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ $VERBOSE -eq 1 ]] && set -x

# ---------------------------------------------------------------------------
# Prefix resolution — default to the dir install.sh lives in.
# ---------------------------------------------------------------------------
if [[ -z "$INSTALL_PREFIX" ]]; then
    INSTALL_PREFIX="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
INSTALL_PREFIX="$(readlink -f "$INSTALL_PREFIX")"
mkdir -p "$INSTALL_PREFIX"

# Sanity: refuse to install into a directory that doesn't look like an
# ofa checkout. Prevents the "user typo'd --prefix and we started
# scribbling into ~/Downloads" failure mode.
if [[ ! -f "$INSTALL_PREFIX/src/ofa_main.py" || ! -f "$INSTALL_PREFIX/bin/ofa" ]]; then
    cat >&2 <<EOF
ERROR: --prefix ($INSTALL_PREFIX) does not look like an ofa checkout.
       Expected to find src/ofa_main.py and bin/ofa there.

If you haven't cloned the repo yet:
    git clone https://github.com/nileshsawant/onfield-assistant.git
    cd onfield-assistant
    ./install.sh
EOF
    exit 1
fi

OFA_ROOT="$INSTALL_PREFIX"
cd "$OFA_ROOT"

# ---------------------------------------------------------------------------
# Arch detection.
# ---------------------------------------------------------------------------
UNAME_M="$(uname -m)"
case "$UNAME_M" in
    x86_64|amd64)   MINIFORGE_ARCH="x86_64";  OLLAMA_ARCH="amd64" ;;
    aarch64|arm64)  MINIFORGE_ARCH="aarch64"; OLLAMA_ARCH="arm64" ;;
    *) echo "ERROR: unsupported arch: $UNAME_M" >&2; exit 1 ;;
esac

log() { echo "[ofa-install] $*" >&2; }

# ---------------------------------------------------------------------------
# 1. Miniforge — self-contained conda distribution. Installed at
#    $OFA_ROOT/env so bin/ofa's `env/bin/python3` lookup finds it.
# ---------------------------------------------------------------------------
install_miniforge() {
    if [[ -x "$OFA_ROOT/env/bin/python3" && $FORCE -eq 0 ]]; then
        log "env/bin/python3 already present; skipping Miniforge bootstrap (--force to redo)"
        return 0
    fi
    local mf_url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${MINIFORGE_ARCH}.sh"
    local mf_installer
    mf_installer="$(mktemp -t miniforge.XXXXXX.sh)"
    log "downloading Miniforge for $MINIFORGE_ARCH"
    curl -fSL "$mf_url" -o "$mf_installer"

    if [[ -d "$OFA_ROOT/env" ]]; then
        log "removing existing env/ (--force)"
        rm -rf "$OFA_ROOT/env"
    fi
    log "installing Miniforge to $OFA_ROOT/env"
    bash "$mf_installer" -b -p "$OFA_ROOT/env"

    if [[ -n "${OFA_INSTALL_PYTHON_VERSION:-}" ]]; then
        log "pinning python=${OFA_INSTALL_PYTHON_VERSION}"
        "$OFA_ROOT/env/bin/mamba" install -y "python=${OFA_INSTALL_PYTHON_VERSION}"
    fi

    if [[ "${OFA_INSTALL_KEEP_INSTALLER:-0}" != "1" ]]; then
        rm -f "$mf_installer"
    fi
    log "Miniforge ready"
}

# ---------------------------------------------------------------------------
# 2. Python deps from requirements.txt.
# ---------------------------------------------------------------------------
install_python_deps() {
    if [[ ! -f "$OFA_ROOT/requirements.txt" ]]; then
        log "WARNING: no requirements.txt found; skipping pip install"
        return 0
    fi
    log "upgrading pip"
    "$OFA_ROOT/env/bin/pip" install --upgrade --quiet pip
    log "pip install -r requirements.txt"
    "$OFA_ROOT/env/bin/pip" install -r "$OFA_ROOT/requirements.txt"
}

# ---------------------------------------------------------------------------
# 3. Ollama static binary. Ships as a tarball whose top-level layout is
#    bin/ + lib/, matching the ofa repo's own layout — we untar directly
#    into $OFA_ROOT.
# ---------------------------------------------------------------------------
install_ollama() {
    local ollama_bin="$OFA_ROOT/bin/ollama"
    if [[ -x "$ollama_bin" && $FORCE -eq 0 ]]; then
        log "bin/ollama already present; skipping (--force to redo)"
        return 0
    fi
    local version="${OFA_INSTALL_OLLAMA_VERSION:-latest}"
    local url
    if [[ "$version" == "latest" ]]; then
        url="https://github.com/ollama/ollama/releases/latest/download/ollama-linux-${OLLAMA_ARCH}.tgz"
    else
        url="https://github.com/ollama/ollama/releases/download/${version}/ollama-linux-${OLLAMA_ARCH}.tgz"
    fi
    log "downloading Ollama ($version) for $OLLAMA_ARCH"
    local tgz
    tgz="$(mktemp -t ollama.XXXXXX.tgz)"
    curl -fSL "$url" -o "$tgz"
    tar -xzf "$tgz" -C "$OFA_ROOT"
    chmod +x "$ollama_bin"
    if [[ "${OFA_INSTALL_KEEP_INSTALLER:-0}" != "1" ]]; then
        rm -f "$tgz"
    fi
    log "ollama installed at $ollama_bin"
}

# ---------------------------------------------------------------------------
# 4. Embedding model — pinned to BAAI/bge-small-en-v1.5, matching what
#    the current Kestrel install uses. Small (~120 MB) and CPU-friendly.
# ---------------------------------------------------------------------------
install_embedding_model() {
    if [[ -f "$OFA_ROOT/embedding_model/config.json" && $FORCE -eq 0 ]]; then
        log "embedding_model/ already populated; skipping (--force to redo)"
        return 0
    fi
    local model_id="${OFA_INSTALL_EMBEDDING_MODEL:-BAAI/bge-small-en-v1.5}"
    log "downloading embedding model $model_id"
    # huggingface-cli is a transitive dep of sentence-transformers, so it
    # should already be on PATH via env/bin/. Explicit fallback install
    # just in case a stripped-down requirements.txt drops it.
    if ! "$OFA_ROOT/env/bin/huggingface-cli" --help >/dev/null 2>&1; then
        "$OFA_ROOT/env/bin/pip" install --quiet 'huggingface-hub[cli]'
    fi
    mkdir -p "$OFA_ROOT/embedding_model"
    "$OFA_ROOT/env/bin/huggingface-cli" download "$model_id" \
        --local-dir "$OFA_ROOT/embedding_model"
    log "embedding model at $OFA_ROOT/embedding_model"
}

# ---------------------------------------------------------------------------
# 5. LLM pull — big, so opt-out. Starts a temporary ollama daemon if none
#    is running, does the pull, then kills the daemon.
# ---------------------------------------------------------------------------
pull_default_model() {
    if [[ $SKIP_MODEL_PULL -eq 1 ]]; then
        log "skipping ollama pull (per --skip-model-pull)"
        return 0
    fi
    local model_id="${OFA_INSTALL_MODEL_ID:-gemma4:31b}"

    cat <<EOF
[ofa-install] Next step: pull LLM '$model_id' (~50-150 GB on disk).
              Storage location: $OFA_ROOT/models
              To skip and configure the model yourself, re-run install.sh
              with --skip-model-pull, or Ctrl+C now and pull later:
                  export OLLAMA_MODELS=$OFA_ROOT/models
                  $OFA_ROOT/bin/ollama serve &
                  $OFA_ROOT/bin/ollama pull $model_id
EOF
    if [[ $NON_INTERACTIVE -eq 0 ]]; then
        local reply
        read -r -p "Continue with model pull? [Y/n] " reply || reply=""
        if [[ "${reply,,}" =~ ^n ]]; then
            log "user declined; skipping model pull"
            return 0
        fi
    else
        log "non-interactive mode: skipping model pull (opt-in only)"
        return 0
    fi

    mkdir -p "$OFA_ROOT/models"
    export OLLAMA_MODELS="$OFA_ROOT/models"

    local started_daemon=0
    local ollama_pid=""
    if "$OFA_ROOT/bin/ollama" list >/dev/null 2>&1; then
        log "existing ollama daemon detected; reusing"
    else
        log "starting temporary ollama daemon (log: /tmp/ofa-install-ollama.log)"
        "$OFA_ROOT/bin/ollama" serve > /tmp/ofa-install-ollama.log 2>&1 &
        ollama_pid=$!
        started_daemon=1
        # Wait for the daemon to accept requests (max ~30 s).
        local i
        for i in $(seq 1 30); do
            if "$OFA_ROOT/bin/ollama" list >/dev/null 2>&1; then
                break
            fi
            sleep 1
        done
    fi

    log "pulling $model_id (large download; be patient)"
    "$OFA_ROOT/bin/ollama" pull "$model_id"

    if [[ $started_daemon -eq 1 && -n "$ollama_pid" ]]; then
        kill "$ollama_pid" 2>/dev/null || true
        wait "$ollama_pid" 2>/dev/null || true
    fi
    log "model pull complete"
}

# ---------------------------------------------------------------------------
# 6. site.toml wizard — interactive; skipped by default in --non-interactive
#    mode (which copies site.example.toml instead so the porter has a
#    starting point). Fields match src/ofa_site.py's DEFAULTS.
# ---------------------------------------------------------------------------
site_wizard() {
    if [[ $SKIP_WIZARD -eq 1 ]]; then
        log "skipping site.toml wizard (per --skip-wizard)"
        return 0
    fi
    if [[ -f "$OFA_ROOT/site.toml" && $FORCE -eq 0 ]]; then
        log "site.toml already present; skipping wizard (--force to overwrite)"
        return 0
    fi
    if [[ $NON_INTERACTIVE -eq 1 ]]; then
        if [[ ! -f "$OFA_ROOT/site.toml" ]]; then
            log "non-interactive: copying site.example.toml -> site.toml (edit before running ofa)"
            cp "$OFA_ROOT/site.example.toml" "$OFA_ROOT/site.toml"
        fi
        return 0
    fi

    echo
    echo "======================================================================"
    echo "                       site.toml wizard"
    echo "======================================================================"
    echo "Answer or press Enter to accept the default in brackets."
    echo "You can edit site.toml later — see site.example.toml for the schema."
    echo

    local site_name site_org site_long site_desc login_host
    local partition gres mem walltime protected
    read -r -p "Site name (short)          [MySiteHPC]: " site_name; site_name="${site_name:-MySiteHPC}"
    read -r -p "Sponsoring org / lab       [MyLab]: "     site_org;  site_org="${site_org:-MyLab}"
    local long_default="the ${site_name} supercomputer"
    read -r -p "Long name for prompts      [${long_default}]: " site_long; site_long="${site_long:-$long_default}"
    read -r -p "GPU descriptor for banner  [single A100]: " site_desc; site_desc="${site_desc:-single A100}"
    local host_default; host_default="$(echo "${site_name}" | tr '[:upper:]' '[:lower:]').example.edu"
    read -r -p "SSH login host             [${host_default}]: " login_host; login_host="${login_host:-$host_default}"
    read -r -p "Slurm partition            [gpu]: "     partition; partition="${partition:-gpu}"
    read -r -p "Slurm GRES                 [gpu:1]: "   gres;      gres="${gres:-gpu:1}"
    read -r -p "Slurm --mem                [80G]: "     mem;       mem="${mem:-80G}"
    read -r -p "Slurm --time default       [00:30:00]: " walltime; walltime="${walltime:-00:30:00}"
    local prot_default; prot_default="/opt/$(echo "${site_name}" | tr '[:upper:]' '[:lower:]')"
    read -r -p "Protected root path        [${prot_default}]: " protected; protected="${protected:-$prot_default}"

    cat > "$OFA_ROOT/site.toml" <<TOML
# ofa site.toml — generated by install.sh $(date -u +%Y-%m-%dT%H:%M:%SZ).
# Edit by hand or re-run 'install.sh --force' to regenerate.
# Schema and porting notes: see site.example.toml.

[site]
name = "${site_name}"
org = "${site_org}"
long_name = "${site_long}"
description = "${site_desc}"
login_host = "${login_host}"
protected_roots = ["${protected}"]

[scheduler]
kind = "slurm"
partition = "${partition}"
gres = "${gres}"
mem = "${mem}"
ntasks_per_node = 32
walltime = "${walltime}"
account_discovery = 'sacctmgr show user "\$USER" format=defaultaccount -nP 2>/dev/null | head -1'

[modules]
# Set to empty string to skip module load (sites without Lmod).
cuda_rhel8 = "cuda"
cuda_rhel9 = "cuda"
TOML
    log "wrote $OFA_ROOT/site.toml"
    echo
    echo "  Review your site.toml, and audit prompts/*.txt for Kestrel-specific"
    echo "  technical content (see site.example.toml § 'ALSO REVIEW when porting')."
    echo
}

# ---------------------------------------------------------------------------
# 7. RAG indices. Only runs if repos/ is populated — otherwise print the
#    manual re-run command so a porter can populate repos/ later.
# ---------------------------------------------------------------------------
rebuild_indices() {
    if [[ $SKIP_INDICES -eq 1 ]]; then
        log "skipping RAG index rebuild (per --skip-indices)"
        return 0
    fi
    if [[ ! -d "$OFA_ROOT/repos" ]] || [[ -z "$(ls -A "$OFA_ROOT/repos" 2>/dev/null)" ]]; then
        log "repos/ is empty; skipping RAG index rebuild"
        log "  populate repos/ with your source dirs (see collections.toml), then run:"
        log "  $OFA_ROOT/env/bin/python3 $OFA_ROOT/src/rebuild_indices.py"
        return 0
    fi
    log "rebuilding RAG indices from collections.toml"
    "$OFA_ROOT/env/bin/python3" "$OFA_ROOT/src/rebuild_indices.py"
}

# ---------------------------------------------------------------------------
# 8. env.sh + Lmod modulefile template.
# ---------------------------------------------------------------------------
emit_env_files() {
    cat > "$OFA_ROOT/env.sh" <<EOF
# ofa environment activation.
# Source this from your shell or from an sbatch script to make 'ofa'
# available without loading a module:
#     source $OFA_ROOT/env.sh
#     ofa --hpc
#
# Generated by install.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ).
export OFA_ROOT="$OFA_ROOT"
export OLLAMA_MODELS="\$OFA_ROOT/models"
export OFA_VECTORDB="\$OFA_ROOT/vectordb"
export PATH="\$OFA_ROOT/bin:\$OFA_ROOT/env/bin:\$PATH"
# Make 'import ofa_client' work in any Python env loaded after this.
export PYTHONPATH="\$OFA_ROOT/src:\${PYTHONPATH:-}"
EOF
    log "wrote $OFA_ROOT/env.sh"

    mkdir -p "$OFA_ROOT/tools"
    cat > "$OFA_ROOT/tools/modulefile.lua.template" <<EOF
-- Lmod module for ofa.
--
-- Generated by install.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ).
-- Deploy by copying under your MODULEPATH, e.g.:
--     cp $OFA_ROOT/tools/modulefile.lua.template <MODULEPATH>/assistant/1.0.lua
--
help([[Name   : ofa (OnField Assistant)]])
help([[Root   : $OFA_ROOT]])
help()
help([[Usage: 'ofa' interactively, 'ofa --serve' for BYOK endpoint.]])
help([[Configure your site by editing $OFA_ROOT/site.toml.]])

local root = "$OFA_ROOT"

prepend_path("PATH",       pathJoin(root, "bin"))
prepend_path("PATH",       pathJoin(root, "env", "bin"))
prepend_path("PYTHONPATH", pathJoin(root, "src"))

setenv("OFA_ROOT",      root)
setenv("OLLAMA_MODELS", pathJoin(root, "models"))
setenv("OFA_VECTORDB",  pathJoin(root, "vectordb"))
EOF
    log "wrote $OFA_ROOT/tools/modulefile.lua.template"
}

# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
main() {
    log "install prefix: $OFA_ROOT (arch: $UNAME_M)"
    install_miniforge
    install_python_deps
    install_ollama
    install_embedding_model
    pull_default_model
    site_wizard
    rebuild_indices
    emit_env_files

    cat <<EOF

======================================================================
[ofa-install] Done.
======================================================================

Next steps:

  1. Activate the environment:
       source $OFA_ROOT/env.sh

     Or deploy the Lmod module template:
       cp $OFA_ROOT/tools/modulefile.lua.template <your MODULEPATH>/assistant/1.0.lua
       module load assistant

  2. Verify:
       ofa --help

  3. First run allocates a GPU node via Slurm. Ensure your site.toml
     [scheduler] section is correct (partition / gres / account discovery).

Files worth reviewing before your first real run:
  $OFA_ROOT/site.toml                    site-specific config
  $OFA_ROOT/site.example.toml            schema + porter's audit checklist
  $OFA_ROOT/collections.toml             RAG source-to-collection mapping
  $OFA_ROOT/prompts/*.txt                site-specific technical content
                                          (see 'ALSO REVIEW' in site.example.toml)
EOF
}

main
