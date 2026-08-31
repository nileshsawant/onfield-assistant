# OnField Assistant 🌵 (ofa)

An AI-powered reasoning and autonomous execution agent tailored for the NLR Kestrel HPC system. OnField Assistant (`ofa`) utilizes a local Large Language Model orchestrator (powered by Ollama and `gemma4:31b-it-q8_0`) alongside Retrieval-Augmented Generation (RAG) using ChromaDB to help researchers build cases, compile complex scientific codebases, organically navigate Kestrel's HPC documentation, and execute multi-step SLURM jobs natively.

## Features

* **Automated Hardware Allocation:** When invoked from a Kestrel login node, the `ofa` wrapper seamlessly detects the environment and spins up an interactive GPU node allocation (targeting H100 partitions). It automatically sets up the appropriate CUDA modules for either RHEL 8 or RHEL 9 environments before launching the LLM.
* **Autonomous ReAct Framework (Plan & Execute):** The assistant is instructed to follow a strict planning phase before acting. It creates persistent `plan` blocks and iterates autonomously using a variety of parsed markdown blocks:
  * `write` and `edit`: For creating and modifying files in the current workspace.
  * `bash`: For interactive terminal commands with real-time `subprocess.Popen` streaming back to the user.
  * `sbatch`: For dispatching background jobs and checking on their SLURM queue status without freezing the conversational loop.
  * `search` and `fetch`: For searching the internet or reading external web documentation when unsure.
* **Domain-Specific Modes:** By passing command-line arguments, the overarching python execution loop swaps the injected system prompts and RAG databases to act as specialized domain experts (e.g., general codebase engineering, specific AMReX compilation, or ReFrame module migrations).
* **Private data indexing:** Each user can index their own data (`ofa --add-private <dir>`) into a per-user store that is retrieved automatically alongside the shared corpora — no write access to the shared install required. See [Index your own private data](#index-your-own-private-data).
* **Intelligent Context Management:** To survive long debugging or compilation sessions, the agent intelligently handles Context Window Collapse. Massive compiler toolchains are dynamically truncated. In deeply extended sessions (over 20 turns), older terminal stdout logs are systematically compressed while preserving the agent's fundamental reasoning and the user's initial instructions to avoid amnesia.
* **Robust Fault Tolerance & Safeguards:** The Python orchestrator natively intercepts hanging shell commands with `/dev/null` stdin piping. It tracks consecutive execution errors, pausing the autonomous loop if the agent hallucinates a failing command 3 times in a row, dropping control back to the human user. The daemon catches `SIGTERM` signals for 30-minute allocation timeouts, shutting down gracefully.

## Usage

Run the `ofa` command from a Kestrel login node **after loading the
`assistant` module**. That module puts `ofa` on `$PATH`, exports
`$OFA_ROOT`, and wires up the Python + ChromaDB + Ollama toolchain
bundled with the deploy.

```bash
module load assistant
ofa --help
```

If you'd rather drive `ofa` from VS Code Chat (the eight modes appear
directly in Copilot Chat's model picker) instead of the shell, skip
straight to [Use `ofa` from VS Code Chat](#use-ofa-from-vs-code-chat-the-onfield-assistant-extension)
below — that section handles the module load automatically inside a
one-click SLURM allocation. The CLI examples in this section are for
the terminal-native path.

```bash
# Default mode: general coding / software-engineering assistant
$ ofa

# Single-shot query (runs without a continuous prompt loop)
$ ofa "explain what a __global__ kernel does in CUDA"

# Specialized Agent Modes
$ ofa --openfoam        # OpenFOAM case generator (was the default in versions <= 1.0)
$ ofa --code            # General coding assistant (redundant — this is the default now)
$ ofa --hpc             # Kestrel HPC, SLURM documentation, and topology expert
$ ofa --amrex           # AMReX C++ framework assistant
$ ofa --marbles         # MARBLES (LBM thermal solver on AMReX) assistant
$ ofa --quantum-computing # Quantum computing (rigorous verification of gates / unitarity / tensor order)
$ ofa --vasp            # VASP (Vienna Ab initio Simulation Package) — tutorials, notes, pinned build/module doc
$ ofa --rhel9_reframe   # ReFrame integration expert strictly adhering to the Kestrel RHEL9 software stack

# Additional Flags
$ ofa --resume          # Resume the previous interactive session (uses ~/.ofa_session.json)
$ ofa --save <dir>      # (with --openfoam) save generated template cases to a directory
$ ofa --no-rag          # Disable ChromaDB context retrieval; relies solely on standard LLM weights
$ ofa --fast            # (with --openfoam) single-shot file generation (skip plan stage)

# BYOK / programmatic server (OpenAI-compatible HTTP endpoint on this node)
$ ofa --serve                    # Start the local HTTP server; see docs/byok-vscode.md
$ ofa --serve --serve-enable-tools   # Also forward OpenAI tool_calls to Ollama (experimental)
$ ofa --serve --serve-quiet      # Silence per-request stderr log (banner + errors still print).
                                 # Handy when --serve shares a shell with an interactive TUI
                                 # (e.g. opencode on RHEL9) so background traffic doesn't leak
                                 # into the foreground display.
```

## Architecture

* **`bin/ofa`**: The frontend Bash CLI wrapper. It routes SLURM jobs, sanitizes nested PMI and SLURM context variables to avoid step allocation deadlocks, natively forces the correct CUDA toolkit loads, and acts as the entrypoint for the Python environment.
* **`src/ofa_main.py`**: The central python controller. Handles Ollama binary lifecycle management via `subprocess`, processes ChromaDB interactions (`_hybrid_search`), loops user input, captures output, and dictates the strict regex parsing logic for the tool-calling mechanism.
* **`src/ofa_server.py`**: OpenAI-compatible HTTP shim used by `ofa --serve` — exposes ofa's system prompts + RAG + memory + multimodal (vision) at `/v1/chat/completions` for VS Code BYOK, `curl`, and the Python client below.
* **`src/ofa_client.py`**: Zero-dependency (stdlib-only) Python client. `from ofa_client import ask, Session` — see the next section.
* **`src/rebuild_indices.py`**: Config-driven RAG (re)ingester. Reads `collections.toml` and refreshes the ChromaDB collections declared there. See [Updating the RAG indices](#updating-the-rag-indices).
* **`collections.toml`**: Declarative source-to-collection mapping consumed by the rebuild script (extend it when you add a new source or a new collection).
* **`examples/`**: Worked end-to-end scripts users can copy verbatim. Currently ships `fit_and_ask.py` — a `curve_fit` demo that shows the `Session` + JSON-extraction patterns in ~230 LOC.
* **`prompts/`**: Directory configuring the personas. `common.txt` establishes the global rules for the agent, establishing the planning pipeline, code syntax standards, and environment constraints. `code.txt`, `hpc.txt`, and others inject the role-specific capabilities.
* **`vectordb/`**: The persistent storage directory for the offline ChromaDB ingestors, containing chunked embeddings for Kestrel's manuals, OpenFOAM examples, and RHEL module stacks.

## Install on a new HPC

ofa was written for NLR's Kestrel but is portable to any Linux HPC with a Slurm scheduler and a GPU node. The installer bootstraps everything you need — a private Miniforge Python distribution, the Ollama binary, an embedding model, and (optionally) the default LLM — into the repo checkout itself. No system-wide changes.

```bash
git clone https://github.com/nileshsawant/onfield-assistant.git
cd onfield-assistant
./install.sh
```

What it does, in order (each step is opt-out via a flag):

1. **Miniforge** — downloads and installs into `env/`, giving you a self-contained Python 3.11+ without touching the host's system Python.
2. **Python deps** — `pip install -r requirements.txt` inside that env (chromadb, sentence-transformers, httpx, and friends).
3. **Ollama binary** — pulls the latest static release from github.com/ollama/ollama into `bin/ollama`. Arch-aware (x86_64 / aarch64).
4. **Embedding model** — snapshots `BAAI/bge-small-en-v1.5` from HuggingFace into `embedding_model/` (~120 MB).
5. **LLM** — prompts to pull `gemma4:31b-it-q8_0` (~34 GB) into `models/`. Skip with `--skip-model-pull` if you want to pull a different model, or curate a smaller one via `OFA_INSTALL_MODEL_ID`.
6. **`site.toml` wizard** — interactive prompts for site name, Slurm partition, GRES, protected paths, etc. Writes `site.toml` at the repo root. Falls back to the annotated `site.example.toml` in `--non-interactive` mode.
7. **RAG indices** — if `repos/` is populated (per `collections.toml`), rebuilds the ChromaDB collections. Skipped otherwise with instructions to run `src/rebuild_indices.py` manually once you've added your source dirs.
8. **`env.sh` + Lmod template** — writes a sourceable activation script and an Lmod modulefile template under `tools/`.

Useful flags:

```bash
./install.sh --skip-model-pull          # don't pull gemma4:31b-it-q8_0
./install.sh --skip-wizard              # don't run the site.toml wizard
./install.sh --skip-indices             # don't rebuild RAG (populate repos/ later)
./install.sh --non-interactive          # scripted install, opt-in steps default to skipped
./install.sh --force                    # redo work even if artifacts already exist
./install.sh --prefix /path/to/checkout # install target (default: install.sh's dir)
./install.sh --help                     # full flag list + env-var overrides
```

**Before you run `ofa` for real on a new site**, audit these files — they're the pieces the installer can't infer:

* [`site.toml`](site.example.toml) — verify partition names, GRES, walltime, account-discovery command.
* [`collections.toml`](collections.toml) — replace the Kestrel-specific RAG corpora (`amrex`, `marbles-papers`, `HPC`, …) with sources that make sense for your users.
* [`prompts/`](prompts/) — the mode prompts template the site *identity* (name, org, long name) automatically via `site.toml`, but leave Kestrel-specific *technical* content (CUDA module versions, partition names like `standard/hbw`, `/nopt/nrel` protected paths, Gila cross-references in `openfoam.txt`, the RHEL8→RHEL9 body in `reframe.txt`) as literal text. Rewrite those bits for your cluster; templating them would produce prompts that lie confidently about your setup.

Once you're satisfied, source the env and go:

```bash
source $OFA_ROOT/env.sh
ofa --help
```

## Updating the RAG indices

The vector store under `vectordb/` is populated from source directories declared in `collections.toml` at the repo root. To keep the indices current when you `git pull` a source repo, add a new source, or drop new documents into an existing one, run the rebuild script:

```bash
python3 $OFA_ROOT/src/rebuild_indices.py                             # rebuild all configured collections
python3 $OFA_ROOT/src/rebuild_indices.py --collection <name>         # scope to one collection
python3 $OFA_ROOT/src/rebuild_indices.py --list                      # show configured collections (no model load)
python3 $OFA_ROOT/src/rebuild_indices.py --dry-run                   # preview additions / skips / orphans
python3 $OFA_ROOT/src/rebuild_indices.py --force                     # ignore mtime cache; re-embed everything
python3 $OFA_ROOT/src/rebuild_indices.py --clear --collection <name> # drop and rebuild from scratch
python3 $OFA_ROOT/src/rebuild_indices.py --incremental               # additive-only: keep chunks in the store even if their source files were removed
```

Behaviour:

* **Idempotent.** Chunk IDs are SHA-256 of `collection + relative path + chunk index`, so re-runs upsert rather than duplicate. Per-file mtime is cached in `vectordb/.rebuild_state.json`; unchanged files are skipped (embedding is by far the slow step).
* **Mixed content per collection.** Each collection can list code directories, PDF directories, or both. Each chunk is tagged with a `source_type` metadata field so retrievers can distinguish source files from documents and cite PDF page numbers.
* **Missing source directories are logged and skipped.** You can declare a source path in `collections.toml` ahead of populating it — the collection activates as soon as content lands.
* **Notebook handling.** `.ipynb` files are parsed as JSON and stripped of cell outputs before chunking, so base64-encoded plot outputs and long stdout dumps don't pollute retrieval.
* **Orphan sweep.** Code files that were previously indexed but no longer exist on disk are removed from the collection on the next rebuild. Two opt-outs are available:
    - **Per-source, in `collections.toml`:** set `keep_missing = true` on a source entry. Missing files under that root are retained in the store while sibling sources in the same collection continue to sweep normally. Typical use: a papers/PDFs source that shares a collection with a git-tracked code source — the code side should still lose deleted files on `git pull`, but the papers side should not.
    - **Global, on the command line:** `--incremental` disables the sweep for every source in this run. Useful when you want a single ingest pass to be additive-only for reasons unrelated to source policy.

Run the rebuild inside a Kestrel GPU allocation so the embedding model uses the H100 — the login node's CUDA driver is older and falls back to CPU, which is considerably slower. Typical wall-times on H100 are a few tens of seconds per thousand chunks.

When rebuilding a collection that was previously indexed by an older ingester, use `--clear` on the first pass so its stale chunk IDs (from the older scheme) are dropped rather than left alongside the new ones. Subsequent rebuilds don't need `--clear`.

Edit `collections.toml` to add a new collection or a new source directory. Paths starting with `/` are absolute; others resolve relative to `$OFA_ROOT`. See the header comment inside that file for the schema.

### Common maintenance workflows

Concrete recipes for the four situations you'll actually run into. Each
one runs `rebuild_indices.py` via the bundled interpreter
(`$OFA_ROOT/env/bin/python`) so you don't need `module load assistant`
in the shell — useful for one-shot maintenance where you don't want to
disturb the interactive environment.

#### 1. Refresh a git-tracked source repo (HPC docs, AMReX, reframe, MARBLES)

The `repos/HPC`, `repos/amrex`, `repos/reframe-universal`, and
`repos/marblesThermal` trees are plain `git clone`s (not submodules).
When upstream publishes new content, pull and rebuild:

```bash
cd $OFA_ROOT/repos/HPC && git pull && cd $OFA_ROOT
./env/bin/python src/rebuild_indices.py --collection hpc_docs
```

`--collection` scopes the run to one collection so an unrelated
gitignored source (say `repos/quantum-papers`) isn't touched. The
mtime cache in `vectordb/.rebuild_state.json` means only files whose
content actually changed on disk get re-embedded — expect
"`[+] N added/updated  [=] M unchanged`" with a small `N` and large
`M` after a routine `git pull`.

For AMReX, reframe, MARBLES: substitute `--collection amrex_src`,
`--collection reframe_src`, or `--collection marbles_src`.

#### 2. Import a new document drop from a shared path

Example: the VASP team hands you a batch of `.md`, `.txt`, or `.html`
files under a shared directory like `/projects/hpcapps/rag-data-for-nilesh/vasp/`.

Copy into `repos/vasp/`, then rebuild:

```bash
# Copy verbatim if the source is already Markdown/text
rsync -av --delete /projects/hpcapps/rag-data-for-nilesh/vasp/*.{md,txt} \
    $OFA_ROOT/repos/vasp/

# HTML files (VASP wiki dumps, etc.) need conversion first — the
# indexer only reads .md/.rst/.txt/.py/.cpp/.h/.f90/.ipynb. Use pandoc
# if available, else a stdlib html.parser one-liner, else drop them
# somewhere non-indexed.

./env/bin/python src/rebuild_indices.py --clear --collection vasp_src
```

`--clear` first drops the existing chunks, then rebuilds — safer than
a plain rebuild when files were renamed or removed en masse (the
mtime cache doesn't detect renames as such, so it can leave stale
chunks with old paths). For an in-place edit of existing files, drop
`--clear`.

`repos/vasp/` is **not** git-tracked (VASP wiki content redistribution
rights aren't clear), so unlike a git-cloned source it lives only on
disk — nothing to commit here. Every fresh clone or new install needs
this rsync step run once against the shared drop path before
`vasp_src` has anything to embed.

#### 3. Update a collection where some source files were deliberately emptied

Example: `repos/quantum-papers/` was cleared to avoid copyright issues,
but you still want the previously-indexed knowledge to remain
available to ofa. The `[[collections.quantum_computing.sources]]`
entry for `repos/quantum-papers` already has `keep_missing = true`
(marbles-papers uses the same pattern), so a plain rebuild is safe:

```bash
./env/bin/python src/rebuild_indices.py --collection quantum_computing
```

Rebuild log will explicitly report:

```
[~] keep_missing: retaining 8 file entries under sources marked keep_missing=true
[+] N added/updated  [=] M unchanged  [-] 0 orphaned
```

The `[-] 0 orphaned` line is the important one — no papers chunks
were swept away. Chunks from the code source alongside are updated
normally.

If you add `keep_missing = true` to a source for the first time, no
`--clear` is needed; the flag only affects future orphan sweeps, not
already-indexed content.

#### 4. Add a new collection or source

Edit `collections.toml`:

```toml
[collections.my_new_collection]
description = "Short human-readable description shown by --list."
[[collections.my_new_collection.sources]]
path       = "repos/my-new-content"       # relative to $OFA_ROOT
type       = "code"                       # or "pdf"
extensions = [".md", ".py"]               # ignored for type="pdf"
# keep_missing = true                     # optional; see workflow 3
# page_ranges  = { "file.pdf" = "23-" }   # optional PDF slicing
```

Then populate `repos/my-new-content/` and rebuild:

```bash
./env/bin/python src/rebuild_indices.py --collection my_new_collection
```

To wire the new collection into an ofa mode's retriever, edit the
`retrieve_<mode>_context()` function in `src/ofa_main.py` and add a
query against the new collection with a sensible top-k budget.
Without that wiring, the collection is queryable via
`src/ofa_client.py` but not automatically injected into chat context.

### Where to run

Both login nodes (`kl6`, `kl7`) and any compute-node GPU allocation
work. On the login node the embedder falls back to CPU because the
GPU driver on `kl6`/`kl7` is older than what the bundled PyTorch
expects — throughput is ~30 chunks/second on CPU, adequate for
routine updates (a full HPC-docs rebuild of ~730 chunks completes in
about 4 minutes). If you already have a compute-node allocation open,
run there for GPU-speed embedding (~5×).

## Index your own private data

Beyond the shared corpora, each user can index their **own** data into a
per-user private store that `ofa` retrieves automatically alongside the
built-in collections — no write access to the shared install required.

```bash
# Index a directory (re-run to refresh after edits)
ofa --add-private ~/my-project-notes
ofa --add-private ~/papers --private-name lit-review   # custom label

# See what you've indexed
ofa --list-private

# Remove one collection, or everything
ofa --forget-private my-project-notes
ofa --forget-private all
```

After indexing, just ask `ofa` (any mode, CLI or VS Code) — private hits are
merged into the retrieved context and clearly marked `PRIVATE DATA` in the
prompt. Supported today: text and code files (`.md`, `.rst`, `.txt`, source
files, `.tex`, `.ipynb`, config files), `.pdf`, and Office `.docx` / `.xlsx`.
Office extraction is text-only and lossy — it drops styling, comments,
tracked changes, and (for spreadsheets) formulas, keeping computed values.

**Scanned / equation-heavy PDFs (vision OCR).** Plain PDF indexing uses fast
text-layer extraction, which mangles dense math and scanned pages (garbled
symbols, `(cid:N)` artifacts). **OCR is off by default** — you turn it on
per-run with `--private-ocr`, which re-reads pages with the local vision
model (nothing leaves the node):

| `--private-ocr` value | Behaviour |
| --- | --- |
| *(omitted)* / `off` | **Default.** Text-layer extraction only. Fast, no GPU pass over images. |
| `auto` | OCR *only* pages whose text layer looks broken; clean pages stay fast. Best for scientific papers. |
| `force` | OCR *every* page. Use for fully-scanned documents. Slow. |

```bash
ofa --add-private ~/papers                       # text-only (default, no OCR)
ofa --add-private ~/papers --private-ocr auto    # OCR degraded pages only
ofa --add-private ~/scans  --private-ocr force   # OCR every page (slow)
```

`auto`/`force` require a vision-capable `OFA_MODEL` (the default
`gemma4:31b-it-q8_0` qualifies) and a GPU allocation for reasonable speed —
each OCR'd page is rendered to an image and transcribed to Markdown + LaTeX
by the model (~seconds per page). To add OCR to a collection you already
indexed, just re-run `--add-private` on the same directory with the flag; it
re-indexes in place.

**Where it lives.** The private store is a separate ChromaDB instance under
`$OFA_SCRATCH/vectordb-private`, created `0700`; its metadata file is `0600`.
It is never written to the shared `vectordb/`. Override the location with
`OFA_PRIVATE_VECTORDB`.

**What to keep in mind for private data:**

- Retrieved private snippets are sent to whatever client is connected. From
  the CLI that stays on the compute node, but via VS Code / BYOK they reach
  your laptop and become part of that editor's chat history.
- `ofa` writes `$OFA_SCRATCH/.ofa_session.json` and `.ofa_history` `0600`, but
  `$OFA_SCRATCH` itself is created by the site — confirm its permissions if
  the data is sensitive.
- Do **not** run `ofa --serve --serve-no-auth` while private data is indexed;
  it would serve that data to anyone who can reach the port. `ofa` prints a
  warning if you do. Auth is on by default.
- The private store rides in your scratch space; treat it with the same care
  as any other file you place there.

## Memory & Session Context

The assistant maintains its transient session state natively in your scratch directory (`~/.ofa_session.json`). Tilde expansion within the tool orchestrator is deliberately handled safely to target your literal home directory instead of generating corrupt relative paths.

If your SLURM allocation expires, the custom signal handlers will safely write the final context to disk. You can then request a new allocation and simply run `ofa --resume` (with any of your targeted flags) to perfectly reconstruct the context window. 
Any permanent global preferences (e.g., "always use 4 spaces for indentation") mentioned to the assistant are extracted into an isolated `~/.ofa_prefs.txt` file and automatically sourced into all future context windows.

## Use `ofa` from VS Code Chat (the OnField Assistant extension)

VS Code's Chat / Copilot Chat picker can drive `ofa` if you install the
bundled OnField Assistant extension on the Kestrel-remote side. The
extension registers the eight `ofa` modes as a `LanguageModelChatProvider`
and handles the SLURM allocation, the login-node TCP bridge, and the API
key for you — **nothing needs to be configured on your laptop**.
End-to-end verified flow (Mac laptop + Kestrel + VS Code Remote-SSH):

1. **Attach VS Code to Kestrel.** Standard Remote-SSH — open a new
   window connected to your Kestrel login node.

2. **Install the extension on the remote.** In the Kestrel-attached
   window: `Cmd+Shift+X` → `⋯` menu at the top of the Extensions
   sidebar → **Install from VSIX...** → pick
   `/nopt/nrel/apps/cpu_stack/software/openfoam/assistant/vscode-ext/ofa-vscode.vsix`.
   Reload when prompted. The extension shows up under **SSH: KESTREL**
   as `ofa-vscode`.

3. **Bring up the server.** `Cmd+Shift+P` → **OFA: Connect**. This
   allocates a debug GPU node, launches
   `ofa --serve --serve-enable-tools`, and opens an `ncat` TCP bridge
   on login-node port `49643`. Watch progress with
   **OFA: Show Logs**. Wait for the connect-success toast.

4. **Chat.** Open Copilot Chat (`Cmd+Ctrl+I`). Click the model picker
   at the bottom of the panel and pick any entry under **OnField
   Assistant (Kestrel)** — e.g. `ofa · hpc`. Ask mode works for
   Q&A; Agent mode additionally lets the model apply edits and run
   commands via tool calls.

**Tear down** with `Cmd+Shift+P` → **OFA: Disconnect** — this releases
the SLURM job and closes the bridge. Subsequent sessions only need
step 3; if a previous allocation is still alive the extension adopts it
automatically on startup instead of queueing a new one.

### How the extension handles keys and ports

You never copy a token. `ofa --serve` generates (or reuses) a bearer
token at `$OFA_SCRATCH/.ofa_api_key` and prints it in its startup
banner; the extension scrapes that line, keeps the value in memory for
the lifetime of the connection, and sends it as an
`Authorization: Bearer …` header on every request. Because both the
extension and the server run on the Kestrel side, the token never
crosses to your laptop and is never written to any VS Code config.

This is the main difference from the older BYOK route, where you pasted
the token into **Chat: Manage Language Models → gear → Update API Key**
because Copilot Chat read it from VS Code's per-provider secret storage.
The extension bypasses secret storage entirely.

Ports likewise need no attention: the compute-node listen port is
chosen per user and persisted in `$OFA_SCRATCH/.ofa_serve_port`, while
the login-node bridge is pinned to `49643`
(`ofa.laptopSideBridgePort`). The bridge is re-established on every
connect, so a new SLURM allocation on a different node needs no
reconfiguration.

Token limits are derived from the model you select in `ofa.model`
rather than hardcoded, so the picker advertises the real context window
(the default `gemma4:31b-it-q8_0` runs at its full 262144-token
context). Changing `ofa.model` requires
**OFA: Disconnect** → **OFA: Connect**, since the backing LLM is chosen
when `ofa --serve` starts.

### Deprecated: the manual BYOK route

Earlier versions required registering the modes yourself in a
laptop-side `chatLanguageModels.json` via
`tools/byok-update-config.py`. The extension replaces that, and running
both leaves you with **two redundant groups** in the model picker —
`OFA (Kestrel)` from BYOK and `OnField Assistant (Kestrel)` from the
extension — pointing at the same server. If you see both, delete the
`OFA (Kestrel)` provider (via **Chat: Manage Language Models**, or by
removing that object from `chatLanguageModels.json`) and reload.

The BYOK path is still documented for non-VS-Code clients and manual
setups: [`docs/byok-vscode.md`](docs/byok-vscode.md).

## Bring your own agent

`ofa --serve` speaks the OpenAI `/v1/chat/completions` API (the same
endpoint the VS Code BYOK setup above uses), so any agent framework
that lets you point at a custom `base_url` + `api_key` can use ofa as
its backend LLM — not just VS Code. Two verified integrations:

### Use ofa from opencode (RHEL9 · access-gated)

[opencode](https://opencode.ai) is a terminal-native coding TUI. On Kestrel
it's shipped as a module (RHEL9 nodes only, granted on request — ask the
maintainers). Once you have access, it can talk to `ofa --serve` as one of
its providers so the whole ofa mode family shows up in its model picker
alongside the built-in `serveai` and `local-ollama` providers.

**One-time setup.** Create `~/.config/opencode/opencode.json` with an `ofa`
provider block whose `baseURL` / `apiKey` reference `{env:OFA_BYOK_URL}` and
`{env:OFA_BYOK_TOKEN}`, listing the seven ofa modes as separate models. The
`{env:…}` substitution happens at opencode startup, so the file never has to
be re-edited when the allocation, port, or token changes. The opencode
maintainers can share a template.

**Per-allocation flow.**

```bash
# 1) Load ofa and start its server. This writes the auto-picked port and
#    bearer token to $OFA_SCRATCH/.ofa_serve_port and $OFA_SCRATCH/.ofa_api_key.
module load assistant
ofa --serve --serve-quiet --serve-enable-tools &

# Wait until "[ofa-serve] Ctrl+C to stop." appears, then press Enter to get
# the shell prompt back. The server keeps running under job control (&).

# 2) Load opencode. Loading it does NOT overwrite your opencode.json.
module load opencode

# 3) Wire ofa's live endpoint into the environment. opencode.json substitutes
#    {env:OFA_BYOK_URL} / {env:OFA_BYOK_TOKEN} at startup, so ofa is picked up
#    dynamically each allocation without editing any config file.
OFA_SCRATCH="${OFA_SCRATCH:-/scratch/$USER}"
export OFA_BYOK_URL="http://localhost:$(cat $OFA_SCRATCH/.ofa_serve_port)/v1"
export OFA_BYOK_TOKEN=$(cat $OFA_SCRATCH/.ofa_api_key)

# 4) Optional — also drop ./opencode.json in the current directory so its
#    built-in serveai + local-ollama providers show up alongside ofa in the
#    model picker for this workdir. Skip this if you only want ofa.
_write_opencode_json

# 5) Launch. The model picker lists "OnField Assistant 🌵" with 7 modes.
opencode
```

`--serve-quiet` matters here because ofa's `--serve` stderr shares the
terminal with opencode's TUI — without it every request prints a
`[ofa-serve] <model> (<mode>): N msg(s)...` line into the foreground.
`--serve-enable-tools` lets opencode chain file edits and shell commands
through ofa's tool-calling passthrough instead of one-shot Q&A.

### Use ofa from AMReX Agent

[AMReX Agent](https://github.com/AMReX-Codes/amrex-agent) is AMReX's
LangGraph-based coding agent. Its `litellm` LLM provider talks to any
OpenAI-compatible endpoint (Ollama, vLLM, a LiteLLM proxy, LM Studio, or
an HPC BYOK server like ofa) — see
[PR #59](https://github.com/AMReX-Codes/amrex-agent/pull/59) /
`docs/api_keys.md` in that repo for the general recipe. Pointed at ofa:

```bash
module load assistant
ofa --serve --serve-enable-tools
export LITELLM_BASE_URL="http://localhost:$(cat $OFA_SCRATCH/.ofa_serve_port)/v1"
export LITELLM_API_KEY="$(cat $OFA_SCRATCH/.ofa_api_key)"
export LITELLM_MODEL="ofa-code"
unset CBORG_API_KEY   # unset whichever other provider key would otherwise take priority
```

```yaml
# ~/.amrex_agent/kestrel-ofa.yaml
llm_provider: litellm
llm_model: ofa-code
embedding_provider: huggingface   # ofa's /v1 endpoint doesn't serve /v1/embeddings
```

```bash
python amrex_agent.py --prompt "..." --config ~/.amrex_agent/kestrel-ofa.yaml
```

`llm_model` accepts any of the eight `ofa-*` mode IDs — `ofa-code`,
`ofa-hpc`, `ofa-amrex`, `ofa-marbles`, `ofa-openfoam`, `ofa-reframe`,
`ofa-quantum-computing`, `ofa-vasp` — pick whichever matches the task.

## Programmatic use from Python (`ofa_client`)

Call `ofa` from your own simulation / diagnostic scripts. The client is
a single stdlib-only Python file that talks to a running `ofa --serve`
over HTTP — no third-party packages, works in any Python 3.8+
interpreter (bare, venv, conda, spack, container).

### One-time setup

```bash
module load assistant                    # puts ofa_client on PYTHONPATH
ofa --serve > ofa-serve.log 2>&1 &       # start the local server (in your allocation)
trap 'kill %1 2>/dev/null' EXIT          # optional: clean up on shell exit
```

The server writes its bearer token to `$OFA_SCRATCH/.ofa_api_key` and
its port to `$OFA_SCRATCH/.ofa_serve_port`. The client reads both
automatically — no URL/token wiring in your script.

### `ask()` — one-shot, stateless

Best fit for sim loops where each call is independent (e.g., summarise
plot N without any memory of plot N-1).

```python
from ofa_client import ask

# 1. Plain text
text = ask("what is a good turbulence model for cavity flow at Re=1e4?")

# 2. Text plus inline context string
text = ask(
    "diagnose this run",
    context="Simulation: cavity flow, Re=1000. Diverged at step 4200.",
)

# 3. Text plus a file (tail-reads last 32 KB by default; full_file=True to override)
text = ask("why is this crashing?", file="output/solver.log")

# 4. Text plus an image (Gemma 4's vision head handles it in any mode)
text = ask("describe this plot", image="output/step_0100_pressure.png")

# 5. Everything at once, and pick a specific mode
text = ask(
    "diagnose this simulation step",
    image="output/step_4200_pressure.png",
    file="output/solver.log",
    context="Re=1000, cavity flow, k-omega SST turbulence model.",
    model="ofa-code",          # ofa-openfoam | ofa-hpc | ofa-code | ofa-amrex | ofa-marbles | ofa-reframe | ofa-quantum-computing | ofa-vasp
    timeout=60,
)
```

Full signature:

```python
ask(
    prompt: str,               # main question (required)
    *,
    image:     str | Path = None,        # attach one image (any PIL-readable format)
    images:    list       = None,        # attach several
    context:   str        = None,        # inline text prepended verbatim
    file:      str | Path = None,        # read one file, fence with its name
    files:     list       = None,        # read several
    model:     str        = "ofa-code",  # any of the five ofa-* modes
    url:       str        = None,        # override auto-detection
    token:     str        = None,        # override auto-detection
    timeout:   float      = 120.0,
    full_file: bool       = False,       # True disables the 32 KB tail-cap
) -> str
```

### `Session()` — multi-turn, client-side history

Use when later turns need the model to remember earlier turns
(interactive code review, iterative parameter refinement,
conversational drill-down). Session accumulates the message list
locally and sends the whole thing on each `.ask()`, so state survives
server restarts and adds zero memory footprint on the server.

```python
from ofa_client import Session

sess = Session(model="ofa-code", timeout=120)

# Turn 1: attach an image and context; model critiques.
sess.ask(
    "look at this fit; is it good?",
    image="fit_before.png",
    context="model: y = A * exp(-b*x) * cos(omega*x + phi); RMS=0.85",
)

# Turn 2: no need to re-attach the image — it's in the history.
sess.ask(
    "based on your critique, propose better initial guesses as JSON: "
    '{"p0": [A, b, omega, phi], "maxfev": <int>, "notes": "..."}'
)

print(sess)                # Session(model='ofa-code', turns=2)
sess.clear()               # forget everything and start fresh
```

`Session.ask()` accepts the same kwargs as `ask()` except `model`,
`url`, `token`, `timeout`, `full_file` (those are fixed at
construction).

### Parsing structured output from the LLM

A reliable pattern for getting parseable numbers back: **ask for a
fenced JSON code block with an explicit schema**, then extract-and-parse
with a fallback ladder. LLMs handle "return this JSON schema" much
better than "return only a number".

```python
import json, re

def extract_json(text):
    for pat in (r"```json\s*(\{.*?\})\s*```", r"```\s*(\{.*?\})\s*```"):
        m = re.search(pat, text, re.DOTALL | re.IGNORECASE)
        if m:
            try: return json.loads(m.group(1))
            except json.JSONDecodeError: pass
    for m in re.finditer(r"\{.*?\}", text, re.DOTALL):
        try: return json.loads(m.group(0))
        except json.JSONDecodeError: continue
    return None
```

See [`examples/fit_and_ask.py`](examples/fit_and_ask.py) for a worked
end-to-end example: fits a noisy damped sinusoid + harmonic (a function
where `scipy.optimize.curve_fit` genuinely gets trapped in a local
minimum from a bad initial guess), sends the plot + fit summary to
`ofa` inside a `Session`, asks for a prose critique on turn 1 and a
strict-JSON refit suggestion on turn 2, extracts the JSON with the
fallback ladder above, and reruns the fit. A typical successful run
prints:

```
RMS: 0.5370 -> 0.11xx  (IMPROVED, delta -0.42)
```

Run it in-place:

```bash
module load assistant
ofa --serve > /tmp/ofa-serve.log 2>&1 &
cd $OFA_ROOT/examples && python3 fit_and_ask.py
```

### Auto-detection order

The client resolves `url` and `token` by falling through:

1. Explicit `url=` / `token=` kwargs.
2. `$OFA_BYOK_URL` and `$OFA_BYOK_TOKEN` environment variables.
3. `$OFA_SCRATCH/.ofa_serve_port` and `$OFA_SCRATCH/.ofa_api_key`.
4. `/scratch/$USER/.ofa_serve_port` and `/scratch/$USER/.ofa_api_key`.
5. Raise `RuntimeError` with a clear "no ofa server detected" hint.

Before each request the client also probes `/healthz` with a 3-second
cap so a dead server (killed allocation, stale port file) surfaces as
an immediate, actionable error — not a two-minute hang.

### Robust wrapping in a sim

A slow model, expired allocation, or dropped connection should skip
the AI call, not crash your sim:

```python
try:
    summary = ask(f"summarise pressure field at step {step}",
                  image=fname, timeout=60)
    with open("output/ai_summary.log", "a") as f:
        f.write(f"[step {step}] {summary}\n")
except Exception as e:
    print(f"[ai summary skipped: {e}]")
```

### Where to go next

* Full BYOK + VS Code walkthrough: [`docs/byok-vscode.md`](docs/byok-vscode.md).
* Deeper explanation of how `ofa --serve` layers RAG, memory, and system
  prompts on top of Ollama: `docs/ofa-technical-overview.md` §5.3.
* Reference implementation: [`src/ofa_client.py`](src/ofa_client.py) is
  ~360 LOC, stdlib only, and the entire public surface is documented
  inline.
