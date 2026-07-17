# `ofa` — Technical Overview

**Project**: OnField Assistant 🌵 (`ofa`)
**Deployment**: NLR Kestrel HPC
**Module**: `assistant` (Lmod)
**Repo**: https://github.com/nileshsawant/onfield-assistant
**Document date**: June 2026

---

## 1. Quickstart

### On Kestrel

```bash
module load assistant
ofa                                  # General coding assistant (default)
ofa --openfoam                       # OpenFOAM dictionary generator (was default in <= 1.0)
ofa --hpc                            # Kestrel HPC documentation assistant
ofa --code                           # General coding assistant (redundant — this is the default)
ofa --amrex                          # AMReX C++ framework
ofa --marbles                        # MARBLES (LBM thermal solver on AMReX)
ofa --quantum-computing              # Quantum computing (rigorous gate / matrix verification)
ofa --rhel9_reframe                  # ReFrame for RHEL9 migration
ofa --resume                         # Resume your last session
ofa --openfoam "set up a cavity case" --save ./case   # Single OpenFOAM query + save case files
```

`ofa` auto-allocates a quarter-node H100 GPU (debug partition, 30 min walltime)
via SLURM on first invocation. Override the defaults *before* running:

```bash
export OFA_ACCOUNT=<your-slurm-account>      # default: your default account
export OFA_PARTITION=gpu-h100                 # default: debug
export OFA_WALLTIME=04:00:00                  # default: 00:30:00
```

Inside any interactive session, type `/help` for the full slash-command menu
(skills, memory inspect/edit, model switch, history, save case, shell escape).

### From VS Code Chat (BYOK)

```bash
# On Kestrel
ofa --serve --serve-enable-tools
```

`ofa --serve` prints a labelled connection block with the exact `ssh -L` line
(compute-node hostname + ports already filled in), the BYOK URL, and the
bearer token. Paste the `ssh -L` in a laptop terminal, then register the URL
and token in VS Code's `chatLanguageModels.json`.

The helper [`tools/byok-update-config.py`](../tools/byok-update-config.py)
generates the VS Code config in one shot. Full walkthrough including the
known VS Code-side gotchas: [`docs/byok-vscode.md`](byok-vscode.md).

### Discoverability

`module help assistant` prints a usage summary; `module load assistant` shows
a short banner with the BYOK quick-start; the GitHub repo
(https://github.com/nileshsawant/onfield-assistant) holds the live source
and this document.

---

## 2. Executive summary

`ofa` is a domain-specialised AI assistant that pairs a local 31-billion-parameter language model (Gemma 4) on Kestrel H100 GPUs with retrieval-augmented generation over indexed Kestrel/OpenFOAM/AMReX/MARBLES/ReFrame/quantum-computing corpora. It exposes two surfaces:

- **Interactive CLI** (`ofa`, `ofa --hpc`, `ofa --code`, `ofa --amrex`, `ofa --marbles`, `ofa --quantum-computing`, `ofa --rhel9_reframe`) — a full agent loop that reads files, executes bash, edits code, and persists session state on Kestrel.
- **OpenAI-compatible HTTP server** (`ofa --serve`) — a Bring-Your-Own-Key (BYOK) endpoint so VS Code Chat, `opencode`, or any OpenAI-compatible client can route requests through the same domain layer.

The codebase is ~6,000 lines of Python (no exotic dependencies — stdlib + httpx + chromadb + rank_bm25 + sentence-transformers + ollama). All inference runs locally on a quarter-node Kestrel GPU allocation; no data leaves NLR's network. 124 commits as of this writing; production-stable on the OpenFOAM/HPC modes.

The remainder of this document covers what's in the repo, how the pieces fit together, and the operational/safety properties anyone evaluating `ofa` for wider use will want to know.

---

## 3. What `ofa` is — and is not

### Is

- A **local-first** LLM assistant: model weights, indexed corpora, and runtime state all live on Kestrel under the user's account or `$OFA_ROOT`.
- **Domain-specialised** through five mode-specific system prompts, six pre-built vector indices (~27,500 documents), per-user long-term memory, and a skill system.
- **Agentic on the CLI side**: it can run bash, read/write files, save OpenFOAM cases, retry on tool errors, and persist its working memory across sessions.
- **Interoperable** via OpenAI's `/v1/chat/completions` shape — usable from VS Code BYOK, `opencode`, `curl`, or any client that speaks the same wire format.

### Is not

- A frontier model. Gemma 4 31B is materially less reliable than GPT-4/Claude at long-horizon agentic work; the prompts and execution-side guardrails compensate but do not match frontier behaviour.
- A multi-user shared service. Each user gets their own SLURM allocation (a quarter of a 4-GPU H100 node) and their own per-user scratch state. Concurrency across users is achieved by everyone running their own `ofa` invocation.
- A general-purpose coding agent. The system prompts and the RAG corpora are tuned for OpenFOAM 13, AMReX/MARBLES, Kestrel HPC, ReFrame RHEL9 migration, and adjacent C++/Slurm work. It will answer general programming questions but its strengths lie in those domains.

---

## 4. Architecture (high level)

```
                ┌─────────────────────────────────────────────────────┐
                │              Kestrel compute node                    │
                │                                                      │
   user shell   │   ┌──────────────┐                                  │
   ────────────▶│   │   ofa CLI    │   bin/ofa wrapper:               │
                │   │  (ofa_main)  │   - auto-allocates GPU via salloc│
                │   └──────┬───────┘   - sets OFA_ROOT, OLLAMA_MODELS │
                │          │                                          │
                │          ├── prompts/    (5 mode prompts + common)  │
                │          ├── vectordb/   (6 ChromaDB collections,   │
                │          │                 ~27.5K indexed docs)     │
                │          ├── repos/      (live git clones for RAG   │
                │          │                 grep + ad-hoc reads)     │
                │          ├── models/     (Ollama weights, e.g.      │
                │          │                 gemma4:31b)              │
                │          └── $OFA_SCRATCH/  (per-user state:        │
                │                 .ofa_session.json, prefs, lessons,  │
                │                 serve port, api key, history)      │
                │          │                                          │
   VS Code      │   ┌──────▼───────┐    ┌──────────────┐             │
   BYOK     ───▶│   │ ofa --serve  │───▶│   Ollama     │  GPU        │
   opencode     │   │ (ofa_server) │    │  (port 11434)│  inference  │
   curl         │   └──────────────┘    └──────────────┘             │
                │      OpenAI-compat                                  │
                │      HTTP shim                                      │
                └─────────────────────────────────────────────────────┘
```

Three layers, in order of how a request flows through them:

1. **Surface** — either the interactive CLI (terminal stdin/stdout, agent loop) or the BYOK HTTP server (`POST /v1/chat/completions`).
2. **Domain layer** — the same in both surfaces: system-prompt selection, long-term memory injection, RAG retrieval, optional skill content.
3. **Inference layer** — Ollama running `gemma4:31b` on an H100. Both surfaces use the same Ollama process via `ofa_main.chat_stream()`.

---

## 5. Interaction surfaces

### 5.1 CLI

| Command | Mode | System prompt | RAG retriever |
|---|---|---|---|
| `ofa` (default) | general code R/W/X | `prompts/code.txt` + `prompts/cpp.txt` | `retrieve_hpc_context` |
| `ofa --openfoam` | OpenFOAM case / dictionary generator | `prompts/openfoam.txt` | `retrieve_context` (OpenFOAM tutorials) |
| `ofa --hpc` | Kestrel HPC docs | `prompts/hpc.txt` | `retrieve_hpc_context` (Kestrel docs) |
| `ofa --code` | general code R/W/X (same as default) | `prompts/code.txt` + `prompts/cpp.txt` | `retrieve_hpc_context` |
| `ofa --amrex` | AMReX C++ framework | `prompts/amrex.txt` | `retrieve_amrex_context` (AMReX source only) |
| `ofa --marbles` | MARBLES LBM thermal solver | `prompts/marbles.txt` | `retrieve_marbles_context` (MARBLES primary + light AMReX) |
| `ofa --quantum-computing` | Quantum computing (code + papers, rigorous math verification) | `prompts/quantum-computing.txt` | `retrieve_quantum_computing_context` |
| `ofa --rhel9_reframe` | ReFrame for RHEL9 | `prompts/reframe.txt` | `_get_reframe_rag` + `retrieve_hpc_context` |

CLI behaviour:

- **Auto-allocation**: `bin/ofa` detects whether it's running inside a SLURM job. If not, it `salloc`s a quarter-node H100 allocation (defaults: debug partition, 30 min, 32 cores, 80 GB RAM, 1 GPU) and re-exec's itself on the compute node.
- **Ollama bootstrap**: `ensure_ollama_running()` starts a per-user `ollama serve` on a UID-derived port if none is already running, then waits for `/api/tags` to respond.
- **Session resume**: `--resume` reloads `$OFA_SCRATCH/.ofa_session.json`. Sessions auto-compress when they grow past 100 KB (see §6.6).
- **Interactive slash commands**: `/help`, `/clear`, `/history`, `/cwd`, `/retry`, `/memory`, `/remember <text>`, `/forget [prefs|lessons|all]`, `/skills`, `/skill <name>`, `/skill off <name>`, `/models`, plus shell escapes (`$ <cmd>`) and file inlining (`@<path>`).
- **Tool execution**: the model emits fenced `=== TOOL ===` blocks; `_run_react_loop` parses them, executes (bash, file read/write, planned-file generation), feeds output back, and continues until the model stops asking for tools or the consecutive-error limit (3) is hit.

### 5.2 BYOK HTTP server (`ofa --serve`)

OpenAI-compatible HTTP shim. Three endpoints:

| Method | Path | Auth | Notes |
|---|---|---|---|
| GET | `/healthz` | none | Liveness probe; returns `{"status":"ok"}`. |
| GET | `/v1/models` | bearer | Lists seven model IDs (`ofa-openfoam`, `ofa-hpc`, `ofa-code`, `ofa-amrex`, `ofa-marbles`, `ofa-reframe`, `ofa-quantum-computing`). |
| POST | `/v1/chat/completions` | bearer | OpenAI format; supports `stream: true` SSE; optional `tools` / `tool_choice` passthrough when `--serve-enable-tools` is set. |

The model ID in the request body selects the mode → system prompt → RAG retriever. Inbound system messages from the client are dropped (the BYOK client's own prompt would override ofa's domain knowledge). RAG retrieval is applied only to the most recent user message.

The full client setup (SSH port-forward + VS Code config) is documented in [docs/byok-vscode.md](byok-vscode.md). A helper at [tools/byok-update-config.py](../tools/byok-update-config.py) generates the VS Code `chatLanguageModels.json` provider entry in one shot.

### 5.3 Python client (`ofa_client`)

For calling ofa from inside a user's own code (typically a simulation
loop that wants AI summaries of plots, log snippets, or config
files), `src/ofa_client.py` is a stdlib-only client that talks to a
running `ofa --serve` on the same node. `module load assistant` adds
`$OFA_ROOT/src` to `PYTHONPATH`, so it imports cleanly from any Python
environment (venv, conda, bare interpreter) — the client has no third
-party dependencies and doesn't import anything from `ofa_main` or
`ofa_server`.

Two API entry points:

- `ask(prompt, ...)` — stateless one-shot; the natural fit for a sim
  loop where each summary is independent.
- `Session(model=...)` — accumulates a message history client-side and
  sends the whole thing on each `.ask()`; the natural fit for multi
  -turn scripts (code diagnosis, exploratory back-and-forth).

Usage:

```python
from ofa_client import ask, Session

# 1. Plain text
text = ask("what is a good turbulence model for cavity flow at Re=1e4?")

# 2. Text with inline context
text = ask(
    "diagnose this run",
    context="Simulation was cavity flow, Re=1000. Diverged at step 4200.",
)

# 3. Attach a file (tail-reads last 32 KB by default; full_file=True to
#    override — useful for huge solver logs)
text = ask("why is this crashing?", file="output/solver.log")

# 4. Attach an image (base64-encoded and sent as OpenAI image_url)
text = ask("describe this plot", image="output/step_0100_pressure.png")

# 5. All at once + explicit model
text = ask(
    "diagnose this simulation",
    image="output/step_4200_pressure.png",
    file="output/solver.log",
    context="Re=1000, cavity flow, k-omega SST turbulence model.",
    model="ofa-code",
    timeout=60,
)

# Multi-turn
sess = Session(model="ofa-code")
sess.ask("what turbulence model for cavity flow at Re=1e4?")
sess.ask("show me a controlDict for that")   # sees the previous turn
```

Auto-detection order for URL and bearer token:

1. Explicit `url=` / `token=` kwargs to `ask` / `Session`.
2. `$OFA_BYOK_URL` and `$OFA_BYOK_TOKEN` environment variables.
3. `$OFA_SCRATCH/.ofa_serve_port` and `$OFA_SCRATCH/.ofa_api_key`.
4. `/scratch/$USER/.ofa_serve_port` and `/scratch/$USER/.ofa_api_key`.

Raises `RuntimeError` with a clear message if no server is reachable —
sim loops should wrap the call in try/except so a slow model or
expired allocation skips the summary rather than crashes the sim:

```python
try:
    summary = ask(f"Summarise pressure field at step {step}", image=fname,
                  timeout=60)
    with open("output/ai_summary.log", "a") as f:
        f.write(f"[step {step}] {summary}\n")
except Exception as e:
    print(f"[ai summary skipped: {e}]")
```

Any of the five `ofa` modes (`ofa-openfoam`, `ofa-hpc`, `ofa-code`,
`ofa-amrex`, `ofa-marbles`, `ofa-reframe`, `ofa-quantum-computing`) can be passed as `model=`. Images pair
with any mode — Gemma 4's vision head handles them regardless of
which system prompt is loaded.

---

## 6. The domain layer (what makes `ofa` more than vanilla Gemma)

Both surfaces share these five components, layered into the request before it reaches Ollama.

### 6.1 System prompts

| File | Lines | Role |
|---|---:|---|
| `prompts/common.txt` | 117 | Cross-mode rules: tool-fence convention, RAG citation policy, two-channel long-term-memory contract (PREFS + LESSONS), output style. Included in every prompt. |
| `prompts/openfoam.txt` | 42 | Dictionary-generator mode — OpenFOAM 13 case file conventions. |
| `prompts/hpc.txt` | 11 | Kestrel HPC documentation assistant. |
| `prompts/code.txt` | 7 | General coding assistant. |
| `prompts/cpp.txt` | 29 | Layered onto `code.txt` for C++ work. |
| `prompts/amrex.txt` | 8 | AMReX C++ framework. |
| `prompts/marbles.txt` | 10 | MARBLES lattice-Boltzmann thermal solver (built on AMReX). |
| `prompts/quantum-computing.txt` | 27 | Quantum computing; enforces per-answer verification of gate matrices, tensor ordering, and unitarity. |
| `prompts/reframe.txt` | 10 | ReFrame RHEL9 migration. |
| `prompts/plan.txt` | 6 | Plan-stage prompt used by `plan_file_list()` in OpenFOAM mode. |

System-prompt construction is in `load_system_prompt(prompt_type)`. After the per-mode prompt and `common.txt` are concatenated, the function:

1. Appends `--- LESSONS LEARNED ---` block from `$OFA_SCRATCH/.ofa_lessons.txt` if present.
2. Appends `--- USER PREFERENCES ---` block from `$OFA_SCRATCH/.ofa_prefs.txt` if present.
3. Substitutes `{OFA_ROOT}` and `{OFA_SCRATCH}` placeholders so prompts can reference deployment-specific paths without hard-coding them.

### 6.2 RAG (retrieval-augmented generation)

Six ChromaDB collections served from `$OFA_ROOT/vectordb/` (Chroma's persistent client, default cosine distance, 384-dim Sentence-Transformers embeddings):

| Collection | Documents | Source |
|---|---:|---|
| `of13_src` | 10,221 | OpenFOAM 13 source tree (`repos/openfoam-13/`) |
| `amrex_src` | 11,305 | AMReX source (`repos/amrex/`) |
| `openfoam` | 4,068 | OpenFOAM tutorials (cleaned via `rebuild_tutorials_*.py`) |
| `hpc_docs` | 953 | Kestrel documentation (Markdown) |
| `reframe_src` | 875 | ReFrame source tree (RHEL9 migration tests) |
| `marbles_src` | 151 | MARBLES source |
| `quantum_computing` | 0† | Quantum-computing code + papers (populated by `src/rebuild_indices.py` when `repos/quantum-code/` and `repos/quantum-papers/` are populated) |
| **Total** | **27,573** | — |

**Hybrid retrieval**: each retriever combines dense (ChromaDB embedding similarity) and sparse (BM25 over tokens) scores. BM25 indices are pre-built at startup via `_init_rag()` and cached in memory for the session — first-query latency was prohibitive before the prebuild was introduced. The merge weights are tuned per retriever (see `retrieve_context`, `retrieve_hpc_context`, `retrieve_amrex_context`, `retrieve_marbles_context`, `retrieve_quantum_computing_context`, `_get_reframe_rag`).

**Greeting bypass**: retrieval is skipped for trivial queries (`hi`, `hello`, `thanks`, etc.) to avoid spending tokens on irrelevant context.

**Fencing**: retrieved snippets are wrapped via `_fence_rag()` in clearly delimited `=== RETRIEVED REFERENCE ===` tags, with a defence-in-depth reminder telling the model that fenced content is data, not instructions. Mitigates prompt-injection risk from documents we index.

### 6.3 Long-term memory (two channels)

Per-user files in `$OFA_SCRATCH`, persistent across sessions:

| File | Channel | Trigger | Cap |
|---|---|---|---|
| `.ofa_prefs.txt` | PREFS — user-explicit standing instructions | User says "always", "never", "prefer", "from now on", etc. | 4 lines/turn |
| `.ofa_lessons.txt` | LESSONS — model-autonomous observations | Command failure understood, user correction, environment quirk discovered | 2 lines/turn |

Both channels share the implementation in `_save_marker_block(text, label, channel, max_per_turn)`:

- Scans for `=== LABEL === ... === END LABEL ===` blocks with `re.findall` (multi-block per turn supported).
- Strips list-bullet prefixes (`-`, `*`, `•`) so the model can write bulleted lists naturally.
- Dedupes new entries against the existing file.
- Caps at 16 KB total per channel; drops oldest entries on overflow.
- Atomic write via temp file + `os.replace` so a crash mid-write can't corrupt the file.
- Logs each save to stderr in magenta: `[memory] saved preference: <line>`.

Both channels are injected into every request's system prompt (§6.1). The model is instructed in `common.txt` that PREFS overrides LESSONS on conflict (the user is ground truth).

The whole memory machinery is also unit-tested — see [test_ofa_memory.py](file:///tmp/test_ofa_memory.py), 14 tests covering extraction, dedup, caps, atomic write under simulated `os.replace` failure, and the byte-cap eviction.

### 6.4 Skills

Markdown files in `prompts/skills/` that the user can inject into the running session on demand:

```
prompts/skills/
├── README.md                      # contract: name, format, lifecycle, security
└── kestrel-debug-jobs.md          # example: Kestrel debug-partition rules
```

Slash commands `/skills` (list), `/skill <name>` (load), `/skill off <name>` / `/skill off all` (unload). When loaded, a skill becomes a system-role message tagged `[SKILL: <name>]` inserted right after the base system prompt; `/clear` and session exit drop loaded skills.

A separate filename-stem allow-list refuses path traversal (`..`, leading dots, `/`, `\`) so users can only load files that actually live in the skills dir.

### 6.5 Safety guards

The CLI surface executes commands; that's where safety lives.

- **Destructive-command pattern matcher**: regex screen catches `rm -rf /`, `dd of=/dev/...`, recursive `chmod`/`chown` on system paths, `mkfs` on real devices, etc. before they reach `subprocess.run`. Match → red-banner approval prompt requiring exact-case confirmation typed by the user (Ctrl+C / EOF treats as "no", not a crash).
- **Consecutive-error pause**: after 3 consecutive tool failures, the react loop hands control back to the user instead of letting the model thrash.
- **Tool output truncation**: any single command's stdout/stderr is capped at 96 KB before being fed back to the model (head + tail keepers); session-wide context is compressed when it exceeds 100 KB (see §6.6).
- **Catastrophic command confirmation phrase**: certain irreversible operations (e.g. `git push --force`, `rm -rf` after the regex screen passes due to a non-system path) require typing an exact phrase, not just `y`.
- **Untested-model warning**: at startup, if the active LLM is not in `TESTED_MODELS` (currently only `gemma4:31b`), a loud red banner explains that destructive-command guards have only been validated against the default. The model registry, picker UI, and the `/models` slash command are deliberately not exposed on the startup banner — see [commit a4f1124](https://github.com/nileshsawant/onfield-assistant/commit/a4f1124) for the rationale.

### 6.6 Session context compression

When `messages` exceeds 100 KB, `manage_session_context()` walks oldest → newest (skipping the system prompt and the last 2 messages) and applies three compression strategies in order:

1. Strip `<thought>...</thought>` deliberation from old assistant turns.
2. Replace old `Output from executed commands: …` user messages with a short placeholder.
3. For any other old user message > 8 KB containing fenced code blocks, replace each ` ``` … ``` ` body with `[Older code/output block omitted by system to preserve context memory.]` — keeps the language tag (`​```python` etc.) so the model knows what kind of content was there.

Compression stops when below target (75 % of cap). If progress is made but the result is still over cap, a yellow `/clear` hint is emitted. If compression can free nothing, the function stays silent — earlier versions printed `[System: Context size (N) near limit. Compressing old logs...]` every turn even when no compression was possible, which was the original UX bug that drove the refactor ([commit df81e1b](https://github.com/nileshsawant/onfield-assistant/commit/df81e1b)).

Eight unit tests cover this in [test_ofa_compress.py](file:///tmp/test_ofa_compress.py).

---

## 7. Code organisation

### 7.1 Repository layout

```
$OFA_ROOT/
├── bin/ofa                      # SLURM-aware shell wrapper
├── src/
│   ├── ofa_main.py             # ~3,600 LOC: CLI, agent loop, RAG, memory
│   ├── ofa_server.py           # ~ 870 LOC: BYOK HTTP shim
│   ├── ofa_client.py           # ~ 360 LOC: stdlib-only Python client
│   ├── build_index.py          # legacy index builder
│   ├── build_index_v2.py       # current index builder
│   ├── ingest_amrex.py         # AMReX source ingestion
│   ├── ingest_reframe.py       # ReFrame source ingestion
│   ├── rebuild_tutorials_clean.py
│   └── rebuild_tutorials_of13.py
├── tools/
│   └── byok-update-config.py   # VS Code chatLanguageModels.json helper
├── prompts/                    # 7 mode prompts + common.txt + skills/
├── vectordb/                   # ChromaDB persistent store (6 collections)
├── repos/                      # live git clones for RAG + grep
├── models/                     # Ollama model weights (gemma4:31b, etc.)
├── env/                        # bundled Python 3.13 virtualenv
├── docs/
│   ├── byok-vscode.md          # BYOK setup walkthrough
│   ├── byok-vscode-chatLanguageModels.example.json
│   └── ofa-technical-overview.md   # this file
├── ARCHITECTURE.md             # high-level architecture notes
└── README.md
```

Total tracked source: **~6,500 LOC** across 10 Python files (excluding tests and prompts).

### 7.2 `src/ofa_main.py` walkthrough (3,544 LOC)

Logical sections, in roughly the order they appear:

| Section | Purpose |
|---|---|
| Path/env constants (~15–60) | `OFA_ROOT`, `OFA_SCRATCH`, `PROMPTS_DIR`, `VECTORDB_PATH`, `SKILLS_DIR`. |
| Model registry (~60–230) | `MODEL_REGISTRY` (sampling params per model), `TESTED_MODELS`, `_print_model_registry`, `_print_active_model_banner`. |
| Behavioural constants (~230–250) | Tool-output cap, session-compress thresholds, max-consecutive-errors. |
| Terminal colouring (~245–280) | `_USE_COLOR`, `_c()`, `_banner()`. Respects `NO_COLOR`. |
| Safe input helpers (~280–305) | `_safe_input()` treats Ctrl+C at approval prompts as "decline" instead of crash. |
| Scratch resolution (~305–340) | `_resolve_scratch()` — `$OFA_SCRATCH` → `/scratch/$USER` → XDG → tmp. |
| Session persistence (~370–510) | `save_session`, `load_session`, `manage_session_context` (compression). |
| Long-term memory (~510–620) | `_save_marker_block`, `extract_and_save_prefs`, `extract_and_save_lessons`, `_read_memory_file`, `_count_memory_lines`. |
| Skills (~620–720) | `_list_skill_files`, `_load_skill_text`, `_active_skill_names`. |
| Thinking-channel filter (~840–920) | Streaming filter that hides `<thought>...</thought>` from display while preserving it in captured text. |
| `_run_react_loop` (~920–1100) | The CLI agent loop: stream response → save prefs/lessons → execute tool calls → loop. |
| Tool-fence parsing (~1100–1700) | `check_and_execute_bash`, `extract_plan`, `_looks_like_unfenced_tool_intent` + nudge. |
| Ollama bootstrap (~1330–1700) | `ensure_ollama_running`, `_shutdown_ollama`. |
| RAG retrievers (~1770–2680) | `retrieve_context` (OpenFOAM), `retrieve_hpc_context` (Kestrel docs), `retrieve_amrex_context` (AMReX-only), `retrieve_marbles_context` (MARBLES + light AMReX), `retrieve_quantum_computing_context` (quantum code + papers), `_get_reframe_rag`. Hybrid dense+BM25. |
| `chat_stream` (~1820–1870) | The Ollama API call. All chat traffic flows through here. |
| `interactive_mode` (~2100–2300) | Banner, slash-command dispatch, REPL loop. |
| `single_query` / `hpc_single_query` (~2300–3070) | One-shot CLI mode and the plan→generate-per-file pattern used by `--save`. |
| `main` (~3350–3540) | Argparse, dispatch to interactive vs single-query vs `--serve`. |

### 7.3 `src/ofa_server.py` walkthrough (815 LOC)

Linear file, easier to read top-to-bottom:

| Function / class | Lines | Purpose |
|---|---|---|
| `_MODEL_MODES` | 50 | `ofa-hpc → "hpc"` mapping; advertises five model IDs. |
| `_retrieve_for_mode` | 66 | Dispatches to the right RAG retriever in `ofa_main`. |
| `_augment_user_message` | 89 | Fences RAG context and prepends it to the user message. |
| `_augment_messages` | 118 | Drops inbound system messages, injects ofa's, RAG-fences last user msg. |
| `_sse_chunk` / `_sse_done` | 147 | OpenAI SSE chunk formatting. |
| `load_or_create_api_key` | 170 | Bearer token persistence with 0o600 file perms. |
| `_ollama_chat_raw` | 213 | Used when `--serve-enable-tools` is on. Talks to Ollama's `/api/chat` directly with `tools` / `tool_choice`, yielding full chunk dicts so the caller sees `message.tool_calls`. |
| `_ollama_tool_call_to_openai` | 252 | Translates an Ollama `tool_call` to OpenAI SSE delta format. |
| `_Handler` | 281 | The `BaseHTTPRequestHandler` subclass. Methods: `_auth_ok` (5 header formats), `_log_auth_failure` (redacted), `do_GET` (healthz + models), `do_POST` (chat-completions). |
| `_handle_stream` | ~370 | Streaming branch. Two paths (tools-on / tools-off) emitting OpenAI SSE chunks. |
| `_handle_blocking` | ~460 | Non-streaming branch (full JSON response). |
| `_read_or_random_port` | 580 | Persisted-or-random port helper, 0o600 file. |
| `_default_serve_port` / `_default_local_port` | 613 / 629 | Per-user stable ports in 40000–49999 (REMOTE) and 49200–64200 (LOCAL). |
| `serve()` | 645 | Bootstrap: ensure Ollama, init RAG, resolve ports, set bearer token, start `ThreadingHTTPServer`. Prints the labelled connection block. Blocks until Ctrl+C. |

The whole thing depends on `ofa_main` only via `_retrieve_for_mode` and `_augment_messages` (RAG + system prompt) and `chat_stream` / `_ollama_chat_raw` (Ollama I/O). A colleague wanting to expose their own offline LLM via BYOK can use `ofa_server.py` as a template and replace those four call sites.

---

## 8. Deployment on Kestrel

### 8.1 Module file

The Lmod modulefile is at `/nopt/nrel/apps/cpu_stack/modules/default/application/assistant.lua` (outside the repo). Two functions:

1. **`help([[...]])`** — shown by `module help assistant`. Lists CLI invocations, env-var overrides, BYOK quick start, slash-command pointer.
2. **`LmodMessage([[...]])` at load time** — banner printed when the user does `module load assistant`. Includes the active-model warning, override env vars, usage table, and the two-line BYOK pointer.

Environment exports:

| Variable | Value |
|---|---|
| `OFA_ROOT` | `/nopt/nrel/apps/cpu_stack/software/openfoam/assistant` |
| `OLLAMA_MODELS` | `$OFA_ROOT/models` (so all users share the same model weights) |
| `OFA_VECTORDB` | `$OFA_ROOT/vectordb` |
| `PATH` | `$OFA_ROOT/bin` prepended (so `ofa` is on `$PATH`) |

The modulefile is updated in-place; changes are live for everyone on the next `module load`.

### 8.2 Per-user runtime state

Everything else lives under `$OFA_SCRATCH` (defaults to `/scratch/$USER`):

| File | Purpose |
|---|---|
| `.ofa_session.json` | Last interactive-mode message history. |
| `.ofa_history` | readline history for the interactive prompt. |
| `.ofa_prefs.txt` | Long-term user preferences (PREFS channel). |
| `.ofa_lessons.txt` | Model-autonomous lessons (LESSONS channel). |
| `.ofa_active_skills` (transient) | (none on disk — skills are session-only.) |
| `.ofa_api_key` | Bearer token for `--serve` (0o600). |
| `.ofa_serve_port` | Per-user persisted REMOTE port (0o600). |
| `.ofa_serve_local_port` | Per-user persisted LOCAL port for the printed ssh -L line. |
| `.ofa_ollama.pid` / `.ofa_ollama.port` | PID + port of the user's `ollama serve`. |
| `.ofa_vectordb.lock` | Per-user lock to serialise heavy ChromaDB inits. |

The `0o600` mode on the api-key / port files matters: `$OFA_SCRATCH` can be group/world-readable depending on filesystem ACLs, and these files contain user-specific secrets or routing info.

### 8.3 SLURM allocation flow

1. User runs `ofa` outside a SLURM job → `bin/ofa` calls `salloc` with the user's default account, requesting `--gres=gpu:1 --ntasks-per-node=32 --mem=80G --time=00:30:00 -p debug`.
2. Within `salloc`, the wrapper `srun --pty`s itself onto the compute node and re-execs `ofa`.
3. The compute-node-side wrapper unsets/cleans Cray PMI and SLURM step env vars (so any `srun` the agent kicks off later creates fresh steps) and finally execs `python3 src/ofa_main.py "$@"`.
4. `ensure_ollama_running()` finds a free per-user port (UID-derived hash), spawns `ollama serve` if no responding instance exists, waits for `/api/tags`.
5. `_init_rag()` loads ChromaDB collections and the BM25 token caches into memory.

The user can override account / partition / walltime via `OFA_ACCOUNT` / `OFA_PARTITION` / `OFA_WALLTIME` before invoking `ofa`. The recommendation for longer sessions is `OFA_PARTITION=gpu-h100 OFA_WALLTIME=04:00:00 ofa`.

---

## 9. Safety and security

| Concern | Mitigation |
|---|---|
| Destructive commands | Regex screen + red approval prompt + exact-phrase confirmation for catastrophic patterns. |
| Tool-output exfiltration | Output cap (96 KB) keeps massive command stdout from being fed back verbatim. Session compression strips old fenced blocks. |
| Prompt injection via indexed docs | `_fence_rag()` wraps every retrieved snippet with explicit "this is data, not instructions" tags. |
| BYOK auth | `Authorization: Bearer <token>` (or `api-key` / `x-api-key` / `openai-api-key` — see `_auth_ok`). Constant-time comparison. Bearer token in 0o600 file under `$OFA_SCRATCH`. 401 logs a redacted summary so debugging is possible without leaking tokens. |
| BYOK network exposure | `--serve-host` default `0.0.0.0` is required for ssh -L through the login node, but bearer token auth is mandatory by default. `--serve-no-auth` combined with 0.0.0.0 emits a loud warning. |
| Untested model risk | Models other than `gemma4:31b` (the only one in `TESTED_MODELS`) trigger an unmissable red startup banner explaining that the safety guards were validated against the default only. |
| Per-user isolation | Each user has their own `$OFA_SCRATCH`, their own Ollama process, their own SLURM allocation. No process-level sharing. |

`ofa --serve` runs entirely within Kestrel's internal network; the SSH port-forward is the only path from outside, and bearer tokens are mandatory unless the user explicitly opts out.

---

## 10. Performance characteristics

| Metric | Value |
|---|---|
| Model | `gemma4:31b` (Apache 2.0, Google) |
| Hardware | NVIDIA H100, 1× per user (quarter-node) |
| Cold-start | ~30–90 s (salloc + ollama load + model into GPU + RAG init) |
| First chat reply | ~5–15 s (model already warm, RAG retrieval included) |
| Subsequent replies | ~1–5 s for short turns, ~10–30 s for plan-then-generate file batches |
| Throughput | ~30–60 tokens/sec generation on a single H100 |
| RAG retrieval | < 200 ms per query (ChromaDB warm; BM25 caches in memory) |
| Index size on disk | ~500 MB (`vectordb/`) |
| Model weights | ~18 GB (`models/`) |

The dominant latency on first reply is GPU warm-up; on subsequent replies it's token generation. RAG and prompt construction are insignificant by comparison.

---

## 11. Testing

Three hermetic test suites, all stdlib `unittest`. Currently run manually (`python3 /tmp/test_ofa_*.py`); not yet wired into CI.

| Suite | Cases | Coverage |
|---|---:|---|
| [test_ofa_memory.py](file:///tmp/test_ofa_memory.py) | 14 | `_save_marker_block` extraction, multi-block, bullet stripping, dedup, caps, byte-cap eviction, atomic write under simulated `os.replace` failure, `extract_and_save_*` wrappers. |
| [test_ofa_skills.py](file:///tmp/test_ofa_skills.py) | 15 | Listing (empty, with README, with non-`.md`), summary parsing + truncation, sorting, missing dir, loading + path-traversal refusal, `_active_skill_names` parsing, integration. |
| [test_ofa_compress.py](file:///tmp/test_ofa_compress.py) | 7 | Session-history compression: old tool outputs, @file pastes, protected-tail silent no-op, under-threshold no-op, partial-progress + `/clear` hint, fence-tag preservation. |
| [test_ofa_server.py](file:///tmp/test_ofa_server.py) | 26 | BYOK HTTP: routing, auth (5 header formats), `/healthz`, `/v1/models`, blocking, streaming SSE, options pass-through, system-msg drop, RAG only on last user msg, mode routing, greetings bypass, key-file lifecycle, no-auth mode, tool_calls passthrough (8 cases). |
| **Total** | **62** | — |

All 62 pass at this commit (`a24a58b`).

---

## 12. Appendix

### 12.1 Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `OFA_ROOT` | `/nopt/nrel/apps/cpu_stack/software/openfoam/assistant` | Code + assets root. |
| `OFA_SCRATCH` | `/scratch/$USER` | Per-user runtime state. |
| `OFA_VECTORDB` | `$OFA_ROOT/vectordb` | ChromaDB store. |
| `OFA_MODEL` | `gemma4:31b` | Override the LLM. |
| `OFA_TEMPERATURE` | 1.0 | Sampling temperature. |
| `OFA_TOP_P` | 0.95 | Top-p sampling. |
| `OFA_TOP_K` | 64 | Top-k sampling. |
| `OFA_REPEAT_PENALTY` | 1.15 | Penalise repetition. |
| `OFA_NUM_PREDICT` | 32768 | Max tokens per response. |
| `OFA_NUM_CTX` | 65536 | Context-window tokens. |
| `OFA_NUM_GPU` | 99 | GPU layers (99 = all). |
| `OFA_ACCOUNT` | user's SLURM default | salloc account. |
| `OFA_PARTITION` | `debug` | salloc partition. |
| `OFA_WALLTIME` | `00:30:00` | salloc walltime. |
| `OFA_PORT` | UID-derived | Ollama port. |
| `NO_COLOR` | (unset) | Disables ANSI colour. |

### 12.2 CLI flags (selected)

| Flag | Effect |
|---|---|
| (none) | OpenFOAM mode, interactive. |
| `--hpc` | Kestrel HPC documentation mode. |
| `--code` | General coding assistant. |
| `--amrex` | AMReX C++ framework assistant. |
| `--marbles` | MARBLES (LBM thermal solver on AMReX) assistant. |
| `--quantum-computing` | Quantum-computing assistant (rigorous math verification). |
| `--rhel9_reframe` | ReFrame testing for RHEL9 migration. |
| `--resume` | Reload `.ofa_session.json`. |
| `--save DIR` | Write the assistant's `=== FILE ===` blocks into `DIR`. |
| `--no-rag` | Skip RAG retrieval. |
| `--fast` | OpenFOAM single-shot (skip plan stage). |
| `--model ID` | Override `gemma4:31b`. |
| `--list-models` | Print model registry and exit. |
| `--serve` | Start BYOK HTTP server. |
| `--serve-port N` | Pin REMOTE port (default: per-user persisted). |
| `--serve-local-port N` | Pin LOCAL port (default: per-user persisted). |
| `--serve-host ADDR` | Bind address (default `0.0.0.0`). |
| `--serve-api-key-file PATH` | Bearer-token file (default `$OFA_SCRATCH/.ofa_api_key`). |
| `--serve-no-auth` | Disable bearer-token auth. **Local dev only.** |
| `--serve-enable-tools` | Forward `tools` / `tool_choice` to Ollama; translate `tool_calls` back. Experimental. |

### 12.3 Slash commands (interactive mode)

```
quit | exit | q       — exit
/clear                — reset conversation (keeps system prompt, drops loaded skills)
/history              — show session size
/cwd                  — show current working directory
/retry                — re-prompt the model and demand a proper tool fence
/memory               — show what's stored in long-term memory
/remember <text>      — manually add a lesson to long-term memory
/forget [prefs|lessons|all]  — clear stored memory
/skills               — list available skill files
/skill <name>         — load a skill into this session
/skill off <name>     — unload a skill (use 'all' to unload every skill)
/models               — list pulled models and how to switch
save <dir>            — save last assistant response into <dir>
$ <shell command>     — run a shell command locally (cd persists)
@<path>               — inline a file into your prompt (relative to cwd)
/help                 — this message
```

### 12.4 Key references

- Repo: https://github.com/nileshsawant/onfield-assistant
- BYOK walkthrough: [`docs/byok-vscode.md`](byok-vscode.md)
- BYOK config helper: [`tools/byok-update-config.py`](../tools/byok-update-config.py)
- VS Code BYOK docs: https://code.visualstudio.com/blogs/2026/06/18/byok-vscode
- Ollama: https://github.com/ollama/ollama
- Gemma 4: https://blog.google/technology/developers/gemma-4/ (Apache 2.0)
- ChromaDB: https://www.trychroma.com/
- ReFrame: https://reframe-hpc.readthedocs.io/

### 12.5 Recent commit highlights

```
a24a58b  docs(byok): rewrite byok-vscode.md to lean 5-step quick start
e45be78  feat(serve): per-user stable Kestrel-side port (default)
9816511  feat(serve): --serve-enable-tools forwards tools/tool_calls to Ollama
3b137e9  feat(tools): byok-update-config.py to register all 5 OFA modes
0fe7eb8  fix(serve): accept multiple auth header formats; bind 0.0.0.0 default
08d7e18  docs(serve): label REMOTE vs LOCAL port explicitly
b933e0b  feat(serve): dynamic port allocation on both sides
79b52dc  feat(serve): print exact ssh tunnel command at startup
a4f1124  refactor(banner): hide model menu from startup, expose via /models
d2a83b4  feat(serve): add `ofa --serve` OpenAI-compatible BYOK shim for VS Code
df81e1b  fix(context): make session compression effective and quiet when no-op
497722d  feat(skills): on-demand skill loading via /skill
00e3f76  feat(memory): autonomous long-term memory with prefs + lessons channels
```

124 commits total at the time of writing.

---

*Document maintained alongside the code. If anything in this overview disagrees with the repo, the repo is the source of truth.*
