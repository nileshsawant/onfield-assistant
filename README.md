# OnField Assistant 🌵 (ofa)

An AI-powered reasoning and autonomous execution agent tailored for the NLR Kestrel HPC system. OnField Assistant (`ofa`) utilizes a local Large Language Model orchestrator (powered by Ollama and `gemma4:31b`) alongside Retrieval-Augmented Generation (RAG) using ChromaDB to help researchers build cases, compile complex scientific codebases, organically navigate Kestrel's HPC documentation, and execute multi-step SLURM jobs natively.

## Features

* **Automated Hardware Allocation:** When invoked from a Kestrel login node, the `ofa` wrapper seamlessly detects the environment and spins up an interactive GPU node allocation (targeting H100 partitions). It automatically sets up the appropriate CUDA modules for either RHEL 8 or RHEL 9 environments before launching the LLM.
* **Autonomous ReAct Framework (Plan & Execute):** The assistant is instructed to follow a strict planning phase before acting. It creates persistent `plan` blocks and iterates autonomously using a variety of parsed markdown blocks:
  * `write` and `edit`: For creating and modifying files in the current workspace.
  * `bash`: For interactive terminal commands with real-time `subprocess.Popen` streaming back to the user.
  * `sbatch`: For dispatching background jobs and checking on their SLURM queue status without freezing the conversational loop.
  * `search` and `fetch`: For searching the internet or reading external web documentation when unsure.
* **Domain-Specific Modes:** By passing command-line arguments, the overarching python execution loop swaps the injected system prompts and RAG databases to act as specialized domain experts (e.g., general codebase engineering, specific AMReX compilation, or ReFrame module migrations).
* **Intelligent Context Management:** To survive long debugging or compilation sessions, the agent intelligently handles Context Window Collapse. Massive compiler toolchains are dynamically truncated. In deeply extended sessions (over 20 turns), older terminal stdout logs are systematically compressed while preserving the agent's fundamental reasoning and the user's initial instructions to avoid amnesia.
* **Robust Fault Tolerance & Safeguards:** The Python orchestrator natively intercepts hanging shell commands with `/dev/null` stdin piping. It tracks consecutive execution errors, pausing the autonomous loop if the agent hallucinates a failing command 3 times in a row, dropping control back to the human user. The daemon catches `SIGTERM` signals for 30-minute allocation timeouts, shutting down gracefully.

## Usage

Simply run the `ofa` command from your Kestrel environment. Make sure to load the corresponding application module first.

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
$ ofa --rhel9_reframe   # ReFrame integration expert strictly adhering to the Kestrel RHEL9 software stack

# Additional Flags
$ ofa --resume          # Resume the previous interactive session (uses ~/.ofa_session.json)
$ ofa --save <dir>      # (with --openfoam) save generated template cases to a directory
$ ofa --no-rag          # Disable ChromaDB context retrieval; relies solely on standard LLM weights
$ ofa --fast            # (with --openfoam) single-shot file generation (skip plan stage)

# BYOK / programmatic server (OpenAI-compatible HTTP endpoint on this node)
$ ofa --serve                    # Start the local HTTP server; see docs/byok-vscode.md
$ ofa --serve --serve-enable-tools   # Also forward OpenAI tool_calls to Ollama (experimental)
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

## Memory & Session Context

The assistant maintains its transient session state natively in your scratch directory (`~/.ofa_session.json`). Tilde expansion within the tool orchestrator is deliberately handled safely to target your literal home directory instead of generating corrupt relative paths.

If your SLURM allocation expires, the custom signal handlers will safely write the final context to disk. You can then request a new allocation and simply run `ofa --resume` (with any of your targeted flags) to perfectly reconstruct the context window. 
Any permanent global preferences (e.g., "always use 4 spaces for indentation") mentioned to the assistant are extracted into an isolated `~/.ofa_prefs.txt` file and automatically sourced into all future context windows.

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
    model="ofa-code",          # ofa-openfoam | ofa-hpc | ofa-code | ofa-amrex | ofa-marbles | ofa-reframe | ofa-quantum-computing
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
