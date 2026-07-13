# OpenFOAM Assistant (ofa)

An AI-powered reasoning and autonomous execution agent tailored for the NREL Kestrel HPC system. The OpenFOAM Assistant (`ofa`) utilizes a local Large Language Model orchestrator (powered by Ollama and `gemma4:31b`) alongside Retrieval-Augmented Generation (RAG) using ChromaDB to help researchers build cases, compile complex scientific codebases, organically navigate Kestrel's HPC documentation, and execute multi-step SLURM jobs natively.

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
# Standard Interactive OpenFOAM case generator
$ ofa

# Single-shot query (runs without a continuous prompt loop)
$ ofa "Write the blockMeshDict for a backward facing step"

# Specialized Agent Modes
$ ofa --code            # General coding assistant and software engineering
$ ofa --hpc             # Kestrel HPC, SLURM documentation, and topology expert
$ ofa --amrex           # AMReX and MARBLES codebase assistant
$ ofa --rhel9_reframe   # ReFrame integration expert strictly adhering to the Kestrel RHEL9 software stack

# Additional Flags
$ ofa --resume          # Resume the previous interactive session (uses ~/.ofa_session.json)
$ ofa --save <dir>      # Save locally generated template cases to a specific directory path
$ ofa --no-rag          # Disable ChromaDB context retrieval; relies solely on standard LLM weights
$ ofa --fast            # Execute single-shot file generation simultaneously

# BYOK / programmatic server (OpenAI-compatible HTTP endpoint on this node)
$ ofa --serve                    # Start the local HTTP server; see docs/byok-vscode.md
$ ofa --serve --serve-enable-tools   # Also forward OpenAI tool_calls to Ollama (experimental)
```

## Architecture

* **`bin/ofa`**: The frontend Bash CLI wrapper. It routes SLURM jobs, sanitizes nested PMI and SLURM context variables to avoid step allocation deadlocks, natively forces the correct CUDA toolkit loads, and acts as the entrypoint for the Python environment.
* **`src/ofa_main.py`**: The central python controller. Handles Ollama binary lifecycle management via `subprocess`, processes ChromaDB interactions (`_hybrid_search`), loops user input, captures output, and dictates the strict regex parsing logic for the tool-calling mechanism.
* **`src/ofa_server.py`**: OpenAI-compatible HTTP shim used by `ofa --serve` — exposes ofa's system prompts + RAG + memory + multimodal (vision) at `/v1/chat/completions` for VS Code BYOK, `curl`, and the Python client below.
* **`src/ofa_client.py`**: Zero-dependency (stdlib-only) Python client. `from ofa_client import ask, Session` — see the next section.
* **`examples/`**: Worked end-to-end scripts users can copy verbatim. Currently ships `fit_and_ask.py` — a `curve_fit` demo that shows the `Session` + JSON-extraction patterns in ~230 LOC.
* **`prompts/`**: Directory configuring the personas. `common.txt` establishes the global rules for the agent, establishing the planning pipeline, code syntax standards, and environment constraints. `code.txt`, `hpc.txt`, and others inject the role-specific capabilities.
* **`vectordb/`**: The persistent storage directory for the offline ChromaDB ingestors, containing chunked embeddings for Kestrel's manuals, OpenFOAM examples, and RHEL module stacks.

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
    model="ofa-code",          # ofa-openfoam | ofa-hpc | ofa-code | ofa-amrex | ofa-reframe
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
