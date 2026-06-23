# OpenFOAM Assistant (`ofa`)

An AI-powered reasoning and autonomous execution agent tailored for Kestrel. The OpenFOAM Assistant (`ofa`) utilizes a local LLM orchestrator (powered by Ollama and `gemma4:31b`) and Retrieval-Augmented Generation (RAG) to help researchers build cases, compile code, navigate HPC documentation, and execute multi-step SLURM jobs natively.

## 🚀 Features
* **Automated Hardware Allocation:** If run from a login node, `ofa` automatically spins up an interactive GPU (`H100`) SLURM allocation to host the local LLM and immediately hands you the prompt.
* **Autonomous ReAct Framework:** The AI doesn't just chat. It makes a plan (`plan` blocks), writes actual files (`write` / `edit`), safely executes shell scripts (`bash` tool with real-time streaming), submits Slurm jobs (`sbatch`), and checks its own output to iteratively accomplish tasks.
* **Domain-Specific Agents:** Start `ofa` with different flags (`--code`, `--hpc`, `--rhel9_reframe`, `--amrex`) to inject different prompt personas and vector databases.
* **Robust Context Management:** Massive compiler outputs are smartly truncated, and older terminal logs in long sessions are intelligently compressed to prevent agent amnesia and context window collapse.
* **Fault Tolerance:** Built-in safeguards prevent the agent from getting stuck in hallucination loops (max consecutive error thresholds), safely handles interactive commands (hang protection), and elegantly catches `SIGTERM` signals for clean SLURM walltime roll-overs.

## 🛠️ Usage

Simply run `ofa` from anywhere on Kestrel (module load required depending on your stack).

```bash
# Interactive OpenFOAM case generator (Default)
$ ofa

# Single-shot query
$ ofa "Write the blockMeshDict for a backward facing step"

# Specialized Agent Modes
$ ofa --code            # Expert software engineer & general code assistant
$ ofa --hpc             # Kestrel HPC / SLURM documentation & topology expert
$ ofa --amrex           # AMReX/MARBLES codebase assistant
$ ofa --rhel9_reframe   # ReFrame integration expert for Kestrel RHEL9 migration

# Useful Flags
$ ofa --resume          # Resume the previous interactive session (uses .ofa_session.json)
$ ofa --save <dir>      # Save generated case files to a specific directory
$ ofa --no-rag          # Disable VectorDB context (raw LLM knowledge only)
```

## 🏗️ Architecture
* **`bin/ofa`**: Bash CLI wrapper. Detects hardware, automatically sets up SLURM allocations (or strips nested step trackers to avoid deadlocks), and loads CUDA drivers dynamically based on the RHEL 8 vs RHEL 9 topology before passing execution to Python.
* **`src/ofa_main.py`**: The central Python orchestration engine. Manages Ollama daemon processes, vectors queries through ChromaDB, loops user input, streams subprocess executions natively, enforces prompt safeguards, and maintains session state.
* **`prompts/`**: Directory containing the `.txt` files defining the personas and strict baseline execution guidelines (e.g., `common.txt`).

## 🧠 Memory & Context
The AI maintains a session scratchpad natively in your `$SCRATCH` directory (`~/.ofa_session.json`). Tilde expansion internally resolves to your literal home directory safely. Use `ofa --resume` to natively reload the AI's exact memory states even after SLURM revokes an earlier GPU allocation.

