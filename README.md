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
```

## Architecture

* **`bin/ofa`**: The frontend Bash CLI wrapper. It routes SLURM jobs, sanitizes nested PMI and SLURM context variables to avoid step allocation deadlocks, natively forces the correct CUDA toolkit loads, and acts as the entrypoint for the Python environment.
* **`src/ofa_main.py`**: The central python controller. Handles Ollama binary lifecycle management via `subprocess`, processes ChromaDB interactions (`_hybrid_search`), loops user input, captures output, and dictates the strict regex parsing logic for the tool-calling mechanism.
* **`prompts/`**: Directory configuring the personas. `common.txt` establishes the global rules for the agent, establishing the planning pipeline, code syntax standards, and environment constraints. `code.txt`, `hpc.txt`, and others inject the role-specific capabilities.
* **`vectordb/`**: The persistent storage directory for the offline ChromaDB ingestors, containing chunked embeddings for Kestrel's manuals, OpenFOAM examples, and RHEL module stacks.

## Memory & Session Context

The assistant maintains its transient session state natively in your scratch directory (`~/.ofa_session.json`). Tilde expansion within the tool orchestrator is deliberately handled safely to target your literal home directory instead of generating corrupt relative paths.

If your SLURM allocation expires, the custom signal handlers will safely write the final context to disk. You can then request a new allocation and simply run `ofa --resume` (with any of your targeted flags) to perfectly reconstruct the context window. 
Any permanent global preferences (e.g., "always use 4 spaces for indentation") mentioned to the assistant are extracted into an isolated `~/.ofa_prefs.txt` file and automatically sourced into all future context windows.
