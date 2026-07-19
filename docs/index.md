# OnField Assistant (`ofa`) 🌵

**Locally-hosted, RAG-augmented LLM assistant** for HPC and scientific-computing workflows. Runs Gemma 4 (31B) on a single GPU via Ollama and ships specialized modes for OpenFOAM, AMReX, MARBLES, ReFrame, quantum computing, HPC support, and general coding.

- **Repository:** [github.com/nileshsawant/onfield-assistant](https://github.com/nileshsawant/onfield-assistant)
- **Ships with:** a `--serve` OpenAI-compatible HTTP endpoint for VS Code BYOK / opencode, a stdlib-only `ofa_client` Python module, and a one-command `install.sh` for standing ofa up on any Linux HPC.

## What lives where

- **User & porting guide:** [Getting started](getting-started.md) — end-to-end usage, CLI flags, installation on a new HPC.
- **BYOK integration:** [Use ofa from VS Code](byok-vscode.md) — pointing VS Code Chat at `ofa --serve` over an ssh tunnel.
- **Architecture:** [High-level layout](architecture.md) and the [in-depth technical overview](ofa-technical-overview.md).
- **API reference:** [Python API](api.rst) — every module, class, and function in `src/`, hyperlinked, with `[source]` jumps to the exact line on GitHub.

```{toctree}
:hidden:
:caption: User guides

getting-started
byok-vscode
architecture
ofa-technical-overview
```

```{toctree}
:hidden:
:caption: Reference

api
```
