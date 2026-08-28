# OnField Assistant (`ofa`) 🌵

**Locally-hosted, RAG-augmented LLM assistant** for HPC and scientific-computing workflows. Runs Gemma 4 (31B, Q8_0, at its full 262144-token context) on a single GPU via Ollama and ships specialized modes for OpenFOAM, AMReX, MARBLES, ReFrame, VASP, quantum computing, HPC support, and general coding.

- **Repository:** [github.com/nileshsawant/onfield-assistant](https://github.com/nileshsawant/onfield-assistant)
- **Ships with:** a VS Code extension that puts all eight modes in Copilot Chat's model picker, a `--serve` OpenAI-compatible HTTP endpoint for other editors / opencode, a stdlib-only `ofa_client` Python module, and a one-command `install.sh` for standing ofa up on any Linux HPC.

## What lives where

- **User & porting guide:** [Getting started](getting-started.md) — end-to-end usage, CLI flags, installation on a new HPC, and the VS Code extension walkthrough.
- **VS Code / other clients:** [Use ofa from VS Code](byok-vscode.md) — the manual BYOK route over an ssh tunnel. Superseded by the extension for Remote-SSH users; still the reference for other editors.
- **Architecture:** [High-level layout](architecture.md) and the [in-depth technical overview](ofa-technical-overview.md).
- **RAG maintenance:** [Updating the RAG corpora](rag-maintenance.md) — operator playbook for git-tracked, vendored, and copyright-restricted corpora.
- **API reference:** [Python API](api.rst) — every module, class, and function in `src/`, hyperlinked, with `[source]` jumps to the exact line on GitHub.

```{toctree}
:hidden:
:caption: User guides

getting-started
byok-vscode
architecture
ofa-technical-overview
rag-maintenance
```

```{toctree}
:hidden:
:caption: Reference

api
```
