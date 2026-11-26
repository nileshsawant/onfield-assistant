# OnField Assistant 🌵 (`ofa`) Architecture & Hybrid Search

## Overview
The OnField Assistant (`ofa`) is an AI-powered CLI wrapper designed to help HPC users navigate complex CFD case setups, explore OpenFOAM C++ source code, and retrieve HPC hardware documentation directly from the console. It leverages locally hosted Large Language Models (via Ollama) augmented with highly specialized domain knowledge through Retrieval-Augmented Generation (RAG).

## Core Components
* **LLM Backend**: Ollama running locally on the user's GPU allocation (default model `gemma4:31b-it-q8_0` at its full 262144-token context; see `MODEL_REGISTRY` in `src/ofa_main.py`). Optional alternatives: an OpenAI-compatible LiteLLM gateway (`--litellm`), a larger model that `ofa --model` starts automatically on its own GPU node(s) with vLLM or llama.cpp (`src/ofa_vllm.py`), or a hand-started vLLM server (`--vllm`). Retrieval and prompts stay local in every case.
* **Vector Database**: ChromaDB (Persistent).
* **Embeddings**: `sentence-transformers` for dense vector semantic representations.
* **Lexical Search**: `rank-bm25` (Okapi BM25) for exact keyword and token matching.

## Retrieval-Augmented Generation (RAG) Collections
There are eight ChromaDB collections (see the technical overview for the full list); each mode queries the ones relevant to it. The OpenFOAM and HPC paths use:
1. **`openfoam`**: OpenFOAM tutorials and case dictionaries to accurately generate and format case files (`system/`, `constant/`, `0/`).
2. **`hpc_docs`**: Hardware, SLURM, and exact HPC software guides (e.g., Kestrel vs Gila configurations).
3. **`of13_src`**: OpenFOAM C++ source code (accessible via keywords like `cpp`, `source code`, `pEqn.H`, etc.).

## Hybrid Search & Reciprocal Rank Fusion (RRF)
Initially, the RAG pipeline relied purely on dense vector similarity (Semantic Search). This posed a problem for highly specific, domain-unique keywords (e.g., "M-Star", "STAR-CCM+", "pEqn.H"). Semantic search models tend to dilute specialized terms into generic concepts, returning irrelevant tutorials and promoting LLM hallucinations.

To resolve this, the architecture employs a **True Hybrid Search** mechanism:

1. **Vector Semantic Search**: Encodes the user's query and retrieves the top `K` most conceptually similar documents. This is excellent for conceptual intent like *"tips for optimizing simulations."*
2. **BM25 Lexical Search**: Tokenizes the query—filtering out common English stopwords (`how`, `to`, `what`, `is`, `here`)—and performs an exact-match term frequency-inverse document frequency search. This is crucial for retrieving specific software acronyms or exact C++ file structures when the semantic model fails to prioritize them.
3. **Reciprocal Rank Fusion**: RRF mathematically merges the ranked result lists from both the Vector and BM25 search queries into a single unified context block:
   ```python
   # Simplification of the fusion logic
   rrf_score = 1.0 / (rank + 60)
   ```
   The merged list ensures that documents appearing high in *either* exact-keyword matches or semantic representation take absolute priority in the LLM's context window.

**Query construction**: the retrieval query is the user's latest request followed by up to two earlier requests (`build_rag_query()`), so follow-ups like "now run it on a GPU" keep the earlier topic. Context that VS Code adds around the request is stripped first (`request_text()`). In HPC retrieval, chunks from the RHEL9 upgrade docs are placed first and labelled CURRENT so they outrank older guidance.

## Primary Execution Paths
- **General coding (`ofa`, the default)**: an agent loop that reads and edits files, runs commands, and submits SLURM jobs, with Kestrel HPC docs retrieved as context. Other modes (`--amrex`, `--marbles`, `--vasp`, `--quantum-computing`, `--rhel9_reframe`) swap in their own prompt and collections.
- **Case Generation (`ofa --openfoam <query>`)**: Processes requested configurations, queries the `openfoam` DB, and predictably maps output to required file hierarchies. Can be run in step-by-step memory tracking mode, or single-shot (`--fast`) for rapid setup. Saved directly to target directories (`--save`).
- **HPC Documentation (`ofa --hpc <query>`)**: Bypasses the CFD configuration pipeline to strictly query `hpc_docs`. Used for generating accurate `salloc` and `sbatch` setups tailored for Kestrel/Gila software modules (like M-Star, PyTorch, Conda).
- **C++ Exploration**: RAG automatically intercepts C++ intent (checking for `.C`, `.H`, `cpp`, or `implementation`) and executes a hybrid query against the `of13_src` DB collection to explain and retrieve actual OpenFOAM class architectures.
