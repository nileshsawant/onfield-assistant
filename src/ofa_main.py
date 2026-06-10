#!/usr/bin/env python3
"""OpenFOAM Assistant (ofa) — RAG-augmented LLM for OpenFOAM case setup."""

import argparse
import atexit
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

OFA_ROOT = os.environ.get("OFA_ROOT", str(Path(__file__).resolve().parent.parent))
OLLAMA_BIN = os.path.join(OFA_ROOT, "bin", "ollama")

# Run each user's Ollama server on a completely isolated port based on their numeric UID to completely eliminate collisions on shared SLURM compute nodes!
USER_UID = os.getuid()
OFA_PORT = 10000 + (USER_UID % 50000)
OLLAMA_HOST = f"http://127.0.0.1:{OFA_PORT}"

MODEL = "gemma4:31b"
PROMPTS_DIR = os.path.join(OFA_ROOT, "prompts")
OPENFOAM_PROMPT_PATH = os.path.join(PROMPTS_DIR, "openfoam.txt")
HPC_PROMPT_PATH = os.path.join(PROMPTS_DIR, "hpc.txt")
PLAN_PROMPT_PATH = os.path.join(PROMPTS_DIR, "plan.txt")
VECTORDB_PATH = os.environ.get("OFA_VECTORDB", os.path.join(OFA_ROOT, "vectordb"))

_embed_model = None       # loaded once at startup
_chroma_collection = None  # loaded once at startup
_hpc_docs_collection = None
_of13_src_collection = None
_amrex_src_collection = None
_marbles_src_collection = None
_reframe_src_collection = None



_ollama_proc = None

SESSION_FILE = f"/scratch/{os.environ.get('USER', 'default')}/.ofa_session.json"

def save_session(messages):
    try:
        with open(SESSION_FILE, "w") as f:
            json.dump(messages, f)
    except Exception:
        pass

def load_session():
    try:
        with open(SESSION_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return None


def manage_session_context(messages, max_chars=100000):
    """
    Intelligently compress session history. Instead of dropping messages (which 
    causes amnesia), we just strip the massive terminal logs from OLD messages, 
    keeping the agent's thought process, plans, and instructions intact.
    """
    import sys
    total_len = sum(len(m.get("content", "")) for m in messages)
    if total_len <= max_chars:
        return
        
    print(f"\n[System: Context size ({total_len} chars) near limit. Compressing old logs...]", file=sys.stderr)
    
    # Iterate from oldest to newest (skipping system prompt at 0 and the latest 3 messages)
    for i in range(1, len(messages) - 3):
        if total_len <= max_chars * 0.75: # Trim down to 75% capacity
            break
            
        msg = messages[i]
        if msg.get("role") == "user" and "Output from executed commands:" in msg.get("content", ""):
            old_len = len(msg["content"])
            if old_len > 400:
                msg["content"] = "[Older terminal output omitted by system to preserve context memory.]"
                total_len -= (old_len - len(msg["content"]))


def extract_and_save_prefs(response_text: str):
    import re
    prefs_match = re.search(r'=== PREFS ===(.*?)=== END PREFS ===', response_text, re.DOTALL)
    if prefs_match:
        new_prefs = prefs_match.group(1).strip()
        prefs_file = f"/scratch/{os.environ.get('USER', 'default')}/.ofa_prefs.txt"
        with open(prefs_file, "a") as f:
            f.write("\n" + new_prefs)
        print(f"  [Saved user preference to {prefs_file}]", file=sys.stderr)
  # set if we started Ollama ourselves


def _shutdown_ollama():
    """Terminate Ollama if this session started it."""
    global _ollama_proc
    if _ollama_proc is not None and _ollama_proc.poll() is None:
        _ollama_proc.terminate()
        try:
            _ollama_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _ollama_proc.kill()
        _ollama_proc = None

SESSION_FILE = f"/scratch/{os.environ.get('USER', 'default')}/.ofa_session.json"

def save_session(messages):
    try:
        with open(SESSION_FILE, "w") as f:
            json.dump(messages, f)
    except Exception:
        pass

def load_session():
    try:
        with open(SESSION_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return None

def extract_and_save_prefs(response_text: str):
    import re
    prefs_match = re.search(r'=== PREFS ===(.*?)=== END PREFS ===', response_text, re.DOTALL)
    if prefs_match:
        new_prefs = prefs_match.group(1).strip()
        prefs_file = f"/scratch/{os.environ.get('USER', 'default')}/.ofa_prefs.txt"
        with open(prefs_file, "a") as f:
            f.write("\n" + new_prefs)
        print(f"  [Saved user preference to {prefs_file}]", file=sys.stderr)




def extract_plan(response_text: str):
    import re
    plan_match = re.search(r'```plan\n(.*?)```', response_text, re.DOTALL | re.IGNORECASE)
    if plan_match:
        plan = plan_match.group(1).strip()
        print(f"\n[Tracking Plan:\n{plan}\n]", file=sys.stderr)
        return plan
    return None

def load_system_prompt(prompt_type="openfoam"):

    import os
    
    common_path = os.path.join(OFA_ROOT, "prompts", "common.txt")
    common_prompt = ""
    if os.path.exists(common_path):
        with open(common_path) as f: common_prompt = f.read().strip()
        
    if prompt_type == "code":
        with open(CODE_PROMPT_PATH) as f: prompt = f.read().strip()
    elif prompt_type == "hpc":
        with open(HPC_PROMPT_PATH) as f: prompt = f.read().strip()
    elif prompt_type == "reframe":
        reframe_prompt_path = os.path.join(OFA_ROOT, "prompts", "reframe.txt")
        if os.path.exists(reframe_prompt_path):
            with open(reframe_prompt_path) as f: prompt = f.read().strip()
        else:
            prompt = "You are a ReFrame testing assistant for Kestrel."
    elif prompt_type == "amrex":
        with open(os.path.join(OFA_ROOT, "prompts", "amrex.txt")) as f: prompt = f.read().strip()
    else:
        with open(OPENFOAM_PROMPT_PATH) as f: prompt = f.read().strip()
        
    if common_prompt:
        prompt = prompt + "\n\n" + common_prompt

    prefs_file = f"/scratch/{os.environ.get('USER', 'default')}/.ofa_prefs.txt"
    if os.path.exists(prefs_file):
        with open(prefs_file) as f:
            prefs = f.read().strip()
        if prefs:
            prompt += "\n\n--- USER PREFERENCES ---\n" + prefs
    return prompt


def ensure_ollama_running():
    """Start Ollama server if not already running."""
    try:
        r = httpx.get(f"{OLLAMA_HOST}/api/tags", timeout=2.0)
        if r.status_code == 200:
            # Verify the required model is actually loaded in this daemon
            tags = r.json().get("models", [])
            if any(m.get("name") == MODEL for m in tags):
                return True
            else:
                # Daemon is running but doesn't have our model! Probably a stale process using the wrong cache.
                print("Warning: Stale Ollama daemon detected. Attempting to kill it...", file=sys.stderr)
                # Removed 'import os, signal' from here to avoid UnboundLocalError since os is imported globally
                os.system(f"killall -u {os.environ.get('USER')} -9 ollama 2>/dev/null")
                time.sleep(1)
    except (httpx.ConnectError, httpx.TimeoutException):
        pass

    # Start Ollama in background
    env = os.environ.copy()
    env["OLLAMA_MODELS"] = os.path.join(OFA_ROOT, "models")
    env["OLLAMA_HOST"] = f"127.0.0.1:{OFA_PORT}"
    env["OLLAMA_FLASH_ATTENTION"] = "1"
    env["OLLAMA_KV_CACHE_TYPE"] = "q8_0"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["LD_LIBRARY_PATH"] = (
        os.path.join(OFA_ROOT, "lib")
        + ":"
        + env.get("LD_LIBRARY_PATH", "")
    )

    global _ollama_proc
    print("Starting Ollama server...", file=sys.stderr)
    _ollama_proc = subprocess.Popen(
        [OLLAMA_BIN, "serve"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setpgrp # <-- Critical: detach Ollama from Bash process group so Ctrl-C doesn't kill it
    )
    atexit.register(_shutdown_ollama)

    # Wait for server to be ready
    for _ in range(30):
        time.sleep(1)
        try:
            r = httpx.get(f"{OLLAMA_HOST}/api/tags", timeout=2.0)
            if r.status_code == 200:
                print("Ollama server ready.", file=sys.stderr)
                return True
        except (httpx.ConnectError, httpx.TimeoutException):
            continue

    print("ERROR: Could not start Ollama server.", file=sys.stderr)
    sys.exit(1)


def _init_rag():
    """Load the embedding model and ChromaDB collection once."""
    global _embed_model, _chroma_collection, _hpc_docs_collection, _of13_src_collection, _amrex_src_collection, _marbles_src_collection, _reframe_src_collection
    if _embed_model is not None:
        return
    import chromadb
    from sentence_transformers import SentenceTransformer
    _embed_model = SentenceTransformer(
        os.path.join(OFA_ROOT, "embedding_model"),
        device="cpu",
    )
    import subprocess
    import shutil
    user = os.environ.get("USER", "default")
    local_db = f"/scratch/{user}/.ofa_vectordb"
    
    # Sync the master vector database to the user's scratch to avoid readonly SQLite lock errors
    try:
        subprocess.run(["rsync", "-a", "--delete", f"{VECTORDB_PATH}/", f"{local_db}/"], check=False)
    except Exception as e:
        print(f"Warning: Failed to sync vector db locally: {e}", file=sys.stderr)
        local_db = VECTORDB_PATH
        
    client = chromadb.PersistentClient(path=local_db)
    _chroma_collection = client.get_collection("openfoam")
    try:
        _hpc_docs_collection = client.get_collection("hpc_docs")
    except Exception:
        _hpc_docs_collection = None
    try:
        _of13_src_collection = client.get_collection("of13_src")
    except Exception:
        _of13_src_collection = None
    try:
        _amrex_src_collection = client.get_collection("amrex_src")
    except Exception:
        _amrex_src_collection = None
    try:
        _marbles_src_collection = client.get_collection("marbles_src")
    except Exception:
        _marbles_src_collection = None
    try:
        _reframe_src_collection = client.get_collection("reframe_src")
    except Exception:
        _reframe_src_collection = None
_of13_src_collection = None



def fetch_url_context(query: str, max_chars: int = 64000) -> str:
    """Extract URLs from query, fetch their content, and return as context."""
    urls = re.findall(r'https?://\S+', query)
    if not urls:
        return ""
    parts = []
    for url in urls[:3]:  # cap at 3 URLs per query
        try:
            print(f"Fetching {url} ...", file=sys.stderr)
            r = httpx.get(url, timeout=15, follow_redirects=True,
                          headers={"User-Agent": "Mozilla/5.0 (OpenFOAM Assistant)"})
            if r.status_code != 200:
                continue
            html = r.text
            # Strip scripts/styles
            html = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', html,
                          flags=re.DOTALL | re.IGNORECASE)
            # Strip tags
            text = re.sub(r'<[^>]+>', ' ', html)
            # Decode entities
            for ent, ch in [('&amp;','&'),('&lt;','<'),('&gt;','>'),
                            ('&nbsp;',' '),('&quot;','"'),('&#39;',"'")]:
                text = text.replace(ent, ch)
            text = re.sub(r'\s+', ' ', text).strip()
            parts.append(f"[webpage: {url}]\n{text[:max_chars]}")
        except Exception as e:
            print(f"Warning: could not fetch {url}: {e}", file=sys.stderr)
    return "\n\n---\n\n".join(parts)


# OpenFOAM dict/solver names and generic words that are NOT tutorial case names
_CASE_SKIP = {
    # Generic words that match the patterns but are not case names
    'tutorial', 'setup', 'constant', 'system', 'solver', 'version',
    # OpenFOAM dict / file names
    'fvSchemes', 'fvSolution', 'blockMesh', 'blockMeshDict', 'controlDict',
    'divSchemes', 'gradSchemes', 'laplacianSchemes', 'ddtSchemes',
    'snappyHexMesh', 'decomposeParDict', 'transportProperties',
    'turbulenceProperties', 'momentumTransport', 'physicalProperties',
    # Solver names
    'foamRun', 'simpleFoam', 'pimpleFoam', 'pisoFoam', 'rhoPimpleFoam',
    'icoFoam', 'interFoam', 'buoyantFoam', 'openFoam', 'OpenFOAM',
    'parallelMesh', 'runApplication', 'runParallel', 'cleanCase',
}



_bm25_indices = {}
_bm25_docs_cache = {}

def _get_bm25_index(collection, name):
    global _bm25_indices, _bm25_docs_cache
    if name not in _bm25_indices and collection is not None:
        try:
            import re
            from rank_bm25 import BM25Okapi
            docs = collection.get()
            _bm25_docs_cache[name] = docs
            token_pattern = r'[a-zA-Z0-9\-]+'
            tokenized_docs = [re.findall(token_pattern, d.lower()) for d in docs["documents"]]
            _bm25_indices[name] = BM25Okapi(tokenized_docs)
        except ImportError:
            _bm25_indices[name] = False
    return _bm25_indices.get(name), _bm25_docs_cache.get(name)

def _hybrid_search(query: str, query_embedding: list, collection, coll_name: str, top_k: int, where_filter=None):
    if collection is None:
        return [], []
        
    # 1. Vector Search
    where_args = where_filter if where_filter else None
    res = collection.query(
        query_embeddings=[query_embedding], 
        n_results=top_k, 
        where=where_args,
        include=["documents", "metadatas"]
    )
    vec_docs = res["documents"][0] if res["documents"] else []
    vec_metas = res["metadatas"][0] if res["metadatas"] else []
    
    # 2. BM25 Search
    bm25_docs = []
    bm25_metas = []
    bm25, hw_docs = _get_bm25_index(collection, coll_name)
    if bm25 and hw_docs:
        import re
        import numpy as np
        token_pattern = r'[a-zA-Z0-9\-]+'
        stopwords = {"how", "to", "use", "here", "what", "is", "the", "a", "in", "on", "for", "with", "and", "do", "of", "can"}
        tokenized_query = [w for w in re.findall(token_pattern, query.lower()) if w not in stopwords]
        
        if tokenized_query:
            scores = bm25.get_scores(tokenized_query)
            top_n_idx = np.argsort(scores)[::-1]
            added = 0
            for idx in top_n_idx:
                if scores[idx] <= 0:
                    break
                meta = hw_docs["metadatas"][idx]
                doc = hw_docs["documents"][idx]
                
                if where_filter:
                    match = True
                    for k, v in where_filter.items():
                        if meta.get(k) != v:
                            match = False
                            break
                    if not match:
                        continue
                        
                bm25_docs.append(doc)
                bm25_metas.append(meta)
                added += 1
                if added >= top_k:
                    break

    # 3. Reciprocal Rank Fusion (RRF)
    rrf_scores = {}
    meta_map = {}
    for rank, (doc, meta) in enumerate(zip(vec_docs, vec_metas)):
        rrf_scores[doc] = rrf_scores.get(doc, 0.0) + 1.0 / (rank + 60)
        meta_map[doc] = meta
        
    for rank, (doc, meta) in enumerate(zip(bm25_docs, bm25_metas)):
        rrf_scores[doc] = rrf_scores.get(doc, 0.0) + 1.0 / (rank + 60)
        meta_map[doc] = meta
        
    sorted_docs = sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)
    top_fused = sorted_docs[:top_k]
    
    return top_fused, [meta_map[d] for d in top_fused]


def _extract_case_names(query: str) -> list:
    """Extract potential OpenFOAM tutorial case directory names from a query.

    Handles both camelCase names (e.g. pitzDaily, motorBike) and lowercase
    single-word names that appear adjacent to OF context keywords (e.g. "cavity
    tutorial", "elbow fvSchemes").
    """
    # camelCase identifiers: pitzDaily, motorBike, damBreak, airFoil2D ...
    camel = re.findall(r'\b([a-z][a-z0-9]*[A-Z][a-zA-Z0-9]*)\b', query)
    # Single lowercase words immediately before OF context keywords
    single = re.findall(
        r'\b([a-z][a-z0-9]+)\s+(?=tutorial|fvSchemes|fvSolution|blockMesh|\bcase\b|system/|constant/|0/)',
        query,
    )
    seen: set = set()
    candidates = []
    for c in camel + single:
        if c not in seen:
            seen.add(c)
            candidates.append(c)
    return [c for c in candidates if c not in _CASE_SKIP and len(c) >= 4]


def retrieve_context(query: str, top_k: int = 10) -> str:
    """Retrieve relevant OpenFOAM file chunks from the vector database using Hybrid Search."""
    try:
        _init_rag()
        
        semantic_query = query
        q_lower = query.lower()
        if 'pytorch' in q_lower or 'tensorflow' in q_lower or 'conda' in q_lower:
             semantic_query = 'Machine Learning, PyTorch, TensorFlow, python, cuda, gpu ' + query
        elif 'm-star' in q_lower or 'mstar' in q_lower:
             semantic_query = 'LBMcfd M-Star mstar mstar-cfd-mgpu Gila GPU CFD mpirun ' + query
        
        query_embedding = _embed_model.encode([semantic_query])[0].tolist()

        context_parts = []
        seen: set = set()

        # Case-specific pre-query
        for case_name in _extract_case_names(query)[:2]:
            try:
                docs, metas = _hybrid_search(
                    query=query, 
                    query_embedding=query_embedding, 
                    collection=_chroma_collection, 
                    coll_name="openfoam", 
                    top_k=8, 
                    where_filter={"case": case_name}
                )
                for doc, meta in zip(docs, metas):
                    key = doc[:60]
                    if key not in seen:
                        seen.add(key)
                        header = f"[{meta.get('version', '?')}/{meta.get('case', '?')}/{meta.get('filename', '?')}]"
                        context_parts.append(f"{header}\n{doc}")
            except Exception:
                pass

        # General hybrid semantic search 
        try:
            docs, metas = _hybrid_search(query=query, query_embedding=query_embedding, collection=_chroma_collection, coll_name="openfoam", top_k=top_k)
            for doc, meta in zip(docs, metas):
                key = doc[:60]
                if key not in seen:
                    seen.add(key)
                    header = f"[{meta.get('version', '?')}/{meta.get('case', '?')}/{meta.get('filename', '?')}]"
                    context_parts.append(f"{header}\n{doc}")
        except Exception:
            pass

        url_ctx = fetch_url_context(query)
        if url_ctx:
            context_parts.append(url_ctx)

        # C++ Source Code Intent Routing (Hybrid Search)
        keywords = ["c++", "source code", "cpp", "implementation", "header file", ".C", ".H"]
        wants_source = any(k in query.lower() for k in keywords)
        
        if wants_source and _of13_src_collection is not None:
            try:
                docs, metas = _hybrid_search(query=query, query_embedding=query_embedding, collection=_of13_src_collection, coll_name="of13_src", top_k=3)
                for s_doc, s_meta in zip(docs, metas):
                    s_header = f"[OpenFOAM 13 C++ Source Code - src/{s_meta.get('filepath', '?')}]"
                    context_parts.append(f"{s_header}\n{s_doc}\n")
            except Exception:
                pass

        return "\n\n---\n\n".join(context_parts)

    except Exception as e:
        import sys
        print(f"Warning: RAG retrieval failed ({e}), proceeding without context.", file=sys.stderr)
        return fetch_url_context(query)

def chat_stream(messages: list, **option_overrides):
    """Stream a chat response from Ollama."""
    opts = {
        "repeat_penalty": 1.15,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "num_predict": 32768,
        "num_ctx": 65536,
        "num_gpu": 99,
    }
    opts.update(option_overrides)
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": True,
        "options": opts,
    }
    try:
        with httpx.stream(
            "POST",
            f"{OLLAMA_HOST}/api/chat",
            json=payload,
            timeout=300.0,
        ) as resp:
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    if data.get("error"):
                        print(f"Ollama error: {data['error']}", file=sys.stderr)
                        break
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content
                    if data.get("done"):
                        break
    except KeyboardInterrupt:
        # Sub-catch inside the stream directly before it tears down httpx
        return


def chat_complete(messages: list) -> str:
    """Non-streaming chat call — returns full response string (for planning)."""
    return "".join(chat_stream(messages, temperature=1.0, top_p=0.95, top_k=64, num_predict=8192))


def plan_file_list(query: str, rag_context: str, system_prompt: str) -> list[str] | None:
    """Ask the LLM to plan which files the case needs, in generation order.

    Returns a list of file paths (e.g. ["system/controlDict", "0/U", ...])
    or None if parsing fails (caller should fall back to single-shot).
    """
    with open(PLAN_PROMPT_PATH) as f:
        plan_prompt = f.read().replace("{query}", query)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": (
            f"Here are relevant OpenFOAM example files for reference:\n\n{rag_context}\n\n---\n\n"
            + plan_prompt
        ) if rag_context else plan_prompt},
    ]
    response = chat_complete(messages)
    match = re.search(r"\[.*?\]", response, re.DOTALL)
    if match:
        try:
            files = json.loads(match.group())
            if isinstance(files, list) and files:
                return [str(f) for f in files]
        except (json.JSONDecodeError, TypeError):
            pass
    return None


def generate_file(
    filepath: str,
    query: str,
    rag_context: str,
    system_prompt: str,
    generated: dict[str, str],
) -> str:
    """Generate a single OpenFOAM file, using already-generated files as context.

    Streams output to stdout and returns the full file content.
    """
    # Trim prior_context to avoid prompt bloat: only first 30 lines per file
    prior_context = ""
    if generated:
        prior_context = "Already generated files for this case (use for cross-file consistency only):\n\n"
        for fpath, content in generated.items():
            trimmed = "\n".join(content.split("\n")[:30])
            prior_context += f"// File: {fpath}\n{trimmed}\n...\n\n"

    user_msg = (
        (f"Relevant OpenFOAM examples:\n\n{rag_context}\n\n---\n\n" if rag_context else "")
        + (prior_context if prior_context else "")
        + f"User request: {query}\n\n"
        f"Generate the complete content of `{filepath}` for an OpenFOAM case.\n"
        "Provide ALL required sub-dictionaries with correct entries. "
        "For fvSchemes include: ddtSchemes, gradSchemes, divSchemes, "
        "laplacianSchemes, interpolationSchemes, snGradSchemes. "
        "For fvSolution include: solvers block and SIMPLE or PIMPLE block. "
        "No explanation, just the file content."
    )
    # Build a clean FoamFile header (no backslash banner to avoid escaping issues)
    obj_name = filepath.split("/")[-1]
    location = "/".join(filepath.split("/")[:-1]) or "."
    field_classes = {
        "U": "volVectorField", "p": "volScalarField", "p_rgh": "volScalarField",
        "k": "volScalarField", "epsilon": "volScalarField", "omega": "volScalarField",
        "nut": "volScalarField", "nuTilda": "volScalarField", "T": "volScalarField",
        "alphat": "volScalarField", "R": "volSymmTensorField",
    }
    foam_class = field_classes.get(obj_name, "dictionary")
    sep = "// " + "* " * 37 + "//"
    # First-line hints seeded into prefill so LLM continues into content
    # rather than looping on the separator '*' pattern.
    first_line_hints = {
        "blockMeshDict":        "convertToMeters   1;\n\nvertices\n(\n",
        "controlDict":          "application         simpleFoam;\nstartFrom       startTime;\n",
        "fvSchemes":            "ddtSchemes\n{\n",
        "fvSolution":           "solvers\n{\n",
        "decomposeParDict":     "numberOfSubdomains  1;\n\nmethod          scotch;\n",
        "transportProperties":  "transportModel  Newtonian;\n\nnu\t\t\t[0 2 -1 0 0 0 0] 1e-05;\n",
        "turbulenceProperties": "simulationType  RAS;\n\nRAS\n{\n",
        "momentumTransport":    "simulationType  RAS;\n\nRAS\n{\n",
        "physicalProperties":   "viscosityModel  constant;\n\nviscosityCoeffs\n{\n",
        "U":       "dimensions      [0 1 -1 0 0 0 0];\n\ninternalField   uniform (0 0 0);\n\nboundaryField\n{\n",
        "p":       "dimensions      [0 2 -2 0 0 0 0];\n\ninternalField   uniform 0;\n\nboundaryField\n{\n",
        "k":       "dimensions      [0 2 -2 0 0 0 0];\n\ninternalField   uniform 0.375;\n\nboundaryField\n{\n",
        "epsilon": "dimensions      [0 2 -3 0 0 0 0];\n\ninternalField   uniform 14.855;\n\nboundaryField\n{\n",
        "nut":     "dimensions      [0 2 -1 0 0 0 0];\n\ninternalField   uniform 0;\n\nboundaryField\n{\n",
        "omega":   "dimensions      [0 2 -3 0 0 0 0];\n\ninternalField   uniform 0;\n\nboundaryField\n{\n",
    }
    first_line = first_line_hints.get(obj_name, "")
    foamfile_header = (
        f"// File: {filepath}\n"
        f"FoamFile\n{{\n"
        f"    format      ascii;\n"
        f"    class       {foam_class};\n"
        f"    location    \"{location}\";\n"
        f"    object      {obj_name};\n"
        f"}}\n"
        f"{sep}\n\n"
    ) + first_line
    # Instruct LLM to generate ONLY the dict content after the FoamFile block
    content_user_msg = (
        user_msg
        + "\n\nIMPORTANT: The FoamFile header block has already been written. "
        "Generate ONLY the dictionary entries that come after the "
        "'// * * * //' separator line. Do NOT repeat the FoamFile header. "
        "Start immediately with the first sub-dictionary "
        "(e.g. for fvSchemes: 'ddtSchemes {'; for controlDict: 'application'; "
        "for field files: 'dimensions'). "
        "End with the footer: // " + "* " * 37 + "//"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content_user_msg},
        {"role": "assistant", "content": foamfile_header},
    ]

    for attempt in range(3):
        content = ""
        for chunk in chat_stream(messages):
            print(chunk, end="", flush=True)
            content += chunk
        print()
        if content.strip():
            # Strip LLM preamble/re-outputs: find last "// File: ..." marker the LLM may have
            # re-output, and keep only what follows it. If it also re-output the FoamFile block,
            # skip ahead to after the separator line.
            marker = f"// File: {filepath}"
            last_idx = content.rfind(marker)
            if last_idx != -1:
                after = content[last_idx + len(marker):].lstrip("\n")
                # If LLM also re-output FoamFile header, skip past the separator line
                if after.lstrip().startswith("FoamFile") or after.lstrip().startswith("/*"):
                    sep_idx = after.find("// * * *")
                    if sep_idx != -1:
                        end_sep = after.find("\n", sep_idx)
                        after = after[end_sep + 1:].lstrip("\n") if end_sep != -1 else after
                content = after
            # If LLM re-output the first_line hint (already in prefill), strip duplicate
            if first_line and content.lstrip("\n").startswith(first_line):
                content = content.lstrip("\n")[len(first_line):]
            # Strip markdown backticks
            content = "\n".join(c for c in content.split("\n") if not c.strip().startswith("```"))
            return foamfile_header + content
        if attempt < 2:
            print(f"  [retry {attempt+1}/2 for {filepath}]", file=sys.stderr)
    print(f"  Warning: empty LLM response for {filepath}", file=sys.stderr)
    return foamfile_header


def save_case(response_text: str, output_dir: str):
    """Parse response and save files to a case directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    current_file = None
    current_content = []
    written_files: set[str] = set()

    for line in response_text.split("\n"):
        if line.strip().startswith("=== FILE:") or line.strip().startswith("// File:"):
            if current_file and current_content and current_file not in written_files:
                fpath = output_path / current_file
                fpath.parent.mkdir(parents=True, exist_ok=True)
                fpath.write_text("\n".join(current_content) + "\n")
                if "Allrun" in fpath.name or "Allclean" in fpath.name or fpath.name.endswith(".sh"):
                    import os
                    os.chmod(fpath, 0o755)
                print(f"  Written: {fpath}", file=sys.stderr)
                written_files.add(current_file)
            
            cf = line.strip()
            if cf.startswith("==="):
                cf = cf.replace("=== FILE:", "").replace("===", "").strip()
            else:
                cf = cf.replace("// File:", "").strip()
            
            current_file = cf
            current_content = []
        elif line.strip().startswith("=== END ==="):
            if current_file and current_content and current_file not in written_files:
                fpath = output_path / current_file
                fpath.parent.mkdir(parents=True, exist_ok=True)
                fpath.write_text("\n".join(current_content) + "\n")
                if "Allrun" in fpath.name or "Allclean" in fpath.name or fpath.name.endswith(".sh"):
                    import os
                    os.chmod(fpath, 0o755)
                print(f"  Written: {fpath}", file=sys.stderr)
                written_files.add(current_file)
                current_file = None
                current_content = []
        elif current_file is not None:
            if not line.strip().startswith("```"):
                current_content.append(line)

    if current_file and current_content and current_file not in written_files:
        fpath = output_path / current_file
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_text("\n".join(current_content) + "\n")
        if "Allrun" in fpath.name or "Allclean" in fpath.name or fpath.name.endswith(".sh"):
            import os
            os.chmod(fpath, 0o755)
        print(f"  Written: {fpath}", file=sys.stderr)


def interactive_mode(save_dir: str = None, resume: bool = False, hpc_mode: bool = False, code_mode: bool = False, amrex_mode: bool = False, reframe_mode: bool = False):
    """Run interactive chat loop."""
    current_plan = ""
    try:
        import readline
        hist_file = f"/scratch/{os.environ.get('USER', 'default')}/.ofa_history"
        if os.path.exists(hist_file):
            readline.read_history_file(hist_file)
        import atexit
        atexit.register(readline.write_history_file, hist_file)
    except Exception:
        pass

    system_prompt = load_system_prompt("reframe") if reframe_mode else (load_system_prompt("amrex") if amrex_mode else (load_system_prompt("code") if code_mode else (load_system_prompt("hpc") if hpc_mode else load_system_prompt("openfoam"))))
    messages = load_session() if resume else None
    if messages:
        messages[0]["content"] = system_prompt
        print("Resumed previous session.", file=sys.stderr)
    else:
        messages = [{"role": "system", "content": system_prompt}]

    print("NLR HPC & OpenFOAM AI Assistant - 3 Primary Modes:")
    print("  1. Dictionary Generator (Default) - Generates & runs cases")
    print("  2. HPC Documentation (--hpc) - Kestrel/Slurm support")
    print("  3. Coding Assistant (--code) - Read/Write/Execute codebase tools")
    print("\nFeatures:\n  - Session Resume (--resume)\n  - History saved to /scratch")
    print("\nType 'quit' to exit, 'save <dir>' to save last response.")
    print("-" * 60)

    last_response = ""

    while True:
        try:
            user_input = input("\n> ").strip()
        except KeyboardInterrupt:
            print("\n(Ctrl+C pressed. Type 'quit' to exit safely.)", file=sys.stderr)
            continue
        except EOFError:
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if user_input.lower().startswith("save "):
            dirname = user_input[5:].strip()
            if last_response:
                save_case(last_response, dirname)
                print(f"Case files saved to: {dirname}")
            else:
                print("No response to save yet.")
            continue

        if user_input.startswith("$"):
            cmd = user_input[1:].strip()
            print(f"[Executing Local Command: {cmd}]")
            import subprocess
            try:
                res = subprocess.run(cmd, shell=True, text=True, capture_output=True)
                cmd_out = ""
                if res.stdout: cmd_out += res.stdout
                if res.stderr: cmd_out += res.stderr
                if not cmd_out.strip(): cmd_out = "(No output)\n"
                
                if len(cmd_out) > 96000:
                    cmd_out = cmd_out[:48000] + "\n...[OUTPUT TRUNCATED]...\n" + cmd_out[-48000:]
            except Exception as e:
                cmd_out = f"Error executing command: {e}"
                
            print(cmd_out)
            augmented_input = f"I manually executed the following command:\n```bash\n{cmd}\n```\nHere is the output:\n```text\n{cmd_out}\n```\nPlease analyze this output or continue your previous thoughts incorporating this context."

        else:
            # Retrieve RAG context
            greetings = {"hi", "hello", "hey", "howdy", "thanks", "thank you"}
            is_greeting = user_input.strip().lower() in greetings
            if is_greeting:
                context = ""
            else:
                if reframe_mode:
                    rhel9_context = _get_reframe_rag(user_input)
                    base_context = retrieve_hpc_context(user_input)
                    context = f"=== RHEL9 SPECIFIC CONTEXT (TAKES PRECEDENCE) ===\n{rhel9_context}\n\n=== GENERAL HPC CONTEXT (RHEL8/Legacy) ===\n{base_context}"
                else:
                    context = retrieve_amrex_context(user_input) if amrex_mode else (retrieve_hpc_context(user_input) if (hpc_mode or code_mode) else retrieve_context(user_input))
            if context:
                if reframe_mode:
                    augmented_input = f"Extracted RHEL9 Stack & RHEL8 Context:\n\n{context}\n\n---\n\nUser request: {user_input}"
                elif hpc_mode or code_mode or amrex_mode:
                    augmented_input = f"Here is relevant context for your reference:\n\n{context}\n\n---\n\nUser request: {user_input}"
                else:
                    augmented_input = (
                        f"Here are relevant OpenFOAM example files for reference:\n\n"
                        f"{context}\n\n---\n\n"
                        f"User request: {user_input}"
                    )
            else:
                augmented_input = user_input

        messages.append({"role": "user", "content": augmented_input})

        # Stream response
        while True:
            last_response = ""
            try:
                for chunk in chat_stream(messages):
                    print(chunk, end="", flush=True)
                    last_response += chunk
            except KeyboardInterrupt:
                print("\n[AI generation interrupted by user (Ctrl+C)]", file=sys.stderr)
                pass
            except httpx.ConnectError:
                print("\n[Error: Connection to Ollama server lost. The backend may have crashed.]", file=sys.stderr)
                break
            print()

            messages.append({"role": "assistant", "content": last_response})
            save_session(messages)
            manage_session_context(messages)
            extract_and_save_prefs(last_response)
            new_plan = extract_plan(last_response)
            if new_plan: current_plan = new_plan
            
            cmd_out = check_and_execute_bash(last_response)
            if cmd_out:
                # If command output is extremely large, truncate it to prevent LLM context collapse
                if len(cmd_out) > 96000:
                    truncated = cmd_out[:48000] + "\n...[OUTPUT TRUNCATED]...\n" + cmd_out[-48000:]
                else:
                    truncated = cmd_out
                
                inject_msg = f"Output from executed commands:\n```text\n{truncated}\n```\nPlease continue to assist the user using this information."
                if current_plan:
                    inject_msg += f"\n\n[SYSTEM REMINDER] Proceed with your active plan:\n```plan\n{current_plan}\n```\nEvaluate what is complete and trigger the next step."
                    
                messages.append({"role": "user", "content": inject_msg})
                save_session(messages)
                manage_session_context(messages)
                print("\n[AI is analyzing the output...]", flush=True)
            else:
                break

        # Auto-save if --save was specified
        if save_dir:
            save_case(last_response, save_dir)
            print(f"\nCase files saved to: {save_dir}", file=sys.stderr)
    


def single_query(query: str, save_dir: str = None, fast: bool = False, resume: bool = False):
    current_plan = ""
    """Run a single query.

    By default uses sequential file generation (plan then generate each file
    with cross-file context).  Pass fast=args.fast to use the original single-shot
    mode (faster but less consistent across files).
    """
    system_prompt = load_system_prompt()
    rag_context = retrieve_context(query)

    if fast:
        # Single-shot: one LLM call generates all case files
        extra = (
            "\n\nGenerate these files: system/blockMeshDict, system/controlDict, "
            "system/fvSchemes, system/fvSolution, constant/transportProperties, "
            "constant/turbulenceProperties, 0/U, 0/p, 0/k, 0/epsilon, 0/nut, "
            "Allrun, Allclean.\n"
            "Mark each file using exactly: === FILE: <path> ===\n"
            "End each file using exactly: === END ===\n"
            "DO NOT OUTPUT separator comment lines like // * * * //\n"
            "For system/fvSchemes you MUST include exactly these 6 sub-dicts "
            "(in this order): ddtSchemes, gradSchemes, divSchemes, "
            "laplacianSchemes, interpolationSchemes, snGradSchemes. "
            "Example for simpleFoam/k-epsilon:\n"
            "ddtSchemes { default steadyState; }\n"
            "gradSchemes { default Gauss linear; grad(U) Gauss linear; }\n"
            "divSchemes { default none; div(phi,U) bounded Gauss linearUpwind grad(U); "
            "div(phi,k) bounded Gauss upwind; div(phi,epsilon) bounded Gauss upwind; "
            "div((nuEff*dev2(T(grad(U))))) Gauss linear; }\n"
            "laplacianSchemes { default Gauss linear corrected; }\n"
            "interpolationSchemes { default linear; }\n"
            "snGradSchemes { default corrected; }\n"
            "For system/fvSolution include: solvers block and SIMPLE block with residualControl."
        ) if save_dir else ""
        augmented = (
            f"Here are relevant OpenFOAM example files for reference:\n\n"
            f"{rag_context}\n\n---\n\n"
            f"User request: {query}{extra}"
        ) if rag_context else (query + extra)

        messages = load_session() if resume else None
        if messages:
            messages[0]["content"] = system_prompt
        else:
            messages = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "user", "content": augmented})
        response = ""
        try:
            for chunk in chat_stream(messages):
                print(chunk, end="", flush=True)
                response += chunk
        except KeyboardInterrupt:
            print("\n[AI generation interrupted by user (Ctrl+C)]", file=sys.stderr)
            pass
        print()
        if save_dir:
            save_case(response, save_dir)
            print(f"\\nCase files saved to: {save_dir}", file=sys.stderr)
        extract_and_save_prefs(response)
        messages.append({"role": "assistant", "content": response})
        save_session(messages)
        manage_session_context(messages)
        return

    # --- Sequential generation ---
    print("Planning case files...", file=sys.stderr)
    file_list = plan_file_list(query, rag_context, system_prompt)

    if not file_list:
        print("Warning: could not parse file plan, falling back to single-shot.", file=sys.stderr)
        single_query(query, save_dir=save_dir, fast=args.fast, resume=resume)
        return

    print(f"Generating {len(file_list)} files: {file_list}", file=sys.stderr)

    generated: dict[str, str] = {}
    full_response = ""

    for filepath in file_list:
        print(f"\n{'='*60}", flush=True)
        print(f"// File: {filepath}", flush=True)
        print(f"{'='*60}", flush=True)
        content = generate_file(filepath, query, rag_context, system_prompt, generated)
        generated[filepath] = content
        full_response += content + "\n\n"

    if save_dir:
        save_case(full_response, save_dir)
        print(f"\nCase files saved to: {save_dir}", file=sys.stderr)
    
    extract_and_save_prefs(full_response)
    messages = load_session() if resume else None
    if not messages:
        messages = [{"role": "system", "content": system_prompt}]
    messages.append({"role": "user", "content": query})
    messages.append({"role": "assistant", "content": f"Successfully generated {len(file_list)} files for the case."})
    save_session(messages)
    manage_session_context(messages)



with open(HPC_PROMPT_PATH) as f:
    HPC_SYSTEM_PROMPT = f.read().strip()

CODE_PROMPT_PATH = os.path.join(PROMPTS_DIR, "code.txt")
with open(CODE_PROMPT_PATH) as f:
    CODE_SYSTEM_PROMPT = f.read().strip()
with open(os.path.join(OFA_ROOT, "prompts", "amrex.txt")) as f:
    AMREX_SYSTEM_PROMPT = f.read()


_bm25_hpc = None
_hpc_all_docs = None

def _get_hpc_bm25():
    global _bm25_hpc, _hpc_all_docs
    if _bm25_hpc is None and _hpc_docs_collection is not None:
        try:
            import re
            from rank_bm25 import BM25Okapi
            _hpc_all_docs = _hpc_docs_collection.get()
            token_pattern = r'[a-zA-Z0-9\-]+'
            tokenized_docs = [re.findall(token_pattern, doc.lower()) for doc in _hpc_all_docs["documents"]]
            _bm25_hpc = BM25Okapi(tokenized_docs)
        except ImportError:
            _bm25_hpc = False # mark as missing
    return _bm25_hpc, _hpc_all_docs


def _get_reframe_rag(query: str, top_k: int = 5):
    try:
        with open("/scratch/nsawant/rhel9_module_structure.txt", "r") as f:
            static_rh9 = f.read().strip()
    except Exception:
        static_rh9 = ""
        
    _init_rag()
    query_embedding = _embed_model.encode([query])[0].tolist()
    context_parts = []
    
    if _reframe_src_collection is not None:
        try:
            docs, metas = _hybrid_search(query=query, query_embedding=query_embedding, collection=_reframe_src_collection, coll_name="reframe_src", top_k=top_k)
            for s_doc, s_meta in zip(docs, metas):
                s_header = f"[ReFrame Repository Code/Docs - {s_meta.get('filepath', '?')}]"
                context_parts.append(f"{s_header}\n{s_doc}\n")
        except Exception:
            pass
            
    dynamic_rh9 = "\n\n---\n\n".join(context_parts)
    return f"{static_rh9}\n\n{dynamic_rh9}".strip()

def retrieve_amrex_context(query: str, top_k: int = 5) -> str:
    _init_rag()
    query_embedding = _embed_model.encode([query])[0].tolist()
    context_parts = []
    
    # Try Marbles
    if _marbles_src_collection is not None:
        try:
            docs, metas = _hybrid_search(query=query, query_embedding=query_embedding, collection=_marbles_src_collection, coll_name="marbles_src", top_k=top_k)
            for s_doc, s_meta in zip(docs, metas):
                s_header = f"[MARBLES thermal C++ Source Code - src/{s_meta.get('filepath', '?')}]"
                context_parts.append(f"{s_header}\n{s_doc}\n")
        except Exception:
            pass

    # Try AMReX
    if _amrex_src_collection is not None:
        try:
            docs, metas = _hybrid_search(query=query, query_embedding=query_embedding, collection=_amrex_src_collection, coll_name="amrex_src", top_k=top_k)
            for s_doc, s_meta in zip(docs, metas):
                s_header = f"[AMReX Core Source Code - {s_meta.get('filepath', '?')}]"
                context_parts.append(f"{s_header}\n{s_doc}\n")
        except Exception:
            pass

    # Also grab generic Kestrel HPC docs so it knows about module paths / Slurm
    hpc_ctx = retrieve_hpc_context(query, top_k=2)
    if hpc_ctx:
        context_parts.append(hpc_ctx)

    return "\n\n---\n\n".join(context_parts)

def retrieve_hpc_context(query: str, top_k: int = 15) -> str:
    _init_rag()
    if _hpc_docs_collection is None:
        return ""
    try:
        import re
        query_embedding = _embed_model.encode([query])[0].tolist()
        res = _hpc_docs_collection.query(
            query_embeddings=[query_embedding], 
            n_results=top_k, 
            include=["documents", "metadatas"]
        )
        vec_docs = res["documents"][0] if res["documents"] else []
        vec_metas = res["metadatas"][0] if res["metadatas"] else []
        
        bm25, hw_docs = _get_hpc_bm25()
        bm25_docs = []
        bm25_metas = []
        if bm25 and hw_docs:
            token_pattern = r'[a-zA-Z0-9\-]+'
            stopwords = {"how", "to", "use", "here", "what", "is", "the", "a", "in", "on", "for"}
            tokenized_query = [w for w in re.findall(token_pattern, query.lower()) if w not in stopwords]
            
            if tokenized_query:
                scores = bm25.get_scores(tokenized_query)
                import numpy as np
                top_n_idx = np.argsort(scores)[::-1][:top_k]
                for idx in top_n_idx:
                    if scores[idx] > 0:
                        bm25_docs.append(hw_docs["documents"][idx])
                        bm25_metas.append(hw_docs["metadatas"][idx])

        rrf_scores = {}
        meta_map = {}
        for rank, (doc, meta) in enumerate(zip(vec_docs, vec_metas)):
            rrf_scores[doc] = rrf_scores.get(doc, 0.0) + 1.0 / (rank + 60)
            meta_map[doc] = meta
            
        for rank, (doc, meta) in enumerate(zip(bm25_docs, bm25_metas)):
            rrf_scores[doc] = rrf_scores.get(doc, 0.0) + 1.0 / (rank + 60)
            meta_map[doc] = meta
            
        sorted_docs = sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)
        top_fused = sorted_docs[:top_k]
        
        parts = []
        for d in top_fused:
            filename = meta_map[d].get('filename', '?')
            parts.append(f"[HPC DOCS / {filename}]\n{d}")
            
        url_ctx = fetch_url_context(query)
        if url_ctx:
            parts.append(url_ctx)

        return "\n\n---\n\n".join(parts)
    except Exception as e:
        import sys
        print(f"Warning: HPC RAG retrieval failed: {e}", file=sys.stderr)
        return fetch_url_context(query)


def check_and_execute_bash(response_text):
    import re, subprocess
    bash_blocks = re.findall(r"```(?:bash|sh|shell)\n(.*?)\n```", response_text, re.DOTALL)
    search_blocks = re.findall(r"```(?:search)(?:\s+|\n)(.*?)\s*```", response_text, re.DOTALL)
    fetch_blocks = re.findall(r"```(?:fetch)(?:\s+|\n)(.*?)\s*```", response_text, re.DOTALL)
    read_blocks = re.findall(r"```(?:read)(?:\s+|\n)(.*?)\s*```", response_text, re.DOTALL)
    write_blocks = re.findall(r"```(?:write)\s+([^\n]+)\n(.*?)\n```", response_text, re.DOTALL)
    edit_blocks = re.findall(r"```(?:edit)\s+([^\n]+)\n(.*?)\n```", response_text, re.DOTALL)

    if not bash_blocks and not search_blocks and not fetch_blocks and not read_blocks and not write_blocks and not edit_blocks:
        # Check if the AI wrote standard code blocks but forgot to use the tool syntax
        import re
        rogue_code = re.findall(r"```(cpp|c\+\+|bash|sh|python|cmake|cmakelists)\n(.*?)```", response_text, re.IGNORECASE | re.DOTALL)
        if rogue_code:
            # Let's see if we can salvage it if it mentioned a filename right before the block
            salvaged = False
            for rctype, rctext in rogue_code:
                # Find where this block is in the response
                block_idx = response_text.find("```" + rctype)
                if block_idx > 0:
                    preceding_text = response_text[:block_idx].split('\n')[-3:] # Get last 3 lines before the block
                    for line in preceding_text:
                        # Common ways LLMs declare filenames: "Here is main.cpp:" or "### `main.cpp`"
                        m = re.search(r'`?([a-zA-Z0-9_\-\./\\]+\.(?:cpp|H|h|c|py|cmake|sh|bash))`?', line)
                        if m:
                            filename = m.group(1)
                            print(f"\n[AI generated a rogue `{rctype}` code block, but mentioned file '{filename}'. Auto-casting to 'write' command...]")
                            write_blocks.append((filename, rctext.strip()))
                            salvaged = True
                            break
                        
            if not salvaged:
                print("\n[Warning] The AI generated raw code blocks but did not use the `write <file>` or `bash` tool syntax.")
                print("It cannot be executed automatically because there is no filepath context. You can copy-paste the code manually.")
        if not write_blocks:
            return None
    
    all_outputs = []

    # Process search blocks
    for q in search_blocks:
        q = q.strip()
        if not q:
            continue
        print(f"\n[Internet Search Suggested]")
        print(f"Query: {q}")
        ans = input("Execute this search? [y/N]: ").strip().lower()
        if ans in ('y', 'yes'):
            print("-" * 60)

            try:
                from ddgs import DDGS
                results = DDGS().text(q, max_results=3, safesearch='strict')
                out_str = f"Search Results for '{q}':\n"
                if results:
                    for i, r in enumerate(results):
                        out_str += f"{i+1}. {r['title']} ({r['href']})\n{r['body']}\n\n"

                else:
                    out_str += "No results found.\n"
                print(out_str)
                all_outputs.append(out_str)
            except Exception as e:

                err_msg = f"Error executing search: {e}"
                print(err_msg)
                all_outputs.append(err_msg)
            print("-" * 60)


    # Process fetch blocks
    for url in fetch_blocks:
        url = url.strip()
        if not url:
            continue
        print(f"\n[Web Fetch Suggested]")
        print(f"URL: {url}")
        ans = input("Execute this fetch? [y/N]: ").strip().lower()
        if ans in ('y', 'yes'):
            print("-" * 60)
            try:
                import httpx
                from lxml import html
                resp = httpx.get(url, timeout=5.0, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'})
                tree = html.fromstring(resp.content)
                for bad in tree.xpath('//script|//style|//header|//footer|//nav|//aside'):
                    bad.getparent().remove(bad)
                
                elements = tree.xpath('//text()')
                cleaned = " ".join([t.strip() for t in elements if t.strip() and len(t.strip()) > 3])
                
                if not cleaned:
                    cleaned = "Unable to read dynamic webpage content cleanly."
                out_str = f"\n--- Fetched URL: {url} ---\n{cleaned[:16000]}\n----------------------------------\n"
                print(out_str)
                all_outputs.append(out_str)
            except Exception as e:
                err_msg = f"\n--- Fetch Failed ---\n{str(e)}\n----------------------------------\n"
                print(err_msg)
                all_outputs.append(err_msg)
            print("-" * 60)


    # Process read blocks
    for file_to_read in read_blocks:
        import os
        file_to_read = os.path.expanduser(file_to_read.strip())
        if not file_to_read: continue
        print(f"\n[File Read Suggested]")
        print(f"File: {file_to_read}")
        
        # Auto-allow reading files from the global repos directory
        if "assistant/repos" in file_to_read or file_to_read.startswith("repos/"):
            print("Auto-approving read from reference repository...")
            ans = 'y'
        else:
            ans = input("Allow reading this file? [Y/n]: ").strip().lower()
            
        if ans in ('y', 'yes', ''):
            print("-" * 60)
            try:
                import os
                if os.path.exists(file_to_read):
                    with open(file_to_read, 'r') as f:
                        content = f.read()
                        out_str = f"\n--- Context from {file_to_read} ---\n{content[:16000]}\n----------------------------------\n"
                        print(f"Read {len(content)} characters.")
                else:
                    out_str = f"\n--- File Read Error ---\nFile not found: {file_to_read}\n----------------------------------\n"
                    print(out_str)
            except Exception as e:
                out_str = f"\n--- File Read Error ---\n{str(e)}\n----------------------------------\n"
                print(out_str)
            all_outputs.append(out_str)
            print("-" * 60)

    # Process write blocks
    for filepath, content in write_blocks:
        import os
        filepath = os.path.expanduser(filepath.strip())
        if not filepath: continue
        print(f"\n[File Write Suggested]")
        print(f"File: {filepath} ({len(content)} chars)")
        ans = input("Allow writing this file? [y/N]: ").strip().lower()
        if ans in ('y', 'yes'):
            print("-" * 60)
            try:
                import os
                os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
                with open(filepath, 'w') as f:
                    f.write(content)
                out_str = f"\n--- File Write Success ---\nSuccessfully wrote to {filepath}\n----------------------------------\n"
                print(f"Wrote to target file.")
            except Exception as e:
                out_str = f"\n--- File Write Error ---\n{str(e)}\n----------------------------------\n"
                print(out_str)
            all_outputs.append(out_str)
            print("-" * 60)

    # Process edit blocks
    for filepath, content in edit_blocks:
        import os
        filepath = os.path.expanduser(filepath.strip())
        if not filepath: continue
        print(f"\n[File Edit Suggested]")
        print(f"File: {filepath}")
        
        # Parse FIND and REPLACE sections
        if "<<FIND>>" in content and "<<REPLACE>>" in content:
            find_str = content.split("<<FIND>>")[1].split("<<REPLACE>>")[0].strip('\n')
            replace_str = content.split("<<REPLACE>>")[1].strip('\n')
            
            ans = input("Allow editing this file? [y/N]: ").strip().lower()
            if ans in ('y', 'yes'):
                print("-" * 60)
                try:
                    with open(filepath, 'r') as f:
                        file_data = f.read()
                    
                    if find_str in file_data:
                        file_data = file_data.replace(find_str, replace_str, 1)
                        with open(filepath, 'w') as f:
                            f.write(file_data)
                        out_str = f"\n--- File Edit Success ---\nSuccessfully edited {filepath}\n----------------------------------\n"
                        print(f"Edited target file successfully.")
                    else:
                        out_str = f"\n--- File Edit Error ---\nCould not find the exact <<FIND>> text in {filepath}. The file was not changed.\n----------------------------------\n"
                        print(out_str)
                except Exception as e:
                    out_str = f"\n--- File Edit Error ---\n{str(e)}\n----------------------------------\n"
                    print(out_str)
                all_outputs.append(out_str)
                print("-" * 60)
        else:
            out_str = f"\n--- File Edit Error ---\nEdit block missing <<FIND>> and <<REPLACE>> section markers.\n----------------------------------\n"
            print(out_str)
            all_outputs.append(out_str)

    # Process bash blocks
    for cmd in bash_blocks:
        cmd = cmd.strip()
        if not cmd:
            continue
            
        # Automatically inject --overlap into srun commands to prevent SLURM step deadlocks
        if "srun " in cmd and "--overlap" not in cmd:
            cmd = cmd.replace("srun ", "srun --overlap ")
            
        # Prevent the AI from executing standalone script files natively in the shell.
        # If the block starts with #!/bin/bash or contains pure #SBATCH lines, and DOES NOT contain `cat << 'EOF'`, it's just raw script text meant for the user.
        if (cmd.startswith("#!/bin/bash") or cmd.startswith("#!/bin/sh") or "#SBATCH" in cmd) and "cat <<" not in cmd:
            continue
            
        lines = cmd.split('\n')
        if all(line.strip().startswith('#') or not line.strip() for line in lines):
             # it is literally just a block of comments or an un-executed script file text.
             continue
        dangerous = False
        lower_cmd = cmd.lower()
        if any(bad in lower_cmd for bad in ["rm -rf", "mkfs", "dd if=", "> /dev/sda", "mv /"]):
            dangerous = True
        
        print(f"\n[System Command Suggested]")
        print(f"> {cmd}")
        if dangerous:
            print("WARNING: This command looks potentially destructive!")
            ans = input("Execute this command? [y/N]: ").strip().lower()
        else:
            # Auto-execute harmless stateless commands, such as module loads and ls.
            lines = [l.strip() for l in cmd.split('\n') if l.strip()]
            
            def is_line_safe(line):
                # No destructive chaining, command substitution, or outputs allowed
                if any(bad in line for bad in [">", ";", "&&", "||", "`", "$(", "|"]):
                    return False
                
                # Commands that are safe anywhere (stateless environment lookups)
                global_safe = ["module avail", "module show", "module list", "ls", "sinfo", "squeue", "pwd", "whoami", "echo", "which", "whereis"]
                if any(line == tool or line.startswith(tool + " ") for tool in global_safe):
                    return True
                    
                # Commands that read files (safe due to ACLs, but we still explicitly whitelist the base binaries)
                read_tools = ["grep", "cat", "find", "tree", "tail", "head", "stat"]
                if any(line == tool or line.startswith(tool + " ") for tool in read_tools):
                    return True
                    
                return False

            if lines and all(is_line_safe(l) for l in lines):
                print("Auto-approving read-only stateless command...")
                ans = 'y'
            else:
                ans = input("Execute this command? [y/N]: ").strip().lower()
                
        if ans in ('y', 'yes'):
            print("-" * 60)
            try:
                out_str = f"$ {cmd}\n"
                captured_text = ""
                
                # Use Popen to stream output in real-time
                process = subprocess.Popen(cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True)
                
                for line in process.stdout:
                    print(line, end="", flush=True)
                    captured_text += line
                    
                process.wait()
                
                # Truncate massive outputs to save context window for the AI (Terminal already saw full output)
                lines = captured_text.split('\n')
                if len(lines) > 100:
                    truncated = "\n".join(lines[:30]) + "\n... (output truncated, " + str(len(lines) - 60) + " lines omitted) ...\n" + "\n".join(lines[-30:])
                    out_str += truncated
                else:
                    out_str += captured_text
                
                # Hard limit character length to prevent context explosion on monolithic lines
                if len(out_str) > 3000:
                    out_str = out_str[:1500] + "\n...[OUTPUT TRUNCATED]...\n" + out_str[-1500:]
                all_outputs.append(out_str)
            except KeyboardInterrupt:
                err_msg = f"\n[Command execution aborted by user (Ctrl+C)]"
                print(err_msg)
                all_outputs.append(err_msg)
            except Exception as e:
                err_msg = f"Error executing command: {e}"
                print(err_msg)
                all_outputs.append(err_msg)
            print("-" * 60)
    
    if all_outputs:
        return "\n".join(all_outputs)
    return None


def hpc_single_query(query: str, resume: bool = False, code_mode: bool = False, amrex_mode: bool = False, reframe_mode: bool = False):
    current_plan = ""
    greetings = {"hi", "hello", "hey", "howdy", "thanks", "thank you"}
    is_greeting = query.strip().lower() in greetings
    if reframe_mode:
        rhel9_context = _get_reframe_rag(query) if not is_greeting else ""
        base_context = retrieve_hpc_context(query) if not is_greeting else ""
        context = f"=== RHEL9 SPECIFIC CONTEXT (TAKES PRECEDENCE) ===\n{rhel9_context}\n\n=== GENERAL HPC CONTEXT (RHEL8/Legacy) ===\n{base_context}" if not is_greeting else ""
    else:
        context = (retrieve_amrex_context(query) if amrex_mode else retrieve_hpc_context(query)) if not is_greeting else ""

    augmented = f"Context Information:\n---\n{context}\n---\n\nUser Query: {query}" if context else query
    messages = load_session() if resume else None
    if messages:
        messages[0]["content"] = HPC_SYSTEM_PROMPT
    else:
        messages = [{"role": "system", "content": load_system_prompt("reframe") if reframe_mode else (load_system_prompt("amrex") if amrex_mode else (load_system_prompt("code") if code_mode else load_system_prompt("hpc")))}]
    messages.append({"role": "user", "content": augmented})
    
    print(f"\n[HPC Documentation Assistant]\nQuerying Kestrel docs...", file=sys.stderr)
    while True:
        response = ""
        try:
            for chunk in chat_stream(messages):
                print(chunk, end="", flush=True)
                response += chunk
        except KeyboardInterrupt:
            print("\n[AI generation interrupted by user (Ctrl+C)]", file=sys.stderr)
            pass
        print("\n")
        messages.append({"role": "assistant", "content": response})
        save_session(messages)
        manage_session_context(messages)
        
        new_plan = extract_plan(response)
        if new_plan: current_plan = new_plan
        
        cmd_out = check_and_execute_bash(response)
        if cmd_out:
            if len(cmd_out) > 96000:
                truncated = cmd_out[:48000] + "\n...[OUTPUT TRUNCATED]...\n" + cmd_out[-48000:]
            else:
                truncated = cmd_out
                
            inject_msg = f"Output from executed commands:\n```text\n{truncated}\n```\nPlease continue to assist the user using this information."
            if current_plan:
                inject_msg += f"\n\n[SYSTEM REMINDER] Proceed with your active plan:\n```plan\n{current_plan}\n```\nEvaluate what is complete and trigger the next step."
                
            messages.append({"role": "user", "content": inject_msg})
            save_session(messages)
            manage_session_context(messages)
            print("\n[AI is analyzing the output...]", flush=True)
        else:
            break
            
    return


def handle_slurm_sigterm(*args):
    import sys
    import os
    print("\n\n[SLURM WALLTIME REACHED]", file=sys.stderr)
    print("SLURM has forcefully revoked the compute node allocation.", file=sys.stderr)
    print("Safely shutting down Ollama and exiting...", file=sys.stderr)
    _shutdown_ollama()
    # Os._exit completely bypasses Python's atexit threading tracebacks (like TMonitor queues) 
    # to guarantee a clean terminal exit when the cluster nukes the process
    os._exit(0)

def main():

    parser = argparse.ArgumentParser(
        description="OpenFOAM Assistant — AI-powered case setup helper"
    )
    parser.add_argument(
        "query", nargs="*", help="Query to ask (omit for interactive mode)"
    )
    parser.add_argument(
        "--save", "-s", metavar="DIR",
        help="Save generated case files to this directory"
    )
    parser.add_argument(
        "--no-rag", action="store_true",
        help="Disable RAG retrieval (use model knowledge only)"
    )
    parser.add_argument(
        "--hpc", action="store_true",
        help="Use the Kestrel HPC Documentation assistant instead of OpenFOAM"
    )
    parser.add_argument(
        "--amrex", action="store_true",
        help="Use AMReX/MARBLES assistant mode"
    )
    parser.add_argument(
        "--code", action="store_true",
        help="General coding assistant mode"
    )
    parser.add_argument(
        "--rhel9_reframe", action="store_true",
        help="ReFrame testing assistant for RHEL9 Kestrel migration"
    )
    parser.add_argument(
        "--resume", "-r", action="store_true",
        help="Resume previous conversation session"
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Single-shot mode: generate all files at once (faster, less consistent)"
    )
    args = parser.parse_args()

    # Handle Ctrl+C and SIGTERM gracefully (sys.exit triggers atexit handlers)
    # Use default KeyboardInterrupt handling for SIGINT
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, handle_slurm_sigterm)

    ensure_ollama_running()

    if args.no_rag:
        # Monkey-patch retrieve_context to return empty
        global retrieve_context
        retrieve_context = lambda *a, **kw: ""
    else:
        print("Loading RAG index...", file=sys.stderr)
        _init_rag()
        print("RAG ready.", file=sys.stderr)

    if args.query:
        if args.rhel9_reframe:
            hpc_single_query(" ".join(args.query), resume=args.resume, code_mode=False, amrex_mode=False, reframe_mode=True)
        elif args.amrex:
            hpc_single_query(" ".join(args.query), resume=args.resume, amrex_mode=True)
        elif args.code:
            # Reusing hpc logic internally but pointing to code prompt later
            hpc_single_query(" ".join(args.query), resume=args.resume, code_mode=True)
        elif args.hpc:
            hpc_single_query(" ".join(args.query), resume=args.resume)
        else:
            single_query(" ".join(args.query), save_dir=args.save, fast=args.fast, resume=args.resume)
    else:
        interactive_mode(save_dir=args.save, resume=args.resume, hpc_mode=args.hpc, code_mode=args.code, amrex_mode=args.amrex, reframe_mode=args.rhel9_reframe)


if __name__ == "__main__":
    main()
