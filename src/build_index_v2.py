#!/usr/bin/env python3
"""
Build the OpenFOAM Assistant RAG index from scratch.

Sources:
  1. OpenFOAM-13 tutorials, src headers, applications headers, etc/caseDicts
     - system/, 0/, constant/ files: whole-file (up to 16 KB), then split
     - .H source headers: 2000-char chunks
  2. 13 Municchi et al. arXiv papers (fetched from arxiv.org/html/...)
  3. Kestrel HPC system documentation (local markdown files)

Run on a GPU/compute node (not login node) for speed.
"""

import os
import re
import sys
import hashlib
import time
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OFA_ROOT = os.environ.get(
    "OFA_ROOT",
    str(Path(__file__).resolve().parent.parent),
)
VECTORDB_PATH = os.environ.get("OFA_VECTORDB", os.path.join(OFA_ROOT, "vectordb"))
EMBEDDING_MODEL_PATH = os.path.join(OFA_ROOT, "embedding_model")

OF13_ROOT = Path(
    "/nopt/nrel/apps/cpu_stack/software/openfoam"
    "/openfoam13_craympich_scotch/OpenFOAM-13"
)

KESTREL_DOCS_ROOT = Path(
    "/projects/hpcapps/nsawant/apps/HPC/docs/Documentation"
)
KESTREL_SUBDIRS = [
    "Systems/Kestrel",
    "Applications",
]

ARXIV_PAPER_IDS = [
    "1811.06960",
    "1811.06972",
    "1906.01316",
    "1909.02818",
    "1909.13767",
    "2006.02704",
    "2105.08853",
    "2203.09305",
    "2212.10961",
    "2212.13519",
    "2301.13160",
    "2304.09180",
    "2404.19636",
]

# Local cache directory for pre-downloaded paper texts (avoids compute-node network issues)
PAPERS_CACHE_DIR = Path(OFA_ROOT) / "papers"

# ---------------------------------------------------------------------------
# Chunking settings
# ---------------------------------------------------------------------------
CASE_DICT_CHUNK   = 16_000   # case dict files (system/0/constant): ≤16 KB = one chunk
CASE_DICT_OVERLAP = 500
SOURCE_CHUNK      = 2_000    # .H source headers
SOURCE_OVERLAP    = 200
KESTREL_CHUNK     = 3_000    # Kestrel markdown
KESTREL_OVERLAP   = 300
PAPER_CHUNK       = 3_000    # arXiv paper text
PAPER_OVERLAP     = 300

BATCH_SIZE = 256

# ---------------------------------------------------------------------------
# Skip rules
# ---------------------------------------------------------------------------
SKIP_DIRS = {
    "postProcessing", "processor0", "processor1", "processor2", "processor3",
    "dynamicCode", ".git", "polyMesh", "lnInclude", "platforms", "build",
    "Make", "wmake", "__pycache__", "test",
}
SKIP_EXTENSIONS = {
    ".gz", ".obj", ".stl", ".vtk", ".vtu", ".png", ".jpg", ".pdf",
    ".so", ".o", ".dep", ".a", ".pyc", ".eps", ".svg",
}
MAX_FILE_SIZE = 64 * 1024  # 64 KB hard cap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def should_skip_dir(name: str) -> bool:
    return name in SKIP_DIRS or name.startswith(".")


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks, preferring newline breaks."""
    if len(text) <= size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        if end < len(text):
            nl = text.rfind("\n", start + size // 2, end + 200)
            if nl > start:
                end = nl + 1
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def stable_id(*parts) -> str:
    """Create a stable, filesystem-safe ID from arbitrary string parts."""
    raw = "_".join(str(p) for p in parts)
    return hashlib.md5(raw.encode()).hexdigest()[:16] + "_" + re.sub(r"[^a-zA-Z0-9_-]", "_", raw)[:80]


def is_case_dict_file(filepath: Path) -> bool:
    """True if file lives in system/, 0/, or constant/ under a tutorial case."""
    parts = filepath.parts
    try:
        tut_idx = parts.index("tutorials")
    except ValueError:
        return False
    if tut_idx + 3 < len(parts):
        return parts[tut_idx + 3] in ("system", "0", "constant")
    return False


# ---------------------------------------------------------------------------
# Source 1: OpenFOAM-13
# ---------------------------------------------------------------------------

def should_index_of_file(filepath: Path, context: str) -> bool:
    if filepath.suffix in SKIP_EXTENSIONS:
        return False
    try:
        st = filepath.stat()
    except OSError:
        return False
    if st.st_size == 0 or st.st_size > MAX_FILE_SIZE:
        return False
    if context == "source":
        return filepath.suffix == ".H"
    if context in ("tutorials", "etc"):
        return True
    return False


def walk_of(root: Path, context: str) -> list[Path]:
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not should_skip_dir(d)]
        dp = Path(dirpath)
        for fname in filenames:
            fp = dp / fname
            try:
                if should_index_of_file(fp, context):
                    files.append(fp)
            except (OSError, PermissionError):
                continue
    return files


def collect_of13() -> list[tuple[Path, str]]:
    """Return (filepath, context) for all OF-13 files."""
    if not OF13_ROOT.exists():
        print(f"ERROR: OF-13 root not found: {OF13_ROOT}")
        sys.exit(1)

    result = []
    for subdir, ctx in [
        ("tutorials",       "tutorials"),
        ("src",             "source"),
        ("applications",    "source"),
        ("etc/caseDicts",   "etc"),
    ]:
        d = OF13_ROOT / subdir
        if d.exists():
            files = walk_of(d, ctx)
            result.extend((f, ctx) for f in files)
            print(f"  OF-13 {subdir}: {len(files)} files")
        else:
            print(f"  OF-13 {subdir}: NOT FOUND (skipping)")
    return result


def make_of_prefix(filepath: Path, context: str) -> str:
    try:
        rel = filepath.relative_to(OF13_ROOT)
    except ValueError:
        rel = filepath
    parts = rel.parts

    if context == "tutorials" and "tutorials" in parts:
        tut_idx = parts.index("tutorials")
        rel_from_tut = "/".join(parts[tut_idx:])
        return f"[openfoam13] [tutorials] FILE: {rel_from_tut}"
    if context == "source":
        return f"[openfoam13] [source] {rel}"
    if context == "etc":
        return f"[openfoam13] [etc] {rel}"
    return f"[openfoam13] [{context}] {filepath.name}"


def index_of13(model, collection) -> int:
    print("\n--- Indexing OpenFOAM-13 ---")
    all_files = collect_of13()
    print(f"  Total files: {len(all_files)}")

    docs, metas, ids = [], [], []
    batch_num = chunk_total = 0

    def flush():
        nonlocal batch_num, docs, metas, ids
        if not docs:
            return
        batch_num += 1
        embs = model.encode(docs, show_progress_bar=False).tolist()
        collection.add(documents=docs, embeddings=embs, metadatas=metas, ids=ids)
        print(f"    batch {batch_num}: {chunk_total} chunks so far", flush=True)
        docs, metas, ids = [], [], []

    for i, (filepath, context) in enumerate(all_files):
        try:
            text = filepath.read_text(encoding="utf-8", errors="ignore").strip()
        except (OSError, PermissionError):
            continue
        if not text:
            continue

        cs, co = (CASE_DICT_CHUNK, CASE_DICT_OVERLAP) if (
            context == "tutorials" and is_case_dict_file(filepath)
        ) else (SOURCE_CHUNK, SOURCE_OVERLAP)

        prefix = make_of_prefix(filepath, context)
        parts = filepath.parts
        meta = {
            "version":  "openfoam13",
            "context":  context,
            "filename": filepath.name,
            "path":     str(filepath),
        }
        if context == "tutorials" and "tutorials" in parts:
            tut_idx = parts.index("tutorials")
            if tut_idx + 1 < len(parts):
                meta["solver"] = parts[tut_idx + 1]
            if tut_idx + 2 < len(parts):
                meta["case"] = parts[tut_idx + 2]

        for ci, chunk in enumerate(chunk_text(text, cs, co)):
            docs.append(f"{prefix}\n{chunk}")
            metas.append(meta)
            try:
                rel_id = str(filepath.relative_to(OF13_ROOT))
            except ValueError:
                rel_id = str(filepath)
            ids.append(stable_id("of13", rel_id, ci))
            chunk_total += 1
            if len(docs) >= BATCH_SIZE:
                flush()

        if (i + 1) % 1000 == 0:
            print(f"    files: {i+1}/{len(all_files)}", flush=True)

    flush()
    print(f"  OF-13 total chunks: {chunk_total}")
    return chunk_total


# ---------------------------------------------------------------------------
# Source 2: Kestrel HPC docs
# ---------------------------------------------------------------------------

def collect_kestrel_docs() -> list[Path]:
    files = []
    for subdir in KESTREL_SUBDIRS:
        d = KESTREL_DOCS_ROOT / subdir
        if not d.exists():
            print(f"  Kestrel docs {d}: NOT FOUND (skipping)")
            continue
        for fp in d.rglob("*.md"):
            try:
                if fp.stat().st_size > 0:
                    files.append(fp)
            except OSError:
                continue
    return files


def index_kestrel(model, collection) -> int:
    print("\n--- Indexing Kestrel HPC docs ---")
    files = collect_kestrel_docs()
    print(f"  Files found: {len(files)}")

    docs, metas, ids = [], [], []
    chunk_total = 0

    for fp in files:
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore").strip()
        except (OSError, PermissionError):
            continue
        if not text:
            continue

        # Determine sub-context
        try:
            rel = fp.relative_to(KESTREL_DOCS_ROOT)
        except ValueError:
            rel = fp
        ctx = "kestrel_system" if "Systems" in str(rel) else "kestrel_docs"
        prefix = f"[kestrel] [{ctx}] FILE: {rel}"

        meta = {
            "version":  "kestrel",
            "context":  ctx,
            "filename": fp.name,
            "path":     str(fp),
        }

        for ci, chunk in enumerate(chunk_text(text, KESTREL_CHUNK, KESTREL_OVERLAP)):
            docs.append(f"{prefix}\n{chunk}")
            metas.append(meta)
            ids.append(stable_id("kestrel", str(rel), ci))
            chunk_total += 1

    if docs:
        embs = model.encode(docs, show_progress_bar=False).tolist()
        collection.add(documents=docs, embeddings=embs, metadatas=metas, ids=ids)

    print(f"  Kestrel total chunks: {chunk_total}")
    return chunk_total


# ---------------------------------------------------------------------------
# Source 3: Municchi arXiv papers
# ---------------------------------------------------------------------------

def load_paper_text(arxiv_id: str) -> str | None:
    """Load paper text from local cache, falling back to arXiv HTML fetch."""
    # 1. Try local cache first (works from compute nodes with no internet)
    cached = PAPERS_CACHE_DIR / f"{arxiv_id}.txt"
    if cached.exists() and cached.stat().st_size > 1000:
        return cached.read_text(encoding="utf-8", errors="ignore").strip()

    # 2. Fall back to live fetch (works from login nodes)
    try:
        import httpx
    except ImportError:
        return None

    for url in [
        f"https://arxiv.org/html/{arxiv_id}",
        f"https://arxiv.org/pdf/{arxiv_id}",
    ]:
        for attempt in range(2):
            try:
                r = httpx.get(url, timeout=60, follow_redirects=True)
                if r.status_code != 200:
                    break
                if "pdf" in url or r.headers.get("content-type", "").startswith("application/pdf"):
                    try:
                        import io
                        import pypdf
                        reader = pypdf.PdfReader(io.BytesIO(r.content))
                        text = "\n\n".join(p.extract_text() or "" for p in reader.pages).strip()
                    except Exception:
                        continue
                else:
                    text = re.sub(r"<[^>]+>", " ", r.text)
                    text = re.sub(r"&[a-zA-Z]+;", " ", text)
                    text = re.sub(r"\s{3,}", "\n\n", text).strip()
                if text:
                    return text
            except Exception as e:
                print(f"    attempt {attempt+1} failed ({url}): {e}")
                time.sleep(2)
    return None


def index_papers(model, collection) -> int:
    print("\n--- Indexing Municchi arXiv papers ---")
    docs, metas, ids = [], [], []
    chunk_total = 0

    for arxiv_id in ARXIV_PAPER_IDS:
        cached = PAPERS_CACHE_DIR / f"{arxiv_id}.txt"
        src = "cache" if (cached.exists() and cached.stat().st_size > 1000) else "fetch"
        print(f"  {arxiv_id} [{src}]...", end=" ", flush=True)
        text = load_paper_text(arxiv_id)
        if not text:
            print("FAILED")
            continue
        print(f"{len(text)} chars")

        prefix = f"[arxiv_paper] Municchi_F arXiv:{arxiv_id}"
        meta = {
            "version":   "?",
            "context":   "arxiv_paper",
            "arxiv_id":  arxiv_id,
            "author":    "Municchi_F",
            "source":    f"https://arxiv.org/html/{arxiv_id}",
        }

        for ci, chunk in enumerate(chunk_text(text, PAPER_CHUNK, PAPER_OVERLAP)):
            docs.append(f"{prefix}\n{chunk}")
            metas.append(meta)
            ids.append(stable_id("arxiv", arxiv_id, ci))
            chunk_total += 1

    if docs:
        # Papers may be large — batch them
        for start in range(0, len(docs), BATCH_SIZE):
            batch_docs  = docs[start:start+BATCH_SIZE]
            batch_metas = metas[start:start+BATCH_SIZE]
            batch_ids   = ids[start:start+BATCH_SIZE]
            embs = model.encode(batch_docs, show_progress_bar=False).tolist()
            collection.add(documents=batch_docs, embeddings=embs,
                           metadatas=batch_metas, ids=batch_ids)

    print(f"  Papers total chunks: {chunk_total}")
    return chunk_total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build():
    print("=" * 60)
    print("OpenFOAM Assistant RAG — full rebuild from scratch")
    print("=" * 60)

    print(f"\nLoading embedding model from {EMBEDDING_MODEL_PATH}")
    model = SentenceTransformer(EMBEDDING_MODEL_PATH)

    print(f"\nInitialising ChromaDB at {VECTORDB_PATH}")
    client = chromadb.PersistentClient(path=VECTORDB_PATH)

    # Drop and recreate collection
    try:
        client.delete_collection("openfoam")
        print("  Deleted existing 'openfoam' collection.")
    except Exception:
        pass
    collection = client.create_collection(
        "openfoam",
        metadata={"hnsw:space": "cosine"},
    )
    print("  Created fresh 'openfoam' collection.")

    t0 = time.time()
    total  = 0
    total += index_of13(model, collection)
    total += index_kestrel(model, collection)
    total += index_papers(model, collection)

    elapsed = time.time() - t0
    final = collection.count()
    print("\n" + "=" * 60)
    print(f"DONE in {elapsed:.0f}s")
    print(f"  Chunks added: {total}")
    print(f"  Collection count: {final}")
    print(f"  Vector DB: {VECTORDB_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    build()
