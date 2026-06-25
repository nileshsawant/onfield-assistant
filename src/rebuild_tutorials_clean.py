#!/usr/bin/env python3
"""
Clean rebuild: wipe the entire 'openfoam' ChromaDB collection and re-index
ONLY OpenFOAM-13 tutorial case input files:

  - tutorials/<solver>/<case>/system/*   (fvSchemes, fvSolution, controlDict, …)
  - tutorials/<solver>/<case>/0/*        (field files: U, p, k, epsilon, …)
  - tutorials/<solver>/<case>/constant/* (physicalProperties, momentumTransport, …)
  - tutorials/<solver>/<case>/Allrun     (run script)

Skips ALL C++ source/headers, papers, HPC docs, and build artifacts.

Whole-file storage (up to 8 KB) so the model sees complete, runnable files
rather than fragments.
"""

import os
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

OFA_ROOT = os.environ.get("OFA_ROOT", str(Path(__file__).resolve().parent.parent))
VECTORDB_PATH = os.environ.get("OFA_VECTORDB", os.path.join(OFA_ROOT, "vectordb"))
EMBEDDING_MODEL_PATH = os.path.join(OFA_ROOT, "embedding_model")

OF13_ROOT = os.environ.get(
    "OFA_OPENFOAM13_ROOT",
    "/nopt/nrel/apps/cpu_stack/software/openfoam/openfoam13_craympich_scotch/OpenFOAM-13",
)

CASE_SUBDIRS = {"system", "0", "constant"}
CASE_SCRIPTS = {"Allrun", "Allclean"}

SKIP_DIRS = {
    "postProcessing", "processor0", "processor1", "processor2", "processor3",
    "dynamicCode", ".git", "polyMesh", "lnInclude", "platforms", "build",
    "Make", "wmake", "__pycache__", "test",
}
SKIP_EXTENSIONS = {
    ".gz", ".obj", ".stl", ".vtk", ".vtu", ".vtp", ".foam",
    ".png", ".jpg", ".jpeg", ".pdf", ".svg",
    ".so", ".o", ".dep", ".a", ".pyc",
    ".C", ".H",          # C++ source — not useful for case generation
}
MAX_FILE_BYTES = 64 * 1024   # 64 KB hard cap
CHUNK_SIZE     = 8000        # whole-file if smaller; else split
CHUNK_OVERLAP  = 400
BATCH_SIZE     = 256


def should_skip_dir(name: str) -> bool:
    return name in SKIP_DIRS or name.startswith(".")


def is_case_input_file(filepath: Path) -> bool:
    """Return True only for files inside system/, 0/, constant/, or top-level run scripts."""
    if filepath.suffix in SKIP_EXTENSIONS:
        return False
    try:
        sz = filepath.stat().st_size
        if sz == 0 or sz > MAX_FILE_BYTES:
            return False
    except OSError:
        return False

    parts = filepath.parts
    try:
        tut_idx = parts.index("tutorials")
    except ValueError:
        return False

    # Depth relative to tutorials/:
    #   [solver, case, subdir, …filename]  → len = tut_idx + 4+
    #   [solver, case, script]             → len = tut_idx + 3
    rel = parts[tut_idx + 1:]   # e.g. ("incompressibleFluid", "pitzDaily", "system", "fvSchemes")

    if len(rel) >= 3 and rel[2] in CASE_SUBDIRS:
        return True
    if len(rel) == 3 and rel[2] in CASE_SCRIPTS:
        return True
    return False


def chunk_text(text: str) -> list:
    if len(text) <= CHUNK_SIZE:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = start + CHUNK_SIZE
        if end < len(text):
            nl = text.rfind("\n", start + CHUNK_SIZE // 2, end + 200)
            if nl > start:
                end = nl + 1
        chunks.append(text[start:end])
        start = end - CHUNK_OVERLAP
    return chunks


def rebuild():
    print("=" * 60)
    print("OpenFOAM RAG — clean tutorials-only rebuild")
    print("=" * 60)

    print(f"\nLoading embedding model: {EMBEDDING_MODEL_PATH}")
    model = SentenceTransformer(EMBEDDING_MODEL_PATH)

    print(f"Opening ChromaDB:        {VECTORDB_PATH}")
    client = chromadb.PersistentClient(path=VECTORDB_PATH)

    # Wipe existing collection completely
    try:
        client.delete_collection("openfoam")
        print("Deleted existing 'openfoam' collection.")
    except Exception as e:
        print(f"No existing collection to delete ({e})")

    collection = client.create_collection("openfoam", metadata={"hnsw:space": "cosine"})
    print("Created fresh 'openfoam' collection.\n")

    # Walk tutorials tree
    tut_dir = Path(OF13_ROOT) / "tutorials"
    if not tut_dir.exists():
        print(f"ERROR: tutorials directory not found: {tut_dir}")
        sys.exit(1)

    print(f"Scanning: {tut_dir}")
    all_files = []
    for dirpath, dirnames, filenames in os.walk(tut_dir):
        dirnames[:] = [d for d in dirnames if not should_skip_dir(d)]
        dp = Path(dirpath)
        for fname in filenames:
            fp = dp / fname
            if is_case_input_file(fp):
                all_files.append(fp)

    print(f"Tutorial input files found: {len(all_files)}\n")

    of13_root = Path(OF13_ROOT)
    documents, metadatas, ids = [], [], []
    chunk_total = 0
    batch_num = 0
    skipped = 0

    for i, filepath in enumerate(all_files):
        try:
            text = filepath.read_text(encoding="utf-8", errors="ignore")
        except (OSError, PermissionError):
            skipped += 1
            continue
        if not text.strip():
            skipped += 1
            continue

        parts = filepath.parts
        tut_idx = parts.index("tutorials")
        rel_from_tut = "/".join(parts[tut_idx:])   # e.g. tutorials/incompressibleFluid/pitzDaily/system/fvSchemes

        meta = {
            "version":  "openfoam13",
            "context":  "tutorials",
            "filename": filepath.name,
            "path":     str(filepath),
        }
        rel = parts[tut_idx + 1:]
        if len(rel) >= 1:
            meta["solver"] = rel[0]   # e.g. "incompressibleFluid"
        if len(rel) >= 2:
            meta["case"] = rel[1]     # e.g. "pitzDaily"

        prefix = f"[openfoam13] [tutorials] FILE: {rel_from_tut}"
        chunks = chunk_text(text)

        try:
            base_id = str(filepath.relative_to(of13_root))
        except ValueError:
            base_id = str(filepath)
        base_id = base_id.replace("/", "_").replace(" ", "_")

        for ci, chunk in enumerate(chunks):
            documents.append(f"{prefix}\n{chunk}")
            metadatas.append(meta)
            ids.append(f"of13_{base_id}_{ci}")
            chunk_total += 1

            if len(documents) >= BATCH_SIZE:
                batch_num += 1
                embeddings = model.encode(documents, show_progress_bar=False).tolist()
                collection.add(documents=documents, embeddings=embeddings,
                               metadatas=metadatas, ids=ids)
                print(f"  Batch {batch_num:3d} — {chunk_total} chunks indexed…", flush=True)
                documents, metadatas, ids = [], [], []

        if (i + 1) % 500 == 0:
            print(f"  Files processed: {i + 1}/{len(all_files)}", flush=True)

    # Final partial batch
    if documents:
        batch_num += 1
        embeddings = model.encode(documents, show_progress_bar=False).tolist()
        collection.add(documents=documents, embeddings=embeddings,
                       metadatas=metadatas, ids=ids)

    total = collection.count()
    print(f"\n{'=' * 60}")
    print(f"Done.")
    print(f"  Files scanned:   {len(all_files)}  (skipped {skipped})")
    print(f"  Chunks indexed:  {chunk_total}")
    print(f"  Total in DB:     {total}")
    print(f"  Content:         tutorials only (system/ + 0/ + constant/ + Allrun)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    rebuild()
