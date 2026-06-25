#!/usr/bin/env python3
"""
Targeted re-index: remove all OF-13 and v2512 tutorial/source chunks,
then re-index OF-13 only with file-level chunking for case dict files.

Case dict files (system/, 0/, constant/) are stored as whole files (no splitting
up to 8 KB) so the model retrieves complete, runnable file content rather than
fragmented chunks.
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

# Chunk sizes
CASE_DICT_CHUNK_SIZE = 8000   # Case dict files: whole-file up to 8 KB, then split
CASE_DICT_OVERLAP    = 400
SOURCE_CHUNK_SIZE    = 2000   # .H source headers: normal chunking
SOURCE_OVERLAP       = 200

SKIP_DIRS = {
    "postProcessing", "processor0", "processor1", "processor2", "processor3",
    "dynamicCode", ".git", "polyMesh", "lnInclude", "platforms", "build",
    "Make", "wmake", "__pycache__", "test",
}
SKIP_EXTENSIONS = {
    ".gz", ".obj", ".stl", ".vtk", ".vtu", ".png", ".jpg", ".pdf",
    ".so", ".o", ".dep", ".a", ".pyc",
}
MAX_FILE_SIZE = 64 * 1024  # 64 KB


def is_case_dict_file(filepath: Path) -> bool:
    """True if the file lives in system/, 0/, or constant/ inside a tutorial case."""
    parts = filepath.parts
    try:
        tut_idx = parts.index("tutorials")
    except ValueError:
        return False
    # Path relative to tutorials: solver/case/system|0|constant/...
    if tut_idx + 3 < len(parts):
        subdir = parts[tut_idx + 3]
        return subdir in ("system", "0", "constant")
    return False


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into chunks with overlap."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            nl = text.rfind("\n", start + chunk_size // 2, end + 200)
            if nl > start:
                end = nl + 1
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def should_skip_dir(dirname: str) -> bool:
    return dirname in SKIP_DIRS or dirname.startswith(".")


def should_index_file(filepath: Path, context: str) -> bool:
    if filepath.suffix in SKIP_EXTENSIONS:
        return False
    try:
        stat = filepath.stat()
    except OSError:
        return False
    if stat.st_size > MAX_FILE_SIZE or stat.st_size == 0:
        return False
    if context == "source":
        return filepath.suffix == ".H"
    elif context == "tutorials":
        return True
    elif context == "etc":
        return filepath.suffix in {"", ".cfg", ".sh", ".csh"} or filepath.name.endswith("Dict")
    return False


def walk_directory(root: Path, context: str) -> list[Path]:
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not should_skip_dir(d)]
        dp = Path(dirpath)
        for fname in filenames:
            fp = dp / fname
            try:
                if should_index_file(fp, context):
                    files.append(fp)
            except (OSError, PermissionError):
                continue
    return files


def make_prefix(filepath: Path, of13_root: Path, context: str) -> str:
    """Build a rich prefix that includes the full relative path for case-specific retrieval."""
    try:
        rel = filepath.relative_to(of13_root)
    except ValueError:
        rel = filepath
    parts = rel.parts
    prefix = f"[openfoam13] [{context}] {filepath.name}"
    if context == "tutorials" and "tutorials" in parts:
        tut_idx = parts.index("tutorials")
        # Include solver/case/subdir so pitzDaily files are clearly labeled
        rel_from_tut = "/".join(parts[tut_idx:])
        prefix = f"[openfoam13] [tutorials] FILE: {rel_from_tut}"
    elif context == "source":
        prefix = f"[openfoam13] [source] {rel}"
    elif context == "etc":
        prefix = f"[openfoam13] [etc] {rel}"
    return prefix


def collect_of13_files():
    """Collect all OF-13 files to index."""
    root = Path(OF13_ROOT)
    if not root.exists():
        print(f"ERROR: OF-13 root not found: {root}")
        sys.exit(1)

    all_files = []  # (filepath, context)

    tut_dir = root / "tutorials"
    if tut_dir.exists():
        files = walk_directory(tut_dir, "tutorials")
        all_files.extend((f, "tutorials") for f in files)
        print(f"  OF-13 tutorials: {len(files)} files")

    src_dir = root / "src"
    if src_dir.exists():
        files = walk_directory(src_dir, "source")
        all_files.extend((f, "source") for f in files)
        print(f"  OF-13 src: {len(files)} .H files")

    apps_dir = root / "applications"
    if apps_dir.exists():
        files = walk_directory(apps_dir, "source")
        all_files.extend((f, "source") for f in files)
        print(f"  OF-13 applications: {len(files)} .H files")

    etc_dir = root / "etc" / "caseDicts"
    if etc_dir.exists():
        files = walk_directory(etc_dir, "etc")
        all_files.extend((f, "etc") for f in files)
        print(f"  OF-13 etc/caseDicts: {len(files)} files")

    return all_files


def rebuild():
    print("Loading embedding model from", EMBEDDING_MODEL_PATH)
    model = SentenceTransformer(EMBEDDING_MODEL_PATH)

    print("Opening ChromaDB at", VECTORDB_PATH)
    client = chromadb.PersistentClient(path=VECTORDB_PATH)

    try:
        collection = client.get_collection("openfoam")
    except Exception:
        collection = client.create_collection("openfoam", metadata={"hnsw:space": "cosine"})

    before = collection.count()
    print(f"Chunks before removal: {before}")

    # Remove existing OF-13 and v2512 chunks (keep Kestrel docs + papers)
    print("Removing openfoam13 chunks...")
    try:
        collection.delete(where={"version": "openfoam13"})
    except Exception as e:
        print(f"  (openfoam13 delete: {e})")

    print("Removing openfoam_v2512 chunks...")
    try:
        collection.delete(where={"version": "openfoam_v2512"})
    except Exception as e:
        print(f"  (openfoam_v2512 delete: {e})")

    after_del = collection.count()
    print(f"Chunks after removal: {after_del} (removed {before - after_del})")

    print("\nCollecting OF-13 files...")
    all_files = collect_of13_files()
    print(f"Total files to index: {len(all_files)}")

    of13_root = Path(OF13_ROOT)
    documents, metadatas, ids = [], [], []
    BATCH_SIZE = 256
    batch_num = 0
    chunk_total = 0

    for i, (filepath, context) in enumerate(all_files):
        try:
            text = filepath.read_text(encoding="utf-8", errors="ignore")
        except (OSError, PermissionError):
            continue
        if not text.strip():
            continue

        # Choose chunk size based on file type
        if context == "tutorials" and is_case_dict_file(filepath):
            cs, co = CASE_DICT_CHUNK_SIZE, CASE_DICT_OVERLAP
        else:
            cs, co = SOURCE_CHUNK_SIZE, SOURCE_OVERLAP

        chunks = chunk_text(text, cs, co)
        prefix = make_prefix(filepath, of13_root, context)

        # Metadata
        meta = {
            "version": "openfoam13",
            "context": context,
            "filename": filepath.name,
            "path": str(filepath),
        }
        # Add solver/case for tutorials
        parts = filepath.parts
        if context == "tutorials" and "tutorials" in parts:
            tut_idx = parts.index("tutorials")
            if tut_idx + 1 < len(parts):
                meta["solver"] = parts[tut_idx + 1]
            if tut_idx + 2 < len(parts):
                meta["case"] = parts[tut_idx + 2]

        for ci, chunk in enumerate(chunks):
            doc_text = f"{prefix}\n{chunk}"
            documents.append(doc_text)
            metadatas.append(meta)
            # Use relative path for stable, unique IDs
            try:
                rel_id = str(filepath.relative_to(of13_root))
            except ValueError:
                rel_id = str(filepath)
            ids.append(f"of13_{rel_id}_{ci}".replace("/", "_").replace(" ", "_"))
            chunk_total += 1

            if len(documents) >= BATCH_SIZE:
                batch_num += 1
                embeddings = model.encode(documents, show_progress_bar=False).tolist()
                collection.add(
                    documents=documents, embeddings=embeddings,
                    metadatas=metadatas, ids=ids,
                )
                print(f"  Batch {batch_num}: {chunk_total} chunks so far...", flush=True)
                documents, metadatas, ids = [], [], []

        if (i + 1) % 500 == 0:
            print(f"  Files processed: {i+1}/{len(all_files)}", flush=True)

    # Final batch
    if documents:
        batch_num += 1
        embeddings = model.encode(documents, show_progress_bar=False).tolist()
        collection.add(
            documents=documents, embeddings=embeddings,
            metadatas=metadatas, ids=ids,
        )

    total = collection.count()
    print(f"\nDone!")
    print(f"  New OF-13 chunks added: {chunk_total}")
    print(f"  Total chunks in DB: {total}")
    print(f"  (Kestrel docs + papers preserved: {after_del})")


if __name__ == "__main__":
    rebuild()
