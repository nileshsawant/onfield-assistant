#!/usr/bin/env python3
"""Build the RAG vector index over OpenFOAM tutorials, source headers, and docs."""

import os
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

OFA_ROOT = os.environ.get("OFA_ROOT", str(Path(__file__).resolve().parent.parent))
VECTORDB_PATH = os.environ.get("OFA_VECTORDB", os.path.join(OFA_ROOT, "vectordb"))
EMBEDDING_MODEL_PATH = os.path.join(OFA_ROOT, "embedding_model")

# The two latest OpenFOAM versions to index. Override per deployment using the
# OFA_OPENFOAM13_ROOT and OFA_OPENFOAM_V2512_ROOT environment variables.
OPENFOAM_ROOTS = {
    "openfoam13": os.environ.get(
        "OFA_OPENFOAM13_ROOT",
        "/nopt/nrel/apps/cpu_stack/software/openfoam/openfoam13_craympich_scotch/OpenFOAM-13",
    ),
    "openfoam_v2512": os.environ.get(
        "OFA_OPENFOAM_V2512_ROOT",
        "/nopt/nrel/apps/cpu_stack/software/openfoam/openfoam_v2512_openmpi/OpenFOAM-v2512",
    ),
}

# Directories to skip everywhere
SKIP_DIRS = {
    "postProcessing", "processor0", "processor1", "processor2", "processor3",
    "dynamicCode", ".git", "polyMesh", "lnInclude", "platforms", "build",
    "Make", "wmake", "__pycache__", "test",
}

# Extensions to skip
SKIP_EXTENSIONS = {
    ".gz", ".obj", ".stl", ".vtk", ".vtu", ".png", ".jpg", ".pdf",
    ".so", ".o", ".dep", ".a", ".pyc",
}

# Max file size to index (skip huge generated files)
MAX_FILE_SIZE = 64 * 1024  # 64 KB

# Chunk size for splitting large files
CHUNK_SIZE = 2000  # characters (~500 tokens)
CHUNK_OVERLAP = 200


def should_skip_dir(dirname: str) -> bool:
    return dirname in SKIP_DIRS or dirname.startswith(".")


def should_index_file(filepath: Path, context: str) -> bool:
    """Decide whether to index this file."""
    if filepath.suffix in SKIP_EXTENSIONS:
        return False
    if filepath.stat().st_size > MAX_FILE_SIZE:
        return False
    if filepath.stat().st_size == 0:
        return False

    if context == "source":
        # Only index header files from source tree
        return filepath.suffix == ".H"
    elif context == "tutorials":
        # Index all text files in tutorials
        return True
    elif context == "etc":
        # Index config templates
        return filepath.suffix in {"", ".cfg", ".sh", ".csh"} or filepath.name.endswith("Dict")
    elif context == "doc":
        return filepath.suffix in {".md", ".org", ".txt", ".html"}
    return False


def chunk_text(text: str, filepath: Path) -> list[str]:
    """Split text into chunks with overlap."""
    if len(text) <= CHUNK_SIZE:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        # Try to break at a newline
        if end < len(text):
            nl = text.rfind("\n", start + CHUNK_SIZE // 2, end + 200)
            if nl > start:
                end = nl + 1
        chunks.append(text[start:end])
        start = end - CHUNK_OVERLAP
    return chunks


def get_metadata(filepath: Path, version: str, context: str) -> dict:
    """Build metadata for a file chunk."""
    meta = {
        "version": version,
        "context": context,
        "filename": filepath.name,
        "path": str(filepath),
    }

    parts = filepath.parts
    if context == "tutorials":
        # Extract solver and case name from path
        try:
            tut_idx = parts.index("tutorials")
            if tut_idx + 1 < len(parts):
                meta["solver"] = parts[tut_idx + 1]
            if tut_idx + 2 < len(parts):
                meta["case"] = parts[tut_idx + 2]
        except ValueError:
            pass
    elif context == "source":
        # Extract module from src/MODULE or applications/TYPE/NAME
        try:
            if "src" in parts:
                idx = parts.index("src")
                if idx + 1 < len(parts):
                    meta["module"] = parts[idx + 1]
            elif "applications" in parts:
                idx = parts.index("applications")
                if idx + 2 < len(parts):
                    meta["app_type"] = parts[idx + 1]  # solvers/utilities
                    meta["app_name"] = parts[idx + 2]
        except (ValueError, IndexError):
            pass

    return meta


def walk_directory(root: Path, context: str) -> list[tuple[Path, str]]:
    """Walk a directory tree, respecting skip rules."""
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Filter out skipped directories in-place
        dirnames[:] = [d for d in dirnames if not should_skip_dir(d)]
        dp = Path(dirpath)
        for fname in filenames:
            fp = dp / fname
            try:
                if should_index_file(fp, context):
                    files.append((fp, context))
            except (OSError, PermissionError):
                continue
    return files


def collect_files() -> list[tuple[Path, str, str]]:
    """Collect all files to index from both OpenFOAM versions."""
    all_files = []  # (filepath, context, version)

    for version, root_str in OPENFOAM_ROOTS.items():
        root = Path(root_str)
        if not root.exists():
            print(f"WARNING: {root} does not exist, skipping")
            continue

        # Tutorials
        tutorials_dir = root / "tutorials"
        if tutorials_dir.exists():
            files = walk_directory(tutorials_dir, "tutorials")
            all_files.extend((f, ctx, version) for f, ctx in files)
            print(f"  {version}/tutorials: {len(files)} files")

        # Source headers
        src_dir = root / "src"
        if src_dir.exists():
            files = walk_directory(src_dir, "source")
            all_files.extend((f, ctx, version) for f, ctx in files)
            print(f"  {version}/src: {len(files)} .H files")

        # Applications (solvers + utilities)
        apps_dir = root / "applications"
        if apps_dir.exists():
            files = walk_directory(apps_dir, "source")
            all_files.extend((f, ctx, version) for f, ctx in files)
            print(f"  {version}/applications: {len(files)} .H files")

        # etc/caseDicts (template configurations)
        etc_dir = root / "etc" / "caseDicts"
        if etc_dir.exists():
            files = walk_directory(etc_dir, "etc")
            all_files.extend((f, ctx, version) for f, ctx in files)
            print(f"  {version}/etc/caseDicts: {len(files)} files")

        # Documentation
        doc_dir = root / "doc"
        if doc_dir.exists():
            files = walk_directory(doc_dir, "doc")
            all_files.extend((f, ctx, version) for f, ctx in files)
            print(f"  {version}/doc: {len(files)} files")

    return all_files


def build_index():
    """Main index building function."""
    print("Loading embedding model from", EMBEDDING_MODEL_PATH)
    model = SentenceTransformer(EMBEDDING_MODEL_PATH)

    print("Initializing ChromaDB at", VECTORDB_PATH)
    client = chromadb.PersistentClient(path=VECTORDB_PATH)
    # Delete existing collection if present
    try:
        client.delete_collection("openfoam")
    except Exception:
        pass
    collection = client.create_collection(
        "openfoam",
        metadata={"hnsw:space": "cosine"},
    )

    print("\nCollecting files...")
    all_files = collect_files()
    print(f"\nTotal files to index: {len(all_files)}")

    if not all_files:
        print("ERROR: No files found!")
        sys.exit(1)

    # Process files in batches
    documents = []
    metadatas = []
    ids = []
    batch_num = 0
    BATCH_SIZE = 256

    for i, (filepath, context, version) in enumerate(all_files):
        try:
            text = filepath.read_text(encoding="utf-8", errors="ignore")
        except (OSError, PermissionError):
            continue

        if not text.strip():
            continue

        chunks = chunk_text(text, filepath)
        meta = get_metadata(filepath, version, context)

        for ci, chunk in enumerate(chunks):
            # Prefix with file info for better retrieval
            prefix = f"[{version}] [{context}] {filepath.name}"
            if "solver" in meta:
                prefix += f" (solver: {meta['solver']}"
                if "case" in meta:
                    prefix += f", case: {meta['case']}"
                prefix += ")"
            doc_text = f"{prefix}\n{chunk}"

            documents.append(doc_text)
            metadatas.append(meta)
            ids.append(f"{version}_{context}_{filepath.name}_{i}_{ci}")

            if len(documents) >= BATCH_SIZE:
                batch_num += 1
                embeddings = model.encode(documents, show_progress_bar=False).tolist()
                collection.add(
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids,
                )
                print(f"  Batch {batch_num}: indexed {batch_num * BATCH_SIZE} chunks...", flush=True)
                documents = []
                metadatas = []
                ids = []

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{len(all_files)} files...", flush=True)

    # Final batch
    if documents:
        batch_num += 1
        embeddings = model.encode(documents, show_progress_bar=False).tolist()
        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

    total = collection.count()
    print(f"\nDone! Indexed {total} chunks total.")
    print(f"Vector DB stored at: {VECTORDB_PATH}")


if __name__ == "__main__":
    build_index()
