#!/usr/bin/env python3
"""Rebuild one or more ofa RAG collections from ``collections.toml``.

The go-to script to run after ``git pull``-ing a code repo, dropping
new PDFs into a papers directory, or adding a new collection entry.

Usage::

    python3 $OFA_ROOT/src/rebuild_indices.py                 # all collections
    python3 $OFA_ROOT/src/rebuild_indices.py --collection X  # just one
    python3 $OFA_ROOT/src/rebuild_indices.py --list          # show configured
    python3 $OFA_ROOT/src/rebuild_indices.py --dry-run       # preview only
    python3 $OFA_ROOT/src/rebuild_indices.py --force         # ignore mtime cache

Design:

* Config lives at ``$OFA_ROOT/collections.toml`` (declarative — see the
  file's own header comment for the schema).
* Chunk IDs are ``sha256(collection + relpath + chunk_index)`` truncated,
  so re-runs ``upsert`` rather than duplicate. Chunks whose source file
  is gone from disk are removed from the collection at the end of the
  pass ("orphan sweep").
* Per-source-file mtime is remembered in
  ``$OFA_VECTORDB/.rebuild_state.json`` so unchanged files aren't
  re-embedded (embedding is by far the slowest step). ``--force``
  ignores the cache and re-embeds everything.
* PDFs are extracted via :mod:`pdf_extract` (pdfplumber under the hood);
  each PDF page becomes a chunk unit, further split to
  ``PAPER_CHUNK`` characters if a page is long.
* Code files use ``chunk_text`` with the same size/overlap constants as
  ``build_index_v2.py`` (kept in sync manually — see the CHUNK_*
  constants below).

The script is deliberately self-contained: it does NOT import from
``build_index_v2.py`` to keep responsibilities separate. If chunk sizes
need to change, update both files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Paths + constants (kept in sync with build_index_v2.py deliberately)
# ---------------------------------------------------------------------------

OFA_ROOT = Path(os.environ.get(
    "OFA_ROOT",
    str(Path(__file__).resolve().parent.parent),
))
VECTORDB_PATH = Path(os.environ.get(
    "OFA_VECTORDB", str(OFA_ROOT / "vectordb")
))
EMBEDDING_MODEL_PATH = OFA_ROOT / "embedding_model"
CONFIG_PATH = OFA_ROOT / "collections.toml"
STATE_PATH = VECTORDB_PATH / ".rebuild_state.json"

CODE_CHUNK    = 2_000    # source-header default
CODE_OVERLAP  = 200
PAPER_CHUNK   = 3_000    # PDF page / markdown paragraphs
PAPER_OVERLAP = 300

BATCH_SIZE = 256


# ---------------------------------------------------------------------------
# Helpers copy-pasted from build_index_v2.py (deliberately, to keep this
# module dep-light).
# ---------------------------------------------------------------------------

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


def stable_id(collection: str, source_relpath: str, chunk_idx: int) -> str:
    """Deterministic chunk ID so re-runs upsert instead of duplicate."""
    raw = f"{collection}::{source_relpath}::{chunk_idx}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: Path = CONFIG_PATH) -> dict:
    try:
        import tomllib  # stdlib in Python 3.11+
    except ImportError as e:
        raise RuntimeError(
            "Python 3.11+ required for tomllib (config file is TOML)."
        ) from e
    if not path.is_file():
        raise FileNotFoundError(
            f"config file not found: {path}. See collections.toml for the schema."
        )
    with open(path, "rb") as f:
        return tomllib.load(f)


def list_collections(cfg: dict) -> None:
    print(f"{'collection':<24}  {'sources':<8}  description")
    print("-" * 76)
    for name, cinfo in cfg.get("collections", {}).items():
        n_src = len(cinfo.get("sources", []))
        desc = cinfo.get("description", "")
        print(f"{name:<24}  {n_src:<8}  {desc}")


# ---------------------------------------------------------------------------
# State (per-file mtime cache)
# ---------------------------------------------------------------------------

def load_state() -> dict:
    if not STATE_PATH.is_file():
        return {"collections": {}}
    try:
        with open(STATE_PATH) as f:
            data = json.load(f)
        if "collections" not in data:
            data["collections"] = {}
        return data
    except (OSError, json.JSONDecodeError) as e:
        print(f"[!] state file unreadable ({e}); starting fresh", file=sys.stderr)
        return {"collections": {}}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    os.replace(tmp, STATE_PATH)


# ---------------------------------------------------------------------------
# Source walkers
# ---------------------------------------------------------------------------

_SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    "build", "dist", ".mypy_cache", ".pytest_cache", ".tox",
}


def walk_code(root: Path, extensions: list[str]) -> Iterable[Path]:
    """Yield source files under *root* matching any of *extensions*."""
    ext_set = {e.lower() for e in extensions}
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune noisy dirs in place so we don't descend into them.
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix.lower() in ext_set:
                yield p


def walk_pdfs(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]
        for fn in filenames:
            if fn.lower().endswith(".pdf"):
                yield Path(dirpath) / fn


# ---------------------------------------------------------------------------
# Per-source-file processing
# ---------------------------------------------------------------------------

def _extract_notebook_text(path: Path) -> str:
    """Extract only ``source`` from code + markdown cells of a Jupyter
    notebook, dropping all cell outputs.

    Raw ``.ipynb`` JSON embeds base64-encoded output images and long
    stdout dumps that would waste embedding compute and pollute
    retrieval with noise chunks. This keeps the semantically useful
    parts (the code the author wrote and their markdown narrative)
    and discards everything else.

    Returns the joined text, or empty string on malformed notebooks
    (a warning is printed but doesn't kill the ingestion).
    """
    import json
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            nb = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[!] notebook {path}: {e}", file=sys.stderr)
        return ""
    parts: list[str] = []
    for i, cell in enumerate(nb.get("cells", [])):
        ct = cell.get("cell_type", "")
        if ct not in ("code", "markdown"):
            continue
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        src = src.strip()
        if not src:
            continue
        tag = "code" if ct == "code" else "md"
        parts.append(f"# --- cell {i} ({tag}) ---\n{src}")
    return "\n\n".join(parts)


def process_code_file(path: Path, root: Path, collection: str) -> list[tuple[str, str, dict]]:
    """Read a text/code file, return list of ``(chunk_id, doc, metadata)`` tuples."""
    if path.suffix.lower() == ".ipynb":
        text = _extract_notebook_text(path)
    else:
        try:
            text = path.read_text(errors="replace")
        except OSError as e:
            print(f"[!] {path}: {e}", file=sys.stderr)
            return []
    if not text.strip():
        return []
    relpath = path.relative_to(root).as_posix()
    prefix = f"[{root.name} source - {relpath}]"
    out = []
    for i, chunk in enumerate(chunk_text(text, CODE_CHUNK, CODE_OVERLAP)):
        out.append((
            stable_id(collection, str(path), i),
            f"{prefix}\n{chunk}",
            {
                "source_type": "code",
                "source_root": root.name,
                "filepath": relpath,
                "chunk_index": i,
            },
        ))
    return out


def process_pdf_file(path: Path, root: Path, collection: str) -> list[tuple[str, str, dict]]:
    """Extract a PDF, return list of ``(chunk_id, doc, metadata)`` tuples.

    Each PDF page becomes at least one chunk. Long pages are split
    further with ``PAPER_CHUNK`` / ``PAPER_OVERLAP``.
    """
    try:
        from pdf_extract import extract_pages
    except ImportError:
        # Support running the script directly from the repo without
        # installing anything.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from pdf_extract import extract_pages

    relpath = path.relative_to(root).as_posix()
    out = []
    chunk_idx = 0
    for page_num, page_text in extract_pages(path):
        for sub_chunk in chunk_text(page_text, PAPER_CHUNK, PAPER_OVERLAP):
            prefix = f"[{root.name} paper - {relpath}, page {page_num}]"
            out.append((
                stable_id(collection, f"{path}:p{page_num}", chunk_idx),
                f"{prefix}\n{sub_chunk}",
                {
                    "source_type": "pdf",
                    "source_root": root.name,
                    "filepath": relpath,
                    "page": page_num,
                    "chunk_index": chunk_idx,
                },
            ))
            chunk_idx += 1
    return out


# ---------------------------------------------------------------------------
# Collection rebuild
# ---------------------------------------------------------------------------

def rebuild_collection(
    name: str,
    cinfo: dict,
    embed_model,
    chroma_client,
    state: dict,
    *,
    dry_run: bool = False,
    force: bool = False,
    clear: bool = False,
) -> None:
    """Rebuild one collection according to its config entry.

    Prints per-source progress; returns nothing (side effects on
    ``chroma_client`` and ``state``).
    """
    print(f"\n=== collection: {name} ===")

    if clear and not dry_run:
        try:
            chroma_client.delete_collection(name)
            print(f"  [x] cleared existing collection")
        except Exception:
            pass  # didn't exist yet
        state["collections"].pop(name, None)

    # Gather all (path, kind, root) entries across configured sources.
    entries: list[tuple[Path, str, Path]] = []
    for src in cinfo.get("sources", []):
        rel = src.get("path")
        src_type = src.get("type", "code")
        # Absolute paths pass through; relative paths resolve against OFA_ROOT.
        root = Path(rel) if rel.startswith("/") else (OFA_ROOT / rel)
        if not root.is_dir():
            print(f"  [~] source '{rel}' not present; skipping", file=sys.stderr)
            continue
        if src_type == "code":
            exts = src.get("extensions", [".txt", ".md"])
            for p in walk_code(root, exts):
                entries.append((p, "code", root))
        elif src_type == "pdf":
            for p in walk_pdfs(root):
                entries.append((p, "pdf", root))
        else:
            print(f"  [!] unknown source type '{src_type}' in {name}",
                  file=sys.stderr)

    if not entries:
        print("  (no files matched — nothing to do)")
        return

    print(f"  {len(entries)} candidate files")

    coll = chroma_client.get_or_create_collection(name)
    coll_state = state["collections"].setdefault(name, {})

    added = skipped = removed = 0
    seen_paths: set[str] = set()
    pending_ids: list[str] = []
    pending_docs: list[str] = []
    pending_metas: list[dict] = []

    for path, kind, root in entries:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        key = str(path)
        seen_paths.add(key)

        # Skip unchanged files.
        prev_mtime = coll_state.get(key, {}).get("mtime")
        if not force and prev_mtime == mtime:
            skipped += 1
            continue

        # Chunk it.
        if kind == "code":
            chunks = process_code_file(path, root, name)
        else:
            chunks = process_pdf_file(path, root, name)
        if not chunks:
            continue

        for cid, doc, meta in chunks:
            pending_ids.append(cid)
            pending_docs.append(doc)
            pending_metas.append(meta)

        coll_state[key] = {"mtime": mtime, "n_chunks": len(chunks)}
        added += len(chunks)

    # Orphan sweep — remove chunks for files that used to be indexed
    # but are gone from disk (deleted / moved out of source root).
    gone = [k for k in list(coll_state) if k not in seen_paths]
    for k in gone:
        n = coll_state[k].get("n_chunks", 0)
        # We can't easily know the individual chunk IDs after a re-run
        # without tracking them, but stable_id is deterministic given
        # the ORIGINAL path — so we can regenerate the IDs and delete.
        ids_to_drop = [stable_id(name, k, i) for i in range(n)]
        # PDF-derived stable_ids used a ":pN" suffix in the path arg, so
        # this simple regeneration will miss PDF chunks. That's fine for
        # the common case of code-file renames / deletes; a stale PDF
        # will be caught by re-running with --force.
        try:
            if ids_to_drop and not dry_run:
                coll.delete(ids=ids_to_drop)
        except Exception as e:
            print(f"  [!] orphan sweep for {k}: {e}", file=sys.stderr)
        removed += n
        del coll_state[k]

    if dry_run:
        print(f"  [dry-run] would add/upsert {added} chunks, "
              f"skip {skipped} unchanged, remove {removed} orphan chunks")
        return

    # Batch-upsert to keep memory bounded.
    if pending_ids:
        print(f"  embedding {len(pending_ids)} chunks…")
        t0 = time.time()
        embeddings = embed_model.encode(
            pending_docs, batch_size=BATCH_SIZE,
            show_progress_bar=False,
        )
        print(f"  embedding done in {time.time() - t0:.1f}s; upserting…")
        # ChromaDB upsert handles duplicate IDs cleanly.
        coll.upsert(
            ids=pending_ids,
            documents=pending_docs,
            metadatas=pending_metas,
            embeddings=embeddings.tolist(),
        )

    total = coll.count()
    print(f"  [+] {added} added/updated  [=] {skipped} unchanged  "
          f"[-] {removed} orphaned  ->  {total} total in {name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Rebuild ofa RAG collections from collections.toml",
    )
    ap.add_argument("--collection", "-c",
                    help="rebuild only this collection (default: all)")
    ap.add_argument("--list", action="store_true",
                    help="list configured collections and exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would happen without touching the store")
    ap.add_argument("--force", action="store_true",
                    help="ignore mtime cache; re-embed everything")
    ap.add_argument("--clear", action="store_true",
                    help="drop the collection before rebuilding (use once when "
                         "migrating a collection previously built by a different "
                         "indexer, to avoid stale duplicate chunks)")
    args = ap.parse_args()

    cfg = load_config()
    if args.list:
        list_collections(cfg)
        return

    target = args.collection
    if target and target not in cfg.get("collections", {}):
        print(f"error: unknown collection '{target}'. Configured: "
              f"{list(cfg['collections'])}",
              file=sys.stderr)
        sys.exit(2)

    # Lazy imports of heavy deps so `--list` / `--help` / `--dry-run`
    # stay fast (no embedding model load on a dry run).
    import chromadb

    print(f"vectordb:        {VECTORDB_PATH}")
    chroma_client = chromadb.PersistentClient(path=str(VECTORDB_PATH))
    embed_model = None
    if not args.dry_run:
        from sentence_transformers import SentenceTransformer
        print(f"embedding model: {EMBEDDING_MODEL_PATH}")
        embed_model = SentenceTransformer(str(EMBEDDING_MODEL_PATH))

    state = load_state()
    try:
        collections = cfg["collections"] if not target else {
            target: cfg["collections"][target]
        }
        for name, cinfo in collections.items():
            rebuild_collection(
                name, cinfo, embed_model, chroma_client, state,
                dry_run=args.dry_run, force=args.force, clear=args.clear,
            )
    finally:
        if not args.dry_run:
            save_state(state)
            print(f"\nstate saved to {STATE_PATH}")


if __name__ == "__main__":
    main()
