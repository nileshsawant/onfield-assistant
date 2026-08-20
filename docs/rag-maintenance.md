# Updating the RAG corpora

Operator playbook for keeping `ofa`'s retrieval-augmented context up to
date. The [`Updating the RAG indices`](getting-started.md) section of
the getting-started guide covers the *command reference* — this doc
covers the *concrete recipes* for the three kinds of updates the RAG
pipeline routinely sees, plus the config-level features that make them
safe to re-run.

## Mental model

Every collection in [`collections.toml`](https://github.com/nileshsawant/onfield-assistant/blob/main/collections.toml)
declares one or more **sources**. A source has:

- `path` — filesystem location (relative to `$OFA_ROOT`), e.g. `repos/vasp`.
- `type` — `code` (text-with-extensions) or `pdf`.
- `extensions` — for `code`-type sources, which file suffixes to ingest.
- `keep_missing` (optional) — protect this source's chunks from the
  orphan sweep even if the underlying files vanish from disk.
- `page_ranges` (optional, PDF sources) — per-file page-range spec to
  skip front-matter or unrelated chapters.

Chunk IDs are `SHA-256(collection + relative_path + chunk_index)`, so
re-runs upsert rather than duplicate. Per-file mtime is cached in
`vectordb/.rebuild_state.json`; unchanged files are skipped (embedding
is by far the slow step).

## Cheat sheet

Run these from `$OFA_ROOT`:

```bash
# Preview: what would each collection do without touching the store?
./env/bin/python src/rebuild_indices.py --dry-run

# Rebuild everything (all collections, mtime-cached).
./env/bin/python src/rebuild_indices.py

# Rebuild one collection.
./env/bin/python src/rebuild_indices.py --collection <name>

# Rebuild from scratch (drop existing chunks first). Use after file
# renames within a source, or when chunker settings changed.
./env/bin/python src/rebuild_indices.py --clear --collection <name>

# Additive-only: disable the orphan sweep GLOBALLY for this run
# (files removed on disk are kept in the store). Prefer per-source
# `keep_missing = true` for durable intent.
./env/bin/python src/rebuild_indices.py --incremental

# List configured collections without loading the embedding model.
./env/bin/python src/rebuild_indices.py --list

# Ignore the mtime cache and re-embed everything.
./env/bin/python src/rebuild_indices.py --force
```

You do **not** need to restart `ofa --serve` after a rebuild. Retrieval
re-opens the Chroma collection per request, so the next chat call
picks up fresh chunks.

## The three recipes

### 1. Git-cloned upstream (HPC docs, source repos)

These are the collections whose source lives in a normal git checkout
under `repos/` (kept out of the parent repo entirely by [`.gitignore`](https://github.com/nileshsawant/onfield-assistant/blob/main/.gitignore)'s
`repos/*` pattern). Any changes upstream propagate via `git pull`:

```bash
cd $OFA_ROOT/repos/HPC       # or repos/amrex, repos/reframe-universal, etc.
git fetch origin
git status --short           # confirm no local drift
git pull                     # fast-forward to upstream

cd $OFA_ROOT
./env/bin/python src/rebuild_indices.py --collection hpc_docs
```

Because the source files' mtimes get bumped by `git pull`, the mtime
cache in `vectordb/.rebuild_state.json` will treat everything as
"changed" and re-embed the whole collection — that's slow (a few
minutes on CPU) but the result is a fully-consistent index. Nothing
needs `--clear` unless something structural changed (file renames,
chunker parameter changes).

If the tree already matches origin, `git pull` is a no-op and the
rebuild becomes an mtime-cache hit; total time drops to seconds.

Applies today to: `hpc_docs`, `amrex_src`, `reframe_src`,
`marbles_src` (code side), `quantum_computing` (code side).

### 2. Vendored / curated corpus (VASP wiki drops)

Some collections aren't tracked upstream — the material is a curated
mix of documents that people drop into a shared spot (e.g. an
application team gives you HTML wiki exports, PDF chapters, notes in
Markdown). The pattern:

```bash
SRC=/projects/hpcapps/rag-data-for-nilesh/vasp     # example
DST=$OFA_ROOT/repos/vasp

# 1. Wipe & repopulate the target dir. Handle format conversion here
#    if the source isn't already in a format rebuild_indices supports
#    (see "HTML conversion" below for a stdlib-only recipe).
rm -f "$DST"/*
cp "$SRC"/*.md "$SRC"/*.txt "$DST"/    # verbatim files
# ... convert any HTML/other to .md if needed ...

# 2. Rebuild the collection with --clear so file renames and
#    deletions land cleanly (mtime-cache alone can't cope with a
#    rename: it treats the old name as orphan-swept and the new name
#    as fresh-embed, which is fine here but --clear makes the state
#    unambiguous).
cd $OFA_ROOT
./env/bin/python src/rebuild_indices.py --clear --collection vasp_src
```

`repos/vasp/` is **not** git-tracked — it's vendored VASP wiki content
whose redistribution rights aren't clear, so (unlike a `git pull`
source) it lives only on disk and is never committed. Every fresh
clone or new site install needs to repopulate it from the shared drop
path before `--collection vasp_src` has anything to embed.

Applies today to: `vasp_src`, `marbles_src` (papers side),
`quantum_computing` (papers side).

### 3. Preserve chunks when the source files disappear

Some corpora — most often PDFs of copyrighted papers or textbooks —
need to be *removed from disk* after ingestion (license/redistribution
reasons) while their chunks stay useful for retrieval. Declare this
intent in [`collections.toml`](https://github.com/nileshsawant/onfield-assistant/blob/main/collections.toml):

```toml
[[collections.quantum_computing.sources]]
path         = "repos/quantum-papers"
type         = "pdf"
keep_missing = true   # protect these chunks from the orphan sweep
                      # even when the underlying files are removed.
```

With `keep_missing = true` on a source:

- Rebuilds report `[~] keep_missing: retaining N file entries under
  sources marked keep_missing=true`.
- Chunks that came from this source are never orphan-swept.
- Sibling sources in the *same collection* (e.g. `repos/quantum-code`)
  continue to sweep normally — deleted files there still drop out of
  the store on the next rebuild.

This is the durable, config-level version of `--incremental`.
`--incremental` disables the sweep for **every** source in a single
run; `keep_missing` is scoped to one source and survives every future
rebuild. Prefer `keep_missing` unless you have a one-off reason to
skip the sweep for a whole rebuild.

Applies today to: `marbles-papers`, `quantum-papers`.

## Verifying a rebuild

Rebuild logs report chunk deltas per collection, e.g.:

```
=== collection: quantum_computing ===
  92 candidate files
  [~] keep_missing: retaining 8 file entries under sources marked keep_missing=true
  embedding 194 chunks…
  embedding done in 63.2s; upserting…
  [+] 194 added/updated  [=] 80 unchanged  [-] 0 orphaned  ->  2115 total in quantum_computing
```

To confirm chunk counts per-source-root inside a collection (useful
when validating `keep_missing` behavior):

```bash
cd $OFA_ROOT
./env/bin/python -c "
import chromadb
from collections import Counter
c = chromadb.PersistentClient(path='vectordb').get_collection('quantum_computing')
print('total:', c.count())
r = c.get(include=['metadatas'], limit=99999)
counter = Counter(m.get('source_root', 'unknown') for m in r['metadatas'])
for k, v in counter.most_common():
    print(f'  {k}: {v}')
"
```

Sample output:

```
total: 2115
  quantum-papers: 1259
  quantum-code: 856
```

## Pinned docs (independent of RAG)

A small number of authoritative pages are *pinned* — read directly
from disk into the mode's context every turn, bypassing the vector
store. Today's pins:

- `repos/HPC/docs/Documentation/LBMcfd.md` — MARBLES mode.
- `repos/HPC/docs/Documentation/quantum_computing.md` — quantum mode.
- `repos/HPC/docs/Documentation/Applications/vasp.md` — VASP mode.

The pinning code lives in
[`_read_pinned_kestrel_doc()`](https://github.com/nileshsawant/onfield-assistant/blob/main/src/ofa_main.py)
and reads from `os.path.join(OFA_ROOT, "repos/HPC/docs/Documentation", relpath)`.
Two consequences:

- Editing a pinned file takes effect **immediately** — no rebuild
  needed. The next request re-reads the file.
- A pinned file's content also lands in `hpc_docs` (via the normal
  `repos/HPC/docs` ingest), so hybrid retrieval finds it too. That's
  belt-and-suspenders on purpose.

If you want to add a new pin for a mode, edit `retrieve_*_context()`
in [`src/ofa_main.py`](https://github.com/nileshsawant/onfield-assistant/blob/main/src/ofa_main.py)
and reference `_read_pinned_kestrel_doc(relpath, label)` with the
relative path under `repos/HPC/docs/Documentation/`.

## HTML conversion (for wiki drops that ship as `.html`)

`rebuild_indices.py` ingests `.md`, `.rst`, `.txt`, and per-collection
code extensions like `.py`, `.cpp`, `.H`, `.ipynb`. It does **not**
natively handle `.html`. If the upstream team sends you HTML wiki
exports (VASP does this — their wiki is MediaWiki-rendered HTML),
convert at copy time.

The Kestrel deploy has no `pandoc`, `html2text`, or `bs4` available, so
the stdlib recipe below is the reliable path. Adapt to your source's
quirks (e.g. VASP HTML retains MediaWiki template syntax like
`{{TAG|X}}` and `[[link|text]]` which is worth stripping for cleaner
retrieval):

```python
#!/usr/bin/env python3
"""Convert an HTML dump directory to .md files for RAG ingest."""
import re
import shutil
from html.parser import HTMLParser
from pathlib import Path

SRC = Path("/path/to/upstream/html/dump")
DST = Path("/nopt/nrel/apps/cpu_stack/software/openfoam/assistant/repos/vasp")


class HTMLToMarkdown(HTMLParser):
    """Minimal HTML -> Markdown-ish converter. Preserves headings,
    code blocks, and list structure enough for good semantic
    retrieval; skips <script>/<style>/<svg>/<img> content."""
    HEADING_TAGS = {"h1": "#", "h2": "##", "h3": "###",
                    "h4": "####", "h5": "#####", "h6": "######"}
    SKIP_TAGS = {"script", "style", "head", "meta", "link",
                 "noscript", "svg", "img", "figure"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.skip_depth, self.pre_depth = [], 0, 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS:
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        if tag == "pre":
            self.pre_depth += 1
            self.out.append("\n```\n")
        elif tag in self.HEADING_TAGS:
            self.out.append(f"\n\n{self.HEADING_TAGS[tag]} ")
        elif tag == "br":
            self.out.append("\n")
        elif tag in {"p", "div", "li"}:
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS:
            self.skip_depth = max(0, self.skip_depth - 1)
            return
        if self.skip_depth:
            return
        if tag == "pre":
            self.out.append("\n```\n")
            self.pre_depth = max(0, self.pre_depth - 1)
        elif tag in self.HEADING_TAGS or tag in {"p", "div", "li"}:
            self.out.append("\n")

    def handle_data(self, data):
        if not self.skip_depth:
            self.out.append(data)

    def result(self):
        text = "".join(self.out)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.rstrip() + "\n"


def strip_mediawiki_markup(text: str) -> str:
    """MediaWiki template stripping — VASP wiki specific but harmless
    on other sources. Iterate to collapse nested templates."""
    for _ in range(6):
        new = re.sub(r"\{\{[Cc]ite[^}]*\}\}", "", text)
        new = re.sub(r"\{\{NB\|[^|}]+\|([^}]+)\}\}", r"\1", new)
        new = re.sub(r"\{\{[A-Za-z_]+\|([^|}]+)\}\}", r"\1", new)
        new = re.sub(r"\{\{[A-Za-z_]+\|([^|}]+)\|([^|}]+)\}\}", r"\2", new)
        new = re.sub(r"\{\{([A-Za-z_]+)\}\}", r"\1", new)
        if new == text:
            break
        text = new
    text = re.sub(r"\[\[[^|\]]+\|([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    return re.sub(r"\[\[:?Category:[^\]]+\]\]", "", text, flags=re.I)


def html_to_md(path: Path) -> str:
    parser = HTMLToMarkdown()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    body = strip_mediawiki_markup(parser.result())
    return f"# {path.stem.replace('-', ' ').title()}\n\n(source: {path.name})\n\n{body}"


def main():
    for existing in DST.iterdir():
        if existing.is_file():
            existing.unlink()
    for src in sorted(SRC.iterdir()):
        if not src.is_file():
            continue
        ext = src.suffix.lower()
        if ext in {".md", ".txt"}:
            shutil.copy2(src, DST / src.name)
        elif ext in {".html", ".htm"}:
            (DST / (src.stem + ".md")).write_text(html_to_md(src), encoding="utf-8")


if __name__ == "__main__":
    main()
```

Save this as a throwaway script (e.g. `/tmp/vasp_sync.py`), run it,
then rebuild with `--clear`. It's not repo-worthy because it embeds
site-specific paths — if HTML ingest ever becomes a recurring pattern
across multiple collections, promote it to a proper subcommand in
`rebuild_indices.py` (add `process_html_file` alongside
`process_code_file` / `process_pdf_file`).

## Common issues

### `SyntaxError: future feature annotations is not defined`

You ran the rebuild with the system Python (`python3`) instead of
the bundled interpreter. `rebuild_indices.py` requires Python 3.7+
for `from __future__ import annotations`; Kestrel's login-node
default is 3.6.

Always invoke through `./env/bin/python` (or the equivalent bundled
Python your install phase materialised):

```bash
cd $OFA_ROOT
./env/bin/python src/rebuild_indices.py ...
```

### `CUDA initialization: driver too old`

Cosmetic warning on Kestrel's login node — its CUDA driver is older
than the version bundled `torch` was compiled against. Embedding
falls back to CPU, which is 2-5x slower but produces bit-identical
results. Ignore, or run the rebuild inside a GPU allocation
(`srun --partition=debug-gpu ...`) if you need the speed.

### Rebuild reports `[-] N orphaned` on a source you didn't touch

The file was renamed, moved, or its mtime changed in a way that
looks like a delete to the mtime cache. Two remedies:

- If the rename was legitimate and you want the new chunks: no
  action; the next run will re-embed under the new name.
- If you did NOT intend the delete: check `git status` on the
  source directory (git-tracked corpora only); for vendored
  corpora, double-check no one else has been editing under
  `repos/vasp/`.

### Model banner shows `Active model: <name>  [UNTESTED — see warning below]`

Unrelated to RAG. The active model isn't in `TESTED_MODELS`
(currently `gemma4:31b` and `gemma4:31b-it-q8_0`, the default). The
safety guards were validated against those models' output style;
other models still work but have not been separately re-validated.
Approve destructive-command prompts carefully.

## Which collection feeds which mode

For reference when deciding which collection to rebuild:

| Mode                    | Primary collection      | Sibling collections retrieved |
|-------------------------|--------------------------|-------------------------------|
| `--openfoam`            | `openfoam`               | `of13_src` (only when the query mentions C++/source code) |
| `--code` (default)      | `hpc_docs`               | —                              |
| `--hpc`                 | `hpc_docs`               | —                              |
| `--amrex`               | `amrex_src`              | `hpc_docs` (light, top_k=2)    |
| `--marbles`             | `marbles_src`            | `amrex_src` (light, top_k=2), `hpc_docs` (light, top_k=2), pinned `LBMcfd.md` |
| `--quantum-computing`   | `quantum_computing`      | `hpc_docs` (light, top_k=2), pinned `quantum_computing.md` |
| `--vasp`                | `vasp_src`               | `hpc_docs` (light, top_k=2), pinned `vasp.md` |
| `--rhel9_reframe`       | `reframe_src`            | `hpc_docs` (full, top_k=15, labeled as RHEL8/legacy context) + static `rhel9_module_structure.txt` |

`--code` and `--hpc` retrieve identically (both call
`retrieve_hpc_context()` directly) — `--code` additionally enables the
coding-agent tool loop (file edits, shell, etc.) on top of the same
RAG context.

The exact retriever weights live in `retrieve_*_context()` functions
in [`src/ofa_main.py`](https://github.com/nileshsawant/onfield-assistant/blob/main/src/ofa_main.py).
