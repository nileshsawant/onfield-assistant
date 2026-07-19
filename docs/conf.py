"""Sphinx configuration for ofa (OnField Assistant).

Builds developer + user documentation from:
  * Python docstrings in ``src/`` (via ``sphinx.ext.autodoc`` +
    ``sphinx.ext.autosummary`` with ``:recursive:``, giving each
    module / class / function its own stub page and hyperlink).
  * The existing markdown pages at the repo root and under ``docs/``
    (via ``myst-parser`` and ``{include}`` shims).

Deployed to GitHub Pages by ``.github/workflows/docs.yml``.

Runs on CI without installing ofa's full runtime — heavy third-party
deps are mocked in ``autodoc_mock_imports`` below so we only need
``sphinx``, ``furo``, and ``myst-parser`` to build the site.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime

# Put src/ on the import path so autodoc can find ``ofa_main`` etc.
# conf.py lives in docs/, so the src/ we want is one level up.
sys.path.insert(0, os.path.abspath("../src"))

# ---------------------------------------------------------------------------
# Project metadata
# ---------------------------------------------------------------------------
project = "OnField Assistant (ofa)"
author = "ofa contributors"
copyright = f"{datetime.now().year}, {author}"
release = "1.0"

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------
extensions = [
    # Autogenerate reference from docstrings.
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    # Parse Google / NumPy / plain docstrings.
    "sphinx.ext.napoleon",
    # Emit "[source]" links next to each rendered symbol so readers
    # can jump to the exact source line — the point of running Sphinx
    # over inline docs.
    "sphinx.ext.viewcode",
    # Cross-reference Python stdlib symbols (Path, Iterable, …).
    "sphinx.ext.intersphinx",
    # Render .md pages as first-class Sphinx documents so we can reuse
    # the repo's existing markdown (README, ARCHITECTURE, byok-vscode,
    # ofa-technical-overview) without duplicating content.
    "myst_parser",
]

autosummary_generate = True
autosummary_imported_members = False

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    # Keep private helpers out of the public reference. Users of the
    # ofa_client public API don't want to trip over _fence_rag &
    # friends; developers can still read them via the [source] link.
    "private-members": False,
}
autodoc_typehints = "description"
autodoc_member_order = "bysource"

# ---------------------------------------------------------------------------
# Mock heavy runtime deps so autodoc can import the ofa modules on a
# stock GitHub Actions runner without a full ``pip install -r
# requirements.txt``. Every third-party package the ofa src/ code
# imports at module top-level goes here; stdlib does not.
# ---------------------------------------------------------------------------
autodoc_mock_imports = [
    "chromadb",
    "ddgs",
    "httpx",
    "huggingface_hub",
    "lxml",
    "numpy",
    "ollama",
    "pdfplumber",
    "pypdf",
    "rank_bm25",
    "sentence_transformers",
    "torch",
    "transformers",
]

# ---------------------------------------------------------------------------
# myst-parser configuration
# ---------------------------------------------------------------------------
myst_enable_extensions = [
    "colon_fence",   # :::note ... ::: admonition blocks
    "deflist",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3  # auto-slug h1/h2/h3 for hyperlinks

# ---------------------------------------------------------------------------
# General config
# ---------------------------------------------------------------------------
source_suffix = {
    ".rst": "restructuredtext",
    ".md":  "markdown",
}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
templates_path = ["_templates"]

# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------
html_theme = "furo"
html_title = "OnField Assistant"
html_static_path = ["_static"]
html_theme_options = {
    "source_repository": "https://github.com/nileshsawant/onfield-assistant/",
    "source_branch": "main",
    "source_directory": "docs/",
    "navigation_with_keys": True,
}

# ---------------------------------------------------------------------------
# intersphinx
# ---------------------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}
