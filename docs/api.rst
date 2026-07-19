API Reference
=============

Every module, class, and function in ``src/`` is documented here. Each
name is hyperlinked, and every rendered signature carries a ``[source]``
button that jumps directly to the definition in GitHub.

Docstrings are extracted verbatim via ``sphinx.ext.autodoc``; any
inaccuracies in an entry below reflect the state of the docstring in
source. Fixing them is a one-line PR: edit the module and re-push, and
the CI pipeline (see ``.github/workflows/docs.yml``) rebuilds this
site.

Modules
-------

The five ofa modules a downstream integrator is most likely to reach for
first:

.. autosummary::
   :toctree: _autosummary
   :recursive:

   ofa_client
   ofa_main
   ofa_server
   ofa_site
   rebuild_indices

Supporting modules
------------------

Ingest / RAG helpers that admins running ``rebuild_indices.py`` may want
to inspect, but that are not part of the ofa runtime hot path:

.. autosummary::
   :toctree: _autosummary
   :recursive:

   build_index
   build_index_v2
   ingest_amrex
   ingest_reframe
   pdf_extract
