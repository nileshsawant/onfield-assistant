# Minimal Makefile for Sphinx documentation.
# Usage:
#   make html       # build HTML into docs/_build/html
#   make clean      # remove docs/_build
#   make linkcheck  # verify all external links resolve
#
# Requires deps from docs/requirements-docs.txt.

SPHINXOPTS    ?= --keep-going
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help clean html linkcheck Makefile

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf "$(BUILDDIR)" _autosummary

html linkcheck:
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
