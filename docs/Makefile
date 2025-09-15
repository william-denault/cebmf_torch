# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Variables
SPHINX_APIDOC = sphinx-apidoc
SPHINX_BUILD = sphinx-build
MODULE_DIR   = ../src/cebmf_torch

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Generate API docs
apidoc:
	rm -r $(SOURCEDIR)/api_files || true
	$(SPHINX_APIDOC) -f -o $(SOURCEDIR)/api_files $(MODULE_DIR)

# Copy notebooks into source/notebooks before building
copy-notebooks:
	mkdir -p $(SOURCEDIR)/notebooks
	cp -r ../examples/*.ipynb $(SOURCEDIR)/notebooks/


.PHONY: help Makefile

html: copy-notebooks apidoc
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

htmlhelp: copy-notebooks apidoc
	@$(SPHINXBUILD) -M htmlhelp "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)


# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
