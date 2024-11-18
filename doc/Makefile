# Minimal makefile for Sphinx documentation

SPHINXOPTS    ?= -nWT --keep-going
SPHINXBUILD   ?= sphinx-build

.PHONY: help Makefile clean html html-noplot linkcheck linkcheck-grep view

first_target: help

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html             to make standalone HTML files"
	@echo "  html-noplot      to make standalone HTML files without plotting"
	@echo "  clean            to clean HTML files"
	@echo "  linkcheck        to check all external links for integrity"
	@echo "  linkcheck-grep   to grep the linkcheck result"
	@echo "  view             to view the built HTML"

html:
	$(SPHINXBUILD) . _build/html -b html $(SPHINXOPTS)

html-noplot:
	$(SPHINXBUILD) . _build/html -b html $(SPHINXOPTS) -D plot_gallery=0

clean:
	rm -rf _build generated

linkcheck:
	$(SPHINXBUILD) . _build/linkcheck -b linkcheck -D plot_gallery=0

linkcheck-grep:
	@! grep -h "^.*:.*: \[\(\(local\)\|\(broken\)\)\]" _build/linkcheck/output.txt

view:
	@python -c "import webbrowser; webbrowser.open_new_tab('file://$(PWD)/_build/html/index.html')"
