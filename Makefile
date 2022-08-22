# simple makefile to simplify repetitive build env management tasks under posix
# caution: testing won't work on windows

PYTHON ?= python
PYTESTS ?= py.test
CTAGS ?= ctags
CODESPELL_SKIPS ?= "*.fif,*.eve,*.gz,*.tgz,*.zip,*.mat,*.stc,*.label,*.w,*.bz2,*.annot,*.sulc,*.log,*.local-copy,*.orig_avg,*.inflated_avg,*.gii,*.pyc,*.doctree,*.pickle,*.inv,*.png,*.edf,*.touch,*.thickness,*.nofix,*.volume,*.defect_borders,*.mgh,lh.*,rh.*,COR-*,FreeSurferColorLUT.txt,*.examples,.xdebug_mris_calc,bad.segments,BadChannels,empty_file,*.orig,*.js,*.map,*.ipynb,searchindex.dat,plot_*.rst,*.rst.txt,*.html,gdf_encodes.txt,*.svg,*.css"
CODESPELL_DIRS ?= mne_icalabel/ doc/ examples/
all: clean inplace test test-doc

clean-pyc:
	find . -name "*.pyc" | xargs rm -f

clean-so:
	find . -name "*.so" | xargs rm -f
	find . -name "*.pyd" | xargs rm -f

clean-build:
	rm -rf _build build dist mne_icalabel.egg-info

clean-ctags:
	rm -f tags

clean-cache:
	find . -name "__pycache__" | xargs rm -rf

clean-test:
	rm -rf .pytest_cache .mypy_cache .ipynb_checkpoints
	rm junit-results.xml

clean: clean-build clean-pyc clean-so clean-ctags clean-cache clean-test

in: inplace # just a shortcut
inplace:
	$(PYTHON) setup.py build_ext -i

pytest: test

test: in
	rm -f .coverage
	$(PYTESTS) mne_icalabel

test-fast: in
	rm -f .coverage
	$(PYTESTS) -m 'not slowtest' mne_icalabel

test-mem: in testing_data
	ulimit -v 1097152 && $(PYTESTS) mne

upload-pipy:
	python setup.py sdist bdist_egg register upload

flake:
	@if command -v flake8 > /dev/null; then \
		echo "Running flake8"; \
		flake8 --count mne_icalabel examples; \
	else \
		echo "flake8 not found, please install it!"; \
		exit 1; \
	fi;
	@echo "flake8 passed"

black:
	@if command -v black > /dev/null; then \
		echo "Running black"; \
		black mne_icalabel examples; \
	else \
		echo "black not found, please install it!"; \
		exit 1; \
	fi;
	@echo "black passed"

codespell:  # running manually
	@codespell -w -i 3 -q 3 -S $(CODESPELL_SKIPS) --ignore-words=.codespellignore $(CODESPELL_DIRS)

codespell-error:  # running on travis
	@codespell -i 0 -q 7 -S $(CODESPELL_SKIPS) --ignore-words=.codespellignore $(CODESPELL_DIRS)

pydocstyle:
	@echo "Running pydocstyle"
	@pydocstyle mne_icalabel

isort:
	@if command -v isort > /dev/null; then \
		echo "Running isort"; \
		isort mne_icalabel examples doc; \
	else \
		echo "isort not found, please install it!"; \
		exit 1; \
	fi;
	@echo "isort passed"

docstyle: pydocstyle

run-checks:
	isort --check .
	black --check mne_icalabel examples
	flake8 .
	mypy ./mne_icalabel
	@$(MAKE) pydocstyle
	@$(MAKE) codespell-error

build-doc:
	@echo "Building documentation"
	make -C doc/ clean
	make -C doc/ html-noplot
	cd doc/ && make view
