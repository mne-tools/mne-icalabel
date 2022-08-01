# simple makefile to simplify repetetive build env management tasks under posix
# caution: testing won't work on windows

PYTHON ?= python
PYTESTS ?= py.test
CTAGS ?= ctags
CODESPELL_SKIPS ?= "*.fif,*.eve,*.gz,*.tgz,*.zip,*.mat,*.stc,*.label,*.w,*.bz2,*.annot,*.sulc,*.log,*.local-copy,*.orig_avg,*.inflated_avg,*.gii,*.pyc,*.doctree,*.pickle,*.inv,*.png,*.edf,*.touch,*.thickness,*.nofix,*.volume,*.defect_borders,*.mgh,lh.*,rh.*,COR-*,FreeSurferColorLUT.txt,*.examples,.xdebug_mris_calc,bad.segments,BadChannels,empty_file,*.orig,*.js,*.map,*.ipynb,searchindex.dat,plot_*.rst,*.rst.txt,*.html,gdf_encodes.txt"
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
	$(PYTESTS) -m 'not slowtest' mne

test-full: in
	rm -f .coverage
	$(PYTESTS) mne

test-no-network: in
	sudo unshare -n -- sh -c 'MNE_SKIP_NETWORK_TESTS=1 py.test mne'

test-no-testing-data: in
	@MNE_SKIP_TESTING_DATASET_TESTS=true \
	$(PYTESTS) mne

test-no-sample-with-coverage: in testing_data
	rm -rf coverage .coverage
	$(PYTESTS) --cov=mne_icalabel --cov-report html:coverage

test-doc: sample_data testing_data
	$(PYTESTS) --doctest-modules --doctest-ignore-import-errors --doctest-glob='*.rst' ./doc/

test-coverage: testing_data
	rm -rf coverage .coverage
	$(PYTESTS) --cov=mne_icalabel --cov-report html:coverage
# what's the difference with test-no-sample-with-coverage?

test-mem: in testing_data
	ulimit -v 1097152 && $(PYTESTS) mne

trailing-spaces:
	find . -name "*.py" | xargs perl -pi -e 's/[ \t]*$$//'

ctags:
	# make tags for symbol based navigation in emacs and vim
	# Install with: sudo apt-get install exuberant-ctags
	$(CTAGS) -R *

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
	@codespell -w -i 3 -q 3 -S $(CODESPELL_SKIPS) --ignore-words=ignore_words.txt $(CODESPELL_DIRS)

codespell-error:  # running on travis
	@codespell -i 0 -q 7 -S $(CODESPELL_SKIPS) --ignore-words=ignore_words.txt $(CODESPELL_DIRS)

pydocstyle:
	@echo "Running pydocstyle"
	@pydocstyle mne_icalabel

check-manifest:
	check-manifest --ignore .circleci*,doc,logo,.DS_Store

check-readme:
	python setup.py check --restructuredtext --strict

isort:
	@if command -v isort > /dev/null; then \
		echo "Running isort"; \
		isort mne_icalabel examples; \
	else \
		echo "isort not found, please install it!"; \
		exit 1; \
	fi;
	@echo "isort passed"

pep:
	@$(MAKE) -k black pydocstyle codespell-error check-manifest

docstyle: pydocstyle

run-checks:
	isort --check .
	black --check mne_icalabel examples
	flake8 .
	mypy ./mne_icalabel
	@$(MAKE) pydocstyle
	check-manifest
	@$(MAKE) codespell-error

build-doc:
	@echo "Building documentation"
	make -C doc/ clean
	make -C doc/ html-noplot
	cd doc/ && make view
