# Contributing

Thanks for considering contributing! Please read this document to learn the various ways you can contribute to this project and how to go about doing it.

## Bug reports and feature requests

### Did you find a bug?

First, do [a quick search](https://github.com/mne-tools/mne-icalabel/issues) to see whether your issue has already been reported.
If your issue has already been reported, please comment on the existing issue.

Otherwise, open [a new GitHub issue](https://github.com/mne-tools/mne-icalabel/issues).  Be sure to include a clear title
and description.  The description should include as much relevant information as possible.  The description should
explain how to reproduce the erroneous behavior as well as the behavior you expect to see.  Ideally you would include a
code sample or an executable test case demonstrating the expected behavior.

### Do you have a suggestion for an enhancement or new feature?

We use GitHub issues to track feature requests. Before you create an feature request:

* Make sure you have a clear idea of the enhancement you would like. If you have a vague idea, consider discussing
it first on a GitHub issue.
* Check the documentation to make sure your feature does not already exist.
* Do [a quick search](https://github.com/mne-tools/mne-icalabel/issues) to see whether your feature has already been suggested.

When creating your request, please:

* Provide a clear title and description.
* Explain why the enhancement would be useful. It may be helpful to highlight the feature in other libraries.
* Include code examples to demonstrate how the enhancement would be used.

## Making a pull request

When you're ready to contribute code to address an open issue, please follow these guidelines to help us be able to review your pull request (PR) quickly.

1. **Initial setup** (only do this once)

    <details><summary>Expand details 👇</summary><br/>

    If you haven't already done so, please [fork](https://help.github.com/en/enterprise/2.13/user/articles/fork-a-repo) this repository on GitHub.

    Then clone your fork locally with

        git clone https://github.com/USERNAME/mne-icalabel.git

    or 

        git clone git@github.com:USERNAME/mne-icalabel.git

    At this point the local clone of your fork only knows that it came from *your* repo, github.com/USERNAME/mne-icalabel.git, but doesn't know anything the *main* repo, [https://github.com/mne-tools/mne-icalabel.git](https://github.com/mne-tools/mne-icalabel). You can see this by running

        # Note you should be in the "mne-icalabel" directory. If you're not
        # run "cd ./mne-icalabel" to change directory into the repo
        git remote -v

    which will output something like this:

        origin https://github.com/USERNAME/mne-icalabel.git (fetch)
        origin https://github.com/USERNAME/mne-icalabel.git (push)

    This means that your local clone can only track changes from your fork, but not from the main repo, and so you won't be able to keep your fork up-to-date with the main repo over time. Therefore you'll need to add another "remote" to your clone that points to [https://github.com/mne-tools/mne-icalabel.git](https://github.com/mne-tools/mne-icalabel). To do this, run the following:

        git remote add upstream https://github.com/mne-tools/mne-icalabel.git

    Now if you do `git remote -v` again, you'll see

        origin https://github.com/USERNAME/mne-icalabel.git (fetch)
        origin https://github.com/USERNAME/mne-icalabel.git (push)
        upstream https://github.com/mne-tools/mne-icalabel.git (fetch)
        upstream https://github.com/mne-tools/mne-icalabel.git (push)

    Finally, you'll need a Python virtual environment suitable for working on this project. We recommend [`uv`](https://docs.astral.sh/uv/), which creates the environment and installs your local clone in "editable mode" with the developer dependencies (declared as dependency groups in `pyproject.toml`) and the ICLabel backends:

        uv sync --extra onnx --extra torch

    The "editable mode" means that any changes you make to the source code in your local clone will be immediately reflected in your virtual environment.

    </details>

2. **Ensure your fork is up-to-date**

    <details><summary>Expand details 👇</summary><br/>

    Once you've added an "upstream" remote pointing to [https://github.com/allenai/python-package-temlate.git](https://github.com/mne-tools/mne-icalabel), keeping your fork up-to-date is easy:

        git checkout main  # if not already on main
        git pull --rebase upstream main
        git push

    </details>

3. **Create a new branch to work on your fix or enhancement**

    <details><summary>Expand details 👇</summary><br/>

    Committing directly to the main branch of your fork is not recommended. It will be easier to keep your fork clean if you work on a separate branch for each contribution you intend to make.

    You can create a new branch with

        # replace BRANCH with whatever name you want to give it
        git checkout -b BRANCH
        git push -u origin BRANCH

    </details>

4. **Test your changes**

    <details><summary>Expand details 👇</summary><br/>

    Our continuous integration (CI) testing runs [a number of checks](https://github.com/mne-tools/mne-icalabel/actions) for each pull request on [GitHub Actions](https://github.com/features/actions). You can run most of these tests locally, which is something you should do *before* opening a PR to help speed up the review process and make it easier for us.

    Style checks (linting and formatting with [`ruff`](https://docs.astral.sh/ruff/), spell checking, ...) are run with [`pre-commit`](https://pre-commit.com/). You can install the git hooks to run them automatically on every commit, or run them on the entire repository:

        pre-commit install
        pre-commit run --all-files

    We also strive to maintain high test coverage, so most contributions should include additions to [the unit tests](https://github.com/mne-tools/mne-icalabel/tree/main/tests). These tests are run with [`pytest`](https://docs.pytest.org/en/latest/), which you can use to locally run any test modules that you've added or changed.

    For example, if you've fixed a bug in `src/mne_icalabel/a/b.py`, you can run the tests specific to that module with

        pytest -v tests/a/test_b.py

    Our CI will automatically check that test coverage stays above a certain threshold (around 90%). To check the coverage locally in this example, you could run

        pytest -v --cov mne_icalabel.a.b tests/a/test_b.py

    If your contribution involves additions to any public part of the API, we require that you write docstrings
    for each function, method, class, or module that you add.
    See the [Writing docstrings](#writing-docstrings) section below for details on the syntax.
    You should test to make sure the API documentation can build without errors by running

        cd doc
        make html

    If the build fails, it's most likely due to small formatting issues. If the error message isn't clear, feel free to comment on this in your pull request.

    And finally, please update the [CHANGELOG](https://github.com/mne-tools/mne-icalabel/blob/main/CHANGELOG.md) with notes on your contribution in the "Unreleased" section at the top.

    After all of the above checks have passed, you can now open [a new GitHub pull request](https://github.com/mne-tools/mne-icalabel/pulls).
    Make sure you have a clear description of the problem and the solution, and include a link to relevant issues.

    We look forward to reviewing your PR!

    </details>

### Writing docstrings

We use [Sphinx](https://www.sphinx-doc.org/en/master/index.html) to build our API docs, which automatically parses all docstrings
of public classes and methods. All docstrings should adhere to the [Numpy styling convention](https://www.sphinx-doc.org/en/master/usage/extensions/example_numpy.html).
