# hipct-reg

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Tests status][tests-badge]][tests-link]
[![License][license-badge]](./LICENSE.md)

<!--
[![PyPI version][pypi-version]][pypi-link]
[![Conda-Forge][conda-badge]][conda-link]
[![PyPI platforms][pypi-platforms]][pypi-link]
-->

<!-- prettier-ignore-start -->
[tests-badge]:              https://github.com/dstansby/hipct-reg/actions/workflows/tests.yml/badge.svg
[tests-link]:               https://github.com/dstansby/hipct-reg/actions/workflows/tests.yml
[linting-badge]:            https://github.com/dstansby/hipct-reg/actions/workflows/linting.yml/badge.svg
[linting-link]:             https://github.com/dstansby/hipct-reg/actions/workflows/linting.yml
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/hipct-reg
[conda-link]:               https://github.com/conda-forge/hipct-reg-feedstock
[pypi-link]:                https://pypi.org/project/hipct-reg/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/hipct-reg
[pypi-version]:             https://img.shields.io/pypi/v/hipct-reg
[license-badge]:            https://img.shields.io/badge/License-BSD_3--Clause-blue.svg
<!-- prettier-ignore-end -->

Code to register regions of interest with full organ datasets.

## Getting Started

### Prerequisites

<!-- Any tools or versions of languages needed to run code. For example specific Python or Node versions. Minimum hardware requirements also go here. -->

`hipct-reg` requires Python 3.11.

### Installation

<!-- How to build or install the application. -->

We recommend installing in a project specific virtual environment created using a environment management tool such as [Mamba](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html) or [Conda](https://conda.io/projects/conda/en/latest/). To install the latest development version of `hipct-reg` using `pip` in the currently active environment run

```sh
pip install git+https://github.com/dstansby/hipct-reg.git
```

Alternatively create a local clone of the repository with

```sh
git clone https://github.com/dstansby/hipct-reg.git
```

and then install in editable mode by running

```sh
pip install -e .
```

### Running Tests

<!-- How to run tests on your local system. -->

Tests can be run across all compatible Python versions in isolated environments using
[`tox`](https://tox.wiki/en/latest/) by running

```sh
tox
```

To run tests manually in a Python environment with `pytest` installed run

```sh
pytest tests
```

again from the root of the repository.
