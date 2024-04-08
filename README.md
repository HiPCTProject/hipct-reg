# hipct-reg

[![License][license-badge]](./LICENSE.md)

[![Tests status][tests-badge]][tests-link]
[![codecov](https://codecov.io/gh/HiPCTProject/hipct-reg/graph/badge.svg?token=KCHTIG4HH5)](https://codecov.io/gh/HiPCTProject/hipct-reg)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/HiPCTProject/hipct-reg/main.svg)](https://results.pre-commit.ci/latest/github/HiPCTProject/hipct-reg/main)
[![Documentation Status](https://readthedocs.org/projects/hipct-reg/badge/?version=latest)](https://hipct-reg.readthedocs.io/en/latest/?badge=latest)

<!--
[![PyPI version][pypi-version]][pypi-link]
[![Conda-Forge][conda-badge]][conda-link]
[![PyPI platforms][pypi-platforms]][pypi-link]
-->

<!-- prettier-ignore-start -->
[tests-badge]:              https://github.com/HiPCTProject/hipct-reg/actions/workflows/tests.yml/badge.svg
[tests-link]:               https://github.com/HiPCTProject/hipct-reg/actions/workflows/tests.yml
[linting-badge]:            https://github.com/HiPCTProject/hipct-reg/actions/workflows/linting.yml/badge.svg
[linting-link]:             https://github.com/HiPCTProject/hipct-reg/actions/workflows/linting.yml
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/hipct-reg
[conda-link]:               https://github.com/conda-forge/hipct-reg-feedstock
[pypi-link]:                https://pypi.org/project/hipct-reg/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/hipct-reg
[pypi-version]:             https://img.shields.io/pypi/v/hipct-reg
[license-badge]:            https://img.shields.io/badge/License-BSD_3--Clause-blue.svg
<!-- prettier-ignore-end -->

Code to register regions of interest with full organ datasets.

## High level overview

The goal of this package is to provide reproducible and tested code to register (align) high-resolution region of interest (ROI) scans with low-resolution full organ scans.
It does this using the registration framework of `SimpleITK`.

Within the `SimpleITK` framework registration maps points from one image to another image (or in our case one volume to another volume - to keep it simple we'll just use image from now on).
In `SimpleITK` language, we are looking for a transformation from a **fixed image** coordinate system to a **moving image** coordinate system.
Because we are looking for a transform to map the ROI to full organ, we will set the fixed image to be the ROI and moving image to be the full organ.

The registration is done in several steps, described in the following sections.

### Manually finding a common point

The first step is to manually find a common point between the two datasetsbegin registered.
The easiest way to do this is using neuroglancer, with both datasets open side by side.
Once you have found a common point, record the pixel coordinates in both the full organ dataset and the ROI dataset.

### Initial rotational registration

The first step of the automated pipeline finds the approximate relative rotation about the z-axis betweeen the two datasets.
It does this by fixing the centre of rotation at the common point identified in both datasets, and then:

1. Scanning through the whole 0 - 360 degree angle range at a resolution of 2 degrees, and identifying the best match.
2. Scanning in a range of +/- 2.5 degrees about this best match at a resolution of 0.1 degrees to identify a more accurate best match.

### Final full registration

The second and final step of the automated pipeline uses a rigid transform that allows for variation in translation (x, y, z compoonents), a rotation around the z-axis (one component), and a scale factor.
The scaling factor is varied to take into account uncertainties in the resolution of each dataset.

## Getting Started

### Prerequisites

<!-- Any tools or versions of languages needed to run code. For example specific Python or Node versions. Minimum hardware requirements also go here. -->

`hipct-reg` requires Python 3.11.

### Installation

<!-- How to build or install the application. -->

We recommend installing in a project specific virtual environment created using a environment management tool such as [Mamba](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html) or [Conda](https://conda.io/projects/conda/en/latest/). To install the latest development version of `hipct-reg` using `pip` in the currently active environment run

```sh
pip install git+https://github.com/HiPCTProject/hipct-reg.git
```

Alternatively create a local clone of the repository with

```sh
git clone https://github.com/HiPCTProject/hipct-reg.git
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
