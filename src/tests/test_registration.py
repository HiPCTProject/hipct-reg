"""An example set of tests."""

from pathlib import Path
from typing import Any
import tifffile

import numpy as np
import pytest
import numpy.typing as npt
from skimage.data import binary_blobs


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=1)


def write_array_to_stack(array: npt.NDArray[Any], folder: Path) -> None:
    """
    Write a 3D array to a stack of images.
    """
    assert array.ndim == 3, "Input array must be 3D"
    for i, plane in enumerate(array):
        tifffile.imwrite(folder / f'{i}.tif', plane, photometric='minisblack')


def test_simple_registration(tmp_path: Path, rng: np.random.Generator) -> None:
    """
    Test a really simple registration where the common point given is exactly the
    correct point, and there is no rotation between the two datasets.
    """
    high_res_folder = tmp_path / "high_res"
    high_res_folder.mkdir()
    high_res_array = binary_blobs(
        length=512, n_dim=3, blob_size_fraction=0.01, volume_fraction=0.5, rng=rng
    )
    write_array_to_stack(high_res_array, high_res_folder)
    assert len(list(high_res_folder.glob("*tif"))) == 512
