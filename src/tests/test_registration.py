"""An example set of tests."""

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import tifffile
from skimage.data import binary_blobs
from skimage.measure import block_reduce

from hipct_reg.ITK_registration import registration_pipeline


@pytest.fixture()
def rng() -> np.random.Generator:
    """
    Setup the default random number generator.
    """
    return np.random.default_rng(seed=1)


def write_array_to_stack(array: npt.NDArray[Any], folder: Path) -> None:
    """
    Write a 3D array to a stack of images.
    """
    assert array.ndim == 3, "Input array must be 3D"
    for i, plane in enumerate(array):
        tifffile.imwrite(folder / f"{i}.tif", plane, photometric="minisblack")


def test_simple_registration(tmp_path: Path, rng: np.random.Generator) -> None:
    """
    Test a really simple registration where the common point given is exactly the
    correct point, and there is no rotation between the two datasets.
    """
    ground_truth_data = binary_blobs(
        length=512, n_dim=3, blob_size_fraction=0.01, volume_fraction=0.5, rng=rng
    ).astype(np.float32)

    # Downsample to mimic a full organ scan
    bin_factor = 4
    full_organ_scan = block_reduce(
        ground_truth_data, (bin_factor, bin_factor, bin_factor), np.mean
    )
    full_organ_folder = tmp_path / "20.0um_full_organ"
    full_organ_folder.mkdir()
    write_array_to_stack(full_organ_scan, full_organ_folder)

    # Take a sub-volume to mimic a high resolution region of interest
    offset = 128
    size = 64
    roi_scan = ground_truth_data[
        offset : offset + size, offset : offset + size, offset : offset + size
    ]
    roi_folder = tmp_path / "5.0um_roi"
    roi_folder.mkdir()
    write_array_to_stack(roi_scan, roi_folder)

    full_organ_point = np.array([offset, offset, offset]) / bin_factor
    roi_point = np.array([0, 0, 0])

    registration_pipeline(
        str(full_organ_folder), str(roi_folder), full_organ_point, roi_point
    )
