"""An example set of tests."""

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import SimpleITK as sitk
import tifffile
from skimage.data import binary_blobs
from skimage.measure import block_reduce

from hipct_reg.helpers import import_im
from hipct_reg.ITK_registration import registration_rot

PIXEL_SIZE_UM = 5
BIN_FACTOR = 4
ROI_OFFSET = 128
ROI_SIZE = 64


@pytest.fixture
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


@pytest.fixture
def ground_truth(rng: np.random.Generator) -> npt.NDArray[np.float32]:
    """
    Ground truth, high resolution data.
    """
    return binary_blobs(
        length=256, n_dim=3, blob_size_fraction=0.01, volume_fraction=0.5, rng=rng
    ).astype(np.uint16)


@pytest.fixture
def full_organ_scan(
    tmp_path: Path, ground_truth: npt.NDArray[np.float32]
) -> sitk.Image:
    """
    Downsampled ground truth data, to mimic a full organ scan.
    """
    full_organ_scan = block_reduce(
        ground_truth, (BIN_FACTOR, BIN_FACTOR, BIN_FACTOR), np.mean
    )
    full_organ_folder = tmp_path / "20.0um_full_organ"
    full_organ_folder.mkdir()
    write_array_to_stack(full_organ_scan, full_organ_folder)

    return import_im(
        str(full_organ_folder), pixel_size=PIXEL_SIZE_UM * BIN_FACTOR * 1000
    )


@pytest.fixture
def roi_scan(tmp_path: Path, ground_truth: npt.NDArray[np.float32]) -> sitk.Image:
    """
    Sub-volume of ground truth data, to mimic ROI data.
    """
    roi_scan = ground_truth[
        ROI_OFFSET : ROI_OFFSET + ROI_SIZE,
        ROI_OFFSET : ROI_OFFSET + ROI_SIZE,
        ROI_OFFSET : ROI_OFFSET + ROI_SIZE,
    ]
    roi_folder = tmp_path / "5.0um_roi"
    roi_folder.mkdir()
    write_array_to_stack(roi_scan, roi_folder)

    return import_im(str(roi_folder), pixel_size=PIXEL_SIZE_UM * 1000)


def test_registration_rot(
    full_organ_scan: sitk.Image, roi_scan: sitk.Image, caplog
) -> None:
    """
    Test a really simple registration where the common point given is exactly the
    correct point, and there is no rotation between the two datasets.
    """
    trans_point = np.array([ROI_OFFSET, ROI_OFFSET, ROI_OFFSET]) / BIN_FACTOR
    rotation_center = (
        trans_point + np.array([ROI_SIZE, ROI_SIZE, ROI_SIZE]) / 2 / BIN_FACTOR
    )

    zrot = 0
    with caplog.at_level(logging.INFO):
        transform = registration_rot(
            fixed_image=full_organ_scan,
            moving_image=roi_scan,
            trans_point=trans_point,
            rotation_center_pix=rotation_center,
            zrot=zrot,
            angle_range=360,
            angle_step=2,
        )

    assert re.match(
        r"""INFO Starting rotational registration
INFO Initial rotation = 0 deg
INFO Range = 360 deg
INFO Step = 2 deg
INFO Translation = \[32. 32. 32.\] pix
INFO Rotation center = \[40. 40. 40.\] pix
INFO Starting registration...
INFO Registration finished!
INFO Final metric value = -0\.[0-9]*
INFO Stopping condition = ExhaustiveOptimizerv4: Completed sampling of parametric space of size 6
INFO Registered rotation angele = 2.0 deg
""",
        caplog.text,
    )

    assert isinstance(transform, sitk.Euler3DTransform)
    assert transform.GetAngleX() == 0
    assert transform.GetAngleY() == 0
    # This value should be close to zero
    zrot = np.rad2deg(transform.GetAngleZ())
    assert zrot == pytest.approx(2)

    # Try a smaller angular range at higher angular resolution
    transform = registration_rot(
        fixed_image=full_organ_scan,
        moving_image=roi_scan,
        trans_point=trans_point,
        rotation_center_pix=rotation_center,
        zrot=zrot,
        angle_range=5,
        angle_step=0.1,
    )

    assert isinstance(transform, sitk.Euler3DTransform)
    assert transform.GetAngleX() == 0
    assert transform.GetAngleY() == 0
    # This value should be close to zero
    zrot = np.rad2deg(transform.GetAngleZ())
    assert zrot == pytest.approx(1.5)
