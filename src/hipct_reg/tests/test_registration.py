import logging
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import SimpleITK as sitk
import tifffile
from skimage.data import binary_blobs
from skimage.measure import block_reduce

from hipct_reg.helpers import (
    arr_to_index_tuple,
    get_central_pixel_index,
    import_im,
    transform_to_dict,
)
from hipct_reg.registration import (
    registration_rigid,
    registration_rot,
    run_registration,
)
from hipct_reg.types import RegistrationInput

# Pixel size of full resolution (ROI) pixels
PIXEL_SIZE_UM = 5
BIN_FACTOR = 4
# Offset in units of full-resolution (ROI) pixels
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
def ground_truth(rng: np.random.Generator) -> npt.NDArray[np.uint16]:
    """
    Ground truth, high resolution data.
    """
    data: npt.NDArray = binary_blobs(
        length=256, n_dim=3, blob_size_fraction=0.01, volume_fraction=0.5, rng=rng
    ).astype(np.uint16)
    return data


@pytest.fixture
def full_organ_scan_folder(
    tmp_path: Path, ground_truth: npt.NDArray[np.float32]
) -> Path:
    full_organ_scan = block_reduce(
        ground_truth, (BIN_FACTOR, BIN_FACTOR, BIN_FACTOR), np.mean
    )
    full_organ_folder = tmp_path / "20.0um_full_organ"
    full_organ_folder.mkdir()
    write_array_to_stack(full_organ_scan, full_organ_folder)
    return full_organ_folder


@pytest.fixture
def full_organ_scan(
    full_organ_scan_folder: Path,
) -> sitk.Image:
    """
    Downsampled ground truth data, to mimic a full organ scan.
    """
    return import_im(full_organ_scan_folder, pixel_size=PIXEL_SIZE_UM * BIN_FACTOR)


@pytest.fixture
def roi_scan_folder(tmp_path: Path, ground_truth: npt.NDArray[np.float32]) -> Path:
    roi_scan = ground_truth[
        ROI_OFFSET : ROI_OFFSET + ROI_SIZE,
        ROI_OFFSET : ROI_OFFSET + ROI_SIZE,
        ROI_OFFSET : ROI_OFFSET + ROI_SIZE,
    ]
    roi_folder = tmp_path / "5.0um_roi"
    roi_folder.mkdir()
    write_array_to_stack(roi_scan, roi_folder)
    return roi_folder


@pytest.fixture
def roi_scan(roi_scan_folder: Path) -> sitk.Image:
    """
    Sub-volume of ground truth data, to mimic ROI data.
    """
    return import_im(roi_scan_folder, pixel_size=PIXEL_SIZE_UM)


@pytest.fixture
def reg_input(roi_scan: sitk.Image, full_organ_scan: sitk.Image) -> RegistrationInput:
    common_point_roi = arr_to_index_tuple(np.array([ROI_SIZE, ROI_SIZE, ROI_SIZE]) / 2)
    common_point_full = arr_to_index_tuple(
        (
            np.array([ROI_OFFSET, ROI_OFFSET, ROI_OFFSET])
            + np.array([ROI_SIZE, ROI_SIZE, ROI_SIZE]) / 2
        )
        / BIN_FACTOR
    )
    return RegistrationInput(
        roi_name="roi_name",
        full_name="full_name",
        roi_image=roi_scan,
        full_image=full_organ_scan,
        common_point_roi=common_point_roi,
        common_point_full=common_point_full,
    )


@pytest.mark.parametrize("zrot", [0, 30])
def test_registration_rot(
    reg_input: RegistrationInput, caplog: pytest.LogCaptureFixture, zrot: float
) -> None:
    """
    Test a really simple registration where the common point given is exactly the
    correct point, and there is no rotation between the two datasets.
    """

    with caplog.at_level(logging.INFO):
        transform, _ = registration_rot(
            reg_input, angle_range=360, angle_step=2, zrot=zrot
        )
        if zrot == 0:
            # Only test log for one zrot case
            expected = r"""INFO Starting rotational registration
INFO Initial rotation = 0 deg
INFO Range = 360 deg
INFO Step = 2 deg
INFO Common point ROI = (32, 32, 32) pix
INFO Common point full = (40, 40, 40) pix
INFO Starting rotational registration...
INFO Registration finished!
INFO Registered rotation angele = 0.0 deg
"""
            assert caplog.text.replace("INFO \n", "") == expected

    # Regardless of initial zrot, we should get the same results as a full
    # 360 deg is sampled each time
    assert isinstance(transform, sitk.Euler3DTransform)
    assert transform.GetAngleX() == 0
    assert transform.GetAngleY() == 0
    # This value should be close to zero
    zrot = np.rad2deg(transform.GetAngleZ())
    assert zrot == pytest.approx(0)

    # Try a smaller angular range at higher angular resolution
    # Also smoke test verbose option for logging at DEBUG level
    transform, _ = registration_rot(
        reg_input,
        zrot=zrot,
        angle_range=5,
        angle_step=0.1,
        verbose=True,
    )

    assert isinstance(transform, sitk.Euler3DTransform)
    assert transform.GetAngleX() == 0
    assert transform.GetAngleY() == 0
    # This value should be close to zero
    zrot = np.rad2deg(transform.GetAngleZ())
    assert zrot == pytest.approx(-0.1)


def test_registration_rigid(
    reg_input: RegistrationInput, caplog: pytest.LogCaptureFixture
) -> None:
    # Rotate the ROI slightly initially to give the registration something to do
    zrot = np.deg2rad(1)
    with caplog.at_level(logging.INFO):
        final_registration = registration_rigid(
            reg_input,
            zrot=zrot,
        )

    # Don't test output lines, as the final values are subject to floating point
    # differences on each run
    expected = r"""INFO Starting full registration...
INFO Common point ROI = (32, 32, 32)
INFO Common point full = (40, 40, 40)
INFO Initial rotation = 0.02 deg
INFO Starting registration...
INFO Registration finished!
"""
    print(caplog.text)
    assert caplog.text.startswith(expected)

    assert isinstance(final_registration, sitk.Similarity3DTransform)
    # Final matrix should be close to the identity matrix
    np.testing.assert_almost_equal(
        np.array(final_registration.GetMatrix()).reshape((3, 3)),
        np.array([[1, 0.02, 0], [-0.02, 1, 0], [0, 0, 1]]),
        decimal=2,
    )


def test_registration_real(full_organ_image: sitk.Image, roi_image: sitk.Image) -> None:
    """
    Test registration with some real data.
    """

    reg_input = RegistrationInput(
        roi_name="roi_name",
        full_name="full_name",
        roi_image=roi_image,
        full_image=full_organ_image,
        common_point_full=get_central_pixel_index(full_organ_image),
        common_point_roi=get_central_pixel_index(roi_image),
    )

    transform = run_registration(reg_input)
    transform_dict = transform_to_dict(transform)
    assert list(transform_dict.keys()) == ["translation", "rotation_matrix", "scale"]

    np.testing.assert_almost_equal(transform_dict["translation"], [3.96, 3.96, 3.96])
    np.testing.assert_almost_equal(
        transform_dict["rotation_matrix"],
        [0.976296, -0.2164396, 0.0, 0.2164396, 0.976296, 0.0, 0.0, 0.0, 1.0],
    )
    np.testing.assert_almost_equal(transform_dict["scale"], 1)
