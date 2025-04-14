from pathlib import Path

import pooch
import pytest
import SimpleITK as sitk
import zarr.convenience

DOI = "doi:10.5281/zenodo.10778257"


@pytest.fixture
def overview_image() -> sitk.Image:
    file_paths = pooch.retrieve(
        # URL to one of Pooch's test files
        url=f"{DOI}/LADAF-2020-27_heart_complete-organ_25.08um_bm05_3557_2171_4455_32.zarr.zip",
        known_hash=None,
        processor=pooch.Unzip(),
    )
    arr = zarr.convenience.load(Path(file_paths[0]).parent)[:]

    spacing = 25.08
    image = sitk.GetImageFromArray(arr.T)
    image.SetSpacing((spacing, spacing, spacing))
    image.SetOrigin((3557, 2171, 4455))
    image = sitk.Cast(image, sitk.sitkFloat32)
    return image


@pytest.fixture
def zoom_image() -> sitk.Image:
    file_paths = pooch.retrieve(
        # URL to one of Pooch's test files
        url=f"{DOI}/LADAF-2020-27_heart_LR-vent-muscles-ramus-interventricularis-anterior_6.05um_bm05_2115_2284_5179_132.zarr.zip",
        known_hash=None,
        processor=pooch.Unzip(),
    )
    arr = zarr.convenience.load(Path(file_paths[0]).parent)[:]

    spacing = 6.05
    image = sitk.GetImageFromArray(arr.T)
    image.SetSpacing((spacing, spacing, spacing))
    image.SetOrigin((2115, 2284, 5179))
    image = sitk.Cast(image, sitk.sitkFloat32)
    return image
