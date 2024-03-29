"""
Helper functions for downloading/managing data locally.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy.typing as npt
import SimpleITK as sitk
import tensorstore as ts
import zarr
from hipct_data_tools import load_datasets
from hipct_data_tools.data_model import HiPCTDataSet

from hipct_reg.types import RegistrationInput

STORAGE_DIR = Path.home() / "hipct" / "reg_data"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

datasets = {d.name: d for d in load_datasets()}


@dataclass
class Cuboid:
    """
    Represents a small cuboid of data located within a full dataset.

    Attributes
    ----------
    ds :
        Dataset.
    centre_point :
        Index of the centre of the cube.
    size_xy :
        Half the size of an edge of cuboid in the x-y plane.
    size_z :
        Half the thickness of the cuboid along the z-axis.
    """

    ds: HiPCTDataSet
    centre_point: tuple[int, int, int]
    size_xy: int
    size_z: int

    @property
    def local_zarr_path(self) -> Path:
        """
        Path to cube saved to a local zarr store.
        """
        return (
            STORAGE_DIR
            / f"{self.ds.name}_{'_'.join([str(i) for i in self.centre_point])}_{self.size_xy}_{self.size_z}.zarr"
        )

    def get_image(self) -> sitk.Image:
        """
        Get cube as a SimpleITK image.
        """
        spacing = self.ds.resolution_um
        origin = [
            (self.centre_point[0] - self.size_xy) * spacing,
            (self.centre_point[1] - self.size_xy) * spacing,
            (self.centre_point[2] - self.size_z) * spacing,
        ]

        image = sitk.GetImageFromArray(self.get_array().T)
        image.SetSpacing((spacing, spacing, spacing))
        image.SetOrigin(origin)
        image = sitk.Cast(image, sitk.sitkFloat32)
        return image

    def get_array(self) -> npt.NDArray:
        """
        Get cube as a numpy array.
        """
        if not self.local_zarr_path.exists():
            self.download_cube()

        return zarr.load(self.local_zarr_path)[:]

    def download_cube(self) -> None:
        """
        Download the cube from GCS.
        """
        gcs_store = self.get_gcs_store()
        data = (
            gcs_store[
                self.centre_point[0] - self.size_xy : self.centre_point[0]
                + self.size_xy,
                self.centre_point[1] - self.size_xy : self.centre_point[1]
                + self.size_xy,
                self.centre_point[2] - self.size_z : self.centre_point[2] + self.size_z,
            ]
            .read()
            .result()
        )
        zarr.convenience.save(self.local_zarr_path, data)

    def get_gcs_store(self) -> Any:
        """
        Get remote GCS store for the cube.
        """
        return ts.open(
            {
                "driver": "n5",
                "kvstore": f"gs://{self.ds.gcs_bucket}/{self.ds.gcs_path}s0",
            }
        ).result()


def get_reg_input(
    *,
    roi_name: str,
    full_name: str,
    roi_point: tuple[int, int, int],
    full_point: tuple[int, int, int],
    full_size_xy: int = 64,
) -> RegistrationInput:
    """
    Given the dataset of a ROI scan, get:
    - a (full_size_xy x 2) x (full_size_xy x 2) x 32 cubiod of it's parent full-organ scan
    - the equivalent (larger) cube of the ROI itself.

    The size of the full-organ scan cube can be changed.

    Data is cahed on disk to ~/hipct/reg_data so it doesn't need to be re-downloaded.
    """
    full_dataset = datasets[full_name]
    assert (
        full_dataset.is_complete_organ
    ), "Full dataset name given is not a full organ dataset"
    full_size_z = 16
    full_cube = Cuboid(
        full_dataset, full_point, size_xy=full_size_xy, size_z=full_size_z
    )

    roi_dataset = datasets[roi_name]
    assert not roi_dataset.is_complete_organ, "ROI dataset name given is a ROI dataset"
    roi_size_xy = int(
        full_size_xy * full_dataset.resolution_um / roi_dataset.resolution_um
    )
    roi_size_z = int(
        full_size_z * full_dataset.resolution_um / roi_dataset.resolution_um
    )
    roi_cube = Cuboid(roi_dataset, roi_point, size_xy=roi_size_xy, size_z=roi_size_z)

    return RegistrationInput(
        roi_image=roi_cube.get_image(),
        full_image=full_cube.get_image(),
        common_point_full=(full_size_xy, full_size_xy, full_size_z),
        common_point_roi=(roi_size_xy, roi_size_xy, roi_size_z),
    )
