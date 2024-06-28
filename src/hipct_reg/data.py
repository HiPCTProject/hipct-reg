"""
Helper functions for downloading/managing data locally.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import hoa_tools
import hoa_tools.inventory
import numpy as np
import numpy.typing as npt
import SimpleITK as sitk
import tensorstore as ts
import zarr.convenience
import zarr.core
from hoa_tools.dataset import Dataset, get_dataset
from hoa_tools.inventory import load_inventory

from hipct_reg.types import RegistrationInput

STORAGE_DIR: Path | None = None

hoa_tools.inventory.INVENTORY_PATH = Path.home() / "hoa_inventory.csv"

inventory = load_inventory()
datasets = {name: get_dataset(name) for name in inventory.index}


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

    ds: Dataset
    centre_point: tuple[int, int, int]
    size_xy: int
    size_z: int

    @property
    def local_zarr_path(self) -> Path:
        """
        Path to cube saved to a local zarr store.
        """
        if STORAGE_DIR is None:
            raise RuntimeError(
                "The hipct_reg.data.STORAGE_DIR variable must be set to a data download folder first."
            )
        return (
            STORAGE_DIR
            / f"{self.ds.name}_{'_'.join([str(i) for i in self.centre_point])}_{self.size_xy}_{self.size_z}.zarr"
        )

    def get_image(self) -> sitk.Image:
        """
        Get cube as a SimpleITK image.
        """
        spacing = self.ds.resolution.to_value("um")
        origin = [
            (self.lower_idx[0]) * spacing,
            (self.lower_idx[1]) * spacing,
            (self.lower_idx[2]) * spacing,
        ]

        image = sitk.GetImageFromArray(self.get_array().T)
        image.SetSpacing((spacing, spacing, spacing))
        image.SetOrigin(origin)
        image = sitk.Cast(image, sitk.sitkFloat32)
        return image

    def get_array(self) -> npt.NDArray[np.uint16]:
        """
        Get cube as a numpy array.
        """
        if not self.local_zarr_path.exists():
            self.download_cube()
        logging.info(f"Reading {self.local_zarr_path}")
        data_local = ts.open({
            "driver": "zarr",
            "dtype": "uint16",
            "kvstore": {
                "driver": "file",
                "path": str(self.local_zarr_path)
            }
        }).result()
        print(data_local.shape)
        return data_local[:].read().result()

    @property
    def lower_idx(self) -> tuple[int, int, int]:
        return (
            max(0, self.centre_point[0] - self.size_xy),
            max(0, self.centre_point[1] - self.size_xy),
            max(0, self.centre_point[2] - self.size_z),
        )

    @property
    def upper_idx(self) -> tuple[int, int, int]:
        remote_array = self.ds.remote_array(level=0)
        shape = remote_array.shape[::-1]

        for i in range(3):
            if self.centre_point[i] > shape[i]:
                raise RuntimeError(
                    f"Centre point ({self.centre_point[i]}) is outside array bounds ({shape[i]}) in dimension {i} for dataset {self.ds.name}"
                )

        return (
            min(self.centre_point[0] + self.size_xy, shape[0]),
            min(self.centre_point[1] + self.size_xy, shape[1]),
            min(self.centre_point[2] + self.size_z, shape[2]),
        )

    def download_cube(self) -> None:
        """
        Download the cube from GCS.
        """
        logging.info(
            f"Downloading cube for {self.ds.name}, {self.lower_idx} --> {self.upper_idx}"
        )
        path = f"/{self.ds.donor}/{self.ds.organ}"
        if self.ds.organ_context:
            path += f"-{self.ds.organ_context}"
        path += (
            f"/{self.ds.resolution.to_value('um')}um_{self.ds.roi}_{self.ds.beamline}"
        )
        path = f"gs://ucl-hip-ct-35a68e99feaae8932b1d44da0358940b{path}/s0"

        dataset_remote = ts.open(
            {
                "driver": "n5",
                "kvstore": path,
            }
        ).result()
        delayed = dataset_remote[
            self.lower_idx[2] : self.upper_idx[2],
            self.lower_idx[1] : self.upper_idx[1],
            self.lower_idx[0] : self.upper_idx[0],
        ].T
        data_local = ts.open(
            {
                "driver": "zarr",
                "create": True,
                "delete_existing": True,
                "dtype": "uint16",
                "metadata": {"shape": delayed.shape},
                "kvstore": {"driver": "file", "path": str(self.local_zarr_path)},
            }
        ).result()

        data_local[:].write(delayed).result()

    def get_remote_arr(self) -> zarr.core.Array:
        """
        Get remote GCS store for the cube.
        """
        return self.ds.remote_array(level=0)


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
        - a ``(full_size_xy * 2), (full_size_xy * 2), 32`` shaped cubiod of it's parent
          full-organ scan.
        - the equivalent (larger) cube of the ROI itself.

    The size of the full-organ scan cube can be changed.

    Data is cached on disk to ~/hipct/reg_data so it doesn't need to be re-downloaded.
    """
    full_dataset = datasets[full_name]
    assert (
        full_dataset.is_full_organ
    ), "Full dataset name given is not a full organ dataset"
    full_size_z = 16
    full_cube = Cuboid(
        full_dataset, full_point, size_xy=full_size_xy, size_z=full_size_z
    )

    roi_dataset = datasets[roi_name]

    full_resolution_um = full_dataset.resolution.to_value("um")
    roi_resolution_um = roi_dataset.resolution.to_value("um")

    assert not roi_dataset.is_full_organ, "ROI dataset name given is a ROI dataset"
    roi_size_xy = int(full_size_xy * full_resolution_um / roi_resolution_um)
    roi_size_z = int(full_size_z * full_resolution_um / roi_resolution_um)
    roi_cube = Cuboid(roi_dataset, roi_point, size_xy=roi_size_xy, size_z=roi_size_z)

    return RegistrationInput(
        roi_name=roi_name,
        full_name=full_name,
        roi_image=roi_cube.get_image(),
        full_image=full_cube.get_image(),
        common_point_full=(full_size_xy, full_size_xy, full_size_z),
        common_point_roi=(roi_size_xy, roi_size_xy, roi_size_z),
    )
