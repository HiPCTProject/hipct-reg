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
import zarr
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
    Represents a small cuboid of data located within an image.

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
    downsample_level: int

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
        spacing = self.ds.resolution.to_value("um") * 2**self.downsample_level
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

        return zarr.load(self.local_zarr_path)[:]

    @property
    def lower_idx(self) -> tuple[int, int, int]:
        return (
            max(0, self.centre_point[0] // 2**self.downsample_level - self.size_xy),
            max(0, self.centre_point[1] // 2**self.downsample_level - self.size_xy),
            max(0, self.centre_point[2] // 2**self.downsample_level - self.size_z),
        )

    @property
    def upper_idx(self) -> tuple[int, int, int]:
        remote_array = self.get_remote_arr()

        if self.ds.gcs_format == "n5":
            shape = remote_array.shape[::-1]
        else:
            shape = remote_array.shape

        for i in range(3):
            if self.centre_point[i] // 2**self.downsample_level > shape[i]:
                raise RuntimeError(
                    f"Centre point ({self.centre_point[i]//2**self.downsample_level}) is outside array bounds ({shape[i]}) in dimension {i} for dataset {self.ds.name}"
                )

        return (
            min(
                self.centre_point[0] // 2**self.downsample_level + self.size_xy,
                shape[0],
            ),
            min(
                self.centre_point[1] // 2**self.downsample_level + self.size_xy,
                shape[1],
            ),
            min(
                self.centre_point[2] // 2**self.downsample_level + self.size_z, shape[2]
            ),
        )

    def download_cube(self) -> None:
        """
        Download the cube from GCS.
        """
        logging.info(
            f"Downloading cube for {self.ds.name}, {self.lower_idx} --> {self.upper_idx}"
        )
        remote_arr = self.get_remote_arr()
        if self.ds.gcs_format == "n5":
            data = remote_arr[
                self.lower_idx[2] : self.upper_idx[2],
                self.lower_idx[1] : self.upper_idx[1],
                self.lower_idx[0] : self.upper_idx[0],
            ].T
        else:
            data = remote_arr[
                self.lower_idx[0] : self.upper_idx[0],
                self.lower_idx[1] : self.upper_idx[1],
                self.lower_idx[2] : self.upper_idx[2],
            ]

        zarr.save(self.local_zarr_path, data)

    def get_remote_arr(self) -> zarr.Array:
        """
        Get remote GCS store for the cube.
        """
        return self.ds.remote_array(level=self.downsample_level)


def get_reg_input(
    *,
    zoom_name: str,
    overview_name: str,
    zoom_point: tuple[int, int, int],
    overview_point: tuple[int, int, int],
    downsample_level: int,
    overview_size_xy: int = 64,
) -> RegistrationInput:
    """
    Given the dataset of a zoom image, get:
        - a ``(overview_size_xy * 2), (overview_size_xy * 2), 32`` shaped cubiod of it's parent
          overview.
        - the equivalent (larger) cube of the zoom itself.

    The size of the overview image cube can be changed.

    Data is cached on disk to ~/hipct/reg_data so it doesn't need to be re-downloaded.
    """
    overview_dataset = datasets[overview_name]
    assert overview_dataset.is_full_organ, (
        "Overview dataset name given is not an overview dataset"
    )
    overview_size_z = 16
    overview_cube = Cuboid(
        overview_dataset,
        overview_point,
        size_xy=overview_size_xy,
        size_z=overview_size_z,
        downsample_level=downsample_level,
    )

    zoom_dataset = datasets[zoom_name]

    overview_resolution_um = overview_dataset.resolution.to_value("um")
    zoom_resolution_um = zoom_dataset.resolution.to_value("um")

    assert not zoom_dataset.is_full_organ, (
        "zoom dataset name given is a overview dataset"
    )
    zoom_size_xy = int(overview_size_xy * overview_resolution_um / zoom_resolution_um)
    zoom_size_z = int(overview_size_z * overview_resolution_um / zoom_resolution_um)
    zoom_cube = Cuboid(
        zoom_dataset,
        zoom_point,
        size_xy=zoom_size_xy,
        size_z=zoom_size_z,
        downsample_level=downsample_level,
    )

    return RegistrationInput(
        zoom_name=zoom_name,
        overview_name=overview_name,
        zoom_image=zoom_cube.get_image(),
        overview_image=overview_cube.get_image(),
        overview_common_point=(overview_size_xy, overview_size_xy, overview_size_z),
        zoom_common_point=(zoom_size_xy, zoom_size_xy, zoom_size_z),
    )
