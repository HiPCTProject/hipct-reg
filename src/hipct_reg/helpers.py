import glob
from typing import Literal, TypedDict

import dask_image
import dask_image.imread
import numpy as np
import numpy.typing as npt
import SimpleITK as sitk
import skimage.io
import skimage.measure


def import_im(
    path: str,
    pixel_size: float,
    crop_z: tuple[int, int] | None = None,
    bin_factor: int = 1,
) -> sitk.Image:
    """
    Load a file into an ITK image.
    - Reads all files into memory
    - Casts data to float32

    Parameters
    ----------
    path :
        Path to TIFF or JP2 folder.
    pixel_size :
        Pixel size in nm.
    crop_z :
        If given, crop the number of slices in the z-direction.
    bin_factor :
        Downsample the image by a binning factor before returning.

    """

    file_type = test_file_type(path)
    img_array_dask = dask_image.imread.imread(f"{path}/*.{file_type}")

    if crop_z is not None:
        img_array_dask = img_array_dask[crop_z[0] : crop_z[1], :, :]

    img_array = img_array_dask.compute()

    bin_factor = int(bin_factor)
    if bin_factor > 1:
        # Binning
        img_array = skimage.measure.block_reduce(
            img_array, (bin_factor, bin_factor, bin_factor), np.mean
        )

    image = sitk.GetImageFromArray(img_array)
    image.SetOrigin([0, 0, 0])
    image.SetSpacing([pixel_size, pixel_size, pixel_size])

    image = sitk.Cast(image, sitk.sitkFloat32)
    return image


def test_file_type(path: str) -> Literal["tif", "jp2"]:
    N_tif = glob.glob(path + "/*.tif")
    N_jp2 = glob.glob(path + "/*.jp2")

    if len(N_tif) > 0 and len(N_jp2) > 0:
        raise RuntimeError("Error: tif and jp2 files in the folder")
    elif len(N_tif) == 0 and len(N_jp2) == 0:
        raise RuntimeError("Error: no tif or jp2 files in the folder")
    elif len(N_tif) > 0 and len(N_jp2) == 0:
        return "tif"
    else:
        return "jp2"


def arr_to_index_tuple(arr: npt.NDArray) -> tuple[int, int, int]:
    """
    Convert a (3, ) shaped numpy array to an index tuple that simpleitk can use.
    """
    assert arr.shape == (3,)
    return (int(arr[0]), int(arr[1]), int(arr[2]))


class TransformDict(TypedDict):
    translation: tuple[float, float, float]
    rotation_matrix: tuple[
        float, float, float, float, float, float, float, float, float
    ]
    scale: float


def transform_to_dict(transform: sitk.Similarity3DTransform) -> TransformDict:
    """
    Serialise the registered transform to a dict (that can be written to JSON).
    """
    return {
        "translation": transform.TransformPoint([0, 0, 0]),
        "rotation_matrix": transform.GetMatrix(),
        "scale": transform.GetScale(),
    }


def get_pixel_size(path: str) -> float:
    """
    Get pixel size in um from a path.
    """
    if path[-1] == "/":
        path = path[:-1]
    return float(path.split("/")[-1].split("um")[0])
