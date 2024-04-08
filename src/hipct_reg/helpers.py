from pathlib import Path
from typing import Any, Literal, TypedDict

import dask_image
import dask_image.imread
import matplotlib.axes
import numpy as np
import numpy.typing as npt
import SimpleITK as sitk


class TransformDict(TypedDict):
    translation: tuple[float, float, float]
    rotation_matrix: tuple[
        float, float, float, float, float, float, float, float, float
    ]
    scale: float


def transform_to_neuroglancer_dict(
    transform: sitk.Similarity3DTransform,
) -> TransformDict:
    """
    Convert a registered transform to a neuroglancer dict.

    Parameters
    ----------
    transform :
        Registered transform.
    """
    # For neuroglancer, the translation needs to be at the corner of the image
    translation = transform.TransformPoint((0, 0, 0))

    new_transform = sitk.Similarity3DTransform()
    new_transform.SetParameters(transform.GetParameters())
    new_transform.SetTranslation(translation)
    transform_dict = transform_to_dict(new_transform)

    return transform_dict


def import_im(
    path: Path,
    pixel_size: float,
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
    img_array = img_array_dask.compute()

    image = sitk.GetImageFromArray(img_array)
    image.SetOrigin([0, 0, 0])
    image.SetSpacing([pixel_size, pixel_size, pixel_size])

    image = sitk.Cast(image, sitk.sitkFloat32)
    return image


def show_image(
    image: sitk.Image, ax: matplotlib.axes.Axes, z: int, **imshow_kwargs: Any
) -> None:
    """
    Show a 2D SimpleITK image in a Matplotlib figure.

    Parameters
    ----------
    image :
        Image to show.
    ax :
        Axes to show it on.
    z :
        z-index at which to slice the image. A 2D x-y plane is displayed
        at this index.
    imshow_kwargs :
        Any additional keyword arguments are handed to ``imshow``.
    """
    origin = np.array(image.GetOrigin())
    top_right = np.array(image.TransformIndexToPhysicalPoint(image.GetSize()))

    ax.imshow(
        sitk.GetArrayFromImage(image)[z, :, :],
        extent=(origin[0], top_right[0], origin[1], top_right[1]),
        origin="lower",
        **imshow_kwargs,
    )


def test_file_type(path: Path) -> Literal["tif", "jp2"]:
    N_tif = len(list(path.glob("*.tif")))
    N_jp2 = len(list(path.glob("*.jp2")))

    if (N_tif) > 0 and (N_jp2) > 0:
        raise RuntimeError("Error: tif and jp2 files in the folder")
    elif (N_tif) == 0 and (N_jp2) == 0:
        raise RuntimeError("Error: no tif or jp2 files in the folder")
    elif N_tif > 0 and (N_jp2) == 0:
        return "tif"
    else:
        return "jp2"


def arr_to_index_tuple(arr: npt.NDArray) -> tuple[int, int, int]:
    """
    Convert a (3, ) shaped numpy array to an index tuple that simpleitk can use.
    """
    assert arr.shape == (3,)
    return (int(arr[0]), int(arr[1]), int(arr[2]))


def transform_to_dict(transform: sitk.Similarity3DTransform) -> TransformDict:
    """
    Serialise the registered transform to a dict (that can be written to JSON).
    """
    return {
        "translation": transform.GetTranslation(),
        "rotation_matrix": transform.GetMatrix(),
        "scale": transform.GetScale(),
    }


def get_pixel_size(path: Path) -> float:
    """
    Get pixel size in um from a path.
    """
    return float(path.name.split("um")[0])


def get_central_pixel_index(image: sitk.Image) -> tuple[int, int, int]:
    """
    Get index of pixel in the centre of an image.
    """
    return arr_to_index_tuple(np.array(image.GetSize()) // 2)
