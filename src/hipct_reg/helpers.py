from pathlib import Path
from typing import Any, Literal, TypedDict, cast

import dask_image
import dask_image.imread
import matplotlib.axes
import numpy as np
import numpy.typing as npt
import SimpleITK as sitk

from hipct_reg.types import RegistrationInput


class PixelTransformDict(TypedDict):
    """
    Parameters for a pixel-to-pixel transform, mapping from a zoom to a overview
    image.
    """

    tx_pix: float
    ty_pix: float
    tz_pix: float
    rotation_deg: float
    scale: float


def get_pixel_transform_params(
    reg_input: RegistrationInput,
    coord_transform: sitk.Similarity3DTransform,
) -> PixelTransformDict:
    """
    Convert the registered transform to a pixel-to-pixel transform mapping from a zoom
    to a overview image.
    """
    translation = coord_transform.TransformPoint((0, 0, 0))

    new_transform = sitk.Similarity3DTransform()
    new_transform.SetParameters(coord_transform.GetParameters())
    new_transform.SetTranslation(translation)

    spacing_overview = reg_input.overview_image.GetSpacing()[0]
    spacing_zoom = reg_input.zoom_image.GetSpacing()[0]

    rotation = np.arctan2(new_transform.GetMatrix()[1], new_transform.GetMatrix()[0])
    scale = new_transform.GetScale() * spacing_zoom / spacing_overview

    transform_dict = PixelTransformDict(
        {
            "tx_pix": new_transform.GetTranslation()[0] / spacing_overview,
            "ty_pix": new_transform.GetTranslation()[1] / spacing_overview,
            "tz_pix": new_transform.GetTranslation()[2] / spacing_overview,
            "rotation_deg": np.rad2deg(rotation),
            "scale": scale,
        }
    )

    return transform_dict


def resample_zoom_image(
    reg_input: RegistrationInput, transform: sitk.Transform
) -> sitk.Image:
    """
    Resample the region of interest image on to the overview image grid using the
    registered transform.

    Parameters
    ----------
    reg_input :
        Registration input (contains the zoom and overview images).
    transform :
        Registered transform.

    Returns
    -------
    zoom_resampled :
        zoom image transformed and resampled on to the same grid as the overview image.
    """
    return sitk.Resample(
        reg_input.zoom_image,
        reg_input.zoom_image.GetSize(),
        outputOrigin=reg_input.overview_image.GetOrigin(),
        outputSpacing=reg_input.zoom_image.GetSpacing(),
        transform=transform.GetInverse(),
        defaultPixelValue=np.nan,
        interpolator=sitk.sitkNearestNeighbor,
    )


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

    im = sitk.GetArrayFromImage(image)[z, :, :]
    lims = np.percentile(im[np.isfinite(im)], [1, 99])
    ax.imshow(
        im,
        extent=(origin[0], top_right[0], origin[1], top_right[1]),
        origin="lower",
        vmin=lims[0],
        vmax=lims[1],
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


def get_central_point(image: sitk.Image) -> tuple[float, float, float]:
    """
    Get the physical point of the pixel at the centre of the image.
    """
    return cast(
        tuple[float, float, float],
        image.TransformIndexToPhysicalPoint(
            arr_to_index_tuple(np.array(image.GetSize()) // 2)
        ),
    )
