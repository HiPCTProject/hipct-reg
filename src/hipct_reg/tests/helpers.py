from pathlib import Path

import dask_image
import dask_image.imread
import SimpleITK as sitk


def import_tiff_stack(
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

    img_array_dask = dask_image.imread.imread(f"{path}/*.tif")
    img_array = img_array_dask.compute()

    image = sitk.GetImageFromArray(img_array)
    image.SetOrigin([0, 0, 0])
    image.SetSpacing([pixel_size, pixel_size, pixel_size])

    image = sitk.Cast(image, sitk.sitkFloat32)
    return image
