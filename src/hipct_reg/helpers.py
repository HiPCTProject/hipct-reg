import glob
import sys

import dask_image
import dask_image.imread
import numpy as np
import SimpleITK as sitk
import skimage.io
import skimage.measure

MAX_THREADS = 0  # 0 if all


def import_im(
    path: str,
    pixel_size: float,
    crop_z: tuple[int, int] | None = None,
    bin_factor: int = 1,
) -> sitk.Image:
    """
    Load a file into an ITK image.

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
    bin_factor = int(bin_factor)
    pixelType = sitk.sitkInt16
    pixelType = sitk.sitkFloat32

    file_type = test_file_type(path)
    print(f"File type is .{file_type}")

    img_array_dask = dask_image.imread.imread(f"{path}/*.{file_type}")
    print(img_array_dask)

    if crop_z is not None:
        print(f"crop_z = {crop_z}")
        img_array = img_array_dask[crop_z[0] : crop_z[1], :, :].compute()
    else:
        img_array = img_array_dask[:, :, :].compute()

    print(img_array.shape)

    if bin_factor > 1:
        print("\nBinning started\n")
        print(f"Bin factor = {bin_factor}")
        # Binning
        img_array = skimage.measure.block_reduce(
            img_array, (bin_factor, bin_factor, bin_factor), np.mean
        )

    print(img_array.shape)
    print(img_array.dtype)
    image = sitk.GetImageFromArray(img_array)
    del img_array
    image = sitk.Cast(image, pixelType)

    image.SetOrigin([0, 0, 0])
    image.SetSpacing([pixel_size, pixel_size, pixel_size])

    return image


def test_file_type(path):
    N_tif = glob.glob(path + "/*.tif")
    N_jp2 = glob.glob(path + "/*.jp2")

    if len(N_tif) > 0 and len(N_jp2) > 0:
        sys.exit("Error: tif and jp2 files in the folder")
    elif len(N_tif) == 0 and len(N_jp2) == 0:
        sys.exit("Error: no tif or jp2 files in the folder")
    elif len(N_tif) > 0 and len(N_jp2) == 0:
        file_type = "tif"
    elif len(N_jp2) > 0 and len(N_tif) == 0:
        file_type = "jp2"
    return file_type
