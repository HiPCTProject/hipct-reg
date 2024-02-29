"""
Script to demonstrate/test the initial rotational registration.

Uses data downloaded by the download_real_data.py script.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import zarr

from hipct_reg.helpers import arr_to_index_tuple
from hipct_reg.ITK_registration import (
    RegistrationInput,
    registration_rot,
    registration_sitk,
)

logging.basicConfig(level=logging.INFO)


def get_image(name: str, spacing: float) -> sitk.Image:
    """
    Get a pre-downloaded image.
    """
    data = zarr.convenience.open(
        f"/Users/dstansby/software/hipct/hipct-reg/scripts/data/{name}.zarr"
    )
    image = sitk.GetImageFromArray(data[:].T)
    image.SetSpacing((spacing, spacing, spacing))
    image = sitk.Cast(image, sitk.sitkFloat32)
    return image


def get_central_pixel(image: sitk.Image) -> tuple[int, int, int]:
    """
    Get pixel in the centre of an image.
    """
    return arr_to_index_tuple(np.array(image.GetSize()) // 2)


roi = get_image("roi_01", 6.36)
full = get_image("whole_organ", 19.85)

reg_input = RegistrationInput(
    roi_image=roi,
    full_image=full,
    common_point_roi=get_central_pixel(roi),
    common_point_full=get_central_pixel(full),
)

transform, data_coarse = registration_rot(
    reg_input, zrot=0, angle_range=360, angle_step=2
)
transform, data_fine = registration_rot(
    reg_input, zrot=np.rad2deg(transform.GetAngleZ()), angle_range=5, angle_step=0.1
)
transform = registration_sitk(reg_input, zrot=np.rad2deg(transform.GetAngleZ()))

print(transform)

roi_resampled = sitk.Resample(
    reg_input.roi_image, transform.GetInverse(), defaultPixelValue=np.nan
)

# Plot metric against rotation angle
fig, ax = plt.subplots(constrained_layout=True)
ax.plot(data_coarse["rotation"], data_coarse["metric"])
ax.plot(data_fine["rotation"], data_fine["metric"])
ax.set_xlabel("Rotation / deg")
ax.set_ylabel("Registration metric")

# Plot registration before/after
fig, axs = plt.subplots(nrows=2, ncols=2, constrained_layout=True)
for im, ax in zip([roi, full], axs[0, :]):
    zmid = get_central_pixel(im)[2]
    ax.imshow(sitk.GetArrayViewFromImage(im)[zmid, :, :], origin="lower")

axs[0, 0].set_title("ROI scan (unregistered)")
axs[0, 1].set_title("Full organ scan")

for im, ax in zip([roi_resampled, full], axs[1, :]):
    zmid = get_central_pixel(im)[2]
    ax.imshow(sitk.GetArrayViewFromImage(im)[zmid, :, :], origin="lower")

axs[1, 0].set_title("ROI scan (registered)")

plt.show()
