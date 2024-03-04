import logging

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

from hipct_reg.data import get_reg_input
from hipct_reg.helpers import get_central_pixel_index
from hipct_reg.ITK_registration import registration_rot, registration_sitk

logging.basicConfig(level=logging.INFO)
reg_input = get_reg_input(
    roi_name="LADAF-2020-27_heart_LR-vent-muscles-ramus-interventricularis-anterior_6.05um_bm05",
    roi_point=(2115, 2284, 5179),
    full_name="LADAF-2020-27_heart_complete-organ_25.08um_bm05",
    full_point=(3557, 2171, 4455),
)

transform: sitk.Transform
transform, data_coarse = registration_rot(
    reg_input, zrot=0, angle_range=360, angle_step=2
)
transform, data_fine = registration_rot(
    reg_input, zrot=np.rad2deg(transform.GetAngleZ()), angle_range=5, angle_step=0.1
)
transform = registration_sitk(reg_input, zrot=np.rad2deg(transform.GetAngleZ()))
print(transform)

# Plot registration before/after
fig, axs = plt.subplots(
    nrows=2, ncols=2, constrained_layout=True, sharex=True, sharey=True
)
for im, ax in zip([reg_input.roi_image, reg_input.full_image], axs[0, :]):
    zmid = get_central_pixel_index(im)[2]
    half_size = im.GetSize()[0] * im.GetSpacing()[0] / 2
    ax.imshow(
        sitk.GetArrayViewFromImage(im)[zmid, :, :],
        origin="lower",
        extent=(-half_size, half_size, -half_size, half_size),
    )

axs[0, 0].set_title("ROI scan (unregistered)")
axs[0, 1].set_title("Full organ scan")

roi_resampled = sitk.Resample(
    reg_input.roi_image, transform.GetInverse(), defaultPixelValue=np.nan
)
for im, ax in zip([roi_resampled, reg_input.full_image], axs[1, :]):
    zmid = get_central_pixel_index(im)[2]
    half_size = im.GetSize()[0] * im.GetSpacing()[0] / 2
    ax.imshow(
        sitk.GetArrayViewFromImage(im)[zmid, :, :],
        origin="lower",
        extent=(-half_size, half_size, -half_size, half_size),
    )

axs[1, 0].set_title("ROI scan (registered)")

# Plot metric against rotation angle
fig, ax = plt.subplots(constrained_layout=True)
ax.plot(data_coarse["rotation"], data_coarse["metric"])
ax.plot(data_fine["rotation"], data_fine["metric"])
ax.set_xlabel("Rotation / deg")
ax.set_ylabel("Registration metric")

plt.show()
