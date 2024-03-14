import json
import logging
from pathlib import Path

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

from hipct_reg.data import get_reg_input
from hipct_reg.helpers import get_central_pixel_index, transform_to_dict
from hipct_reg.registration import registration_rot, registration_sitk

roi_name = (
    "LADAF-2020-27_heart_LR-vent-muscles-ramus-interventricularis-anterior_6.05um_bm05"
)
full_name = "LADAF-2020-27_heart_complete-organ_25.08um_bm05"
logging.basicConfig(level=logging.INFO)
# Get input data
reg_input = get_reg_input(
    roi_name=roi_name,
    roi_point=(2115, 2284, 5179),
    full_name=full_name,
    full_point=(3557, 2171, 4455),
    full_size=64,
)

# Do the registration
transform: sitk.Transform
transform, data_coarse = registration_rot(
    reg_input, zrot=0, angle_range=360, angle_step=2
)
transform, data_fine = registration_rot(
    reg_input, zrot=np.rad2deg(transform.GetAngleZ()), angle_range=5, angle_step=0.1
)
# Transform maps from ROI image to the full-organ image.
transform = registration_sitk(reg_input, zrot=np.rad2deg(transform.GetAngleZ()))


# Plot registration before/after
def show_image(image: sitk.Image, ax: matplotlib.axes.Axes, z: int) -> None:
    """
    Function to show a SimpleITK image in a Matplotlib figure.

    Parameters
    ----------
    image :
        Image to show.
    ax :
        Axes to show it on.
    z :
        z-index at which to slice the image. A 2D x-y plane is displayed
        at this index.
    """
    scale = image.GetSpacing()[0]
    origin = np.array(image.GetOrigin()) / scale
    top_right = np.array(image.TransformIndexToPhysicalPoint(image.GetSize())) / scale

    ax.imshow(
        sitk.GetArrayFromImage(image)[z, :, :],
        extent=(origin[0], top_right[0], origin[1], top_right[1]),
        origin="lower",
    )


# Before
fig, axs = plt.subplots(
    nrows=2,
    ncols=2,
    constrained_layout=True,
)
for im, ax in zip([reg_input.roi_image, reg_input.full_image], axs[0, :]):
    zmid = get_central_pixel_index(im)[2]
    show_image(im, ax, zmid)

axs[0, 0].set_title("ROI scan (unregistered)")
axs[0, 1].set_title("Full organ scan")

# After
roi_resampled = sitk.Resample(
    reg_input.roi_image,
    reg_input.full_image,
    transform.GetInverse(),
    defaultPixelValue=np.nan,
    interpolator=sitk.sitkNearestNeighbor,
)
for im, ax in zip([roi_resampled, reg_input.full_image], axs[1, :]):
    zmid = get_central_pixel_index(im)[2]
    show_image(im, ax, zmid)

axs[1, 0].set_title("ROI scan (registered)")

# Plot metric against rotation angle
fig, ax = plt.subplots(constrained_layout=True)
ax.plot(data_coarse["rotation"], data_coarse["metric"])
ax.plot(data_fine["rotation"], data_fine["metric"])
ax.set_xlabel("Rotation / deg")
ax.set_ylabel("Registration metric")

# Print and save the transform
print(transform)
# For neuroglancer, the translation needs to be at the corner of the image
translation = transform.TransformPoint((0, 0, 0))

print(f"Old translation = {transform.GetTranslation()} um")
print(f"New translation = {translation} um")
print(
    f"New translation = {np.array(translation) / reg_input.full_image.GetSpacing()[0]} pix"
)

new_transform = sitk.Similarity3DTransform()
new_transform.SetParameters(transform.GetParameters())
new_transform.SetTranslation(translation)
transform_dict: dict = transform_to_dict(new_transform)  # type: ignore[assignment]
transform_dict["full_dataset"] = full_name
transform_dict["roi_datset"] = roi_name

with open(Path(__file__).parent / f"transform_{roi_name}.json", "w") as f:
    f.write(json.dumps(transform_dict, indent=4))

plt.show()
