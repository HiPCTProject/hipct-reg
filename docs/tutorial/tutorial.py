"""
Tutorial
========

This page steps through running the HiP-CT registration pipeline.
"""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

from hipct_reg.data import get_reg_input
from hipct_reg.helpers import (
    get_central_pixel_index,
    show_image,
    transform_to_neuroglancer_dict,
)
from hipct_reg.registration import run_registration

# %%
# Get data
# --------
# Start by defining the datasets, and fetching them.

roi_name = "LADAF-2020-27_heart_ROI-02_6.5um_bm18"
full_name = "LADAF-2020-27_heart_complete-organ_19.89um_bm18"
roi_point = (3478, 2398, 4730)
full_point = (6110, 5025, 4117)

logging.basicConfig(level=logging.INFO)
# Get input data
reg_input = get_reg_input(
    roi_name=roi_name,
    roi_point=roi_point,
    full_name=full_name,
    full_point=full_point,
    full_size_xy=64,
)

# %%
# Run registration
# ----------------
# Do the registration
transform = run_registration(reg_input)


# %%
# Plot results
# ------------

# %%
# Before
fig, axs = plt.subplots(nrows=2, ncols=2, constrained_layout=True, figsize=(10, 10))
for im, ax in zip([reg_input.roi_image, reg_input.full_image], axs[0, :]):
    zmid = get_central_pixel_index(im)[2]
    show_image(im, ax, zmid)

axs[0, 0].set_title("ROI scan (unregistered)")
axs[0, 1].set_title("Full organ scan")

# After
roi_resampled = sitk.Resample(
    reg_input.roi_image,
    reg_input.roi_image.GetSize(),
    outputOrigin=reg_input.full_image.GetOrigin(),
    outputSpacing=reg_input.roi_image.GetSpacing(),
    transform=transform.GetInverse(),
    defaultPixelValue=np.nan,
    interpolator=sitk.sitkNearestNeighbor,
)
for im, ax in zip([roi_resampled, reg_input.full_image], axs[1, :]):
    zmid = get_central_pixel_index(im)[2]
    show_image(im, ax, zmid)
    ax.grid(color="k")

axs[1, 0].set_title("ROI scan (registered)")

# %%
# Save the transform
# ------------------
neuroglancer_dict = dict(transform_to_neuroglancer_dict(transform))
neuroglancer_dict["full_dataset"] = full_name
neuroglancer_dict["roi_datset"] = roi_name

print(neuroglancer_dict)

# Save to a JSON file
with open(Path(__file__).parent / f"transform_{roi_name}.json", "w") as f:
    f.write(json.dumps(neuroglancer_dict, indent=4))

plt.show()
