"""
Tutorial
========

This page steps through running the HiP-CT registration pipeline.
"""

import logging

import matplotlib.pyplot as plt

from hipct_reg.data import get_reg_input
from hipct_reg.helpers import (
    get_central_pixel_index,
    get_central_point,
    get_pixel_transform_params,
    resample_roi_image,
    show_image,
)
from hipct_reg.registration import run_registration

logging.basicConfig(level=logging.INFO)

# %%
# Get data
# --------
# The registration pipeline makes use of simpleITK, both for the actual registration
# and for inputting the images. Using it for image input allows a smaller volume than
# the whole image volume to be used to register, while keeping track of its offset from
# the origin of the original (full) data volume.
#
# hipct-reg takes a `RegistrationInput` object as input to the registration pipeline.
# This class stores the two images, the (pixel) indices of a common point in them, and
# their names.
#
# `get_reg_input` provides an easy way to get the registration input for two datasets:

roi_name = "LADAF-2021-64_heart_VOI-01_6.51um_bm18"
full_name = "LADAF-2021-64_heart_complete-organ_19.89um_bm18"
roi_point = (2240, 1770, 1546)
full_point = (4040, 5136, 502)

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
# This is all the input we need - we can just pass this to `run_registration` to run
# the full registration pipeline:
transform, reg_metric = run_registration(reg_input)

# %%
# This returns two objects - the registered transform, and the registration metric.
# The transform maps from the physical coordinates of the ROI image to the physical
# coordinates of the full-organ image. For HiP-CT data these are just the voxel
# coordinates multiplied by the image resolution.
#
# To get the parameters for a pixel-to-pixel mapping from the ROI image to the
# full-organ image we can use `get_pixel_transform_params`:

pixel_transform_params = get_pixel_transform_params(reg_input, transform)
print(pixel_transform_params)

# %%
# Plot results
# ------------

# Get physical point in centre of full organ image
central_point = get_central_point(reg_input.full_image)

fig, axs = plt.subplots(nrows=1, ncols=3, constrained_layout=True, figsize=(10, 5))

# Before
central_idx = get_central_pixel_index(reg_input.roi_image)
show_image(reg_input.roi_image, axs[0], central_idx[2])
axs[0].set_title("ROI before registration")

# After
roi_resampled = resample_roi_image(reg_input, transform)
resampled_idx = roi_resampled.TransformPhysicalPointToIndex(central_point)
show_image(roi_resampled, axs[1], resampled_idx[2])
axs[1].set_title("ROI after registration")
axs[1].grid(color="black")

# Full-organ image
central_idx = get_central_pixel_index(reg_input.full_image)
show_image(reg_input.full_image, axs[2], central_idx[2])
axs[2].set_title("Full-organ")
axs[2].grid(color="black")


plt.show()
