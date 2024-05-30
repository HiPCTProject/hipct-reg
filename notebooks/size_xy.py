# %% [markdown]
# # Test varying x-y size on registration
#
# This notebook tests the effect of the size of data in the x-y plane on image registration results.

# mypy: ignore-errors
# %%
import json
import logging
import math

import SimpleITK as sitk

from hipct_reg.data import get_reg_input
from hipct_reg.helpers import transform_to_dict
from hipct_reg.registration import run_registration
from hipct_reg.types import RegistrationInput

# %% [markdown]
# # Load data

# %%
roi_name = "A135_lung_VOI-01_2.0um_bm18"
full_name = "A135_lung_complete-organ_10.005um_bm18"
roi_point = (3462, 2447, 4107)
full_point = (4580, 4750, 6822)

logging.basicConfig(level=logging.INFO)
# Get input data
reg_input = get_reg_input(
    roi_name=roi_name,
    roi_point=roi_point,
    full_name=full_name,
    full_point=full_point,
    full_size_xy=64,
)

# %% [markdown]
# # Do the registration

# %%
xyz_roi = reg_input.roi_image.GetSize()
xyz_full = reg_input.full_image.GetSize()


def run_reg(dxy_full: int) -> tuple[RegistrationInput, sitk.Similarity3DTransform]:
    """
    Parameters
    ----------
    dxy_full :
        Amount to reduce the sides of the full image by.
    """
    if dxy_full < 0:
        raise ValueError("dxy_full must be >= 0")

    dxy_roi = math.floor(
        dxy_full
        * reg_input.full_image.GetSpacing()[0]
        / reg_input.roi_image.GetSpacing()[0]
    )

    full_image = sitk.Crop(
        reg_input.full_image, (dxy_full, dxy_full, 0), (dxy_full, dxy_full, 0)
    )
    roi_image = sitk.Crop(
        reg_input.roi_image,
        (dxy_roi, dxy_roi, 0),
        (dxy_roi, dxy_roi, 0),
    )
    common_point_roi = (
        reg_input.common_point_roi[0] - dxy_roi,
        reg_input.common_point_roi[1] - dxy_roi,
        reg_input.common_point_roi[2],
    )
    common_point_full = (
        reg_input.common_point_full[0] - dxy_full,
        reg_input.common_point_full[1] - dxy_full,
        reg_input.common_point_full[2],
    )

    new_reg_input = RegistrationInput(
        roi_name,
        full_name,
        roi_image,
        full_image,
        common_point_roi,
        common_point_full,
    )

    transform = run_registration(new_reg_input)
    transform_dict: dict = dict(transform_to_dict(transform))
    transform_dict["full_dataset"] = full_name
    transform_dict["roi_datset"] = roi_name
    transform_dict["full_size"] = new_reg_input.full_image.GetSize()
    transform_dict["roi_size"] = new_reg_input.roi_image.GetSize()
    with open(f"data/xy/transform_{roi_name}_{full_image.GetSize()[0]}.json", "w") as f:
        f.write(json.dumps(transform_dict, indent=4))

    return new_reg_input, transform


for i in range(20, 54):
    print(i)
    print()
    print()
    try:
        new_reg_input, transform = run_reg(i)
    except RuntimeError:
        continue

# %%
