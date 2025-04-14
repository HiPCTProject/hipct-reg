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
zoom_name = "A135_lung_VOI-01_2.0um_bm18"
overview_name = "A135_lung_complete-organ_10.005um_bm18"
zoom_point = (3462, 2447, 4107)
overview_point = (4580, 4750, 6822)

logging.basicConfig(level=logging.INFO)
# Get input data
reg_input = get_reg_input(
    zoom_name=zoom_name,
    zoom_point=zoom_point,
    overview_name=overview_name,
    overview_point=overview_point,
    overview_size_xy=64,
)

# %% [markdown]
# # Do the registration

# %%
xyz_zoom = reg_input.zoom_image.GetSize()
xyz_overview = reg_input.overview_image.GetSize()


def run_reg(dxy_full: int) -> tuple[RegistrationInput, sitk.Similarity3DTransform]:
    """
    Parameters
    ----------
    dxy_full :
        Amount to reduce the sides of the full image by.
    """
    if dxy_full < 0:
        raise ValueError("dxy_full must be >= 0")

    dxy_zoom = math.floor(
        dxy_full
        * reg_input.overview_image.GetSpacing()[0]
        / reg_input.zoom_image.GetSpacing()[0]
    )

    full_image = sitk.Crop(
        reg_input.overview_image, (dxy_full, dxy_full, 0), (dxy_full, dxy_full, 0)
    )
    zoom_image = sitk.Crop(
        reg_input.zoom_image,
        (dxy_zoom, dxy_zoom, 0),
        (dxy_zoom, dxy_zoom, 0),
    )
    common_point_zoom = (
        reg_input.zoom_common_point[0] - dxy_zoom,
        reg_input.zoom_common_point[1] - dxy_zoom,
        reg_input.zoom_common_point[2],
    )
    common_point_overview = (
        reg_input.overview_common_point[0] - dxy_full,
        reg_input.overview_common_point[1] - dxy_full,
        reg_input.overview_common_point[2],
    )

    new_reg_input = RegistrationInput(
        zoom_name,
        overview_name,
        zoom_image,
        full_image,
        common_point_zoom,
        common_point_overview,
    )

    transform = run_registration(new_reg_input)
    transform_dict: dict = dict(transform_to_dict(transform))
    transform_dict["overview_dataset"] = overview_name
    transform_dict["zoom_datset"] = zoom_name
    transform_dict["overview_size"] = new_reg_input.overview_image.GetSize()
    transform_dict["zoom_size"] = new_reg_input.zoom_image.GetSize()
    with open(
        f"data/xy/transform_{zoom_name}_{full_image.GetSize()[0]}.json", "w"
    ) as f:
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
