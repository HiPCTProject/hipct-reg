# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Plot results of varying size in x-y plane
#
# See `size_xy.py` for source code to run the registration and save the results, which are plotted here.

# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %% [markdown]
# ## Load data

# %%
all_datasets = {}
datasets = [
    "LADAF-2020-27_kidney_left_central-column_6.05um_bm05",
    "LADAF-2020-27_heart_ROI-02_6.5um_bm18",
    "A135_lung_VOI-01_2.0um_bm18",
]

for dataset in datasets:
    reg_files = Path("data/xy").glob(f"transform_{dataset}_*.json")
    full_res = 6.5

    all_data_json = []
    for file in reg_files:
        with file.open() as f:
            all_data_json.append(json.loads(f.read()))

    all_data = pd.DataFrame(all_data_json)

    def get_rot(df: pd.DataFrame) -> float:
        return float(np.rad2deg(np.arccos(df["rotation_matrix"][0] / df["scale"])))

    all_data["full_size"] = all_data["full_size"].map(lambda x: x[0])
    all_data["roi_size"] = all_data["roi_size"].map(lambda x: x[0])
    all_data["tx"] = all_data["translation"].map(lambda x: x[0] / full_res)
    all_data["ty"] = all_data["translation"].map(lambda x: x[1] / full_res)
    all_data["tz"] = all_data["translation"].map(lambda x: x[2] / full_res)
    all_data["rotation"] = all_data.apply(get_rot, axis=1)
    all_data = all_data.sort_values("full_size")

    all_datasets[dataset] = all_data

# %% [markdown]
# ## Plot data

# %%
kwargs = {
    "marker": "o",
    "markersize": 3,
}

fig, axs = plt.subplots(nrows=3, ncols=2, constrained_layout=True, figsize=(9, 7))

for dataset in datasets:
    all_data = all_datasets[dataset]
    ax = axs[0, 0]
    ax.set_title("Translation (x)")
    ax.plot(
        all_data["full_size"],
        all_data["tx"] - all_data["tx"].iloc[-1],
        label=dataset.split("_")[1],
        **kwargs,
    )
    ax.set_ylabel("ROI pix")
    ax.legend()

    ax = axs[1, 0]
    ax.set_title("Translation (y)")
    ax.plot(
        all_data["full_size"],
        all_data["ty"] - all_data["ty"].iloc[-1],
        **kwargs,
    )
    ax.set_ylabel("ROI pix")

    ax = axs[2, 0]
    ax.set_title("Translation (z)")
    ax.plot(
        all_data["full_size"],
        all_data["tz"] - all_data["tz"].iloc[-1],
        **kwargs,
    )
    ax.set_ylabel("ROI pix")
    ax.set_xlabel("Full organ x-y size")

    ax = axs[0, 1]
    ax.set_title("Rotation")
    ax.plot(
        all_data["full_size"],
        all_data["rotation"] - all_data["rotation"].iloc[-1],
        **kwargs,
    )
    ax.set_ylabel("deg")
    ax.set_ylim(-0.5, 0.5)

    ax = axs[1, 1]
    ax.set_title("Scale")
    ax.plot(all_data["full_size"], all_data["scale"] - 1, **kwargs)

for ax in axs.ravel():
    ax.xaxis.grid(color="k", linewidth=0.5, alpha=0.5)
    ax.yaxis.grid(color="k", linewidth=0.5, alpha=0.5)
axs[2, 1].axis("off")

# %% [markdown]
# ## Context
# The goal of this experiement is to verify the variation in registration parameters (rotation, translation, scale) as the size of the cuboids being registered increases. This should show at what point the registration converges (ie at what point adding more pixels isn't adding anything to the registration).
#
# ## Figure overview
# Each registration parameter (y-axis) is plotted against the size of the full-organ cuboid in the x-y plane. All cuboids have a thickness along the z-axis of 32 pixels. The high-resolution ROI cuboids have a larger number of pixels so they are the same physical size as the full-organ cuboids. Each colour represents a different dataset, taken from three different organs.
#
# ## Interpretation
# 1. The registration parameters converge (ie stop varying) at different sizes for each dataset - after 100 pixels for the kidney, 40 pixels for the heart, and 60 pixels for the lung. Since running a registration with 100 pixels does not take too long (~minutes on my laptop), choosing an **x-y size of 110** is a safe choice.
# 2. The **translation parameters** converge to **sub-pixel values** with minimal variation.
# 3. The **rotation parameter** converges to a value that has an **error of ~+/- 0.2 degrees**. This makes sense as an order of magnitude for how accurate the rotation can be determined - the approximate theoretical error is 1 / 120 rad ~= 0.4 degrees.

# %%
