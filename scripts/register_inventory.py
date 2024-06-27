"""
Run registration from entries in the inventory,
saving results to the inventory.
"""

import logging
import tkinter as ttk
from tkinter import Button, Tk

import SimpleITK as sitk
from matplotlib.axes import Axes
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from hipct_reg.data import get_reg_input
from hipct_reg.helpers import (
    get_central_pixel_index,
    get_central_point,
    get_pixel_transform_params,
    resample_roi_image,
    show_image,
)
from hipct_reg.inventory import Dataset, load_datasets, save_datasets
from hipct_reg.registration import run_registration
from hipct_reg.types import RegistrationInput

# Uncomment these lines, and set the path to where you want data downloaded:
# import hipct_reg.data
# import hipct_reg.data
# from pathlib import Path
# hipct_reg.data.STORAGE_DIR = Path(...)


logging.basicConfig(level=logging.INFO)

root = logging.getLogger()
root.setLevel(logging.INFO)

DATASETS = {d.name: d for d in load_datasets()}


def register(ds: Dataset) -> tuple[RegistrationInput, sitk.Similarity3DTransform]:
    """
    Register a single dataset.

    The registration parameters are modified in place on the dataset.
    """
    # return None, None
    assert (
        ds.common_point_x is not None
        and ds.common_point_y is not None
        and ds.common_point_z is not None
    ), "Common point values not set"
    assert (
        ds.common_point_parent_x is not None
        and ds.common_point_parent_y is not None
        and ds.common_point_parent_z is not None
    ), "Common point parent values not set"

    reg_input = get_reg_input(
        roi_name=ds.name,
        roi_point=(ds.common_point_x, ds.common_point_y, ds.common_point_z),
        full_name=ds.parent_name,
        full_point=(
            ds.common_point_parent_x,
            ds.common_point_parent_y,
            ds.common_point_parent_z,
        ),
        full_size_xy=64,
    )
    transform, metric = run_registration(reg_input)
    pix_transform_params = get_pixel_transform_params(reg_input, transform)

    ds.tx = pix_transform_params["tx_pix"]
    ds.ty = pix_transform_params["ty_pix"]
    ds.tz = pix_transform_params["tz_pix"]
    ds.scale = pix_transform_params["scale"]
    ds.rotation = pix_transform_params["rotation_deg"]

    ds.reg_metric = metric

    return reg_input, transform


def plot_central_slice(
    reg_input: RegistrationInput, transform: sitk.Transform, axes: list[Axes]
) -> None:
    """
    Plot central slice of images before/after registration.
    """
    for ax in axes:
        title = ax.get_title()
        ax.cla()
        ax.set_title(title)

    # Before
    central_idx = get_central_pixel_index(reg_input.roi_image)
    show_image(reg_input.roi_image, axes[0], central_idx[2])

    # After
    roi_resampled = resample_roi_image(reg_input, transform)
    # Get physical point in centre of full organ image
    central_z = get_central_point(reg_input.full_image)
    resampled_idx = roi_resampled.TransformPhysicalPointToIndex(central_z)

    show_image(roi_resampled, axes[1], resampled_idx[2])
    central_idx = get_central_pixel_index(reg_input.full_image)
    show_image(reg_input.full_image, axes[2], central_idx[2])
    axes[1].set_xlim(*axes[2].get_xlim())
    axes[1].set_ylim(*axes[2].get_ylim())

    for ax in axes[1:]:
        ax.grid(color="k")


def keep_dataset(d: Dataset) -> bool:
    """
    Whether to keep a dataset for listing in the registration widget.
    """
    return d.common_point_x is not None and d.rotation is None and d.notes is None


class RegWindow(Tk):
    def __init__(self) -> None:
        super().__init__()

        dataset_names = [d for d in DATASETS if keep_dataset(DATASETS[d])]

        self.title("HiP-CT registration")
        self.fig = Figure(figsize=(10, 4), dpi=100)
        for i, title in zip(
            [1, 2, 3], ["Pre-registration", "Post-registration", "Full-organ"]
        ):
            ax = self.fig.add_subplot(1, 3, i)
            ax.set_title(title)
            ax.set_aspect("equal", adjustable="box")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=ttk.TOP, fill=ttk.BOTH, expand=True)

        self.dataset_name = ttk.StringVar()
        self.dataset_menu = ttk.OptionMenu(self, self.dataset_name, *dataset_names)
        self.dataset_name.set(dataset_names[0])
        self.dataset_menu.pack()

        self.button_register = Button(
            self, text="Run registration", command=self.handle_register_button
        )
        self.button_register.pack()

        self.button_good = Button(
            self, text="Good", state="disabled", command=self.handle_good_button
        )
        self.button_good.pack()

        self.button_bad = Button(
            self, text="Bad", state="disabled", command=self.handle_bad_button
        )
        self.button_bad.pack()

        self.text_bad = ttk.Entry(self)
        self.text_bad.pack()

    def handle_register_button(self) -> None:
        dataset = DATASETS[self.dataset_name.get()]
        reg_input, transform = register(dataset)
        plot_central_slice(reg_input, transform, self.fig.axes)
        print(dataset.neuroglancer_link)
        self.canvas.draw()

        self.button_good["state"] = "normal"
        self.button_bad["state"] = "normal"

    def reset(self) -> None:
        for ax in self.fig.axes:
            ax.cla()
        self.button_good["state"] = "disabled"
        self.button_bad["state"] = "disabled"
        self.button_register["state"] = "normal"

    def handle_good_button(self) -> None:
        # Registration params already saved in place on relevant dataset
        save_datasets(list(DATASETS.values()))
        self.reset()

    def handle_bad_button(self) -> None:
        text = self.text_bad.get()
        if text == "":
            print("Need to enter reason why registration is bad")
            return

        dataset = DATASETS[self.dataset_name.get()]
        dataset.notes = text
        dataset.reset_reg_params()
        DATASETS[self.dataset_name.get()] = dataset
        save_datasets(list(DATASETS.values()))
        self.reset()


if __name__ == "__main__":
    # Set up GUI

    # Start the event loop.
    window = RegWindow()
    window.mainloop()
