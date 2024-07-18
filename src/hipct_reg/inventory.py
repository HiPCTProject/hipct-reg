import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel

INVENTORY_FILE = Path(__file__).parent / "reg_inventory.csv"


class RegDataset(BaseModel):
    """
    A dataset that can be, or has been registered.
    """

    # Name of dataset
    name: str
    # Name of dataset registered to
    parent_name: str

    common_point_x: int | None = None
    common_point_y: int | None = None
    common_point_z: int | None = None

    common_point_parent_x: int | None = None
    common_point_parent_y: int | None = None
    common_point_parent_z: int | None = None

    # Free text notes, mainly to note bad data quality
    notes: str | None = None

    # Registration parameters
    tx: float | None = None
    ty: float | None = None
    tz: float | None = None
    rotation: float | None = None  # In degrees
    scale: float | None = None
    reg_metric: float | None = None

    @property
    def has_common_points(self) -> bool:
        return (
            self.common_point_x is not None
            and self.common_point_y is not None
            and self.common_point_z is not None
            and self.common_point_parent_x is not None
            and self.common_point_parent_y is not None
            and self.common_point_parent_z is not None
        )

    def reset_reg_params(self) -> None:
        self.tx = None
        self.ty = None
        self.tz = None
        self.rotation = None
        self.scale = None
        self.reg_metric = None

    @property
    def neuroglancer_link(self) -> str:
        from hipct_data_tools import load_datasets
        from hipct_data_tools.neuroglancer import (
            NEUROGLANCER_INSTANCE,
            dataset_to_layer,
        )

        assert (
            self.tx is not None
            and self.ty is not None
            and self.tz is not None
            and self.rotation is not None
            and self.scale is not None
        )

        datasets = {d.name: d for d in load_datasets()}

        roi_dataset = datasets[self.name]
        full_dataset = datasets[self.parent_name]

        dimensions = {
            dim: (full_dataset.resolution_um, "um") for dim in ["x", "y", "z"]
        }

        full_layer = dataset_to_layer(full_dataset).to_json()
        roi_layer = dataset_to_layer(roi_dataset).to_json()

        # Add transform information to ROI layer
        res_ratio = full_dataset.resolution_um / roi_dataset.resolution_um
        matrix = (
            self.scale
            * res_ratio
            * np.array(
                [
                    [
                        np.cos(np.deg2rad(self.rotation)),
                        np.sin(np.deg2rad(self.rotation)),
                        0,
                    ],
                    [
                        -np.sin(np.deg2rad(self.rotation)),
                        np.cos(np.deg2rad(self.rotation)),
                        0,
                    ],
                    [0, 0, 1],
                ]
            )
        )

        # Add transformation matrix to ROI layer
        roi_layer["source"] = {
            "url": roi_layer["source"][0]["url"],
            "transform": {
                "matrix": [
                    [
                        *list(matrix[0, :]),
                        self.tx * res_ratio,
                    ],
                    [
                        *list(matrix[1, :]),
                        self.ty * res_ratio,
                    ],
                    [
                        *list(matrix[2, :]),
                        self.tz * res_ratio,
                    ],
                ],
                "outputDimensions": {
                    "x": (roi_dataset.resolution_um, "um"),
                    "y": (roi_dataset.resolution_um, "um"),
                    "z": (roi_dataset.resolution_um, "um"),
                },
            },
        }

        ng_dict = {
            "layers": [full_layer, roi_layer],
            "dimensions": dimensions,
            "layout": "4panel",
            "projectionOrientation": (0.3, 0.2, 0, -0.9),
            "projectionScale": 4096,
            "selectedLayer": {"layer": full_dataset.name, "visible": True},
        }

        link = f"{NEUROGLANCER_INSTANCE}/#!{json.dumps(ng_dict, separators=(',', ':'))}"
        return link


def save_datasets(datasets: list[RegDataset]) -> None:
    """
    Save a list of datasets to the .csv inventory file.
    """
    df = pd.DataFrame([dataset.model_dump() for dataset in datasets])
    df = df.sort_values("name")
    df.to_csv(INVENTORY_FILE, index=False, lineterminator="\n")


def load_datasets() -> list[RegDataset]:
    """
    Load a list of datasets from the .csv inventory file.
    """
    df = pd.read_csv(INVENTORY_FILE, na_filter=False)

    def parse_row(row: "pd.Series[Any]") -> RegDataset:
        d = dict(row)
        for key in d:
            if d[key] == "":
                d[key] = None
        return RegDataset(**d)

    return [parse_row(row) for i, row in df.iterrows()]
