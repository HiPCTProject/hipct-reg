import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel

INVENTORY_FILE = Path(__file__).parent / "reg_inventory.csv"
from hipct_data_tools.neuroglancer import NEUROGLANCER_INSTANCE, dataset_to_layer


class Dataset(BaseModel):
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

        full_layer = dataset_to_layer(full_dataset, add_transform=False).to_json()
        roi_layer = dataset_to_layer(roi_dataset, add_transform=False).to_json()
        matrix = self.scale * np.array(
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

        # Add transformation matrix to ROI layer
        roi_layer["source"] = {
            "url": roi_layer["source"][0]["url"],
            "transform": {
                "matrix": [
                    [
                        *list(matrix[0, :]),
                        self.tx / roi_dataset.resolution_um,
                    ],
                    [
                        *list(matrix[1, :]),
                        self.ty / roi_dataset.resolution_um,
                    ],
                    [
                        *list(matrix[2, :]),
                        self.tz / roi_dataset.resolution_um,
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


def save_datasets(datasets: list[Dataset]) -> None:
    """
    Save a list of datasets to the .csv inventory file.
    """
    df = pd.DataFrame([dataset.model_dump() for dataset in datasets])
    df.to_csv(INVENTORY_FILE, index=False, lineterminator="\n")


def load_datasets() -> list[Dataset]:
    """
    Load a list of datasets from the .csv inventory file.
    """
    df = pd.read_csv(INVENTORY_FILE, na_filter=False)

    def parse_row(row: "pd.Series[Any]") -> Dataset:
        d = dict(row)
        for key in d:
            if d[key] == "":
                d[key] = None
        return Dataset(**d)

    return [parse_row(row) for i, row in df.iterrows()]
