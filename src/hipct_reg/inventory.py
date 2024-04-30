from pathlib import Path
from typing import Any

import pandas as pd
import SimpleITK as sitk
from pydantic import BaseModel

from hipct_reg.data import get_reg_input
from hipct_reg.registration import run_registration

INVENTORY_FILE = Path(__file__).parent / "reg_inventory.csv"


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
    rotation: float | None = None
    scale: float | None = None
    reg_metric: float | None = None

    def register(self) -> None:
        assert (
            self.common_point_x is not None
            and self.common_point_y is not None
            and self.common_point_z is not None
        ), "Common point values not set"
        assert (
            self.common_point_parent_x is not None
            and self.common_point_parent_y is not None
            and self.common_point_parent_z is not None
        ), "Common point parent values not set"

        reg_input = get_reg_input(
            roi_name=self.name,
            roi_point=(self.common_point_x, self.common_point_y, self.common_point_z),
            full_name=self.parent_name,
            full_point=(
                self.common_point_parent_x,
                self.common_point_parent_y,
                self.common_point_parent_z,
            ),
            full_size_xy=64,
        )
        transform, metric = run_registration(reg_input)

        # Convert so transform is from (0, 0, 0) instead of around the rotation point
        translation = transform.TransformPoint((0, 0, 0))
        new_transform = sitk.Similarity3DTransform()
        new_transform.SetParameters(transform.GetParameters())
        new_transform.SetTranslation(translation)

        translation = new_transform.GetTranslation()
        self.tx = translation[0]
        self.ty = translation[1]
        self.tz = translation[2]
        self.scale = new_transform.GetScale()
        self.rotation = new_transform.GetVersor()[0]

        self.reg_metric = metric


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
