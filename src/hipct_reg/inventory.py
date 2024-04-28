from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel

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
    notes: str = ""

    # Registration parameters
    tx: float | None = None
    ty: float | None = None
    tz: float | None = None
    rotation: float | None = None
    scale: float | None = None


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
