"""
Generate a spreadsheet with neuroglancer links that can be used to find
common points needed for registration.
"""

import hipct_data_tools
import hipct_data_tools.inventory
import pandas as pd
from hipct_data_tools.neuroglancer import neuroglancer_link

import hipct_reg.inventory

if __name__ == "__main__":
    inventory = {d.name: d for d in hipct_data_tools.load_datasets()}
    inventory_reg = hipct_reg.inventory.load_datasets()

    data: dict[str, list[str]] = {
        "Name": [],
        "ROI name": [],
        "ROI link": [],
        "ROI x": [],
        "ROI y": [],
        "ROI z": [],
        "Full name": [],
        "Full link": [],
        "Full x": [],
        "Full y": [],
        "Full z": [],
        "Notes": [],
    }

    for dataset in inventory_reg:
        if dataset.tx is not None:
            continue

        parent_ng = neuroglancer_link(inventory[dataset.parent_name])
        if parent_ng is None:
            print(f"No neuroglancer link for {dataset.parent_name}")
            continue

        ng = neuroglancer_link(inventory[dataset.name])
        if ng is None:
            print(f"No neuroglancer link for {dataset.name}")
            continue

        data["Name"].append("")
        data["Full name"].append(dataset.parent_name)
        data["Full link"].append(parent_ng)
        for dim in ["x", "y", "z"]:
            data[f"Full {dim}"].append("")

        data["ROI name"].append(dataset.name)
        data["ROI link"].append(ng)
        for dim in ["x", "y", "z"]:
            data[f"ROI {dim}"].append("")

        if dataset.notes:
            data["Notes"].append(dataset.notes)
        else:
            data["Notes"].append("")

    df = pd.DataFrame(data)
    df.to_csv("common_points_new.csv", index=False)
