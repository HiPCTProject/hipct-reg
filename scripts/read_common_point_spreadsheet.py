import pandas as pd

import hipct_reg.inventory

if __name__ == "__main__":
    df = pd.read_csv("common_points.csv")
    inventory = {d.name: d for d in hipct_reg.inventory.load_datasets()}

    for idx, row in df.dropna(
        subset=["ROI x", "ROI y", "ROI z", "Full x", "Full y", "Full z"]
    ).iterrows():
        # Rows with common points
        if (name := row["ROI name"]) not in inventory:
            print(f"Did not find {name} in inventory")
            continue

        dataset = inventory[name]
        dataset.common_point_x = int(row["ROI x"])
        dataset.common_point_y = int(row["ROI y"])
        dataset.common_point_z = int(row["ROI z"])

        dataset.common_point_parent_x = int(row["Full x"])
        dataset.common_point_parent_y = int(row["Full y"])
        dataset.common_point_parent_z = int(row["Full z"])
        dataset.notes = ""

    for idx, row in df.iterrows():
        if (name := row["ROI name"]) not in inventory:
            print(f"Did not find {name} in inventory")
            continue

        dataset = inventory[name]

        if row["Notes"]:
            dataset.notes = row["Notes"]

    hipct_reg.inventory.save_datasets(list(inventory.values()))
