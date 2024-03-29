"""
Create a neuroglancer link from a registered dataset.

First the script new_reg_test.py needs to be run to generate the registration
data in a JSON file.

Then run this script to generate a neuroglancer link.
"""

import json
from pathlib import Path

from hipct_data_tools import load_datasets
from hipct_data_tools.neuroglancer import NEUROGLANCER_INSTANCE, dataset_to_layer

roi_name = "LADAF-2020-27_kidney_left_central-column_6.05um_bm05"
full_name = "LADAF-2020-27_kidney_left_complete-organ_25.08um_bm05"
datasets = {d.name: d for d in load_datasets()}

with open(Path(__file__).parent / f"transform_{roi_name}.json") as f:
    registration = json.loads(f.read())

roi_dataset = datasets[roi_name]
full_dataset = datasets[full_name]

dimensions = {dim: (full_dataset.resolution_um, "um") for dim in ["x", "y", "z"]}

full_layer = dataset_to_layer(full_dataset, add_transform=False)
roi_layer = dataset_to_layer(roi_dataset, add_transform=False)

# Add transformation matrix to ROI layer
roi_layer["source"] = {
    "url": roi_layer["source"],
    "transform": {
        "matrix": [
            registration["rotation_matrix"][0:3]
            + [registration["translation"][0] / roi_dataset.resolution_um],
            registration["rotation_matrix"][3:6]
            + [registration["translation"][1] / roi_dataset.resolution_um],
            registration["rotation_matrix"][6:9]
            + [registration["translation"][2] / roi_dataset.resolution_um],
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
print(link)
