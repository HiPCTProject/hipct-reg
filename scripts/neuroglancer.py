import json
from pathlib import Path

from hipct_data_tools import load_datasets
from hipct_data_tools.neuroglancer import NEUROGLANCER_INSTANCE, dataset_to_layer

datasets = {d.name: d for d in load_datasets()}
roi_name = (
    "LADAF-2020-27_heart_LR-vent-muscles-ramus-interventricularis-anterior_6.05um_bm05"
)
full_name = "LADAF-2020-27_heart_complete-organ_25.08um_bm05"

with open(Path(__file__).parent / f"transform_{roi_name}.json") as f:
    registration = json.loads(f.read())

roi_dataset = datasets[roi_name]
full_dataset = datasets[full_name]
res_ratio = full_dataset.resolution_um / roi_dataset.resolution_um

dimensions = {dim: (full_dataset.resolution_um / 1e6, "m") for dim in ["x", "y", "z"]}
selectedLayer = {"layer": full_dataset.name, "visible": True}

full_layer = dataset_to_layer(full_dataset, add_transform=False)
roi_layer = dataset_to_layer(roi_dataset, add_transform=False)
# Add transformation matrix to ROI layer
roi_layer["source"] = {
    "url": roi_layer["source"],
    "transform": {
        "matrix": [
            # Note that translation has to be in units of `dimensions`,
            # which gives the size of a single un-transformed pixel in the global viewer
            registration["rotation_matrix"][0:3]
            + [registration["translation"][0] / full_dataset.resolution_um],
            registration["rotation_matrix"][3:6]
            + [registration["translation"][1] / full_dataset.resolution_um],
            registration["rotation_matrix"][6:9]
            + [registration["translation"][2] / full_dataset.resolution_um],
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
    "selectedLayer": selectedLayer,
}

print(ng_dict)


link = f"{NEUROGLANCER_INSTANCE}/#!{json.dumps(ng_dict, separators=(',', ':'))}"
print(link)
