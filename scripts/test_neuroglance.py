from hipct_reg.inventory import load_datasets

DATASETS = {d.name: d for d in load_datasets()}

dataset = DATASETS["FO-20-124-OL_lung_column3_6.24um_bm05"]
print(dataset.neuroglancer_link)
