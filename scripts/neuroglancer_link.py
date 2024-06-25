from hipct_reg.inventory import load_datasets

datasets = {d.name: d for d in load_datasets()}
for d in datasets.values():
    try:
        print(f"{d.name}\t{d.neuroglancer_link}")
    except AssertionError:
        pass
