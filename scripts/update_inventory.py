import hipct_data_tools

import hipct_reg.inventory

datasets = hipct_data_tools.load_datasets()
datasets_reg = hipct_reg.inventory.load_datasets()
datasets_reg_names = [d.name for d in datasets_reg]

for d in datasets:
    if d.name not in datasets_reg_names:
        parents = d.parent_datasets()
        if not len(parents):
            continue
        parent = min(parents, key=lambda p: p.resolution_um)
        datasets_reg.append(
            hipct_reg.inventory.Dataset(name=d.name, parent_name=parent.name)
        )

hipct_reg.inventory.save_datasets(datasets_reg)
