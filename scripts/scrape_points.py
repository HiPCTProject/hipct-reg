import hipct_data_tools

from hipct_reg.inventory import load_datasets, save_datasets

datasets = hipct_data_tools.load_datasets()
datasets_reg = {d.name: d for d in load_datasets()}

for d in datasets:
    if d.registration_log_path is None:
        continue

    with d.registration_log_path.open() as f:
        lines = f.readlines()

    point_fixed = None
    point_moved = None
    for l in lines:
        if l.startswith("Point fixed"):
            point_fixed = eval(l.split("=")[1].strip())
        if l.startswith("Point moved"):
            point_moved = eval(l.split("=")[1].strip())

    try:
        dataset = datasets_reg[d.name]
    except KeyError:
        print(f"Didn't find {d.name} in registration database")
        continue
    dataset.common_point_x = int(point_moved[0] * 2)
    dataset.common_point_y = int(point_moved[1] * 2)
    dataset.common_point_z = int(point_moved[2] * 2)

    dataset.common_point_parent_x = int(point_fixed[0] * 2)
    dataset.common_point_parent_y = int(point_fixed[1] * 2)
    dataset.common_point_parent_z = int(point_fixed[2] * 2)

    save_datasets(list(datasets_reg.values()))
