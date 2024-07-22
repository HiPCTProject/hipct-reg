"""
Add new datasets from the hipct-data-tools inventory to the hipct-reg inventory.
"""

from datetime import UTC, datetime

import hipct_data_tools
import hipct_data_tools.inventory
from hipct_data_tools.data_model import HiPCTDataSet

import hipct_reg.inventory


def keep_dataset(d: HiPCTDataSet) -> bool:
    """
    Return whether to keep the dataset and add it to the registration inventory.
    """
    for parent in d.parent_datasets():
        if parent.release == "complete-hearts":
            return True
    return False


def parent_date(parent: HiPCTDataSet) -> datetime:
    """
    Helper to return date of a parent dataset, returning a date far in the
    past if the dataset doesn't have a date.
    """
    if parent.date_reconstruction is not None:
        return parent.date_reconstruction.replace(tzinfo=UTC)  # type: ignore[no-any-return]
    else:
        return datetime(1900, 1, 1, tzinfo=UTC)


def select_parent(
    dataset: HiPCTDataSet,
    parents: list[HiPCTDataSet],
) -> HiPCTDataSet:
    """
    Given a dataset and its parents, choose one of the parents to register to.
    """
    # Prefer same beamline
    parents_beamline = [p for p in parents if p.beamline == dataset.beamline]
    if len(parents_beamline):
        parents = parents_beamline
    date = dataset.date_reconstruction
    if date is not None:
        # Select closest in time
        return min(
            parents, key=lambda p: abs(date.replace(tzinfo=UTC) - parent_date(p))
        )
    else:
        # Fallback on minimum resolution
        return min(parents, key=lambda p: p.resolution_um)


if __name__ == "__main__":
    all_datasets = hipct_data_tools.load_datasets()
    new_datasets = [d for d in all_datasets if keep_dataset(d)]
    if not len(new_datasets):
        raise RuntimeError("No datasets kept to add to inventory")

    new_dataset_names = [d.name for d in new_datasets]
    datasets_reg_names = {d.name: d for d in hipct_reg.inventory.load_datasets()}

    new_datasets = [d for d in new_datasets if not d.is_complete_organ]
    if not len(new_datasets):
        raise RuntimeError("All new datasets are complete organ datasets")

    for d in new_datasets:
        parents = d.parent_datasets()

        if not len(parents):
            print(f"Could not find any parent datasets for {d.name}")
            datasets_reg_names.pop(d.name, None)
            continue

        parent = select_parent(d, parents)

        if d.name not in datasets_reg_names:
            print(f"Adding {d.name} to inventory")
            # Create new dataset
            datasets_reg_names[d.name] = hipct_reg.inventory.RegDataset(
                name=d.name, parent_name=parent.name
            )

    hipct_reg.inventory.save_datasets(list(datasets_reg_names.values()))
