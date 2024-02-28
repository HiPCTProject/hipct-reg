from pathlib import Path
from typing import Any

import numpy as np
import tensorstore as ts
import zarr

bucket = "ucl-hip-ct-35a68e99feaae8932b1d44da0358940b"


def get_dataset(scan: str) -> Any:
    return ts.open(
        {
            "driver": "n5",
            "kvstore": f"gs://{bucket}/LADAF-2021-17/heart/{scan}/s0",
            "context": {"cache_pool": {"total_bytes_limit": 100_000_000}},
        }
    ).result()


ratio = 1 / (1 - np.sqrt(3))

SIZE_FULL = 64
SIZE_ROI = int(SIZE_FULL * 19.85 / 6.36)


def save_zoom(
    dataset_name: str, point: tuple[int, int, int], size: int, fname: str
) -> None:
    ds = get_dataset(dataset_name)
    data = (
        ds[
            point[0] - size : point[0] + size,
            point[1] - size : point[1] + size,
            point[2] - size : point[2] + size,
        ]
        .read()
        .result()
    )
    zarr.convenience.save(Path(__file__).parent / "data" / f"{fname}.zarr", data)


save_zoom(
    "19.85um_complete-organ_bm18",
    (1918 * 2, 1525 * 2, 794 * 2),
    SIZE_FULL,
    "whole_organ",
)
save_zoom("6.36um_ROI-01_bm18", (955 * 2, 1078 * 2, 1715 * 2), SIZE_ROI, "roi_01")
