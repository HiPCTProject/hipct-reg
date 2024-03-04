import numpy as np
import SimpleITK as sitk

from hipct_reg.helpers import get_central_pixel_index


def test_get_central_pix() -> None:
    shape = (10, 20, 30)
    im = sitk.GetImageFromArray(np.zeros(shape).T)
    assert get_central_pixel_index(im) == (5, 10, 15)
