import numpy as np
import SimpleITK as sitk

from hipct_reg.helpers import get_central_pixel_index, get_central_point


def test_get_central_pix() -> None:
    shape = (10, 20, 30)
    im = sitk.GetImageFromArray(np.zeros(shape).T)
    assert get_central_pixel_index(im) == (5, 10, 15)


def test_get_central_point() -> None:
    shape = (10, 20, 30)
    im = sitk.GetImageFromArray(np.zeros(shape).T)
    im.SetSpacing((2, 3, 4))
    assert get_central_point(im) == (10.0, 30.0, 60.0)
