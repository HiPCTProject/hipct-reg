from dataclasses import dataclass

import SimpleITK as sitk


@dataclass
class RegistrationInput:
    """
    A container for all the data needed to register a ROI to a full-organ scan.

    Contains both images, and pixel indices of a common point in both images.
    """

    roi_image: sitk.Image
    full_image: sitk.Image
    # Common points are in units of pixels
    common_point_roi: tuple[int, int, int]
    common_point_full: tuple[int, int, int]
