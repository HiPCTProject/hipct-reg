from dataclasses import dataclass

import SimpleITK as sitk


@dataclass
class RegistrationInput:
    """
    A container for all the data needed to register a zoom to an overview image.

    Contains both images, and pixel indices of a common point in both images.
    """

    zoom_name: str
    overview_name: str
    zoom_image: sitk.Image
    overview_image: sitk.Image
    # Common points are in units of pixels
    zoom_common_point: tuple[int, int, int]
    overview_common_point: tuple[int, int, int]
