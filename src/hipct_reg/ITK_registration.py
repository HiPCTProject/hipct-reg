"""
Author: Joseph Brunet (joseph.brunet@ucl.ac.uk)

This script does the registration of HiP-CT high-resolution volumes to low-resolution
volumes.
"""

import ast
import logging
import math
import sys
import time
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from typing import TypedDict

import numpy as np
import numpy.typing as npt
import psutil
import SimpleITK as sitk
import skimage.io
import skimage.measure
from scipy.spatial.transform import Rotation as ROT

from hipct_reg.helpers import get_pixel_size, import_im
from hipct_reg.types import RegistrationInput

MAX_THREADS = 0  # 0 if all


class RotRegMetrics(TypedDict):
    rotation: list[float]
    metric: list[float]


def show_fiji(
    reg_input: RegistrationInput, transform: sitk.Transform, message: str
) -> None:
    moving_resampled = sitk.Resample(
        reg_input.full_image,
        reg_input.roi_image,
        transform,
        sitk.sitkLinear,
        0,
        reg_input.roi_image.GetPixelID(),
    )
    sitk.Show(0.6 * moving_resampled + 0.4 * reg_input.roi_image, message)


def registration_rot(
    reg_input: RegistrationInput,
    *,
    zrot: float,
    angle_range: float,
    angle_step: float,
    fiji: bool = False,
    verbose: bool = False,
) -> tuple[sitk.Euler3DTransform, RotRegMetrics]:
    """
    Run a registration using a rotational transform about the z-axis.
    The angles of the transform which are sampled are set by manually by function
    arguments.

    Parameters
    ----------
    zrot :
        Initial rotation for the registration. In units of degrees.
    angle_range :
        Range of angles to scan. In units of degrees.
    angle_step :
        Step to take when scanning range of angles. In units of degrees.
    verbose :
        If True, log every step of the optimisation at debug level.

    Returns
    -------
    transform :
        Final registered transform.
    metrics :
        Dict containing the registration metric as a function of angle.

    """
    R = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.01, seed=1)

    R.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    logging.info("Starting rotational registration")
    logging.info(f"Initial rotation = {zrot} deg")
    logging.info(f"Range = {angle_range} deg")
    logging.info(f"Step = {angle_step} deg")
    logging.info(f"Common point ROI = {reg_input.common_point_roi} pix")
    logging.info(f"Common point full = {reg_input.common_point_full} pix")

    R.SetOptimizerAsExhaustive(
        numberOfSteps=[0, 0, int((angle_range / 2) / angle_step), 0, 0, 0],
        stepLength=np.deg2rad(angle_step),
    )

    R.SetOptimizerScalesFromPhysicalShift()
    # These variables are in physical coordinates
    # Rotation centre of transform in ROI image
    rotation_center = reg_input.roi_image.TransformIndexToPhysicalPoint(
        reg_input.common_point_roi
    )
    # Translation from ROI image to full image
    translation = -np.array(rotation_center) + np.array(
        reg_input.full_image.TransformIndexToPhysicalPoint(reg_input.common_point_full)
    )
    logging.debug(f"rotation center = {rotation_center}")
    logging.debug(f"translation from ROI to full organ = {translation}")

    # Initial transform angles in radians
    theta_x = 0.0
    theta_y = 0.0
    theta_z = np.deg2rad(zrot)

    initial_transform = sitk.Euler3DTransform(
        rotation_center, theta_x, theta_y, theta_z, translation
    )
    R.SetInitialTransform(initial_transform, inPlace=True)

    if fiji:
        show_fiji(reg_input, initial_transform, "Before rotation")

    data: RotRegMetrics = {"rotation": [], "metric": []}

    def command_iteration(
        method: sitk.ImageRegistrationMethod,
        translation: npt.NDArray,
    ) -> None:
        rotation = np.rad2deg(method.GetOptimizerPosition()[0:3])
        translation = method.GetOptimizerPosition()[3:6]
        metric = method.GetMetricValue()

        data["metric"].append(metric)
        data["rotation"].append(rotation[2])

        if verbose:
            logging.debug(f"iteration = {method.GetOptimizerIteration()}")
            logging.debug(f"metric = {metric}")
            logging.debug(f"translation = {translation}")
            logging.debug(f"rotation = {rotation} deg")
            logging.debug(f"CPU usage: {psutil.cpu_percent()}")
            logging.debug(f"RAM usage: {psutil.virtual_memory().percent}")
            logging.debug("")

    R.AddCommand(
        sitk.sitkIterationEvent,
        lambda: command_iteration(
            R,
            translation,
        ),
    )

    logging.info("Starting rotational registration...")
    transform_rotation: sitk.Euler3DTransform = R.Execute(
        reg_input.roi_image, reg_input.full_image
    )
    logging.info("Registration finished!")

    if fiji:
        show_fiji(reg_input, initial_transform, "After rotation")

    new_zrot = np.rad2deg(transform_rotation.GetAngleZ())
    logging.debug(f"Final metric value = {R.GetMetricValue()}")
    logging.debug(f"Stopping condition = {R.GetOptimizerStopConditionDescription()}")
    logging.info(f"Registered rotation angele = {new_zrot} deg")
    logging.info("")

    return transform_rotation, data


def registration_sitk(
    reg_input: RegistrationInput,
    *,
    zrot: float,
    fiji: bool = False,
) -> sitk.Similarity3DTransform:
    """
    Run a registration using a full rigid transform.

    The returned transform maps from the ROI iamge to the full-organ image.

    Parameters
    ----------
    zrot :
        Initial rotation for the registration. In units of degrees.

    """
    logging.info("Starting full registration...")
    logging.info(f"Common point ROI = {reg_input.common_point_roi}")
    logging.info(f"Common point full = {reg_input.common_point_full}")
    logging.info(f"Initial rotation = {zrot:.02f} deg")
    pixel_size_roi: int = reg_input.roi_image.GetSpacing()[0]

    R = sitk.ImageRegistrationMethod()

    # Set registration metric settings
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.1, seed=1)

    # Set registration interpolator
    R.SetInterpolator(sitk.sitkLinear)

    # Set registartion optimiser settings
    R.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-7,
        numberOfIterations=2000,
        maximumNumberOfCorrections=20,
        maximumNumberOfFunctionEvaluations=2000,
        costFunctionConvergenceFactor=1e9,
    )
    # Parameters are:
    # - Three rotation angles
    # - Three translation components
    # - Scale factor
    # R.SetOptimizerScalesFromPhysicalShift()
    w = 10
    R.SetOptimizerWeights([w, w, w, w, w, w, w / 1000])

    # Setup for the multi-resolution framework.
    R.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 2, 1, 1, 1])
    R.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 1, 1, 1, 0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # These variables are in physical coordinates
    # Rotation centre of transform in ROI image
    rotation_center = reg_input.roi_image.TransformIndexToPhysicalPoint(
        reg_input.common_point_roi
    )
    # Translation from ROI image to full image
    translation = -np.array(rotation_center) + np.array(
        reg_input.full_image.TransformIndexToPhysicalPoint(reg_input.common_point_full)
    )
    logging.debug(f"rotation center = {rotation_center}")
    logging.debug(f"translation from ROI to full organ = {translation}")

    theta_x = 0.0
    theta_y = 0.0
    theta_z = np.deg2rad(zrot)

    rigid_euler = sitk.Euler3DTransform(
        rotation_center, theta_x, theta_y, theta_z, translation
    )
    initial_transform = sitk.Similarity3DTransform()
    initial_transform.SetMatrix(rigid_euler.GetMatrix())
    initial_transform.SetTranslation(rigid_euler.GetTranslation())
    initial_transform.SetCenter(rigid_euler.GetCenter())

    R.SetInitialTransform(initial_transform, inPlace=True)

    if fiji:
        show_fiji(reg_input, initial_transform, "Before registration")

    metric = []

    def command_iteration(
        method: sitk.ImageRegistrationMethod,
        transform: sitk.Similarity3DTransform,
        pixel_size: int,
    ) -> None:
        metric.append(method.GetMetricValue())

        q0, q1, q2, q3 = transform.GetVersor()
        r = ROT.from_quat([q0, q1, q2, q3])
        theta_x, theta_y, theta_z = np.rad2deg(r.as_rotvec())
        translation = np.array(transform.GetTranslation()) / pixel_size
        scale = transform.GetScale()

        logging.debug(f"Iteration {method.GetOptimizerIteration()}")
        logging.debug(f"metric = {method.GetMetricValue()}")
        logging.debug(f"translation = {translation}")
        logging.debug(f"rotation = {theta_x}, {theta_y}, {theta_z} deg")
        logging.debug(f"scale = {scale}")
        logging.debug("")

    R.AddCommand(
        sitk.sitkIterationEvent,
        lambda: command_iteration(R, initial_transform, pixel_size_roi),
    )

    logging.info("Starting registration...")
    final_transform: sitk.Similarity3DTransform = R.Execute(
        reg_input.roi_image, reg_input.full_image
    )
    logging.info("Registration finished!")

    if fiji:
        show_fiji(reg_input, initial_transform, "After registration")

    logging.debug(f"Final metric value: {R.GetMetricValue()}")
    logging.debug(
        f"Optimizer's stopping condition, {R.GetOptimizerStopConditionDescription()}"
    )

    translation_pix = reg_input.roi_image.TransformPhysicalPointToContinuousIndex(
        final_transform.GetTranslation()
    )
    rotation = np.rad2deg(np.array(final_transform.GetVersor()))
    logging.info(f"translation = {translation_pix} pix")
    logging.info(f"rotation = {rotation} deg")

    return final_transform


def registration_pipeline(
    *,
    path_roi: Path,
    path_full: Path,
    pt_roi: tuple[int, int, int],
    pt_full: tuple[int, int, int],
) -> sitk.Similarity3DTransform:
    """
    Run the full registration pipeline on two folders of images.

    Parameters
    ----------
    path_roi :
        Path to the ROI image.
    path_full :
        Path to the full organ scan image.
    pt_roi :
        Point that the ROI image will be rotated around, in the coordinate
        frame of the ROI image.
    pt_full :
        Point that the ROI image will be rotated around, in the coordinate
        frame of the full organ scan image.
    """
    # Crop the zoom scan to transform the circle fov into a square, thus avoiding the
    # NaN part in the image
    crop_circle_roi = False
    logging.info("Starting registration pipeline...")
    logging.info(f"Crop the ROI iamge to avoid external circle = {crop_circle_roi}")
    logging.info(f"Center of rotation (in frame of ROI image) = {pt_roi}")
    logging.info(f"Center of rotation (in frame of full organ image) = {pt_full}")
    pixel_size_roi = get_pixel_size(path_roi)
    pixel_size_full = get_pixel_size(path_full)

    n_roi = len(list(path_roi.glob("*.tif")))
    n_full = len(list(path_full.glob("*.tif")))

    img = skimage.io.imread(next(path_roi.glob("*.tif")))
    logging.info(f"{n_roi} images of shape {img.shape} in the ROI image folder")

    img = skimage.io.imread(next(path_full.glob("*.tif")))
    x_dim = img.shape[1]
    y_dim = img.shape[0]
    logging.info(f"{n_full} images of shape {img.shape} in the full organ image folder")

    del img

    size_full = n_full * x_dim * y_dim * 2 / (1024 * 1024 * 1024)
    logging.info(f"Total size of full organ image is {size_full} GB")

    # binning_moved = 1 + size_moved // 50  # I put a limit of 50 GB
    # binning_moved = int((pixel_size_fixed // pixel_size_moved) / 2) * 2  # other way
    binning_roi = 1
    binning_full = 1

    if binning_roi != 1:
        n_roi = math.ceil(n_roi / binning_roi)
        logging.info(f"After binning: I found {n_roi} images in the ROI image folder")
    logging.info(f"ROI image will be binned by {binning_roi}")

    if binning_full != 1:
        logging.info(f"A binning of {binning_full} will be applied to the ROI image")
        n_full = math.ceil(n_full / binning_full)
        logging.info(
            f"After binning: I found {n_full} images in the full organ image folder"
        )
    logging.info(f"Full organ image will be binned by {binning_full}")
    logging.info("")

    # pixel_size_fixed = pixel_size_fixed * binning_fixed
    # pixel_size_moved = pixel_size_moved * binning_moved

    # pt_roi = pt_roi / binning_fixed
    # pt_full = pt_full / binning_moved

    logging.info("Importing full organ image...")
    logging.info(f"folder = {path_full}")
    image_full = import_im(
        path_full,
        pixel_size_full,
        bin_factor=binning_full,
    )
    logging.info("Imported full organ image!")
    logging.info("Full organ image parameters:")
    logging.info(f"size = {image_full.GetSize()}")
    logging.info(f"dtype = {image_full.GetPixelIDTypeAsString()}")
    logging.info(f"spacing = {image_full.GetSpacing()}")
    logging.info("")

    """
    # Get values to crop the fixed dataset
    # TODO: check this maths
    moved_z = (0, N_moved)
    zmin = int(pt_roi[2] - 1.2 * ((pt_full[2]) * pixel_size_moved / pixel_size_fixed))
    zmax = int(
        pt_roi[2]
        + 1.2
        * (
            ((moved_z[1] - moved_z[0]) - pt_full[2])
            * pixel_size_moved
            / pixel_size_fixed
        )
    )
    zmin = max(zmin, 0)
    zmax = min(zmax, int(N_fixed))

    fixed_z = (zmin, zmax)
    pt_roi[2] = pt_roi[2] - fixed_z[0]
    """

    logging.info("Importing ROI image...")
    logging.info(f"folder = {path_roi}")
    # logging.info(f"Fixed scan crop z = {fixed_z}")
    image_roi = import_im(
        path_roi,
        pixel_size_roi,
        # crop_z=fixed_z,
        bin_factor=binning_roi,
    )
    logging.info("Finished importing ROI image!")
    logging.info("ROI image parameters:")
    logging.info(f"size = {image_roi.GetSize()}")
    logging.info(f"dtype = {image_roi.GetPixelIDTypeAsString()}")
    logging.info(f"spacing = {image_roi.GetSpacing()}")
    logging.info("")

    logging.info("Normalising images...")
    image_roi = sitk.Normalize(image_roi)
    image_full = sitk.Normalize(image_full)
    logging.info("Images normalised!")
    logging.info("")

    reg_input = RegistrationInput(
        roi_image=image_roi,
        full_image=image_full,
        common_point_roi=pt_roi,
        common_point_full=pt_full,
    )
    return run_registration(reg_input)


def run_registration(reg_input: RegistrationInput) -> sitk.Similarity3DTransform:
    """
    Run registration pipeline on pre-loaded/pre-processed images.
    """

    logging.info("Runinng registration...")
    logging.info(f"Full array size: {reg_input.full_image.GetSize()} pix")
    logging.info(f"Full voxel size = {reg_input.full_image.GetSpacing()} um")
    logging.info(f"ROI array size: {reg_input.roi_image.GetSize()} pix")
    logging.info(f"ROI voxel size = {reg_input.roi_image.GetSpacing()} um")
    logging.info("")
    zrot = 0

    # Try a full 360 deg first at a coarse step
    angle_range = 360
    angle_step = 2.0
    transform_rotation, _ = registration_rot(
        reg_input,
        zrot=zrot,
        angle_range=angle_range,
        angle_step=angle_step,
    )
    zrot = np.rad2deg(np.array(transform_rotation.GetAngleZ()))

    # Now try a smaller angular step
    angle_range = 5
    angle_step = 0.1
    transform_rotation, _ = registration_rot(
        reg_input,
        zrot=zrot,
        angle_range=angle_range,
        angle_step=angle_step,
    )
    zrot = np.rad2deg(np.array(transform_rotation.GetAngleZ()))

    if zrot < 0:
        zrot = zrot + 360

    logging.info("Similarity registration started...")
    final_transform = registration_sitk(
        reg_input,
        zrot=zrot,
    )
    logging.info("Similarity registration finished!")
    logging.info("")

    logging.info("Results from similarity registration:")
    logging.info(f"Translation = {final_transform.GetTranslation()}")
    logging.info(f"Center of rot = {final_transform.GetCenter()}")
    logging.info(f"Matrix = {final_transform.GetMatrix()}")
    logging.info(f"Versor = {final_transform.GetVersor()}")
    logging.info(f"Scale = {final_transform.GetScale()}")
    logging.info("")

    return final_transform


def process_line(line: str) -> tuple[Path, Path, npt.NDArray, npt.NDArray]:
    """
    Process a single line in a registration list file.
    """
    ls = line.split()
    pt_full = np.array(ast.literal_eval(ls[2]))
    pt_roi = np.array(ast.literal_eval(ls[3]))
    return Path(ls[0]), Path(ls[1]), pt_full, pt_roi


def get_registration_list() -> list[tuple[Path, Path, npt.NDArray, npt.NDArray]]:
    """
    Get list of datasets to register.

    Returns
    -------
    list
        Each item in the list contains:
        - Path to full organ dataset
        - Path to dataset being registered
        - Common point in full organ dataset
        - Common point in dataset being registered

    """
    registration_list: list[tuple[Path, Path, npt.NDArray, npt.NDArray]] = []

    if len(sys.argv) == 1:
        Tk().withdraw()
        registration_file = askopenfilename(
            initialdir="/data/projects/hop/data_repository/Various/neuroglancer_pipeline/registration/registration_list/",
            title="Select list to register",
        )
        with open(registration_file) as file:
            lines = [line.rstrip("\n") for line in file]
        for line in lines:
            try:
                registration_list.append(process_line(line))
            except Exception:
                print("Could not read line")
    elif len(sys.argv) == 2:
        registration_file = sys.argv[1]
        with open(registration_file) as file:
            lines = [line.rstrip("\n") for line in file]
        for line in lines:
            try:
                registration_list.append(process_line(line))
            except Exception:
                print("Could not read line")

    elif len(sys.argv) < 4 and len(sys.argv) > 1:
        sys.exit(
            "ERROR: not enough arguments\nThe arguments must be: path fixed image, path moved image, coordinate point fixed image ([x,y,z]), coordinate point moved image ([x,y,z])\n"
            "python3 ITK_registration.py /data/projects/hop/data_repository/LADAF-2021-17/heart/19.85um_complete-organ_bm18/39.70um_LADAF-2021-17_heart_pag-0.04_0.12_/ "
            "/data/projects/hop/data_repository/LADAF-2021-17/heart/6.5um_ROI-07_bm18/50.88um_LADAF-2021-17_heart_VOI-07_pag-0.05_0.24_/ [2087,1985,2377] [970,1465,47]"
        )

    elif len(sys.argv) == 5:
        path_full = Path(sys.argv[1])
        path_roi = Path(sys.argv[2])
        pt_full = np.array(ast.literal_eval(sys.argv[3]))
        pt_roi = np.array(ast.literal_eval(sys.argv[4]))
        registration_list.append((path_full, path_roi, pt_full, pt_roi))

    else:
        raise RuntimeError("Too many arguments")

    return registration_list


if __name__ == "__main__":
    print("Starting registration")

    registration_list = get_registration_list()

    for path_full, path_roi, pt_full, pt_roi in registration_list:
        log_fname = f"registration_{path_roi.name}.log"
        log_file = list(path_roi.parent.glob(log_fname))
        if len(log_file) > 1:
            print("More than one log file found, skipping dataset")
            continue

        with open(log_file[0]) as file:
            lines = [line.rstrip("\n") for line in file]
        if "Total time =" in lines[-1]:
            print(f"Log file {log_file[0]} already present and finished")
            continue

        start_time = time.time()

        # Setup logging
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        handler = logging.FileHandler(
            str(path_roi.parent / log_fname),
            "w",
            "utf-8",
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        root_logger.addHandler(handler)

        logging.info(
            f"Command: python3 ITK_registration.py {path_full} {path_roi} [{pt_full[0]},{pt_full[1]},{pt_full[2]}] [{pt_roi[0]},{pt_roi[1]},{pt_roi[2]}]\n"
        )
        logging.info("REGISTRATION:\n")
        logging.info(f"Full organ image = {path_full}")
        logging.info(f"ROI image = {path_roi}")
        logging.info(f"Full organ common point = {pt_full}")
        logging.info(f"ROI common point = {pt_roi}")

        # Run registration
        registration_pipeline(
            path_roi=path_roi,
            path_full=path_full,
            pt_roi=tuple(pt_roi),
            pt_full=tuple(pt_full),
        )

        print("--- %s seconds ---" % (time.time() - start_time))
        logging.info(f"\nTotal time = {time.time() - start_time}")

        logging.getLogger().removeHandler(logging.getLogger().handlers[0])
    print("List finished")

    # EXAMPLE
# python3 ITK_registration.py /data/projects/hop/data_repository/LADAF-2021-17/kidney_right/25um_complete-organ_bm05/50um_LADAF-2021-17_kidney-right_pag-0.02_0.04_ /data/projects/hop/data_repository/LADAF-2021-17/kidney_right/2.6um_VOI-03.1/5.2um_LADAF-2021-17_kidney-right_VOI-03.1_V2_pag-0.06_0.08_ [230,535,627] [1126,384,560]

# python3 ITK_registration.py /data/projects/hop/data_repository/LADAF-2021-64/heart/19.89um_complete-organ_bm18/159.12um_LADAF-2021-64_heart_pag-0.09_1.06_/ /data/projects/hop/data_repository/LADAF-2021-64/heart/6.51um_VOI-04/52.08um_LADAF-2021-64_heart_VOI-04_pag-0.03_0.26_/ [441,531,420] [274,196,319]
