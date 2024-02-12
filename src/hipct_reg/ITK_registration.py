"""
Author: Joseph Brunet (joseph.brunet@ucl.ac.uk)

This script does the registration of HiP-CT high-resolution volumes to low-resolution
volumes.
"""

import ast
import glob
import logging
import math
import multiprocessing
import os
import sys
import time
from tkinter import Tk  # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename

import itk
import numpy as np
import numpy.typing as npt
import psutil
import SimpleITK as sitk
import skimage.io
import skimage.measure
from scipy.spatial.transform import Rotation as ROT

from .helpers import import_im, test_file_type

MAX_THREADS = 0  # 0 if all


def registration_rot(
    fixed_image: sitk.Image,
    moving_image: sitk.Image,
    trans_point: npt.NDArray,
    pt_fixed: npt.NDArray,
    zrot: float,
    angle_range: float,
    angle_step: float,
    fiji: bool = True,
):
    """
    Parameters
    ----------
    fixed_image, moving_image :
        The images being registered.
    trans_point :
        Vector from [0, 0, 0] voxel in fixed image to [0, 0, 0] voxel in moving image.
    pt_fixed :
        Common point in the fixed image. In units of pixels.
    zrot :
        Initial rotation for the registration. In units of degrees.
    angle_range :
        Range of angles to scan. In units of degrees.
    angle_step :
        Step to take when scanning range of angles. In units of degrees.

    """
    pixel_size_fixed = fixed_image.GetSpacing()[0]

    R = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetMetricSamplingStrategy(R.NONE)
    R.SetMetricSamplingPercentage(0.01)

    R.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    logging.info(f"\nStep= {angle_step} deg")

    R.SetOptimizerAsExhaustive(
        numberOfSteps=[0, 0, int((angle_range / 2) / angle_step), 0, 0, 0],
        stepLength=np.deg2rad(angle_step),
    )

    R.SetOptimizerScalesFromPhysicalShift()

    offset = pixel_size_fixed * trans_point

    rotation_center = pt_fixed * pixel_size_fixed

    theta_x = 0.0
    theta_y = 0.0
    theta_z = np.deg2rad(zrot)
    translation = -offset

    print(
        "OFFSET IS ",
        np.append(np.rad2deg(np.array([theta_x, theta_y, theta_z])), translation),
    )
    print("TRANSLATION X: ", -translation[0] / pixel_size_fixed)
    print("TRANSLATION Y: ", -translation[1] / pixel_size_fixed)
    print("TRANSLATION Z: ", -translation[2] / pixel_size_fixed)

    print("ROTATION CENTER: ", rotation_center / pixel_size_fixed)

    print("CPU usage: ", psutil.cpu_percent())
    print("RAM usage: ", psutil.virtual_memory().percent)

    initial_transform = sitk.Euler3DTransform(
        rotation_center, theta_x, theta_y, theta_z, translation
    )

    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    R.SetInitialTransform(initial_transform, inPlace=True)

    if fiji:
        moving_resampled = sitk.Resample(
            moving_image,
            fixed_image,
            initial_transform,
            sitk.sitkLinear,
            0,
            fixed_image.GetPixelID(),
        )
        sitk.Show(0.6 * moving_resampled + 0.4 * fixed_image, "before rot")

    global metric
    metric = []  # type: ignore[name-defined]

    def command_iteration(
        method,
        pixel_size_fixed,
        trans_point,
        translation,
        rotation_center,
        moving_image,
        fixed_image,
        pt_fixed,
    ):
        global metric
        metric.append(method.GetMetricValue())
        print(
            f"{method.GetOptimizerIteration()} "
            + f"= {method.GetMetricValue()} "
            + f"\nTRANSLATION: {np.array(method.GetOptimizerPosition())[3:]/pixel_size_fixed}"
            + f"\nROTATION: {np.rad2deg(np.array(method.GetOptimizerPosition()))[0:3]}"
            + f"\nCPU usage: {psutil.cpu_percent()}"
            + f"\nRAM usage: {psutil.virtual_memory().percent}"
            + f"\nCPU COUNT: {multiprocessing.cpu_count()}"
        )

    R.AddCommand(
        sitk.sitkIterationEvent,
        lambda: command_iteration(
            R,
            pixel_size_fixed,
            trans_point,
            translation,
            rotation_center,
            moving_image,
            fixed_image,
            pt_fixed,
        ),
    )

    transform_rotation = R.Execute(fixed_image, moving_image)

    if fiji:
        moving_resampled = sitk.Resample(
            moving_image,
            fixed_image,
            transform_rotation,
            sitk.sitkLinear,
            0,
            fixed_image.GetPixelID(),
        )
        sitk.Show(0.6 * moving_resampled + 0.4 * fixed_image, "after rot")

    # Always check the reason optimization terminated.
    print("OFFSET IS ", trans_point, "\n")
    print(
        f"nTRANSLATION: {np.array(transform_rotation.GetParameters()[3:])/fixed_image.GetSpacing()[0]}",
        "\n",
    )
    print(
        f"nROTATION: {np.rad2deg(np.array(transform_rotation.GetParameters()[0:3]))}",
        "\n",
    )

    print(f"Final metric value: {R.GetMetricValue()}")
    print(f"Optimizer's stopping condition, {R.GetOptimizerStopConditionDescription()}")

    print(transform_rotation)

    logging.info(f"Final metric value: {R.GetMetricValue()}")
    logging.info(
        f"nResult Angle: {np.rad2deg(np.array(transform_rotation.GetParameters()[2]))}"
    )

    rigid_euler = sitk.Euler3DTransform(
        rotation_center, theta_x, theta_y, theta_z, translation
    )
    basic_transform = sitk.Similarity3DTransform()
    basic_transform.SetMatrix(rigid_euler.GetMatrix())
    basic_transform.SetTranslation(rigid_euler.GetTranslation())
    basic_transform.SetCenter(rigid_euler.GetCenter())

    return transform_rotation


def registration_ITKelastix(fixed_image, moving_image, trans_point, zrot, pt_fixed):
    print("\n-----------------")
    print("\nBeginning of initialization")
    print("\n-----------------\n")

    pixel_size_fixed = fixed_image.GetSpacing()[0]

    iniParameterMap = itk.ParameterObject.New()
    parameterMap = iniParameterMap.GetDefaultParameterMap("rigid")

    parameterMap["FixedImageDimension"] = ["3"]
    parameterMap["MovingImageDimension"] = ["3"]
    parameterMap["MaximumNumberOfIterations"] = ["1"]
    parameterMap["NumberOfResolutions"] = ["1"]
    parameterMap["WriteResultImage"] = ["false"]

    iniParameterMap.AddParameterMap(parameterMap)

    def convertSI(fixed_image):
        new_sitk_image = sitk.GetImageFromArray(
            itk.GetArrayFromImage(fixed_image),
            isVector=fixed_image.GetNumberOfComponentsPerPixel() > 1,
        )
        new_sitk_image.SetOrigin(tuple(fixed_image.GetOrigin()))
        new_sitk_image.SetSpacing(tuple(fixed_image.GetSpacing()))
        new_sitk_image.SetDirection(
            itk.GetArrayFromMatrix(fixed_image.GetDirection()).flatten()
        )
        return new_sitk_image

    MAX_THREADS = psutil.cpu_count() - 2

    # Call registration function
    result_image, initial_transform = itk.elastix_registration_method(
        moving_image,
        fixed_image,
        parameter_object=iniParameterMap,
        number_of_threads=MAX_THREADS,
        log_to_console=True,
    )

    print("\n\n\nENNND\n")
    print(initial_transform)

    print(type(result_image))

    print(result_image)

    print("\n\n\nENNND222222222\n")

    print("\n-----------------")
    print(r"End of initialization")
    print("\n-----------------\n")

    # use a Python trick to create the offset list based on the dimension
    offset = pixel_size_fixed * trans_point

    rotation_center = pt_fixed * pixel_size_fixed

    theta_x = 0.0
    theta_y = 0.0
    theta_z = np.deg2rad(zrot)
    translation = -offset

    print(
        "OFFSET IS ",
        np.append(np.rad2deg(np.array([theta_x, theta_y, theta_z])), translation),
    )
    print("TRANSLATION X: ", -translation[0] / pixel_size_fixed)
    print("TRANSLATION Y: ", -translation[1] / pixel_size_fixed)
    print("TRANSLATION Z: ", -translation[2] / pixel_size_fixed)

    print("ROTATION CENTER: ", rotation_center / pixel_size_fixed)

    print("ANGLE X: ", theta_x)
    print("ANGLE Y: ", theta_y)
    print("ANGLE Z: ", theta_z)

    print(type(initial_transform))
    print(initial_transform["CenterOfRotationPoint"])

    initial_transform["CenterOfRotationPoint"] = (
        str(rotation_center[0]),
        str(rotation_center[1]),
        str(rotation_center[2]),
    )
    initial_transform["TransformParameters"] = (
        str(theta_x),
        str(theta_y),
        str(theta_z),
        str(translation[0]),
        str(translation[1]),
        str(translation[2]),
    )

    print("TESTTTTTT")

    print(initial_transform)

    from itkwidgets import checkerboard

    checkerboard(moving_image, initial_transform, pattern=5)

    sys.exit()


def registration_sitk(
    fixed_image, moving_image, trans_point, zrot, pt_fixed, fiji=True
):
    pixel_size_fixed = fixed_image.GetSpacing()[0]
    pixel_size_moved = moving_image.GetSpacing()[0]

    R = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetMetricSamplingStrategy(R.RANDOM)

    R.SetMetricSamplingPercentage(0.01)

    R.SetInterpolator(sitk.sitkLinear)

    R.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-7,
        numberOfIterations=2000,
        maximumNumberOfCorrections=20,
        maximumNumberOfFunctionEvaluations=2000,
        costFunctionConvergenceFactor=1e9,
    )

    R.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    R.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 2, 1, 1, 1])
    R.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 1, 1, 1, 0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    w = 10
    R.SetOptimizerWeights([w, w, w, w, w, w, w / 1000])

    offset = pixel_size_fixed * trans_point

    # Get coordinate of centre of rotation of the moving image
    rotation_center = offset.copy()
    rotation_center[0] = rotation_center[0] + int(
        pixel_size_moved * moving_image.GetSize()[0] / 2
    )
    rotation_center[1] = rotation_center[1] + int(
        pixel_size_moved * moving_image.GetSize()[1] / 2
    )

    rotation_center = pt_fixed * pixel_size_fixed

    theta_x = 0.0
    theta_y = 0.0
    theta_z = np.deg2rad(zrot)
    translation = -offset

    print("OFFSET IS ", np.append([theta_x, theta_y, theta_z], translation))
    print("TRANSLATION X: ", -translation[0] / pixel_size_fixed)
    print("TRANSLATION Y: ", -translation[1] / pixel_size_fixed)
    print("TRANSLATION Z: ", -translation[2] / pixel_size_fixed)

    print("ROTATION CENTER: ", rotation_center)

    rigid_euler = sitk.Euler3DTransform(
        rotation_center, theta_x, theta_y, theta_z, translation
    )
    initial_transform = sitk.Similarity3DTransform()
    initial_transform.SetMatrix(rigid_euler.GetMatrix())
    initial_transform.SetTranslation(rigid_euler.GetTranslation())
    initial_transform.SetCenter(rigid_euler.GetCenter())

    del rigid_euler

    R.SetInitialTransform(initial_transform, inPlace=True)

    if fiji:
        moving_resampled = sitk.Resample(
            moving_image,
            fixed_image,
            initial_transform,
            sitk.sitkLinear,
            0,
            fixed_image.GetPixelID(),
        )
        sitk.Show(0.6 * moving_resampled + 0.4 * fixed_image, "ini")

    global metric
    metric = []

    def command_iteration(method, pixel_size, trans_point):
        global metric
        metric.append(method.GetMetricValue())

        q0, q1, q2, q3 = method.GetOptimizerPosition()[0:4]
        r = ROT.from_quat([q0, q1, q2, q3])
        theta_x, theta_y, theta_z = np.rad2deg(r.as_rotvec())

        print(
            f"{method.GetOptimizerIteration()} "
            + f" = {method.GetMetricValue()} "
            + f"\nGetOptimizerPosition: {method.GetOptimizerPosition()}"
            + f"\nTRANSLATION: {np.array(method.GetOptimizerPosition())[3:6]/pixel_size}"
            + f"\nROTATION: {theta_x}  {theta_y}  {theta_z}"
            + f"\nSCALE: {np.array(method.GetOptimizerPosition())[-1]}"
            + f"\nCPU usage: {psutil.cpu_percent()}"
            + f"\nRAM usage: {psutil.virtual_memory().percent}"
        )

    R.AddCommand(
        sitk.sitkIterationEvent,
        lambda: command_iteration(R, pixel_size_fixed, trans_point),
    )

    final_transform = R.Execute(fixed_image, moving_image)

    if fiji:
        moving_resampled = sitk.Resample(
            moving_image,
            fixed_image,
            final_transform,
            sitk.sitkLinear,
            0,
            fixed_image.GetPixelID(),
        )
        sitk.Show(0.6 * moving_resampled + 0.4 * fixed_image, "Final")

    # Always check the reason optimization terminated.
    print("OFFSET IS ", trans_point, "\n")
    print(
        f"TRANSLATION: {np.array(final_transform.GetParameters()[3:])/fixed_image.GetSpacing()[0]}",
        "\n",
    )
    print(
        f"ROTATION: {np.rad2deg(np.array(final_transform.GetParameters()[0:3]))}", "\n"
    )

    print(f"Final metric value: {R.GetMetricValue()}")
    print(f"Optimizer's stopping condition, {R.GetOptimizerStopConditionDescription()}")

    logging.info(
        f"TRANSLATION: {np.array(final_transform.GetParameters()[3:])/fixed_image.GetSpacing()[0]}\n"
    )
    logging.info(
        f"ROTATION: {np.rad2deg(np.array(final_transform.GetParameters()[0:3]))}\n"
    )
    logging.info(f"Final metric value: {R.GetMetricValue()}")

    print(final_transform)
    return initial_transform, final_transform


def registration_simpleElastix(
    fixed_image, moving_image, trans_point, zrot, pt_fixed, fiji=True
):
    print("\n-----------------")
    print("\nBeginning of initialization")
    print("\n-----------------\n")

    pixel_size_fixed = fixed_image.GetSpacing()[0]

    R_ini = sitk.ElastixImageFilter()
    R_ini.SetFixedImage(fixed_image)
    R_ini.SetMovingImage(fixed_image)

    if MAX_THREADS != 0:
        R_ini.SetNumberOfThreads(MAX_THREADS)

    iniParameterMap = sitk.GetDefaultParameterMap("rigid")
    iniParameterMap["MaximumNumberOfIterations"] = ["1"]
    iniParameterMap["NumberOfResolutions"] = ["1"]
    iniParameterMap["WriteResultImage"] = ["false"]
    R_ini.SetParameterMap(iniParameterMap)

    print(sitk.PrintParameterMap(iniParameterMap))

    R_ini.LogToConsoleOn()
    R_ini.Execute()

    print(
        f"\nCPU usage: {psutil.cpu_percent()}"
        + f"\nRAM usage: {psutil.virtual_memory().percent}"
    )

    print("\n-----------------")
    print(r"End of initialization")
    print("\n-----------------\n")

    # use a Python trick to create the offset list based on the dimension
    offset = pixel_size_fixed * trans_point
    rotation_center = pt_fixed * pixel_size_fixed

    theta_x = 0.0
    theta_y = 0.0
    theta_z = np.deg2rad(zrot)
    translation = -offset

    print(
        "OFFSET IS ",
        np.append(np.rad2deg(np.array([theta_x, theta_y, theta_z])), translation),
    )
    print("TRANSLATION X: ", -translation[0] / pixel_size_fixed)
    print("TRANSLATION Y: ", -translation[1] / pixel_size_fixed)
    print("TRANSLATION Z: ", -translation[2] / pixel_size_fixed)

    print("ROTATION CENTER: ", rotation_center / pixel_size_fixed)

    initial_transform = R_ini.GetTransformParameterMap()
    del R_ini

    print("ANGLE X: ", theta_x)
    print("ANGLE Y: ", theta_y)
    print("ANGLE Z: ", theta_z)

    initial_transform[0]["CenterOfRotationPoint"] = (
        str(rotation_center[0]),
        str(rotation_center[1]),
        str(rotation_center[2]),
    )
    initial_transform[0]["TransformParameters"] = (
        str(theta_x),
        str(theta_y),
        str(theta_z),
        str(translation[0]),
        str(translation[1]),
        str(translation[2]),
    )

    moving_image = sitk.Transformix(moving_image, initial_transform)
    sitk.Show(0.5 * moving_image + 0.5 * fixed_image, "3")

    print(moving_image.GetSize())
    print(moving_image.GetPixelIDTypeAsString())

    print(fixed_image.GetSize())
    print(fixed_image.GetPixelIDTypeAsString())

    if fiji:
        sitk.Show(0.5 * moving_image + 0.5 * fixed_image, "Initialisation")

    print("\n-----------------")
    print("\nBeginning of registration")
    print("\n-----------------\n")

    R = sitk.ElastixImageFilter()

    R.SetFixedImage(fixed_image)
    R.SetMovingImage(moving_image)

    R.SetInitialTransformParameterFileName("initial_transform_parameter_map.txt")

    parameterMap = sitk.GetDefaultParameterMap("rigid")

    parameterMap["FixedImageDimension"] = ["3"]
    parameterMap["MovingImageDimension"] = ["3"]
    parameterMap["WriteIterationInfo"] = ["true"]
    parameterMap["WriteResultImage"] = ["false"]

    parameterMap["Registration"] = ["MultiResolutionRegistration"]
    parameterMap["FixedImagePyramid"] = ["FixedSmoothingImagePyramid"]
    parameterMap["MovingImagePyramid"] = ["MovingSmoothingImagePyramid"]
    parameterMap["Interpolator"] = ["LinearInterpolator"]
    parameterMap["ResampleInterpolator"] = ["FinalBSplineInterpolator"]
    parameterMap["Resampler"] = ["DefaultResampler"]

    parameterMap["Metric"] = ["AdvancedMattesMutualInformation"]
    parameterMap["Optimizer"] = ["AdaptiveStochasticGradientDescent"]

    parameterMap["NumberOfResolutions"] = ["3"]
    parameterMap["ImagePyramidSchedule"] = ["4", "4", "2", "2", "2", "1", "1", "1", "1"]

    parameterMap["FinalGridSpacingInVoxels"] = ["8.0", "8.0", "8.0"]

    parameterMap["GridSpacingSchedule"] = [
        "4.0" "4.0" "2.0" "2.0" "2.0" "1.0" "1.0" "1.0" "1.0"
    ]

    ##Sampling
    parameterMap["ImageSampler"] = ["RandomCoordinate"]
    parameterMap["NewSamplesEveryIteration"] = ["true", "true", "true"]
    parameterMap["NumberOfSpatialSamples"] = ["2048", "2048", "4000"]

    # for Matte's Mutual Information
    parameterMap["NumberOfHistogramBins"] = [
        "32",
        "32",
        "32",
    ]
    parameterMap["BSplineInterpolationOrder"] = ["1", "1", "1"]
    parameterMap["FinalBSplineInterpolationOrder"] = ["3"]
    parameterMap["DefaultPixelValue"] = ["0"]

    parameterMap["MaximumNumberOfIterations"] = ["2000"]
    parameterMap["MaximumNumberOfIterations"] = ["100"]

    parameterMap["MaximumStepLength"] = [str(pixel_size_fixed)]

    parameterMap["CenterOfRotationPoint"] = (
        str(rotation_center[0]),
        str(rotation_center[1]),
        str(rotation_center[2]),
    )

    parameterMap["Transform"] = ["SimilarityTransform"]

    R.SetParameterMap(parameterMap)
    R.LogToConsoleOn()

    if MAX_THREADS != 0:
        R.SetNumberOfThreads(MAX_THREADS)

    R.Execute()

    sitk.PrintParameterMap(R.GetTransformParameterMap()[0])
    print("\n-----------------")
    print(r"End of registration")
    print("\n-----------------\n")

    R.PrintParameterMap()

    final_transform = R.GetTransformParameterMap()
    print(final_transform[0]["TransformParameters"])

    print("\n\n-------------------\n\n")
    print("ROTATION CENTER: ", rotation_center / pixel_size_fixed)

    # Always check the reason optimization terminated.
    print(
        "\nOFFSET IS ",
        np.rad2deg(np.array([theta_x, theta_y, theta_z])),
        offset / pixel_size_fixed,
        "\n",
    )
    translation_transform = np.array(
        final_transform[0]["TransformParameters"][3:6]
    ).astype(float)
    print(f"\nTRANSLATION: {translation_transform/pixel_size_fixed}", "\n")
    rotation_transform = np.array(
        final_transform[0]["TransformParameters"][0:3]
    ).astype(float)
    print(f"\nROTATION: {np.rad2deg(rotation_transform)}", "\n")
    rotation_transform[2] = rotation_transform[2] + theta_z
    print(f"\nROTATION: {np.rad2deg(rotation_transform)}", "\n")
    print(f"\nSCALE: {final_transform[0]['TransformParameters'][6]}", "\n")

    final_transform[0]["CenterOfRotationPoint"] = (
        str(rotation_center[0]),
        str(rotation_center[1]),
        str(rotation_center[2]),
    )

    if fiji:
        moved_image = sitk.Transformix(moving_image, final_transform)
        sitk.Show(0.5 * moved_image + 0.5 * fixed_image, "Final")

    Rx = float(final_transform[0]["TransformParameters"][0])
    Ry = float(final_transform[0]["TransformParameters"][1])
    Rz = float(final_transform[0]["TransformParameters"][2]) + theta_z
    Tx = -float(final_transform[0]["TransformParameters"][3]) - offset[0]
    Ty = -float(final_transform[0]["TransformParameters"][4]) - offset[1]
    Tz = -float(final_transform[0]["TransformParameters"][5]) - offset[2]
    scale = float(final_transform[0]["TransformParameters"][6])
    RCx = rotation_center[0]
    RCy = rotation_center[1]
    RCz = rotation_center[2]

    print(RCx, RCy, RCz, Rx, Ry, Rz, Tx, Ty, Tz)

    rigid_euler = sitk.Euler3DTransform([RCx, RCy, RCz], Rx, Ry, Rz, [Tx, Ty, Tz])

    similarity_transform = sitk.Similarity3DTransform()
    similarity_transform.SetMatrix(rigid_euler.GetMatrix())
    similarity_transform.SetTranslation(rigid_euler.GetTranslation())
    similarity_transform.SetCenter(rigid_euler.GetCenter())
    similarity_transform.SetScale(scale)

    del rigid_euler

    moved_image = sitk.Transformix(moving_image, final_transform)
    sitk.Show(0.6 * moved_image + 0.4 * fixed_image, "Final")

    return final_transform


def get_pixel_size(path: str) -> float:
    """
    Get pixel size in nm from a path.
    """
    if path[-1] == "/":
        path = path[:-1]
    return float(path.split("/")[-1].split("um")[0]) / 1000


def registration_pipeline(
    path_fixed: str, path_moved: str, pt_fixed: npt.NDArray, pt_moved: npt.NDArray
) -> None:
    """
    Parameters
    ----------
    path_fixed :
        Path to the fixed dataset.
    path_moved :
        Path to the dataset that is being registered.
    pt_fixed :
        Common point in the fixed dataset. In units of pixels.
    pt_moved :
        Common point in the dataset being moved. In units of pixels.

    """
    # Crop the zoom scan to transform the circle fov into a square, thus avoiding the
    # NaN part in the image
    crop_circle_moved = False

    pixel_size_fixed = get_pixel_size(path_fixed)
    pixel_size_moved = get_pixel_size(path_moved)

    print(f"Path fixed = {path_fixed}")
    print(f"Path zoom = {path_moved}")

    print(
        f"""The pixel sizes I found are:\n\n
          - {pixel_size_fixed} mm for the low-resolution volume\n\n
          - {pixel_size_moved} mm for the high-resolution volume\n\n"""
    )

    logging.info(f"\nPixel size Fixed scan = {round(pixel_size_fixed,5)} mm")
    logging.info(f"Pixel size Moving scan = {round(pixel_size_moved,5)} mm")

    logging.info(
        f"\nCrop the zoom volume to avoid external circle = {crop_circle_moved}"
    )

    # Gather info on images
    file_type_fixed = test_file_type(path_fixed)
    file_type_moved = test_file_type(path_moved)

    N_fixed = len(glob.glob(f"{path_fixed}/*.{file_type_fixed}"))
    N_moved = len(glob.glob(f"{path_moved}/*.{file_type_moved}"))

    img = skimage.io.imread(glob.glob(path_fixed + "/*.tif")[0])
    print(f"\nI found {N_fixed} images of shape {img.shape} in the fixed volume folder")
    logging.info(
        f"\nI found {N_fixed} images of shape {img.shape} in the fixed volume folder"
    )

    img = skimage.io.imread(glob.glob(path_moved + "/*.tif")[0])
    x_dim = img.shape[1]
    y_dim = img.shape[0]
    print(f"I found {N_moved} images of shape {img.shape} in the moved volume folder")
    logging.info(
        f"I found {N_moved} of shape {img.shape} images in the moved volume folder"
    )

    del img

    size_moved = N_moved * x_dim * y_dim * 2 / (1024 * 1024 * 1024)
    print(f"Total size of zoom scan is is {size_moved} GB")
    logging.info(f"Total size of zoom scan is is {size_moved} GB")

    binning_moved = 1 + size_moved // 50  # I put a limit of 50 GB
    binning_moved = int((pixel_size_fixed // pixel_size_moved) / 2) * 2  # other way

    # binning_moved = 4
    binning_fixed = 1

    if binning_fixed != 1:
        print(f"\nA binning of {binning_fixed} will be applied to the fixed volume\n")
        N_fixed = math.ceil(N_fixed / binning_fixed)
        logging.info(
            f"\nAfter binning: I found {N_fixed} images in the fixed volume folder"
        )
    logging.info(f"\nFixed volume - Binning factor = {binning_fixed}")

    if binning_moved != 1:
        print(f"\nA binning of {binning_moved} will be applied to the zoom volume\n")
        N_moved = math.ceil(N_moved / binning_moved)
        logging.info(
            f"After binning: I found {N_moved} images in the moved volume folder"
        )
    logging.info(f"\nZoom volume - Binning factor = {binning_moved}")

    pixel_size_fixed = pixel_size_fixed * binning_fixed
    pixel_size_moved = pixel_size_moved * binning_moved

    pt_fixed = pt_fixed / binning_fixed
    pt_moved = pt_moved / binning_moved

    # Vector from [0, 0, 0] voxel in fixed image to [0, 0, 0] voxel in moving image
    trans_point = pt_fixed - pt_moved * (pixel_size_moved / pixel_size_fixed)

    print("---------------------------------------------------")
    print("\nIMPORTATION OF SCAN TO REGISTER\n")
    logging.info("\n---------------------------------------------------")
    logging.info("IMPORTATION OF SCAN TO REGISTER\n")

    moved_z = (0, N_moved)
    logging.info(f"Moving scan crop z = {moved_z}")

    logging.info("\n---Start importation of Moving image")
    moving_image = import_im(
        path_moved,
        pixel_size_moved,
        crop_z=moved_z,
        bin_factor=binning_moved,
    )
    logging.info("---Moving image imported successfully")

    print("moving_image")
    print(moving_image.GetSize())
    print(moving_image.GetPixelIDTypeAsString())
    print(moving_image.GetSpacing())

    # Get values to crop the fixed dataset
    # TODO: check this maths
    zmin = int(
        pt_fixed[2] - 1.2 * ((pt_moved[2]) * pixel_size_moved / pixel_size_fixed)
    )
    zmax = int(
        pt_fixed[2]
        + 1.2
        * (
            ((moved_z[1] - moved_z[0]) - pt_moved[2])
            * pixel_size_moved
            / pixel_size_fixed
        )
    )
    zmin = max(zmin, 0)
    zmax = min(zmax, int(N_fixed))

    fixed_z = (zmin, zmax)
    trans_point[2] = trans_point[2] - fixed_z[0]
    pt_fixed[2] = pt_fixed[2] - fixed_z[0]
    logging.info(f"\nFixed scan crop z = {fixed_z}")

    print("---------------------------------------------------")
    print("\nIMPORTATION OF REFERENCE SCAN\n")

    logging.info("\n---Start importation of Fixed image")
    fixed_image = import_im(
        path_fixed,
        pixel_size_fixed,
        crop_z=fixed_z,
        bin_factor=binning_fixed,
    )
    logging.info("---Fixed image imported successfully")

    print("fixed_image")
    print(fixed_image.GetSize())
    print(fixed_image.GetPixelIDTypeAsString())
    print(fixed_image.GetSpacing())

    print("IMPORTATION DONE\n\n")
    print("---------------------------------------------------")
    print("REGISTRATION\n\n")
    print("---------------------------------------------------")
    logging.info(
        "\n---------------------------------------------------\nREGISTRATION\n"
    )

    fixed_image = sitk.Normalize(fixed_image)
    moving_image = sitk.Normalize(moving_image)
    logging.info("\nImages normalized")

    pixelType = sitk.sitkFloat32
    fixed_image = sitk.Cast(fixed_image, pixelType)
    moving_image = sitk.Cast(moving_image, pixelType)

    print("\nFIND Z ANGLE\n")
    logging.info("\n---\nRotation registration started\n---")
    zrot = 0

    # Try a full 360 deg first at a coarse step
    angle_range = 360
    angle_step = 2.0
    transform_rotation = registration_rot(
        fixed_image, moving_image, trans_point, pt_fixed, zrot, angle_range, angle_step
    )
    zrot = np.rad2deg(np.array(transform_rotation.GetParameters()[0:3]))[2]

    # Now try a smaller angular step
    angle_range = 5
    angle_step = 0.1
    transform_rotation = registration_rot(
        fixed_image, moving_image, trans_point, pt_fixed, zrot, angle_range, angle_step
    )
    zrot = np.rad2deg(np.array(transform_rotation.GetParameters()[0:3]))[2]

    if zrot < 0:
        zrot = zrot + 360
    print("Zrot = ", zrot)
    logging.info(f"\nRotation registration finished successfully (z rotation = {zrot})")

    print("\nSTART SIMILARITY REGISTRATION")

    logging.info("\n---\nSimilarity registration started\n---")
    initial_transform, final_transform = registration_sitk(
        fixed_image, moving_image, trans_point, zrot, pt_fixed
    )

    print("\n\n\nRESULTS\n")
    logging.info("\n---------------------------------------------------\nRESULTS\n")

    logging.info("RAW RESULTS FROM SIMILARITY TRANSFORM :\n")
    logging.info(f"Translation = {final_transform.GetTranslation()}")
    logging.info(f"Center of rot = {final_transform.GetCenter()}")
    logging.info(f"Matrix = {final_transform.GetMatrix()}")
    logging.info(f"Versor = {final_transform.GetVersor()}")
    logging.info(f"Scale = {final_transform.GetScale()}")

    print("-------------------------------------------")
    logging.info(
        "\n---------------------------------------------------\nFORMATED PARAMETERS FOR NEUROGLANCER\n"
    )
    logging.info("\nTRANSLATIONS")

    final_transform_inverse = final_transform.GetInverse()

    # set point to match dimension
    point = [0, 0, 0]
    transformed_point = final_transform_inverse.TransformPoint(point)

    Tx = transformed_point[0]
    Ty = transformed_point[1]
    Tz = transformed_point[2] + fixed_z[0] * pixel_size_fixed

    print("Tx=", Tx)
    print("Ty=", Ty)
    print("Tz= ", Tz)
    logging.info(f"Translation (Tx,Ty,Tz) = {Tx},{Ty},{Tz}")

    print("\n\n")
    print("-------------------------------------------")
    print("ROTATION :\n")
    logging.info("\nROTATION")

    logging.info(f"Inverse rotation matrix = {final_transform_inverse.GetMatrix()}")

    print("\n\n")
    logging.info("\nSCALE")

    scale = final_transform_inverse.GetScale()
    print("\nScale= pixel size * ", scale)
    print("NEW PIXEL SIZE= ", pixel_size_moved * scale)
    logging.info(f"Scale= pixel size * {scale}")
    logging.info(f"New pixel size= {pixel_size_moved * scale}")


def process_line(line: str) -> tuple[str, str, npt.NDArray, npt.NDArray]:
    """
    Process a single line in a registration list file.
    """
    ls = line.split()
    pt_fixed = np.array(ast.literal_eval(ls[2]))
    pt_moved = np.array(ast.literal_eval(ls[3]))
    return ls[0], ls[1], pt_fixed, pt_moved


def get_registration_list() -> list[tuple[str, str, npt.NDArray, npt.NDArray]]:
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
    registration_list: list[tuple[str, str, npt.NDArray, npt.NDArray]] = []

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
        path_fixed = sys.argv[1]
        path_moved = sys.argv[2]
        pt_fixed = np.array(ast.literal_eval(sys.argv[3]))
        pt_moved = np.array(ast.literal_eval(sys.argv[4]))
        registration_list.append((path_fixed, path_moved, pt_fixed, pt_moved))

    else:
        raise RuntimeError("Too many arguments")

    return registration_list


if __name__ == "__main__":
    print("Starting registration")

    registration_list = get_registration_list()

    for path_fixed, path_moved, pt_fixed, pt_moved in registration_list:
        log_file = glob.glob(
            f"{os.path.dirname(path_moved.rstrip('/'))}/registration_{os.path.basename(path_moved)}.log"
        )
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
            f"{os.path.dirname(path_moved.rstrip('/'))}/registration_{os.path.basename(os.path.normpath(path_moved))}.log",
            "w",
            "utf-8",
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        root_logger.addHandler(handler)

        logging.info(
            f"Command: python3 ITK_registration.py {path_fixed} {path_moved} [{pt_fixed[0]},{pt_fixed[1]},{pt_fixed[2]}] [{pt_moved[0]},{pt_moved[1]},{pt_moved[2]}]\n"
        )
        logging.info("REGISTRATION:\n")
        logging.info(f"Fixed image = {path_fixed}")
        logging.info(f"Moving image = {path_moved}")
        logging.info(f"\nPoint fixed = {pt_fixed}")
        logging.info(f"Point moved = {pt_moved}")

        # Run registration
        registration_pipeline(path_fixed, path_moved, pt_fixed, pt_moved)

        print("--- %s seconds ---" % (time.time() - start_time))
        logging.info(f"\nTotal time = {time.time() - start_time}")

        logging.getLogger().removeHandler(logging.getLogger().handlers[0])
    print("List finished")

    # EXAMPLE
# python3 ITK_registration.py /data/projects/hop/data_repository/LADAF-2021-17/kidney_right/25um_complete-organ_bm05/50um_LADAF-2021-17_kidney-right_pag-0.02_0.04_ /data/projects/hop/data_repository/LADAF-2021-17/kidney_right/2.6um_VOI-03.1/5.2um_LADAF-2021-17_kidney-right_VOI-03.1_V2_pag-0.06_0.08_ [230,535,627] [1126,384,560]

# python3 ITK_registration.py /data/projects/hop/data_repository/LADAF-2021-64/heart/19.89um_complete-organ_bm18/159.12um_LADAF-2021-64_heart_pag-0.09_1.06_/ /data/projects/hop/data_repository/LADAF-2021-64/heart/6.51um_VOI-04/52.08um_LADAF-2021-64_heart_VOI-04_pag-0.03_0.26_/ [441,531,420] [274,196,319]
