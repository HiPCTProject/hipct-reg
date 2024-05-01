import logging
from typing import TypedDict

import numpy as np
import numpy.typing as npt
import psutil
import SimpleITK as sitk
from scipy.spatial.transform import Rotation as ROT

from hipct_reg.types import RegistrationInput

MAX_THREADS = 0  # 0 if all


class RotRegMetrics(TypedDict):
    rotation: list[float]
    metric: list[float]


def registration_rot(
    reg_input: RegistrationInput,
    *,
    zrot: float,
    angle_range: float,
    angle_step: float,
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

    new_zrot = np.rad2deg(transform_rotation.GetAngleZ())
    logging.debug(f"Final metric value = {R.GetMetricValue()}")
    logging.debug(f"Stopping condition = {R.GetOptimizerStopConditionDescription()}")
    logging.info(f"Registered rotation angele = {new_zrot} deg")
    logging.info("")

    return transform_rotation, data


def registration_rigid(
    reg_input: RegistrationInput,
    *,
    zrot: float,
) -> tuple[sitk.Similarity3DTransform, float]:
    """
    Run a registration using a full rigid transform.

    The returned transform maps from the ROI iamge to the full-organ image.

    Parameters
    ----------
    zrot :
        Initial rotation for the registration. In units of degrees.

    Returns
    -------
    transform :
        Registered transform from ROI iamge to the full-organ image.

    metric :
        Registration metric at the final step. The registration metric is
        Mattes mutual information.

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
    R.SetMetricSamplingPercentage(0.01, seed=1)

    # Set registration interpolator
    R.SetInterpolator(sitk.sitkLinear)

    # Set registration optimiser settings
    R.SetOptimizerAsLBFGS2()

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
    logging.info(f"translation from ROI to full organ = {translation}")

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
    # Parameters are:
    # - Three rotation angles
    # - Three translation components
    # - Scale factor
    w = 10
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetOptimizerWeights([0, 0, w, w, w, w, w / 1000])

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

    translation_pix = reg_input.roi_image.TransformPhysicalPointToContinuousIndex(
        initial_transform.GetTranslation()
    )
    logging.info(f"initial translation = {translation_pix} pix")
    logging.info("Starting registration...")
    final_transform: sitk.Similarity3DTransform = R.Execute(
        reg_input.roi_image, reg_input.full_image
    )
    logging.info("Registration finished!")
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

    return final_transform, R.GetMetricValue()


def run_registration(
    reg_input: RegistrationInput,
) -> tuple[sitk.Similarity3DTransform, float]:
    """
    Run registration pipeline on pre-loaded/pre-processed images.

    Returns
    -------
    transform :
        Registered transform from ROI iamge to the full-organ image.

    metric :
        Registration metric at the final step. The registration metric is
        Mattes mutual information.
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
    final_transform, final_metric = registration_rigid(
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
    logging.info(f"Metric = {final_metric}")
    logging.info("")

    return final_transform, final_metric
