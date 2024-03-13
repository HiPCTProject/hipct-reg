# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Understanding `SimpleITK` transforms
#
# This notebook is a tutorial on understanding how `SimpleITK` transforms work.

# %%
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import skimage.data
import skimage.filters

# mypy: ignore-errors

# %% [markdown]
# ## Setting up example data
#
# To start with we generate some "full resolution" example data. This is considered our ground truth data, which we will later downsample and/or take cutouts of to simulate other datasets.

# %%
data_full = skimage.data.binary_blobs(
    length=500, n_dim=2, volume_fraction=0.3, rng=np.random.default_rng(seed=1)
)
# Slice data so the size along each axis isn't equal
data_full = data_full[:, :250]
# Blur so it looks a bit nicer
data_full = skimage.filters.gaussian(data_full, sigma=2)
# Take a quick look at the data
plt.imshow(data_full.T, cmap="Grays", origin="lower")


# %% [markdown]
# `SimpleITK` stores data in `Image` objects, which as well as storing the data array also stores:
# - The pixel spacing in physical distance units.
# - The physical coordinate of the (0, 0) pixel.
#
# Here we put the raw data array into an `Image`, but for now leave the spacing and origin as default values. We also define a helper funciton to plot an `Image`, which we'll use throughout this notebook to understand what's going on.


# %%
def show_image(image: sitk.Image, ax: matplotlib.axes.Axes, **imshow_kwargs):
    """
    Function to show a SimpleITK image in a Matplotlib figure.
    """
    origin = image.GetOrigin()
    top_left = image.TransformIndexToPhysicalPoint(image.GetSize())
    cmap = imshow_kwargs.pop("cmap", "Grays")

    ax.imshow(
        sitk.GetArrayFromImage(image),
        extent=(origin[0], top_left[0], origin[1], top_left[1]),
        cmap=cmap,
        origin="lower",
        **imshow_kwargs,
    )


# %%
image_full = sitk.GetImageFromArray(data_full.T)

fig, ax = plt.subplots()
show_image(image_full, ax)
ax.set_title("Full resolution image")

# %% [markdown]
# To start with we'll just try registring a cut out of this image to the full image. This will illustrate how the translation element of transformations works.

# %%
offset_cutout = (150, 100)
size_cutout = (150, 100)
data_cutout = data_full[
    offset_cutout[0] : offset_cutout[0] + size_cutout[0],
    offset_cutout[1] : offset_cutout[1] + size_cutout[1],
].copy()
# Add some noise so the cutout isn't exact
data_cutout = data_cutout + np.random.rand(*data_cutout.shape) * 0.05
image_cutout = sitk.GetImageFromArray(data_cutout.T)

fig, ax = plt.subplots()
show_image(image_cutout, ax)
ax.set_title(f"Cutout image\noffset = {offset_cutout}")


# %% [markdown]
# Now we're ready to try doing some registration. The most general transform we want to use is a rigid transform that includes a translation, rotation, and (isotropic) scaling. In `SimpleITK` this is represented by a `Similarity2DTransform`.
#
# We are seeking a transform that maps the pixels in our cutout to the pixels in the full image. In this initial simple case we know the transform is a simple offset of (150, 100). Lets setup a transform that is a simple offset, but with slightly incorrect values.
#
# The transform can be visualised by resampling the full image on to the cutout. Here we can see that the transform is slightly wrong.


# %%
def compare_images(*, image_full, image_cutout, transform):
    """
    Compare full and cutout images.

    `transform` maps pixels in image_full to image_cutout.
    """
    transformed_cutout = sitk.Resample(image_cutout, image_full, transform)

    fig, ax = plt.subplots()
    show_image(image_full, ax, alpha=0.5, cmap="Grays")
    show_image(transformed_cutout, ax, alpha=0.6, cmap="Blues")


# %%
transform_translate = sitk.Similarity2DTransform()
transform_translate.SetTranslation((-140, -90))

compare_images(
    image_full=image_full, image_cutout=image_cutout, transform=transform_translate
)


# %% [markdown]
# ## First registration example
#
# The goal of image registration is to optimize the transform parameters so the two images overlap as closly as possible. To start image registration we setup a `SimpleITK` `ImageRegistrationMethod` that already has the metric, sampling strategy, interpolator, and optimizer set. We won't be modifying any of these properties in this notebook.


# %%
def get_reg_method() -> sitk.ImageRegistrationMethod:
    """
    Get a registration method, with the metric, interpolator, and optimizer set.
    """
    R = sitk.ImageRegistrationMethod()

    # Set metric.
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    # Sample all the pixels
    R.SetMetricSamplingStrategy(R.NONE)
    # Set interpolator
    R.SetInterpolator(sitk.sitkLinear)
    # Set optimizer
    R.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=300,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )

    return R


# %% [markdown]
# Now we can get an instance of the registration method, and set the initial transform. In order to avoid changing the scaling or rotation angle during optimization we set the optimizer weights for these paramters to zero.
#
# Then we run the registration, and check the results...

# %%
R = get_reg_method()
R.SetInitialTransform(transform_translate, inPlace=True)
print(f"Initial translation: {transform_translate.GetTranslation()}")
# Parameters are:
# - Scale
# - Rotation angle
# - Two translation components
# Don't change the scale or rotation angle for now
R.SetOptimizerWeights([0, 0, 1, 1])

# Run the transform
R.Execute(image_full, image_cutout)
print(f"Final translation: {transform_translate.GetTranslation()}")

compare_images(
    image_full=image_full,
    image_cutout=image_cutout,
    transform=transform_translate,
)


# %% [markdown]
# The two images overlap much better, so a successful registration!
#
# Because the registration maps the full resolution dataset to the ROI, to convert the other way we can just use the inverse of the transform:


# %%
def print_transform(transform):
    print(f"Translation = {transform.GetTranslation()}")
    print(f"Angle = {np.rad2deg(transform.GetAngle())} deg")
    print(f"Scale = {transform.GetScale()}")


# %%
print_transform(transform_translate.GetInverse())

# %% [markdown]
# ## Adding in rotation
#
# Now lets make things a bit complicated by adding in a rotation. To make a new cutout we rotate the full image by 10 degrees, and then take the cutout.

# %%
scale = 1
rotation = np.deg2rad(20)
translation = (0, 0)
rotation_center = (225, 150)

rotated_full = sitk.Resample(
    image_full,
    sitk.Similarity2DTransform(scale, rotation, translation, rotation_center),
)
rotated_full_data = sitk.GetArrayFromImage(rotated_full).T
rotated_cutout_data = rotated_full_data[150:300, 100:200].copy()

rotated_cutout = sitk.GetImageFromArray(rotated_cutout_data.T)

fig, ax = plt.subplots()
show_image(rotated_cutout, ax)

# %% [markdown]
# As before, lets set an initial transform and compare the two images. This time we set the translation to be correct, but leave the rotation at zero.

# %%
transform_rotate = sitk.Similarity2DTransform()
transform_rotate.SetTranslation((-145, -102))
transform_rotate.SetCenter(rotation_center)
transform_rotate.SetAngle(np.deg2rad(-24))

compare_images(
    image_full=image_full, image_cutout=rotated_cutout, transform=transform_rotate
)

# %% [markdown]
# Now we do the registration, disabling only the scalaing parameter of the similarity transform:

# %%
R = get_reg_method()
R.SetInitialTransform(transform_rotate, inPlace=True)
print(f"Initial translation: {transform_rotate.GetTranslation()}")
print(f"Initial rotation: {np.rad2deg(transform_rotate.GetAngle())} deg")
# Parameters are:
# - Scale
# - Rotation angle
# - Two translation components
# Don't change the scale or rotation angle for now
R.SetOptimizerScalesFromPhysicalShift()
R.SetOptimizerWeights([0, 1, 1, 1])

# Run the transform
R.Execute(image_full, rotated_cutout)
print(f"Final translation: {transform_rotate.GetTranslation()}")
print(f"Final rotation: {np.rad2deg(transform_rotate.GetAngle())} deg")
print(R.GetOptimizerStopConditionDescription())

compare_images(
    image_full=image_full, image_cutout=rotated_cutout, transform=transform_rotate
)

# %%
print("Registered transform:")
print_transform(transform_rotate)
print()
print("Inverse:")
print_transform(transform_rotate.GetInverse())

# %% [markdown]
# Note how the translation of the inverse is not simply minus the original translation.

# %%
