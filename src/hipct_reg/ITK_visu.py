#!/usr/bin/env python

## Author: Joseph Brunet (joseph.brunet@ucl.ac.uk)
## This script allows to visialize the results from the registration performed with ITK_registration.py ##


import ast
import glob
import math
import os
import sys
import time
from tkinter import Tk  # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename

import numpy as np
import SimpleITK as sitk
import skimage.io
import skimage.measure
import tifffile

from .helpers import import_im, test_file_type

MAX_THREADS = 0  # 0 if all


def get_str(row, string):
    return row[row.startswith(string) and len(string) :].strip("\n")


def visu_from_file(log_file, res_factor):
    with open(log_file) as f:
        lines = f.readlines()

    for row in lines:
        if "Fixed image = " in row:
            path_fixed = get_str(row, "Fixed image = ")
        if "Moving image = " in row:
            path_moved = get_str(row, "Moving image = ")
        if "Pixel size Fixed scan = " in row:
            pixel_size_fixed = float(get_str(row, "Pixel size Fixed scan = ")[:-3])
        if "Pixel size Moving scan = " in row:
            pixel_size_moved = float(get_str(row, "Pixel size Moving scan = ")[:-3])
        if "Fixed volume - Binning factor = " in row:
            binning_fixed = int(float(get_str(row, "Fixed volume - Binning factor = ")))
        else:
            binning_fixed = 1
        if "Zoom volume - Binning factor = " in row:
            binning_moved = int(float(get_str(row, "Zoom volume - Binning factor = ")))

        if "Crop the zoom volume to avoid external circle =" in row:
            if "True" in row:
                crop_circle_moved = True
            else:
                crop_circle_moved = False

        if "Moving scan crop z = " in row:
            moved_z = get_str(row, "Moving scan crop z = ")
        if "Fixed scan crop z" in row:
            fixed_z = get_str(row, "Fixed scan crop z")

        if "Point fixed = " in row:
            pt_fixed = get_str(row, "Point fixed = ")
            pt_fixed = np.fromstring(pt_fixed[1:-1], sep=",")

        if "Point moved = " in row:
            pt_moved = get_str(row, "Point moved = ")
            pt_moved = np.fromstring(pt_moved[1:-1], sep=",")

        if "height_factor" in row:
            # height_factor = float(row[row.startswith('height_factor is ') and len('height_factor is '):].strip('\n') )
            height_factor = get_str(row, "height_factor")

            height_factor = float(row.split("= ")[-1])

        if "Translation = " in row:
            translation = get_str(row, "Translation = ")

            translation = np.fromstring(translation[1:-1], sep=",")
        if "Center of rot = " in row:
            rot_center = get_str(row, "Center of rot = ")
            rot_center = np.fromstring(rot_center[1:-1], sep=",")
        if "Matrix = " in row:
            rot_matrix = get_str(row, "Matrix = ")
            rot_matrix = np.fromstring(rot_matrix[1:-1], sep=",")
            rot_matrix = rot_matrix.reshape((3, 3))
        if "Scale = " in row:
            scale = float(get_str(row, "Scale = "))

    # binning_fixed = binning_fixed * res_factor
    # binning_moved = binning_moved * res_factor

    # crop_circle_moved = True #Crop the zoom scan to transform the circle fov into a square, thus avoiding the NaN part in the image
    crop_circle_moved = False  # Crop the zoom scan to transform the circle fov into a square, thus avoiding the NaN part in the image
    crop_z = True  # Crop the z of both scans to use less ram (the parameter height_factor control the height)

    print(
        f"""The pixel sizes I found are:\n\n
          - {pixel_size_fixed} um for the low-resolution volume\n\n
          - {pixel_size_moved} um for the high-resolution volume\n\n"""
    )

    # Gather info on images
    file_type_fixed = test_file_type(path_fixed)
    file_type_moved = test_file_type(path_moved)

    N_fixed = len(glob.glob(f"{path_fixed}/*.{file_type_fixed}"))
    N_moved = len(glob.glob(f"{path_moved}/*.{file_type_moved}"))

    img = skimage.io.imread(glob.glob(path_fixed + "/*.tif")[0])
    print(f"\nI found {N_fixed} images of shape {img.shape} in the fixed volume folder")

    img = skimage.io.imread(glob.glob(path_moved + "/*.tif")[0])
    x_dim = img.shape[1]
    y_dim = img.shape[0]
    print(f"I found {N_moved} images of shape {img.shape} in the moved volume folder")

    del img

    size_moved = N_moved * x_dim * y_dim * 2 / (1024 * 1024 * 1024)
    print(f"Total size of zoom scan is is {size_moved} GB")

    if binning_fixed != 1:
        print(f"\nA binning of {binning_fixed} will be applied to the fixed volume\n")
        N_fixed = math.ceil(N_fixed / binning_fixed)

    if binning_moved != 1:
        print(f"\nA binning of {binning_moved} will be applied to the zoom volume\n")
        N_moved = math.ceil(N_moved / binning_moved)

    pixel_size_fixed = pixel_size_fixed * binning_fixed
    pixel_size_moved = pixel_size_moved * binning_moved

    pt_fixed = np.array(pt_fixed)
    pt_moved = np.array(pt_moved)

    pt_fixed = pt_fixed / binning_fixed
    pt_moved = pt_moved / binning_moved

    trans_point = pt_fixed - pt_moved * (pixel_size_moved / pixel_size_fixed)

    print("---------------------------------------------------")
    print("\nIMPORTATION OF SCAN TO REGISTER\n")

    moved_z = [0, N_moved]

    moving_image = import_im(
        path_moved,
        pixel_size_moved,
        crop_z=moved_z,
        bin_factor=binning_moved,
    )

    print("moving_image")
    print(moving_image.GetSize())
    print(moving_image.GetPixelIDTypeAsString())
    print(moving_image.GetSpacing())

    # if crop_circle_moved:
    #     trans_point[0] = trans_point[0] + int((1-0.708)/2 * (moving_image.GetSize()[0] / (0.708)) * pixel_size_moved/pixel_size_fixed)
    #     trans_point[1] = trans_point[1] + int((1-0.708)/2 * (moving_image.GetSize()[1] / (0.708)) * pixel_size_moved/pixel_size_fixed)

    fixed_z = []
    if crop_z:
        fixed_z = [
            int(
                pt_fixed[2]
                - 1.2 * ((pt_moved[2]) * pixel_size_moved / pixel_size_fixed)
            ),
            int(
                pt_fixed[2]
                + 1.2
                * (
                    ((moved_z[1] - moved_z[0]) - pt_moved[2])
                    * pixel_size_moved
                    / pixel_size_fixed
                )
            ),
        ]
        if fixed_z[0] < 0:
            fixed_z[0] = 0
        if fixed_z[1] > N_fixed:
            fixed_z[1] = int(N_fixed)
        trans_point[2] = trans_point[2] - fixed_z[0]
        pt_fixed[2] = pt_fixed[2] - fixed_z[0]

    print("---------------------------------------------------")
    print("\nIMPORTATION OF REFERENCE SCAN\n")

    fixed_image = import_im(
        path_fixed,
        pixel_size_fixed,
        crop_z=fixed_z,
        bin_factor=binning_fixed,
    )

    print("fixed_image")
    print(fixed_image.GetSize())
    print(fixed_image.GetPixelIDTypeAsString())
    print(fixed_image.GetSpacing())

    print("IMPORTATION DONE\n\n")
    print("---------------------------------------------------")
    print("\nVisualization\n\n")
    print("---------------------------------------------------")

    print("translation: ", translation)
    print("matrix: ", rot_matrix.reshape([9]))
    print("center: ", rot_center)

    affine = sitk.AffineTransform(3)
    affine.SetMatrix(rot_matrix.reshape([9]))
    affine.SetTranslation(translation)
    affine.SetCenter(rot_center)

    print("Before oversampling")
    print(fixed_image.GetSize())
    print(fixed_image.GetPixelIDTypeAsString())
    print(fixed_image.GetSpacing())
    fixed_image = sitk.Expand(fixed_image, [res_factor, res_factor, res_factor])

    pixel_size_fixed = pixel_size_fixed / res_factor
    fixed_image.SetSpacing([pixel_size_fixed, pixel_size_fixed, pixel_size_fixed])

    print("New pixel size: ", pixel_size_fixed)

    print("After oversampling")
    print(fixed_image.GetSize())
    print(fixed_image.GetPixelIDTypeAsString())
    print(fixed_image.GetSpacing())

    # sitk.Show(moving_image, '1111')
    # moving_resampled = sitk.Resample(moving_image, fixed_image, affine, sitk.sitkLinear, 0, fixed_image.GetPixelID())
    # sitk.Show(0.6*moving_resampled+0.4*fixed_image, 'With oversampling')

    fixed_mean = np.mean(fixed_image[0:100, 0:100, int(fixed_image.GetSize()[2] / 2)])
    print(fixed_mean)

    moving_mean = np.mean(
        moving_image[0:100, 0:100, int(moving_image.GetSize()[2] / 2)]
    )
    print(moving_mean)

    moving_image = moving_image * fixed_mean / moving_mean

    # new_value = ( ( old_value - old_min ) / (old_max - old_min) ) * (new_max - new_min) + new_min

    print(fixed_image.GetPixelIDTypeAsString())

    pixelType = sitk.sitkFloat32
    fixed_image = sitk.Cast(fixed_image, pixelType)
    print(fixed_image.GetPixelIDTypeAsString())

    moving_resampled = sitk.Resample(
        moving_image, fixed_image, affine, sitk.sitkLinear, 0, fixed_image.GetPixelID()
    )

    output_dir = f"./output_visu/{os.path.basename(path_moved)}/"
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    img_fixed = adjust_contrast(fixed_image[:, :, int(fixed_image.GetSize()[2] / 2)])
    img_moved = adjust_contrast(
        moving_resampled[:, :, int(moving_resampled.GetSize()[2] / 2)]
    )
    img = np.stack([img_fixed, img_moved], axis=0)
    tifffile.imsave(f"{output_dir}/z_slice-image.tif", img)

    img_fixed = adjust_contrast(
        fixed_image[
            :,
            int(
                -translation[1] / pixel_size_fixed
                + (moving_image.GetSize()[1] / 2) * pixel_size_moved / pixel_size_fixed
            ),
            :,
        ]
    )
    img_moved = adjust_contrast(
        moving_resampled[
            :,
            int(
                -translation[1] / pixel_size_fixed
                + (moving_image.GetSize()[1] / 2) * pixel_size_moved / pixel_size_fixed
            ),
            :,
        ]
    )
    img = np.stack([img_fixed, img_moved], axis=0)
    tifffile.imsave(f"{output_dir}/y_slice-image.tif", img)

    img_fixed = adjust_contrast(
        fixed_image[
            int(
                -translation[0] / pixel_size_fixed
                + (moving_image.GetSize()[0] / 2) * pixel_size_moved / pixel_size_fixed
            ),
            :,
            :,
        ]
    )
    img_moved = adjust_contrast(
        moving_resampled[
            int(
                -translation[0] / pixel_size_fixed
                + (moving_image.GetSize()[0] / 2) * pixel_size_moved / pixel_size_fixed
            ),
            :,
            :,
        ]
    )
    img = np.stack([img_fixed, img_moved], axis=0)
    tifffile.imsave(f"{output_dir}/x_slice-image.tif", img)

    # try:
    #     sitk.Show(sitk.CheckerBoard(moving_resampled, fixed_image, [20,20,20]), 'test')
    # except:
    #     print('Error with imagej')


def adjust_contrast(img):
    img = sitk.GetArrayFromImage(img)
    # img_vmin,img_vmax = np.percentile(img, (3, 97))
    # img = exposure.rescale_intensity(img, in_range=(img_vmin, img_vmax))
    img = img.astype(np.uint16)

    return img


if __name__ == "__main__":
    print("Starting registration check")

    registration_list = []

    if len(sys.argv) == 1:
        Tk().withdraw()
        registration_file = askopenfilename(
            initialdir="/data/projects/hop/data_repository/Various/neuroglancer_pipeline/registration/registration_list/",
            title="Select list to register",
        )
        with open(registration_file) as file:
            lines = [line.rstrip("\n") for line in file]
        for l in lines:
            try:
                ll = l.split()
                ll[2] = ast.literal_eval(ll[2])
                ll[3] = ast.literal_eval(ll[3])
                registration_list.append(ll)
            except:
                print("Could not read line")
    elif len(sys.argv) == 2:
        registration_file = sys.argv[1]
        with open(registration_file) as file:
            lines = [line.rstrip("\n") for line in file]
        for l in lines:
            try:
                ll = l.split()
                ll[2] = ast.literal_eval(ll[2])
                ll[3] = ast.literal_eval(ll[3])
                registration_list.append(ll)
            except:
                print("Could not read line")

    else:
        sys.exit("ERROR: too much arguments\n")

    for idx, r in enumerate(registration_list):
        path_fixed = r[0]
        path_moved = r[1]
        pt_fixed = r[2]
        pt_moved = r[3]

        output_dir = f"./output_visu/{os.path.basename(path_moved)}/"
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        log_file = glob.glob(
            f"{os.path.dirname(path_moved.rstrip('/'))}/registration_{os.path.basename(path_moved)}.log"
        )
        flag = ""
        if log_file:
            if len(log_file) > 1:
                flag = "Too many log files"
            else:
                with open(log_file[0]) as file:
                    lines = [line.rstrip("\n") for line in file]
                if "Total time =" in lines[-1]:
                    flag = "Registered"
                else:
                    flag = "Registration started but not finished"
        else:
            flag = "No log file"

        f = open(output_dir + flag + ".txt", "w")
        f.close()
        file_summary = "./output_visu/summary.txt"
        with open(file_summary, "a") as f:
            f.write(f"{os.path.basename(path_moved)}: {flag}\n")

        print(
            f"\n[{idx}]: {flag}\nFixed volume: {os.path.basename(path_fixed)}\nRegistered volume: {os.path.basename(path_moved)}"
        )

    # ans = input('Which one to check ?')
    # r = registration_list[int(ans)]
    # path_fixed = r[0]
    # path_moved = r[1]
    # pt_fixed = r[2]
    # pt_moved = r[3]
    # log_file = glob.glob(f"{os.path.dirname(path_moved.rstrip('/'))}/registration_*.log")

    # ans = input('What factor to the resolution ? [1]')

    for r in registration_list:
        print("\n\n----------------------\nNew Visu\n----------------------\n\n")
        print(r)
        try:
            path_fixed = r[0]
            path_moved = r[1]
            pt_fixed = r[2]
            pt_moved = r[3]
            log_file = glob.glob(
                f"{os.path.dirname(path_moved.rstrip('/'))}/registration_*.log"
            )
            if len(log_file) > 1:
                print("Too many log files")
            log_file = max(log_file, key=os.path.getctime)
            print(log_file)

            ans = "1"

            if ans == "":
                res_factor = 1
            else:
                res_factor = int(ans)
            print("Resolution factor = ", res_factor)
            start_time = time.time()
            visu_from_file(log_file, res_factor)
            print("--- %s seconds ---" % (time.time() - start_time))
        except Exception as e:
            print(f"Error during visu_from_file: {e}")

    print("Finished")
