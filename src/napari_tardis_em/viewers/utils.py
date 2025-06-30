#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2024                                                   #
#######################################################################
from os import mkdir, listdir, getcwd
from os.path import join, isdir
from shutil import rmtree
from typing import Tuple, Optional

import numpy as np
import scipy
from PyQt5.QtWidgets import QFileDialog

from napari_tardis_em.viewers import colormap_for_display, face_colormap, IMG_FORMAT
from tardis_em.cnn.datasets.build_dataset import build_train_dataset
from tardis_em.utils.dataset import build_test_dataset
from tardis_em.utils.errors import TardisError
from tardis_em.utils.setup_envir import check_dir


def update_viewer_prediction(viewer, image: np.ndarray, position: dict):
    img = viewer.layers["Prediction"]

    ndim = img.data.ndim
    shape_ = img.data.shape

    try:
        if ndim == 3:
            img.data[
                position["z"][0] : position["z"][1],
                position["y"][0] : position["y"][1],
                position["x"][0] : position["x"][1],
            ] = image
        else:
            img.data[
                position["y"][0] : position["y"][1],
                position["x"][0] : position["x"][1],
            ] = image
    except ValueError:
        if ndim == 3:
            diff = [
                image.shape[0] - (position["z"][1] - shape_[0]),
                image.shape[0] - (position["y"][1] - shape_[1]),
                image.shape[0] - (position["x"][1] - shape_[2]),
            ]
            diff = [
                diff[0] if 0 < diff[0] < image.shape[0] else image.shape[0],
                diff[1] if 0 < diff[1] < image.shape[0] else image.shape[0],
                diff[2] if 0 < diff[2] < image.shape[0] else image.shape[0],
            ]
            position["z"][1] = position["z"][0] + diff[0]
            position["y"][1] = position["y"][0] + diff[1]
            position["x"][1] = position["x"][0] + diff[2]

            img.data[
                position["z"][0] : position["z"][1],
                position["y"][0] : position["y"][1],
                position["x"][0] : position["x"][1],
            ] = image[: diff[0], : diff[1], : diff[2]]
        else:
            diff = [
                image.shape[0] - (position["y"][1] - shape_[0]),
                image.shape[0] - (position["x"][1] - shape_[1]),
            ]
            diff = [
                diff[0] if 0 < diff[0] < image.shape[0] else image.shape[0],
                diff[1] if 0 < diff[1] < image.shape[0] else image.shape[0],
            ]
            position["y"][1] = position["y"][0] + diff[0]
            position["x"][1] = position["x"][0] + diff[1]

            img.data[
                position["y"][0] : position["y"][1],
                position["x"][0] : position["x"][1],
            ] = image[
                : diff[0],
                : diff[1],
            ]
    viewer.layers["Prediction"].visible = False
    viewer.layers["Prediction"].visible = True


def create_point_layer(
    viewer,
    points: np.ndarray,
    name: str,
    add_properties: dict = None,
    visibility=True,
    as_filament=False,
    size_=10,
    select_layer=None,
    keep_view_by=False,
    color_=None,
):
    """
    Create a point layer in napari.

    Args:
        viewer (napari.Viewer): Napari viewer
        points (np.ndarray): Image array to display
        name (str): Layer name
        visibility (bool):
        as_filament (bool):
        size_ (float):
    """
    current_center = viewer.camera.center
    current_orient = viewer.camera.angles
    current_zoom = viewer.camera.zoom

    try:
        point_features = viewer.layers[name].properties
    except:
        point_features = {}

    try:
        size_ = viewer.layers[name].current_size
    except:
        size_ = size_

    if keep_view_by:
        view_by = viewer.layers[select_layer].color_by
    else:
        view_by = "ID"

    try:
        viewer.layers.remove(name)
    except:
        pass

    ids = points[:, 0].astype(np.int16)
    points = np.array(points[:, 1:])

    # Assert points in 3D
    if points.shape[1] == 2:
        z = np.zeros((len(points), 1))
        points = np.hstack((points, z))

    # Convert xyz to zyx
    points = np.vstack((points[:, 2], points[:, 1], points[:, 0])).T

    face_colormap_ = "hsv"
    point_features["ID"] = tuple(ids)
    if add_properties is not None:
        for k, v in add_properties.items():
            point_features[k] = tuple(v)
            if not keep_view_by:
                view_by = k

        if not keep_view_by:
            if view_by != "ID":
                face_colormap_ = "inferno"
        else:
            face_colormap_ = "inferno"

    if color_ is not None:
        face_colormap_ = color_

    id_len = len(points)
    point_features = {
        k: v for k, v in point_features.items() if len(v) == id_len and k != "track_id"
    }

    if not as_filament:
        viewer.add_points(
            points,
            name=name,
            features=point_features,
            visible=visibility,
            face_color=face_colormap,
            size=size_,
            out_of_slice_display=True,
        )
    else:
        t = np.zeros_like(ids)
        points = np.vstack((ids, t, points[:, 0], points[:, 1], points[:, 2])).T
        viewer.add_tracks(
            points,
            name=name,
            features=point_features,
            visible=visibility,
        )
        viewer.layers[name].color_by = view_by
        viewer.layers[name].colormap = face_colormap_

    viewer.camera.center = current_center
    viewer.camera.zoom = current_zoom
    viewer.camera.angles = current_orient

    if select_layer is not None:
        viewer.layers[select_layer].visible = False
        while viewer.layers.selection.active.name != select_layer:
            viewer.layers.select_previous()

    try:
        viewer.layers["Predicted_Instances"].visible = True
    except Exception as e:
        pass

    try:
        viewer.layers["Predicted_Instances_filter"].visible = True
    except Exception as e:
        pass


def create_image_layer(
    viewer,
    image=None,
    name="Prediction",
    transparency=True,
    visibility=True,
    range_: Optional[Tuple] = (0, 1),
    zero_dim=False,
):
    """
    Create an image layer in napari.

    Args:
        viewer (napari.Viewer): Napari viewer
        image (np.ndarray): Image array to display
        name (str): Layer name
        transparency (bool): If True, show image as transparent layer
        visibility (bool):
        range_(None, tuple):
    """
    if isinstance(viewer, tuple) or isinstance(viewer, list):
        image = viewer[1]
        viewer = viewer[0]

    try:
        viewer.layers.remove(name)
    except Exception as e:
        pass

    if transparency:
        viewer.add_image(image, name=name, colormap=colormap_for_display, opacity=0.5)
    else:
        viewer.add_image(image, name=name, colormap="gray", opacity=1.0)

    if range_ is not None:
        try:
            viewer.layers[name].contrast_limits = (
                range_[0],
                range_[1],
            )
        except Exception as e:
            pass
    else:
        try:
            viewer.layers[name].contrast_limits = (
                image.min(),
                image.max(),
            )
        except Exception as e:
            pass

    if visibility:
        # set layer as not visible
        viewer.layers[name].visible = True
    else:
        viewer.layers[name].visible = False

    if zero_dim:
        viewer.dims.current_step = (0,) * (image.ndim - 2)


def setup_environment_and_dataset(
    dir_, mask_size, pixel_size, patch_size, correct_pixel_size=None
):
    """Set environment"""
    TRAIN_IMAGE_DIR = join(dir_, "train", "imgs")
    TRAIN_MASK_DIR = join(dir_, "train", "masks")
    TEST_IMAGE_DIR = join(dir_, "test", "imgs")
    TEST_MASK_DIR = join(dir_, "test", "masks")

    DATASET_TEST = check_dir(
        dir_s=dir_,
        with_img=True,
        train_img=TRAIN_IMAGE_DIR,
        train_mask=TRAIN_MASK_DIR,
        img_format=IMG_FORMAT,
        test_img=TEST_IMAGE_DIR,
        test_mask=TEST_MASK_DIR,
        mask_format=(
            "_mask.am",
            ".CorrelationLines.am",
            "_mask.mrc",
            "_mask.tif",
            "_mask.csv",
        ),
    )

    """Optionally: Set-up environment if not existing"""
    if not DATASET_TEST:
        # Check and set-up environment
        if not len([f for f in listdir(dir_) if f.endswith(IMG_FORMAT)]) > 0:
            TardisError(
                "100",
                "tardis_em/train_cnn.py",
                "Indicated folder for training do not have any compatible "
                "data or one of the following folders: "
                "test/imgs; test/masks; train/imgs; train/masks",
            )

        if isdir(join(dir_, "train")):
            rmtree(join(dir_, "train"))

        mkdir(join(dir_, "train"))
        mkdir(TRAIN_IMAGE_DIR)
        mkdir(TRAIN_MASK_DIR)

        if isdir(join(dir_, "test")):
            rmtree(join(dir_, "test"))

        mkdir(join(dir_, "test"))
        mkdir(TEST_IMAGE_DIR)
        mkdir(TEST_MASK_DIR)

        # Build train and test dataset
        build_train_dataset(
            dataset_dir=dir_,
            circle_size=mask_size,
            resize_pixel_size=pixel_size,
            trim_xy=patch_size,
            trim_z=patch_size,
            correct_pixel_size=correct_pixel_size,
        )

        no_dataset = int(len([f for f in listdir(dir_) if f.endswith(IMG_FORMAT)]) / 2)
        build_test_dataset(dataset_dir=dir_, dataset_no=no_dataset)


def calculate_position(patch_size, name):
    name = name.split("_")
    name = {
        "z": int(name[1]),
        "y": int(name[2]),
        "x": int(name[3]),
        "stride": int(name[4]),
    }

    x_start = (name["x"] * patch_size) - (name["x"] * name["stride"])
    x_end = x_start + patch_size
    name["x"] = [x_start, x_end]

    y_start = (name["y"] * patch_size) - (name["y"] * name["stride"])
    y_end = y_start + patch_size
    name["y"] = [y_start, y_end]

    z_start = (name["z"] * patch_size) - (name["z"] * name["stride"])
    z_end = z_start + patch_size
    name["z"] = [z_start, z_end]

    return name


def _update_checkpoint_dir(filter_=False):
    filename_, _ = QFileDialog.getOpenFileName(
        caption="Open File",
        directory=getcwd(),
    )

    if filter_:
        out_ = [
            i
            for i in filename_.split("/")
            if not i.endswith((".mrc", ".rec", ".map", ".tif", ".tiff", ".am"))
        ]
        return filename_, out_
    return filename_


def convert_to_float(strings):
    strings = strings.replace(",", ".") if "," in strings else strings
    return float(strings)


def shift_image(frame, shift_y, shift_x):
    shifted_frame = np.zeros_like(frame)

    # Calculate the slicing indices for the original and shifted frames
    orig_x_start = max(0, -shift_x)
    orig_x_end = frame.shape[1] - max(0, shift_x)
    orig_y_start = max(0, -shift_y)
    orig_y_end = frame.shape[0] - max(0, shift_y)

    shifted_x_start = max(0, shift_x)
    shifted_x_end = frame.shape[1] - max(0, -shift_x)
    shifted_y_start = max(0, shift_y)
    shifted_y_end = frame.shape[0] - max(0, -shift_y)

    # Apply the shift using slicing
    shifted_frame[shifted_y_start:shifted_y_end, shifted_x_start:shifted_x_end] = frame[
        orig_y_start:orig_y_end, orig_x_start:orig_x_end
    ]

    return shifted_frame


def phase_correct(image1, image2):
    """
    Correct the phase of image2 to match image1 using only NumPy.

    Parameters:
    image1, image2 : np.ndarray
        Input images (must be the same shape).

    Returns:
    np.ndarray
        Phase-corrected version of image2.
    """
    assert image1.shape == image2.shape, "Images must have the same shape."

    F1 = np.fft.fft2(image1)
    F2 = np.fft.fft2(image2)

    # Compute phase difference
    phase_diff = np.angle(F1) - np.angle(F2)

    # Compute cross-power spectrum
    cross_power_spectrum = (F1 * np.conj(F2)) / np.abs(
        F1 * np.conj(F2)
    )  # Normalize phase correlation
    correlation = np.fft.ifft2(cross_power_spectrum).real  # Get correlation matrix

    # Find peak correlation value (shift estimation)
    max_idx = np.unravel_index(np.argmax(correlation), correlation.shape)
    y_shift, x_shift = max_idx
    return y_shift, x_shift


def frames_phase_correlation(frames):
    _, h, w = frames.shape

    from tardis_em.utils.normalization import RescaleNormalize, MeanStdNormalize

    norm = RescaleNormalize((0.1, 99.9))
    mstd = MeanStdNormalize()

    stabilized_frames = []
    init_frame = mstd(norm(frames[0]))  # Use the first frame as the reference
    stabilized_frames.append(frames[0])  # Add the first frame as-is

    for i in range(1, len(frames)):
        curr_frame = mstd(norm(frames[i]))

        # calculate the correlation image; note the flipping of onw of the images
        shift = scipy.signal.fftconvolve(
            init_frame, curr_frame[::-1, ::-1], mode="full"
        )
        shift = np.unravel_index(np.argmax(shift), shift.shape)
        shift = (shift[0] - h, shift[1] - w)

        # Compute translation using phase correlation
        stabilized_frame = shift_image(frames[i], shift[0], shift[1])

        # Append stabilized frame to the list
        stabilized_frames.append(stabilized_frame)

    return np.asarray(stabilized_frames)
