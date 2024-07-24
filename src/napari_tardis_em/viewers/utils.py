#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2024                                                   #
#######################################################################
from os import mkdir, listdir
from os.path import join, isdir
from shutil import rmtree

import numpy as np

from napari_tardis_em.viewers import colormap_for_display, face_colormap, IMG_FORMAT

from tardis_em.utils.dataset import build_test_dataset
from tardis_em.cnn.datasets.build_dataset import build_train_dataset
from tardis_em.utils.setup_envir import check_dir
from tardis_em.utils.errors import TardisError


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
    visibility=True,
):
    """
    Create a point layer in napari.

    Args:
        viewer (napari.Viewer): Napari viewer
        points (np.ndarray): Image array to display
        name (str): Layer name
        visibility (bool):
    """
    try:
        viewer.layers.remove(name)
    except Exception as e:
        pass

    point_features = {
        "confidence": tuple(points[:, 0].flatten() * np.random.randint(100)),
    }
    points = np.array(points[:, 1:])

    # Assert points in 3D
    if points.shape[1] == 2:
        z = np.zeros((len(points), 1))
        points = np.hstack((points, z))

    # Convert xyz to zyx
    points = np.vstack((points[:, 2], points[:, 1], points[:, 0])).T
    viewer.layers.select_all()
    viewer.layers.toggle_selected_visibility()

    viewer.add_points(
        points,
        features=point_features,
        face_color="confidence",
        face_colormap=face_colormap,
        visible=visibility,
        size=10,
    )

    try:
        viewer.layers["Predicted_Instances"].visible = True
    except Exception as e:
        pass

    try:
        viewer.layers["Predicted_Instances_filter"].visible = True
    except Exception as e:
        pass

    viewer.dims.ndisplay = 3


def create_image_layer(
    viewer,
    image=None,
    name="Prediction",
    transparency=True,
    visibility=True,
    range_: None = (0, 1),
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


def setup_environment_and_dataset(
    dir_, mask_size, pixel_size, patch_size, correct_pixel_size=None
):
    """Set environment"""
    TRAIN_IMAGE_DIR = join(dir_, "train", "imgs")
    TRAIN_MASK_DIR = join(dir_, "train", "masks")
    TEST_IMAGE_DIR = join(dir_, "test", "imgs")
    TEST_MASK_DIR = join(dir_, "test", "masks")

    DATASET_TEST = check_dir(
        dir_=dir_,
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
