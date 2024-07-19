#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2024                                                   #
#######################################################################
import numpy as np

from napari_tardis_em.viewers import colormap_for_display


def update_viewer_prediction(viewer, image: np.ndarray, position: dict):
    img = viewer.layers["Prediction"]

    try:
        img.data[
            position["z"][0] : position["z"][1],
            position["y"][0] : position["y"][1],
            position["x"][0] : position["x"][1],
        ] = image
    except ValueError:
        shape_ = img.data.shape
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


def create_image_layer(
    viewer,
    image: np.ndarray,
    name: str,
    transparency=False,
    visibility=True,
    range_=None,
):
    """
    Create an image layer in napari.

    Args:
        image (np.ndarray): Image array to display
        name (str): Layer name
        transparency (bool): If True, show image as transparent layer
        visibility (bool):
        range_(tuple):
    """
    try:
        viewer.layers.remove(name)
    except Exception as e:
        pass

    if transparency:
        viewer.add_image(image, name=name, colormap='red', opacity=0.5)
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
