#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2024                                                   #
#######################################################################
from os.path import splitext, basename

from tardis_em.utils.load_data import ImportDataFromAmira
from tardis_em.utils.load_data import load_image

import numpy as np

extensions_points = (".csv", ".am")

extensions_images = (".rec", ".mrc", ".tiff", ".tif", ".nd2", ".am")


def napari_get_reader(path):
    if isinstance(path, list):
        path = path[0]

    if not path.endswith(extensions_images) and not path.endswith(extensions_points):
        return None

    if path.endswith(".am"):
        try:
            with open(path, "r") as f:
                df = next(f).split(" ")
        except UnicodeDecodeError:
            with open(path, "rb") as f:
                df = str(next(f)).split(" ")

        if {"ASCII", "ASCI"}.intersection(df):
            return reader_function_points
        return reader_function_images
    else:
        if path.endswith(extensions_images):
            return reader_function_images
        return reader_function_points


def reader_function_images(path):
    paths = [path] if isinstance(path, str) else path

    file_names = []
    for p in paths:
        file_names.append(splitext(basename(p))[0])

    layer_type = "image"

    #   load all files into array
    layer_data = []
    for _path, _name in zip(paths, file_names):
        data = import_data(_path)

        add_kwargs = {"name": _name}

        layer_data.append((data, add_kwargs, layer_type))

    return layer_data


def reader_function_points(path):
    paths = [path] if isinstance(path, str) else path

    file_names = []
    for p in paths:
        file_names.append(splitext(basename(p))[0])

    # load all files into array
    layer_data = []
    for _path, _name in zip(paths, file_names):
        data, colors, ids = import_data(_path, coord=True)

        layer_type = "tracks"
        t = np.zeros_like(ids)

        data = np.vstack((ids, t, data[:, 0], data[:, 1], data[:, 2])).T

        add_kwargs = {"colormap": "hsv", "name": _name}

        layer_data.append((data, add_kwargs, layer_type))

    return layer_data


def generate_colors(n):
    # Generate n distinct colors using a simple algorithm
    np.random.seed(42)  # Seed for reproducibility

    colors = np.random.rand(n, 3)  # Generate n random RGB colors
    return colors


def import_data(filepath, coord=False):
    if not coord:
        img = load_image(filepath, False, False)

        return img
    else:
        if filepath.endswith(".csv"):
            data = np.genfromtxt(filepath, skip_header=1, delimiter=",")
        elif filepath.endswith(".am"):
            data = ImportDataFromAmira(filepath).get_segmented_points()
        else:
            return

        ids = data[:, 0].astype(np.int16)
        data = data[:, 1:]  # data in XYZ format
        if data.shape[1] == 2:
            data = np.vstack(
                (np.zeros_like(data[:, 0]), data[:, 1], data[:, 0])
            ).T  # Convert 2D to Napari ZYX standard

        else:
            data = np.vstack(
                (data[:, 2], data[:, 1], data[:, 0])
            ).T  # Convert to Napari ZYX standard

        unique_ids = np.unique(ids)
        colors_for_ids = generate_colors(len(unique_ids))
        color_mapping = {uid: colors_for_ids[i] for i, uid in enumerate(unique_ids)}

        colors = np.array([color_mapping[uid] for uid in ids])

        return data, colors, ids
