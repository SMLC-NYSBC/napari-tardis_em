from os.path import join, dirname

import numpy as np
from napari.utils.notifications import show_error

from napari_tardis_em.viewers.utils import create_image_layer, create_point_layer
from tardis_em.analysis.filament_utils import sort_by_length
from tardis_em.cnn.data_processing.trim import trim_with_stride
from tardis_em.cnn.datasets.dataloader import PredictionDataset
from tardis_em.utils.aws import get_all_version_aws
from tardis_em.utils.normalization import adaptive_threshold
from tardis_em.utils.predictor import GeneralPredictor


def _update_cnn_threshold(viewer, dir_, img, cnn_threshold):
    viewer.layers[dir_.split("/")[-1]].visible = True

    if cnn_threshold == 1.0:
        img_threshold = adaptive_threshold(img).astype(np.uint8)
    elif cnn_threshold == 0.0:
        img_threshold = np.copy(img)
    else:
        img_threshold = np.where(img >= cnn_threshold, 1, 0).astype(np.uint8)

    create_image_layer(
        viewer,
        image=img_threshold,
        name="Prediction",
        transparency=True,
        range_=(0, 1),
    )

    return img_threshold


def _update_dist_layer(viewer, segments, segments_filter):
    if segments is not None:
        create_point_layer(
            viewer=viewer,
            points=segments,
            name="Predicted_Instances",
            visibility=True,
        )

    if segments_filter is not None:
        create_point_layer(
            viewer=viewer,
            points=segments_filter,
            name="Predicted_Instances_filter",
            visibility=True,
        )


def _update_dist_graph(filament, segments, GraphToSegment, graphs, pc_ld, output_idx):
    if filament:
        sort = True
        prune = 5
    else:
        sort = False
        prune = 15

    try:
        segments = GraphToSegment.patch_to_segment(
            graph=graphs,
            coord=pc_ld,
            idx=output_idx,
            sort=sort,
            prune=prune,
        )
        segments = sort_by_length(segments)
    except:
        segments = None

    return segments


def _update_versions(model_version, cnn_type, type_model):
    for i in range(model_version.count()):
        model_version.removeItem(0)

    versions = get_all_version_aws(cnn_type, "32", type_model)

    if len(versions) == 0:
        model_version.addItems(["None"])
    else:
        model_version.addItems(["None"] + [i.split("_")[-1] for i in versions])

    return model_version


def semantic_preprocess(viewer, dir_, output_semantic, output_instance, model_params):
    """Pre-settings"""

    if model_params["correct_px"] == "None":
        model_params["correct_px"] = None
    else:
        model_params["correct_px"] = float(model_params["correct_px"])

    msg = (
        f"Predicted file is without pixel size metadate {model_params['correct_px']}."
        "Please correct correct_px argument with a correct pixel size value."
    )
    if model_params["correct_px"] is None:
        show_error(msg)
        return

    if model_params["normalize_px"] == "None":
        model_params["normalize_px"] = None
    else:
        model_params["normalize_px"] = float(model_params["normalize_px"])

    output_formats = f"{output_semantic}_{output_instance}"

    if output_instance == "None":
        instances = False
    else:
        instances = True

    model_params["cnn_threshold"] = (
        "auto"
        if model_params["cnn_threshold"] == 1.0
        else model_params["cnn_threshold"]
    )

    if model_params["model_version"] == "None":
        model_params["model_version"] = None
    else:
        model_params["model_version"] = int(model_params["model_version"])

    if model_params["filter_by_length"] == "None":
        model_params["filter_by_length"] = None
    else:
        model_params["filter_by_length"] = int(model_params["filter_by_length"])

    if model_params["connect_splines"] == "None":
        model_params["connect_splines"] = None
    else:
        model_params["connect_splines"] = int(model_params["connect_splines"])

    if model_params["connect_cylinder"] == "None":
        model_params["connect_cylinder"] = None
    else:
        model_params["connect_cylinder"] = int(model_params["connect_cylinder"])

    if model_params["amira_prefix"] == "None":
        model_params["amira_prefix"] = None
    else:
        model_params["amira_prefix"] = str(model_params["amira_prefix"])

    if model_params["amira_compare_distance"] == "None":
        model_params["amira_compare_distance"] = None
    else:
        model_params["amira_compare_distance"] = int(
            model_params["amira_compare_distance"]
        )

    if model_params["amira_inter_probability"] == "None":
        model_params["amira_inter_probability"] = None
    else:
        model_params["amira_inter_probability"] = float(
            model_params["amira_inter_probability"]
        )

    predictor = GeneralPredictor(
        predict=model_params["predict_type"],
        dir_s=dir_,
        binary_mask=model_params["mask"],
        correct_px=model_params["correct_px"],
        normalize_px=model_params["normalize_px"],
        convolution_nn=model_params["cnn_type"],
        checkpoint=model_params["checkpoint"],
        model_version=model_params["model_version"],
        output_format=output_formats,
        patch_size=model_params["patch_size"],
        cnn_threshold=model_params["cnn_threshold"],
        dist_threshold=model_params["dist_threshold"],
        points_in_patch=model_params["points_in_patch"],
        predict_with_rotation=model_params["rotate"],
        amira_prefix=model_params["amira_prefix"],
        filter_by_length=model_params["filter_by_length"],
        connect_splines=model_params["connect_splines"],
        connect_cylinder=model_params["connect_cylinder"],
        amira_compare_distance=model_params["amira_compare_distance"],
        amira_inter_probability=model_params["amira_inter_probability"],
        instances=instances,
        device_s=model_params["device"],
        debug=False,
        tardis_logo=False,
    )
    predictor.in_format = len(dir_.split(".")[-1]) + 1

    predictor.get_file_list()
    predictor.create_headers()
    predictor.load_data(id_name=predictor.predict_list[0])

    if model_params["image_type"] is not None:
        if model_params["image_type"] == "2D":
            predictor.expect_2d = True
            predictor.cnn._2d = True
        else:
            predictor.expect_2d = False
            predictor.cnn._2d = False

    if not model_params["mask"]:
        trim_with_stride(
            image=predictor.image,
            scale=predictor.scale_shape,
            trim_size_xy=predictor.patch_size,
            trim_size_z=predictor.patch_size,
            output=join(dirname(dir_), "temp", "Patches"),
            image_counter=0,
            clean_empty=False,
            stride=round(predictor.patch_size * 0.125),
        )

        create_image_layer(
            viewer,
            image=predictor.image,
            name=dir_.split("/")[-1],
            range_=(np.min(predictor.image), np.max(predictor.image)),
            visibility=False,
            transparency=False,
        )

        create_image_layer(
            viewer,
            image=np.zeros(predictor.scale_shape, dtype=np.float32),
            name="Prediction",
            transparency=True,
            range_=None,
        )

        predictor.image = None
        scale_shape = predictor.scale_shape

        img_dataset = PredictionDataset(join(dirname(dir_), "temp", "Patches", "imgs"))

        return output_formats, predictor, scale_shape, img_dataset
    else:
        return output_formats, predictor, None, None
