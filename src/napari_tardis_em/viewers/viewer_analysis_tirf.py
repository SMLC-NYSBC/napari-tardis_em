#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2024                                                   #
#######################################################################
import os
from os import getcwd, listdir, mkdir
from os.path import join, splitext, isdir, isfile

import numpy as np
import pandas as pd
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QPushButton,
    QFormLayout,
    QComboBox,
    QLabel,
    QDoubleSpinBox,
    QFileDialog,
    QLineEdit,
    QCheckBox,
)
from napari import Viewer
from napari.utils.notifications import show_info
from qtpy.QtWidgets import QWidget

from napari_tardis_em.utils.utils import get_list_of_device
from napari_tardis_em.viewers import IMG_FORMAT, CheckableComboBox
from napari_tardis_em.viewers.styles import border_style
from napari_tardis_em.viewers.utils import (
    create_image_layer,
    create_point_layer,
    convert_to_float,
    frames_phase_correlation,
)
from tardis_em.analysis.analysis import analyse_filaments_list
from tardis_em.analysis.filament_utils import (
    reorder_segments_id,
    sort_segment,
    sort_by_length,
    sort_segments,
    resample_filament,
)
from tardis_em.utils.load_data import load_image
from tardis_em.utils.predictor import GeneralPredictor


class TardisWidget(QWidget):
    """
    Easy to use plugin for general Microtubule prediction.

    Plugin integrate TARDIS-em and allow to easily set up training. To make it more
    user-friendly, this plugin guid user what to do, and during training display
     results from validation loop.

    ToDo:
        Co-localization for movies and frames
            - data mean but also per pixel on the MT length
            - for all microtubules
            - statistic? which mT are not correlated

    ToDo:
        Try using MTs chanells to get predictions
    """

    def __init__(self, viewer_mt_tirf: Viewer):
        super().__init__()

        self.predicted = False
        self.viewer = viewer_mt_tirf

        self.predictor = None
        self.img_threshold, self.scale_shape = None, None

        self.img, self.px = None, None
        self.out_ = getcwd()
        self.image_name_list, self.nd2_list = None, None
        self.image_list = None

        self.semantic, self.instances, self.instances_filter = None, None, None
        self.nd2_current_frame = 0

        """""" """""" """
          UI Elements
        """ """""" """"""
        self.directory = QPushButton(f"...{getcwd()[-30:]}")
        self.directory.setToolTip(
            "Select directory with image or single file you would like to predict. \n "
            "\n"
            "Supported formats:\n"
            "images: *.mrc, *.rec, *.map, *.am, *.tif, *nd2"
        )
        self.directory.clicked.connect(self.load_directory)
        self.dir = getcwd()

        self.no_instances = QCheckBox()
        self.no_instances.setCheckState(2)

        self.select_data_view = QComboBox()
        self.select_data_view.addItems(
            [
                "None",
            ]
        )
        self.select_data_view.currentTextChanged.connect(self.view_selected_data)

        self.analysis_ch = QComboBox()
        self.analysis_ch.addItems(
            [
                "0",
            ]
        )

        self.stabilize_button = QPushButton("Stabilize movie")
        self.stabilize_button.setMinimumWidth(225)
        self.stabilize_button.clicked.connect(self.stabilize_movie)

        self.thickness_bt1 = QComboBox()
        self.thickness_bt1.addItems(["1", "3", "5", "7", "9"])
        self.thickness_bt1.setCurrentIndex(1)

        self.thickness_bt2 = QComboBox()
        self.thickness_bt2.addItems(["1", "3", "5", "7", "9"])
        self.thickness_bt2.setCurrentIndex(1)

        ##############################
        # Setting user should change #
        ##############################
        label_2 = QLabel("                                      ")
        label_2.setStyleSheet(border_style("green"))

        self.cnn_threshold = QDoubleSpinBox()
        self.cnn_threshold.setDecimals(2)
        self.cnn_threshold.setMinimum(0)
        self.cnn_threshold.setMaximum(1)
        self.cnn_threshold.setSingleStep(0.01)
        self.cnn_threshold.setValue(0.10)
        self.cnn_threshold.setToolTip(
            "Threshold value for binary prediction. Lower value will increase \n"
            "recall [retrieve more of predicted object] but also may increase \n"
            "false/positives. Higher value will result in cleaner output but may \n"
            "reduce recall.\n"
            "\n"
            "If selected 0.0 - Output probability mask \n"
            "If selected 1.0 - Use adaptive threshold."
        )

        self.dist_threshold = QDoubleSpinBox()
        self.dist_threshold.setDecimals(2)
        self.dist_threshold.setMinimum(0)
        self.dist_threshold.setMaximum(1)
        self.dist_threshold.setSingleStep(0.01)
        self.dist_threshold.setValue(0.50)
        self.dist_threshold.setToolTip(
            "Threshold value for instance prediction. Lower value will increase \n"
            "recall [retrieve more of predicted object] but also may increase \n"
            "false/positives. Higher value will result in cleaner output but may \n"
            "reduce recall."
        )

        self.device = QComboBox()
        self.device.addItems(get_list_of_device())
        self.device.setCurrentIndex(0)
        self.device.setToolTip(
            "Select available device on which you want to train your model."
        )

        self.pixel_size_bt = QLineEdit("1.0")
        self.pixel_size_bt.setToolTip("Select correct pixel size for analysis")

        label_run = QLabel("Start Prediction                 ")
        label_run.setStyleSheet(border_style("blue"))

        self.predict_1_button = QPushButton("Predict with workflow...")
        self.predict_1_button.setMinimumWidth(225)
        self.predict_1_button.clicked.connect(self.predict_workflow)

        self.predict_2_button = QPushButton("Analyse frames...")
        self.predict_2_button.setMinimumWidth(225)
        self.predict_2_button.clicked.connect(self.re_analise_data)

        self.predict_3_button = QPushButton("Analyse movies...")
        self.predict_3_button.setMinimumWidth(225)
        self.predict_3_button.clicked.connect(self.re_analise_data_movies)

        self.analysis_list = CheckableComboBox()
        self.analysis_list.addItems(
            [
                "length",
                "curvature",
                "tortuosity",
                "avg_intensity",
                "avg_length_intensity",
                "sum_intensity",
                "sum_length_intensity",
                "correlation",
            ]
        )

        self.add_new_filament_bt = QPushButton("Add new filament")
        self.add_new_filament_bt.setMinimumWidth(225)
        self.add_new_filament_bt.clicked.connect(self.add_new_filament)

        self.nd2_file_frame_bt = QComboBox()
        self.nd2_file_frame_bt.currentTextChanged.connect(self.nd2_file_frame_change)

        self.edite_mode_bt_1 = QPushButton("Selected layer to points")
        self.edite_mode_bt_1.setMinimumWidth(225)
        self.edite_mode_bt_1.clicked.connect(self.point_layer)

        self.edite_mode_bt_2 = QPushButton("Selected layer to filament")
        self.edite_mode_bt_2.setMinimumWidth(225)
        self.edite_mode_bt_2.clicked.connect(self.track_layer)

        self.filament_id_rm = QLineEdit("None")
        self.filament_id_rm.setValidator(QDoubleValidator(0, 10000, 1))
        self.filament_id_rm.setToolTip("Select filament ID to remove")
        self.remove_filament_bt = QPushButton("Remove filament")
        self.remove_filament_bt.setMinimumWidth(225)
        self.remove_filament_bt.clicked.connect(self.remove_filament)

        self.join_filaments_id_1 = QLineEdit("None")
        self.join_filaments_id_1.setValidator(QDoubleValidator(0, 10000, 1))
        self.join_filaments_id_1.setToolTip("Select 1st filament ID to join")
        self.join_filaments_id_2 = QLineEdit("None")
        self.join_filaments_id_2.setValidator(QDoubleValidator(0, 10000, 1))
        self.join_filaments_id_2.setToolTip("Select 2nd filament ID to join")

        self.join_filaments_bt = QPushButton("Join filament")
        self.join_filaments_bt.setMinimumWidth(225)
        self.join_filaments_bt.clicked.connect(self.join_filaments)
        self.join_selected_filaments_bt = QPushButton("Join Selected")
        self.join_selected_filaments_bt.setMinimumWidth(225)
        self.join_selected_filaments_bt.clicked.connect(self.join_selected_filaments)

        self.save_edited_instances_bt = QPushButton("Save selected instance file")
        self.save_edited_instances_bt.clicked.connect(self.save_edited_instances)

        """
        Key Binding
        """
        self.edit_mode = False
        self.viewer.bind_key("e", self.activate_edit, overwrite=True)

        self.viewer.bind_key("b", self.b_event, overwrite=True)
        self.viewer.bind_key("n", self.n_event, overwrite=True)
        self.viewer.bind_key("m", self.m_event, overwrite=True)

        """
        Initialized UI
        """
        layout = QFormLayout()
        layout.addRow("Select Directory", self.directory)
        layout.addRow("Show Instances if possible", self.no_instances)
        layout.addRow("Select Image", self.select_data_view)

        layout.addRow("Analysis Channel", self.analysis_ch)
        layout.addRow("", self.stabilize_button)

        layout.addRow("---- Workflow ----", label_2)
        layout.addRow("Pixel size", self.pixel_size_bt)
        layout.addRow("CNN threshold", self.cnn_threshold)
        layout.addRow("DIST threshold", self.dist_threshold)
        layout.addRow("Device", self.device)

        layout.addRow("---- Run Prediction -----", label_run)
        layout.addRow("Line Thickness Image", self.thickness_bt1)
        layout.addRow("Line Thickness Mask", self.thickness_bt2)
        layout.addRow("", self.predict_1_button)
        layout.addRow("", self.predict_2_button)
        layout.addRow("", self.predict_3_button)
        layout.addRow("", self.analysis_list)

        layout.addRow("---- Correct-filaments ----", label_2)
        layout.addRow("Select nd2 frame", self.nd2_file_frame_bt)
        # Trigger addition mode with a new ID
        layout.addRow("", self.edite_mode_bt_1)
        layout.addRow("", self.edite_mode_bt_2)
        layout.addRow("Add Filament", self.add_new_filament_bt)
        # Remove filament with self.current_filament_id updated in self.filament_id
        layout.addRow("Remove Filament ID", self.filament_id_rm)
        layout.addRow("Remove Filament", self.remove_filament_bt)
        # A specific mode to join filaments when pressed activates the "J" button.
        # Mode then waits for the user to point at the filament and press "J" then the user
        # needs to press "J" again point at the second filament
        layout.addRow("Join Filaments ID 1", self.join_filaments_id_1)
        layout.addRow("Join Filaments ID 2", self.join_filaments_id_2)
        layout.addRow("Join Filaments", self.join_filaments_bt)
        layout.addRow("", self.join_selected_filaments_bt)

        layout.addRow("", label_2)
        layout.addRow("Save", self.save_edited_instances_bt)
        self.setLayout(layout)

    def activate_edit(self, viewer):
        if not self.edit_mode:
            self.edit_mode = True
            show_info("Edit mode activated")
        else:
            self.edit_mode = False
            show_info("Edit mode disable")

    def b_event(self, viewer):
        if self.edit_mode:
            self.remove_selected_filament()
        else:
            return

    def m_event(self, viewer):
        if self.edit_mode:
            self.join_selected_filaments()
        else:
            return

    def n_event(self, viewer):
        if self.edit_mode:
            self.add_new_filament()
        else:
            return

    def load_directory(self):
        filename = QFileDialog.getExistingDirectory(
            caption="Open Files",
            directory=getcwd(),
            # filter="Image Files (*.mrc *.rec *.map, *.tif, *.tiff, *.am)",
        )

        self.directory.setText(filename[-30:])
        self.dir = filename

        self.img = None
        self.load_data()
        self.select_view_data()

        if not isdir(join(self.dir, "Predictions")):
            mkdir(join(self.dir, "Predictions"))

        if not isdir(join(self.dir, "Predictions", "Analysis")):
            mkdir(join(self.dir, "Predictions", "Analysis"))

    def load_data(self):
        image_list = listdir(self.dir)
        self.nd2_list = [i for i in image_list if i.endswith(".nd2")]

        if len(self.nd2_list) == 0:
            self.nd2_list = None

        if self.nd2_list is not None:
            import tifffile.tifffile as save_tiff
            from tardis_em.utils.load_data import load_nd2_file

            for i in self.nd2_list:
                image, _ = load_nd2_file(join(self.dir, i))

                for j in range(image.shape[1]):
                    name_file = join(self.dir, i[:-4]) + f"_{j}.tiff"

                    save_tiff.imwrite(name_file, image[0, j, 0, ...])

        image_list = listdir(self.dir)

        if self.nd2_list is None:
            image_list = [i for i in image_list if i.endswith(IMG_FORMAT)]
        else:
            image_list = self.nd2_list

        self.image_name_list = image_list

    def select_view_data(self):
        self.loading_data_list = True
        self.select_data_view.clear()
        self.select_data_view.addItems(list(self.image_name_list))

        self.loading_data_list = False
        self.view_selected_data()

    def view_selected_data(self):
        if self.loading_data_list:
            return

        name_ = self.select_data_view.currentText()
        img = load_image(join(self.dir, name_), px=False)
        range_ = (
            (np.min(img[0, 0, 0, ...]), np.max(img[0, 0, 0, ...]))
            if name_.endswith(".nd2")
            else (np.min(img.flatten()), np.max(img.flatten()))
        )

        self.analysis_ch.clear()
        if name_.endswith(".nd2"):
            self.analysis_ch.addItems([str(i) for i in range(img.shape[0])])
        else:
            self.analysis_ch.addItems(["0"])

        self.viewer.layers.clear()
        self.nd2_current_frame = 0
        create_image_layer(
            self.viewer,
            image=img,
            range_=range_,
            name=splitext(name_)[0],
            transparency=False,
            zero_dim=True if name_.endswith(".nd2") else False,
        )
        self.nd2_file_frame()

        mask, instance = None, None
        # Find files starting with name_ and ending with extension
        # if self.nd2_list load all files and place them in [C, F, T, Y, X] else load file
        nd2_in = f"_0" if self.nd2_list is not None else ""
        is_mask_tif = isfile(
            join(self.dir, "Predictions", splitext(name_)[0] + nd2_in + "_semantic.tif")
        )
        is_mask_mrc = isfile(
            join(self.dir, "Predictions", splitext(name_)[0] + nd2_in + "_semantic.mrc")
        )

        is_instance = isfile(
            join(
                self.dir,
                "Predictions",
                splitext(name_)[0] + nd2_in + "_instances.csv",
            )
        )

        is_instance_filter = isfile(
            join(
                self.dir,
                "Predictions",
                splitext(name_)[0] + nd2_in + "_instances_filter.csv",
            )
        )

        if is_instance_filter:
            is_instance = False

        if self.no_instances.checkState() != 2:
            is_instance = False
            is_instance_filter = False

        if is_mask_tif:
            if self.nd2_list is None:
                mask = load_image(
                    join(self.dir, "Predictions", splitext(name_)[0] + "_semantic.tif"),
                    px=False,
                )
            else:
                mask = np.zeros(img.shape, dtype=np.uint8)

                for i in range(img.shape[1]):
                    mask_array = load_image(
                        join(
                            self.dir,
                            "Predictions",
                            splitext(name_)[0] + f"_{i}" + "_semantic.tif",
                        ),
                        px=False,
                    )
                    mask[:, i, :, ...] = mask_array
        elif is_mask_mrc:
            if self.nd2_list is None:
                mask = load_image(
                    join(
                        self.dir,
                        "Predictions",
                        splitext(name_)[0] + f"_0" + "_semantic.mrc",
                    ),
                    px=False,
                )
            else:
                mask = load_image(
                    join(
                        self.dir,
                        "Predictions",
                        splitext(name_)[0] + nd2_in + "_semantic.mrc",
                    ),
                    px=False,
                )  # Y, Z
                dir_list = [
                    i
                    for i in listdir(join(self.dir, "Predictions"))
                    if i.startswith(splitext(name_)[0]) and i.endswith("_semantic.mrc")
                ]
                mask = np.zeros((1, len(dir_list), 1, *mask.shape), dtype=np.uint8)

                for i in range(len(dir_list)):
                    mask[0, i, 0, ...] = load_image(
                        join(
                            self.dir,
                            "Predictions",
                            splitext(name_)[0] + f"_{i}" + "_semantic.mrc",
                        ),
                        px=False,
                    )

        if is_instance:
            if self.nd2_list is None:
                instance = np.genfromtxt(
                    join(
                        self.dir,
                        "Predictions",
                        splitext(name_)[0] + "_instances.csv",
                    ),
                    delimiter=",",
                    skip_header=1,
                )
            else:
                id_ = f"_{self.nd2_file_frame_bt.currentText()}"
                instance = np.genfromtxt(
                    join(
                        self.dir,
                        "Predictions",
                        splitext(name_)[0] + id_ + "_instances.csv",
                    ),
                    delimiter=",",
                    skip_header=1,
                )

        if is_instance_filter:
            if self.nd2_list is None:
                instance = np.genfromtxt(
                    join(
                        self.dir,
                        "Predictions",
                        splitext(name_)[0] + "_instances_filter.csv",
                    ),
                    delimiter=",",
                    skip_header=1,
                )
            else:
                id_ = f"_{self.nd2_file_frame_bt.currentText()}"
                instance = np.genfromtxt(
                    join(
                        self.dir,
                        "Predictions",
                        splitext(name_)[0] + id_ + "_instances_filter.csv",
                    ),
                    delimiter=",",
                    skip_header=1,
                )

        if mask is not None:
            create_image_layer(
                self.viewer,
                image=mask,
                name=splitext(name_)[0] + "_semantic",
                range_=(0, 1),
                transparency=True,
            )

        if instance is not None and self.no_instances:
            create_point_layer(
                self.viewer,
                points=instance,
                name=splitext(name_)[0] + "_instance",
                visibility=True,
                as_filament=True,
            )

    def predict_workflow(self):
        predictor = GeneralPredictor(
            predict="Microtubule_tirf",
            dir_s=self.dir,
            binary_mask=False,
            correct_px=None,
            normalize_px=1.0,
            convolution_nn="fnet_attn",
            checkpoint=[None, None],
            model_version=None,
            output_format="tif_csv",
            patch_size=256,
            cnn_threshold=self.cnn_threshold.text(),
            dist_threshold=convert_to_float(self.dist_threshold.text()),
            points_in_patch=900,
            predict_with_rotation=True,
            amira_prefix=None,
            filter_by_length=100,
            connect_splines=25,
            connect_cylinder=12,
            amira_compare_distance=None,
            amira_inter_probability=None,
            instances=True,
            device_s=self.device.currentText(),
            debug=False,
        )

        predictor()
        self.view_selected_data()

    def re_analise_data(self):
        name_ = self.select_data_view.currentText()
        data = self.get_selected_data()
        frame_ = self.nd2_current_frame
        dim_ = int(self.analysis_ch.currentText())

        data = sort_segments(data)
        data = sort_by_length(reorder_segments_id(data))

        analysis_list = self.analysis_list.currentData()

        img = None
        if name_.endswith(".nd2"):
            img = self.viewer.layers[splitext(name_)[0]].data

        pixel_size = convert_to_float(self.pixel_size_bt.text())
        if pixel_size == 1.0:
            pixel_size = None
        else:
            pixel_size = [pixel_size]

        analyse_filaments_list(
            data=[data],
            names_l=[splitext(name_)[0] + f"_{frame_}"],
            path=join(self.dir, "Predictions", "Analysis"),
            images=[img[dim_, frame_, ...]] if img is not None else None,
            image_corr=img[:, frame_, 0, ...],
            frame_id=dim_,
            px=pixel_size,
            thicknesses=[
                int(self.thickness_bt1.currentText()),
                int(self.thickness_bt2.currentText()),
            ],
            anal_list=analysis_list,
        )

        # # Co-localization
        # if "colocalize" in analysis_list:
        #     if img.shape[0] > 2:
        #         pass

    def re_analise_data_movies(self):
        name_ = self.select_data_view.currentText()
        frame_ = self.nd2_current_frame
        dim_ = int(self.analysis_ch.currentText())
        analysis_list = self.analysis_list.currentData()
        analysis_list = [d for d in analysis_list if d != "correlation"]
        try:
            data = self.viewer.layers[splitext(name_)[0] + "_instance"].data
            if data.shape[1] == 5:
                data = np.array((data[:, 0], data[:, 4], data[:, 3], data[:, 2])).T
            else:
                try:
                    ids = self.viewer.layers[
                        splitext(name_)[0] + "_instance"
                    ].properties["ID"]
                    data = np.array((ids, data[:, 2], data[:, 1], data[:, 0])).T
                except KeyError:
                    data = np.zeros((0, 4))
        except AttributeError:
            return

        data = sort_segments(data)
        data = sort_by_length(reorder_segments_id(data))

        image = self.viewer.layers[splitext(name_)[0]].data
        T = image.shape[2]  # No. of time points

        pixel_size = convert_to_float(self.pixel_size_bt.text())
        if pixel_size == 1.0:
            pixel_size = None
        else:
            if T is not None:
                pixel_size = [pixel_size for _ in range(T)]
            else:
                pixel_size = [pixel_size]

        analyse_filaments_list(
            data=[data for _ in range(T)],
            names_l=[splitext(name_)[0] + f"_{i}" for i in range(T)],
            path=join(self.dir, "Predictions", "Analysis"),
            images=[image[dim_, frame_, i, ...] for i in range(T)],
            px=pixel_size,
            anal_list=analysis_list,
            thicknesses=[
                int(self.thickness_bt1.currentText()),
                int(self.thickness_bt2.currentText()),
            ],
        )
        # # Co-localization
        # if "colocalize" in analysis_list:
        #     """
        #     - Get all channels, and dim_ as a changel to which we colocalizing
        #     - get thicness from bt1 and bt2 as the bigger one
        #     - For each frame_
        #         - for each MT
        #             - For each Time point
        #                 - extract MT track on all channels
        #                 - Compute co-localized signal between all channels
        #                 - append resoults
        #     """
        #     pass

    def nd2_file_frame(self):
        self.nd2_update = True
        name_ = self.select_data_view.currentText()
        self.nd2_file_frame_bt.clear()
        if name_.endswith(".nd2"):
            C = self.viewer.layers[splitext(name_)[0]].data.shape[1]
            self.nd2_file_frame_bt.addItems([str(i) for i in range(C)])

        self.nd2_update = False

    def nd2_file_frame_change(self):
        if not self.nd2_update:
            name_ = self.select_data_view.currentText()
            try:
                self.save_edited_instances()
            except AttributeError:
                pass
            self.nd2_current_frame = int(self.nd2_file_frame_bt.currentText())

            id_ = f"_{self.nd2_file_frame_bt.currentText()}"
            try:
                instance = np.genfromtxt(
                    join(
                        self.dir,
                        "Predictions",
                        splitext(name_)[0] + id_ + "_instances_filter.csv",
                    ),
                    delimiter=",",
                    skip_header=1,
                )
            except FileNotFoundError:
                instance = np.genfromtxt(
                    join(
                        self.dir,
                        "Predictions",
                        splitext(name_)[0] + id_ + "_instances.csv",
                    ),
                    delimiter=",",
                    skip_header=1,
                )

            create_point_layer(
                self.viewer,
                points=instance,
                name=splitext(name_)[0] + "_instance",
                visibility=True,
                as_filament=True,
            )

    def point_layer(self):
        data, name = self.get_selected_data(name=True)

        create_point_layer(
            viewer=self.viewer,
            points=data,
            name=name,
            visibility=True,
            as_filament=False,
        )
        self.viewer.layers[name].mode = "select"

    def track_layer(self):
        data, name = self.get_selected_data(name=True)

        data = sort_segments(data)
        data = sort_by_length(reorder_segments_id(data))
        create_point_layer(
            viewer=self.viewer,
            points=data,
            name=name,
            visibility=True,
            as_filament=True,
        )

    def add_new_filament(self):
        """
        Logic to add new filament
        """
        data, name, type_ = self.get_selected_data(name=True, type_=True)
        self.point_layer()

        if len(data) == 0:
            new_id = 1
        else:
            new_id = int(np.max(np.unique(data[:, 0])) + 1)
            self.resample()

        self.viewer.layers[name].feature_defaults["ID"] = new_id
        self.viewer.layers[name].mode = "add"

        show_info(f"Adding new filament id: {new_id}")

    def remove_filament(self):
        data, name, type_ = self.get_selected_data(name=True, type_=True)
        id_to_remove = self.filament_id_rm.text()

        if id_to_remove == "None":
            return

        id_to_remove = int(id_to_remove)
        data = data[~np.isin(data[:, 0], [id_to_remove])]

        data = reorder_segments_id(data)
        create_point_layer(
            viewer=self.viewer,
            points=data,
            name=name,
            visibility=True,
            as_filament=True if type_ == "tracks" else False,
        )
        show_info(f"Removed filament id: {id_to_remove}")

    def remove_selected_filament(self):
        data, name, type_ = self.get_selected_data(name=True, type_=True)
        indices = self.get_selected_ids()
        indices = [int(i) for i in np.unique(data[indices, 0])]

        if len(indices) == 0:
            return

        for i in indices:
            data = data[~np.isin(data[:, 0], [i])]
        data = reorder_segments_id(data)
        create_point_layer(
            viewer=self.viewer,
            points=data,
            name=name,
            visibility=True,
            as_filament=True if type_ == "tracks" else False,
        )
        show_info(f"Removed filament ids: {indices}")

    def join_filaments(self):
        data, name, type_ = self.get_selected_data(name=True, type_=True)
        id_1_to_join = self.join_filaments_id_1.text()
        id_2_to_join = self.join_filaments_id_2.text()

        if id_1_to_join != "None" and id_2_to_join != "None":
            id_1_to_join = int(id_1_to_join)
            id_2_to_join = int(id_2_to_join)
            if id_1_to_join == id_2_to_join:
                return

            joined = data[np.isin(data[:, 0], [id_1_to_join, id_2_to_join]), 1:]
            joined = sort_segment(joined)

            next_id = np.max(np.unique(data[:, 0])).item(0) + 1
            next_id = np.repeat(next_id, len(joined))

            joined = np.array((next_id, joined[:, 0], joined[:, 1], joined[:, 2])).T

            data = data[~np.isin(data[:, 0], [id_1_to_join, id_2_to_join])]
            data = np.concatenate((data, joined))

            data = sort_by_length(reorder_segments_id(data))

            create_point_layer(
                viewer=self.viewer,
                points=data,
                name=name,
                visibility=True,
                as_filament=True if type_ == "tracks" else False,
            )
            show_info(f"Joined filament ids: {id_1_to_join} and {id_2_to_join}")
        else:
            return

    def join_selected_filaments(self):
        data, name, type_ = self.get_selected_data(name=True, type_=True)
        indices = self.get_selected_ids()
        indices = [int(i) for i in np.unique(data[indices, 0])]

        if len(indices) <= 1:
            return

        joined = data[np.isin(data[:, 0], indices), 1:]
        joined = sort_segment(joined)

        next_id = np.max(np.unique(data[:, 0])).item(0) + 1
        next_id = np.repeat(next_id, len(joined))

        joined = np.array((next_id, joined[:, 0], joined[:, 1], joined[:, 2])).T

        data = data[~np.isin(data[:, 0], indices)]
        data = np.concatenate((data, joined))

        data = sort_by_length(reorder_segments_id(data))

        create_point_layer(
            viewer=self.viewer,
            points=data,
            name=name,
            visibility=True,
            as_filament=True if type_ == "tracks" else False,
        )
        show_info(f"Joined filament ids: {indices}")

    def get_selected_ids(self):
        active_layer = self.viewer.layers.selection.active.name
        return list(self.viewer.layers[active_layer].selected_data)

    def get_selected_data(self, name=False, type_=False, image=False):
        active_layer = self.viewer.layers.selection.active.name
        data = self.viewer.layers[active_layer].data

        if image:
            return data, active_layer

        # Convert IDxTimexZxYxX to IDxXxYxZ
        if data.shape[-1] == 5:
            data = np.array((data[:, 0], data[:, 4], data[:, 3], data[:, 2])).T
            type_layer = "tracks"
        else:
            try:
                ids = self.viewer.layers[active_layer].properties["ID"]
                data = np.array((ids, data[:, 2], data[:, 1], data[:, 0])).T
            except KeyError:
                data = np.zeros((0, 4))

            type_layer = "points"

        if name:
            if type_:
                return data, active_layer, type_layer
            return data, active_layer
        if type_:
            return data, type_layer
        return data

    def save_edited_instances(self):
        # ToDo check if files are different, if not than do not save
        data = self.get_selected_data()

        name_ = splitext(self.select_data_view.currentText())[0]
        f_name = name_ + f"_{self.nd2_current_frame}" + "_instances_filter"

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, f_format = QFileDialog.getSaveFileName(
            caption="Save Files",
            directory=join(getcwd(), f_name),
            filter="CSV File (*.csv)",
            options=options,
        )
        filename = os.path.splitext(filename)[0]

        filename = filename + ".csv"

        segments = pd.DataFrame(data)
        segments.to_csv(
            join(self.dir, "Predictions", filename),
            header=["IDs", "X [A]", "Y [A]", "Z [A]"],
            index=False,
            sep=",",
        )

    def resample(self):
        data, name, type_ = self.get_selected_data(name=True, type_=True)

        px = np.ceil(25 / convert_to_float(self.pixel_size_bt.text()))
        data = sort_segments(data)
        data = resample_filament(data, px)
        create_point_layer(
            viewer=self.viewer,
            points=data,
            name=name,
            visibility=True,
            as_filament=True if type_ == "tracks" else False,
        )
        show_info("Resampled all filament with 25A spacing.")

    def stabilize_movie(self):
        data, name = self.get_selected_data(image=True)
        channel = int(self.analysis_ch.currentText())

        if data.ndim != 5:
            show_info("Not a movie, nothing to stabilize.")
            return

        for i in range(data.shape[1]):
            data[channel, i, ...] = frames_phase_correlation(data[channel, i, ...])

        create_image_layer(
            self.viewer,
            image=data,
            range_=(np.min(data[0, 0, 0, ...]), np.max(data[0, 0, 0, ...])),
            name=name,
            transparency=False,
            zero_dim=True,
        )
