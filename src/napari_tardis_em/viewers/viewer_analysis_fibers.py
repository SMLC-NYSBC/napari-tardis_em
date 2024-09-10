#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2024                                                   #
#######################################################################
from os import getcwd, listdir
from os.path import join, splitext

from napari import Viewer
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QPushButton,
    QFormLayout,
    QComboBox,
    QLabel,
    QDoubleSpinBox,
    QFileDialog,
    QLineEdit,
)
from qtpy.QtWidgets import QWidget

import numpy as np

from napari_tardis_em.viewers import IMG_FORMAT
from napari_tardis_em.viewers.styles import border_style
from napari_tardis_em.utils.utils import get_list_of_device
from napari_tardis_em.viewers.utils import create_image_layer, create_point_layer

from tardis_em.analysis.analysis import analise_filaments_list
from tardis_em.utils.load_data import load_image
from tardis_em.utils.predictor import GeneralPredictor
from tardis_em.analysis.filament_utils import (
    reorder_segments_id,
    sort_segment,
    sort_by_length,
    sort_segments,
)


class TardisWidget(QWidget):
    """
    Easy to use plugin for general Microtubule prediction.

    Plugin integrate TARDIS-em and allow to easily set up training. To make it more
    user-friendly, this plugin guid user what to do, and during training display
     results from validation loop.
    """

    def __init__(self, viewer_mt_tirf: Viewer):
        super().__init__()

        self.predicted = False
        self.viewer = viewer_mt_tirf

        self.predictor = None
        self.img_threshold, self.scale_shape = None, None

        self.img, self.px = None, None
        self.out_ = getcwd()
        self.image_list = None

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

        """""" """""" """
           UI Setup
        """ """""" """"""
        layout = QFormLayout()
        layout.addRow("Select Directory", self.directory)

        ##############################
        # Setting user should change #
        ##############################
        label_2 = QLabel("                                      ")
        label_2.setStyleSheet(border_style("green"))

        self.workflow = QComboBox()
        self.workflow.addItems(
            [
                "Actin",
                "Microtubule",
                "Microtubule_tirf",
            ]
        )
        self.workflow.setToolTip("Select workflow.")
        self.workflow.setCurrentIndex(1)

        self.output_semantic = QComboBox()
        self.output_semantic.addItems(["mrc", "tif", "npy", "am"])
        self.output_semantic.setToolTip("Select semantic output format file.")
        self.output_semantic.setCurrentIndex(1)

        self.output_instance = QComboBox()
        self.output_instance.addItems(["None", "csv", "npy", "amSG", "mrc"])
        self.output_instance.setToolTip("Select instance output format file.")
        self.output_instance.setCurrentIndex(1)

        self.cnn_threshold = QDoubleSpinBox()
        self.cnn_threshold.setDecimals(2)
        self.cnn_threshold.setMinimum(0)
        self.cnn_threshold.setMaximum(1)
        self.cnn_threshold.setSingleStep(0.01)
        self.cnn_threshold.setValue(0.25)
        self.cnn_threshold.setToolTip(
            "Threshold value for binary prediction. Lower value will increase \n"
            "recall [retrieve more of predicted object] but also may increase \n"
            "false/positives. Higher value will result in cleaner output but may \n"
            "reduce recall.\n"
            "\n"
            "If selected 0.0 - Output probability mask \n"
            "If selected 1.0 - Use adaptive threshold."
        )
        self.workflow.currentIndexChanged.connect(self.update_cnn_threshold)

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

        label_run = QLabel("Start Prediction                 ")
        label_run.setStyleSheet(border_style("blue"))

        self.predict_1_button = QPushButton("Predict with workflow...")
        self.predict_1_button.setMinimumWidth(225)
        self.predict_1_button.clicked.connect(self.predict_workflow)

        self.predict_2_button = QPushButton("Re-analyse...")
        self.predict_2_button.setMinimumWidth(225)
        self.predict_2_button.clicked.connect(self.re_analise_data)

        self.add_new_filament_bt = QPushButton("Add new filament")
        self.add_new_filament_bt.setMinimumWidth(225)
        self.add_new_filament_bt.clicked.connect(self.add_new_filament)

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

        """
        Initialized UI
        """
        layout = QFormLayout()
        layout.addRow("Select Directory", self.directory)

        layout.addRow("---- Workflow ----", label_2)
        layout.addRow("Workflow type", self.workflow)
        layout.addRow("Semantic output", self.output_semantic)
        layout.addRow("Instance output", self.output_instance)
        layout.addRow("CNN threshold", self.cnn_threshold)
        layout.addRow("DIST threshold", self.dist_threshold)
        layout.addRow("Device", self.device)

        layout.addRow("---- Run Prediction -----", label_run)
        layout.addRow("", self.predict_1_button)
        layout.addRow("", self.predict_2_button)

        layout.addRow("---- Correct-filaments ----", label_2)
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

        self.setLayout(layout)

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

    def load_data(self):
        image_list = listdir(self.dir)
        nd2_list = [i for i in image_list if i.endswith(".nd2")]

        if len(nd2_list) > 0:
            import tifffile.tifffile as save_tiff
            from tardis_em.utils.load_data import load_nd2_file

            for i in nd2_list:
                image, _ = load_nd2_file(join(self.dir, i), channels=True)

                for j in range(image.shape[0]):
                    name_file = join(self.dir, i[:-4]) + f"_{j}.tiff"

                    save_tiff.imwrite(name_file, image[j, ...])

        image_list = listdir(self.dir)
        image_list = [i for i in image_list if i.endswith(IMG_FORMAT)]

        for i in image_list:
            img = load_image(join(self.dir, i), px_=False)
            name_, _ = splitext(i)

            create_image_layer(
                self.viewer,
                image=img,
                name=name_,
                range_=(np.min(img).item(0), np.max(img).item(0)),
                transparency=False,
            )
        self.image_list = image_list

    def predict_workflow(self):
        workflow = self.workflow.currentText()
        predictor = GeneralPredictor(
            predict=workflow,
            dir_=self.dir,
            binary_mask=False,
            correct_px=None,
            normalize_px=1.0 if workflow == "Microtubule_tirf" else None,
            convolution_nn="fnet_attn",
            checkpoint=[None, None],
            model_version=None,
            output_format="return_return",
            patch_size=256 if workflow == "Microtubule_tirf" else 96,
            cnn_threshold=self.cnn_threshold.text(),
            dist_threshold=float(self.dist_threshold.text()),
            points_in_patch=900,
            predict_with_rotation=True,
            amira_prefix=None,
            filter_by_length=100 if workflow == "Microtubule_tirf" else 1000,
            connect_splines=25 if workflow == "Microtubule_tirf" else 2500,
            connect_cylinder=12 if workflow == "Microtubule_tirf" else 250,
            amira_compare_distance=None,
            amira_inter_probability=None,
            instances=True,
            device_=self.device.currentText(),
            debug=False,
        )

        images, instances, instances_filter = predictor()

        names_ = []
        for id_, i in enumerate(images):
            if self.image_list is not None:
                name_ = names_[-1] + "_semantic"
            else:
                name_ = f"Prediction_semantic_{id_}"

            create_image_layer(
                self.viewer,
                image=i,
                name=name_,
                range_=(0, 1),
                transparency=False,
            )

        for id_, i in enumerate(instances_filter):
            if self.image_list is not None:
                name_ = names_[id_] + "_instances_filter"
            else:
                name_ = f"Prediction_instances_filter_{id_}"
            names_.append(name_)

            create_point_layer(
                self.viewer,
                points=i,
                name=name_,
                visibility=True,
                as_filament=True,
            )
        self.predicted = True

        analise_filaments_list(
            data=instances_filter,
            names_=names_,
            path=join(self.dir, "Predictions"),
            images=images,
            px_=None,
        )

    def re_analise_data(self):
        data, name = self.get_selected_data(name=True)

        data = sort_segments(data)
        data = sort_by_length(reorder_segments_id(data))

        if not self.predicted:
            dir_ = QFileDialog.getExistingDirectory(
                caption="Open Files",
                directory=getcwd(),
                # filter="Image Files (*.mrc *.rec *.map, *.tif, *.tiff, *.am)",
            )
        else:
            dir_ = self.dir

        if name.endswith("_instances_filter"):
            img_name = name[:-17]
            image = self.viewer.layers[img_name].data
        elif name.endswith("_instances"):
            img_name = name[:-10]
            image = self.viewer.layers[img_name].data

        analise_filaments_list(
            data=[data],
            names_=[name],
            path=dir_,
            images=[image],
            px_=None,
        )

    def update_cnn_threshold(self):
        if self.workflow.currentText() == "Microtubule_tirf":
            self.cnn_threshold.setValue(0.1)
        else:
            self.cnn_threshold.setValue(0.25)

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

        self.viewer.layers[name].feature_defaults["ids"] = (
            np.max(np.unique(data[:, 0])) + 1
        )

    def remove_filament(self):
        data, name, type_ = self.get_selected_data(name=True, type_=True)
        id_to_remove = self.filament_id_rm.text()

        if id_to_remove != "None":
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
        else:
            return

    def join_filaments(self):
        data, name, type_ = self.get_selected_data(name=True, type_=True)
        id_1_to_join = self.join_filaments_id_1.text()
        id_2_to_join = self.join_filaments_id_2.text()

        if id_1_to_join != "None" and id_2_to_join != "None":
            id_1_to_join = int(id_1_to_join)
            id_2_to_join = int(id_2_to_join)

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
        else:
            return

    def get_selected_data(self, name=False, type_=False):
        active_layer = self.viewer.layers.selection.active.name
        data = self.viewer.layers[active_layer].data

        # Convert IDxTimexZxYxX to IDxXxYxZ
        if data.shape[-1] == 5:
            data = np.array((data[:, 0], data[:, 4], data[:, 3], data[:, 2])).T
            type_layer = "tracks"
        else:
            ids = self.viewer.layers[active_layer].properties["ids"]
            data = np.array((ids, data[:, 2], data[:, 1], data[:, 0])).T
            type_layer = "points"
        if name:
            if type_:
                return data, active_layer, type_layer
            return data, active_layer
        if type_:
            return data, type_layer
        return data
