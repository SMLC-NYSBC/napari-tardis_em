#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2024                                                   #
#######################################################################
from os import getcwd

import numpy as np
import torch
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator, QDoubleValidator
from PyQt5.QtWidgets import (
    QPushButton,
    QFormLayout,
    QLineEdit,
    QComboBox,
    QLabel,
    QCheckBox,
    QDoubleSpinBox,
)
from napari import Viewer
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info, show_error
from qtpy.QtWidgets import QWidget

from napari_tardis_em.utils.utils import get_list_of_device
from napari_tardis_em.viewers.styles import border_style
from napari_tardis_em.viewers.utils import (
    create_image_layer,
    update_viewer_prediction,
    calculate_position,
    _update_checkpoint_dir,
)
from napari_tardis_em.viewers.viewer_utils import (
    _update_cnn_threshold,
    _update_dist_layer,
    _update_dist_graph,
    _update_versions,
    semantic_preprocess,
)
from tardis_em.cnn.data_processing.scaling import scale_image
from tardis_em.utils.load_data import load_image
from tardis_em.utils.setup_envir import clean_up


class TardisWidget(QWidget):
    """
    Easy to use plugin for general Microtubule prediction.

    Plugin integrate TARDIS-em and allow to easily set up training. To make it more
    user-friendly, this plugin guid user what to do, and during training display
     results from validation loop.
    """

    def __init__(self, viewer_mt_3d: Viewer):
        super().__init__()

        self.viewer = viewer_mt_3d

        self.predictor = None
        self.img_threshold, self.scale_shape = None, None

        self.img, self.px = None, None
        self.out_ = getcwd()

        """""" """""" """
          UI Elements
        """ """""" """"""
        self.directory = QPushButton(f"...{getcwd()[-30:]}")
        self.directory.setToolTip(
            "Select directory with image or single file you would like to predict. \n "
            "\n"
            "Supported formats:\n"
            "images: *.mrc, *.rec, *.map, *.am, *.tif"
        )
        self.directory.clicked.connect(self.load_directory)
        self.dir = getcwd()

        self.output = QPushButton(f"...{getcwd()[-17:]}/Predictions/")
        self.output.setToolTip(
            "Select directory in which plugin will save train model, checkpoints and training logs."
        )
        self.output.clicked.connect(self.load_output)
        self.output_folder = f"...{getcwd()[-17:]}/Predictions/"

        ##############################
        # Setting user should change #
        ##############################
        label_2 = QLabel("Setting user should change          ")
        label_2.setStyleSheet(border_style("green"))

        self.output_semantic = QComboBox()
        self.output_semantic.addItems(["mrc", "tif", "npy", "am"])
        self.output_semantic.setToolTip("Select semantic output format file.")

        self.output_instance = QComboBox()
        self.output_instance.addItems(["None", "csv", "npy", "amSG", "mrc"])
        self.output_instance.setToolTip("Select instance output format file.")
        self.output_instance.setCurrentIndex(1)

        self.output_formats = (
            f"{self.output_semantic.currentText()}_{self.output_instance.currentText()}"
        )

        ###########################
        # Setting user may change #
        ###########################
        label_3 = QLabel("Setting user may change             ")
        label_3.setStyleSheet(border_style("yellow"))

        self.mask = QCheckBox()
        self.mask.setCheckState(Qt.CheckState.Unchecked)
        self.mask.setToolTip(
            "Define if you input tomograms images or binary mask \n"
            "with pre segmented microtubules."
        )

        self.correct_px = QLineEdit("None")
        self.correct_px.setValidator(QDoubleValidator(0.00, 100.00, 3))
        self.correct_px.setToolTip(
            "Set correct pixel size value, if image header \n"
            "do not contain or stores incorrect information."
        )

        self.cnn_type = QComboBox()
        self.cnn_type.addItems(["unet", "resnet", "unet3plus", "fnet", "fnet_attn"])
        self.cnn_type.setCurrentIndex(4)
        self.cnn_type.setToolTip("Select type of CNN you would like to train.")
        self.cnn_type.currentIndexChanged.connect(self.update_versions)

        self.checkpoint = QPushButton("None")
        self.checkpoint.setToolTip(
            "Optional, directory to CNN checkpoint to restart training."
        )
        self.checkpoint.clicked.connect(self.update_checkpoint_dir)
        self.checkpoint_dir = None

        self.patch_size = QComboBox()
        self.patch_size.addItems(
            ["32", "64", "96", "128", "160", "192", "256", "512", "1024"]
        )
        self.patch_size.setCurrentIndex(3)
        self.patch_size.setToolTip(
            "Select patch size value that will be used to split \n"
            "all images into smaller patches."
        )

        self.rotate = QCheckBox()
        self.rotate.setCheckState(Qt.CheckState.Checked)
        self.rotate.setToolTip(
            "Select if you want to switch on/of rotation during the prediction. \n"
            "If selected, during CNN prediction image is rotate 4x by 90 degrees.\n"
            "This will increase prediction time 4x. \n"
            "However may lead to more cleaner output."
        )

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
        self.cnn_threshold.valueChanged.connect(self.update_cnn_threshold)

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
        self.dist_threshold.valueChanged.connect(self.update_dist_graph)

        self.device = QComboBox()
        self.device.addItems(get_list_of_device())
        self.device.setCurrentIndex(0)
        self.device.setToolTip(
            "Select available device on which you want to train your model."
        )

        ########################################
        # Setting user is not advice to change #
        ########################################
        label_4 = QLabel("Setting user is not advice to change")
        label_4.setStyleSheet(border_style("red"))

        self.normalize_px = QLineEdit("25.0")
        # self.normalize_px.setValidator(QDoubleValidator(1, 10000, 3))
        self.normalize_px.setToolTip(
            "Optionally, if select normalization pixel size value if you specified it"
            "during your CNN training. Type None to disable normalization, or 0.0 to use default"
        )

        self.points_in_patch = QLineEdit("600")
        self.points_in_patch.setValidator(QDoubleValidator(100, 10000, 1))
        self.points_in_patch.setToolTip(
            "Number of point in patch. Higher number will increase how may points \n"
            "DIST model will process at the time. This is usually only the memory GPU constrain."
        )

        self.model_version = QComboBox()
        self.model_version.addItems(["None"])
        self.update_versions()
        self.model_version.setToolTip("Optional version of the model from 1 to inf.")

        label_run = QLabel("Start Prediction                 ")
        label_run.setStyleSheet(border_style("blue"))

        self.predict_1_button = QPushButton("Predict Semantic...")
        self.predict_1_button.setMinimumWidth(225)
        self.predict_1_button.clicked.connect(self.predict_semantic)

        self.predict_2_button = QPushButton("Predict Instances...")
        self.predict_2_button.setMinimumWidth(225)
        self.predict_2_button.clicked.connect(self.predict_instance)

        self.stop_button = QPushButton("Stop Prediction")
        self.stop_button.setMinimumWidth(225)

        self.export_command = QPushButton("Export command for high-throughput")
        self.export_command.setMinimumWidth(225)
        self.export_command.clicked.connect(self.show_command)

        #################################
        # Optional Microtubules Filters #
        #################################
        label_5 = QLabel("Optional Microtubules Filters      ")
        label_5.setStyleSheet(border_style("orange"))

        self.amira_prefix = QLineEdit(".CorrelationLines")
        self.amira_prefix.setToolTip(
            "If dir/amira foldr exist, TARDIS will search for files with \n"
            "given prefix (e.g. file_name.CorrelationLines.am). If the correct \n"
            "file is found, TARDIS will use its instance segmentation with \n"
            "ZiB Amira prediction, and output additional file called \n"
            "file_name_AmiraCompare.am."
        )

        self.amira_compare_distance = QLineEdit("175")
        self.amira_compare_distance.setValidator(QIntValidator(0, 10000))
        self.amira_compare_distance.setToolTip(
            "The comparison with Amira prediction is done by evaluating \n"
            "filaments distance between Amira and TARDIS. This parameter defines the maximum \n"
            "distance to the similarity between two splines. Value given in Angstrom [A]."
        )
        self.amira_inter_probability = QDoubleSpinBox()
        self.amira_inter_probability.setDecimals(2)
        self.amira_inter_probability.setMinimum(0)
        self.amira_inter_probability.setMaximum(1)
        self.amira_inter_probability.setSingleStep(0.05)
        self.amira_inter_probability.setValue(0.25)
        self.amira_inter_probability.setToolTip(
            "This parameter define normalize between 0 and 1 overlap \n"
            "between filament from TARDIS na Amira sufficient to identifies microtubule as \n"
            "a match between both software."
        )

        self.filter_by_length = QLineEdit("1000")
        self.filter_by_length.setValidator(QIntValidator(0, 10000))
        self.filter_by_length.setToolTip(
            "Filtering parameters for microtubules, defining maximum microtubule \n"
            "length in angstrom. All filaments shorter then this length \n"
            "will be deleted."
        )
        self.filter_by_length.textChanged.connect(self.update_dist_graph)

        self.connect_splines = QLineEdit("2500")
        self.connect_splines.setValidator(QIntValidator(0, 10000))
        self.connect_splines.setToolTip(
            "To address the issue where microtubules are mistakenly \n"
            "identified as two different filaments, we use a filtering technique. \n"
            "This involves identifying the direction each filament end points towards and then \n"
            "linking any filaments that are facing the same direction and are within \n"
            "a certain distance from each other, measured in angstroms. This distance threshold \n"
            "determines how far apart two microtubules can be, while still being considered \n"
            "as a single unit if they are oriented in the same direction."
        )
        self.connect_splines.textChanged.connect(self.update_dist_graph)

        self.connect_cylinder = QLineEdit("250")
        self.connect_cylinder.setValidator(QIntValidator(0, 10000))
        self.connect_cylinder.setToolTip(
            "To minimize false positives when linking microtubules, we limit \n"
            "the search area to a cylindrical radius specified in angstroms. \n"
            "For each spline, we find the direction the filament end is pointing in \n"
            "and look for another filament that is oriented in the same direction. \n"
            "The ends of these filaments must be located within this cylinder \n"
            "to be considered connected."
        )
        self.connect_cylinder.textChanged.connect(self.update_dist_graph)

        """""" """""" """
           UI Setup
        """ """""" """"""
        layout = QFormLayout()
        layout.addRow("Select Directory", self.directory)
        layout.addRow("Output Directory", self.output)

        layout.addRow("---- CNN Options ----", label_2)
        layout.addRow("Semantic output", self.output_semantic)
        layout.addRow("Instance output", self.output_instance)

        layout.addRow("----- Extra --------", label_3)
        layout.addRow("Input as a mask", self.mask)
        layout.addRow("Correct pixel size", self.correct_px)
        layout.addRow("CNN type", self.cnn_type)
        layout.addRow("Checkpoint", self.checkpoint)
        layout.addRow("Patch size", self.patch_size)
        layout.addRow("Rotation", self.rotate)
        layout.addRow("CNN threshold", self.cnn_threshold)
        layout.addRow("DIST threshold", self.dist_threshold)
        layout.addRow("Device", self.device)

        layout.addRow("---- MT Filters -----", label_5)
        layout.addRow("Amira file prefix", self.amira_prefix)
        layout.addRow("Compare distance with Amira [A]", self.amira_compare_distance)
        layout.addRow("Compare similarity probability", self.amira_inter_probability)
        layout.addRow("Filter MT length [A]", self.filter_by_length)
        layout.addRow("Connect splines within distance [A]", self.connect_splines)
        layout.addRow("Connect splines within diameter [A]", self.connect_cylinder)

        layout.addRow("---- Advance -------", label_4)
        layout.addRow("Normalization pixel size [A]", self.normalize_px)
        layout.addRow("No. of points [DIST]", self.points_in_patch)
        layout.addRow("Model Version", self.model_version)

        layout.addRow("---- Run Prediction -----", label_run)
        layout.addRow("", self.predict_1_button)
        layout.addRow("", self.predict_2_button)
        layout.addRow("", self.export_command)

        self.setLayout(layout)

    def update_checkpoint_dir(self):
        filename = _update_checkpoint_dir()

        self.checkpoint.setText(filename[-30:])
        self.checkpoint_dir = filename

    def update_versions(self):
        self.model_version = _update_versions(
            self.model_version, self.cnn_type.currentText(), "microtubules_3d"
        )

    def update_cnn_threshold(self):
        if self.img is not None:
            self.img_threshold = _update_cnn_threshold(
                self.viewer, self.dir, self.img, float(self.cnn_threshold.text())
            )

            self.predictor.image = self.img_threshold
            self.predictor.save_semantic_mask(self.dir.split("/")[-1])

    def update_dist_layer(self):
        self.predictor.image = self.img_threshold

        _update_dist_layer(
            self.viewer, self.predictor.segments, self.predictor.segments_filter
        )

    def update_dist_graph(self):
        if self.predictor is not None:
            if self.predictor.graphs is not None:
                self.predictor.segments = _update_dist_graph(
                    bool(self.filament.checkState()),
                    self.predictor.segments,
                    self.predictor.GraphToSegment,
                    self.predictor.graphs,
                    self.predictor.pc_ld,
                    self.predictor.output_idx,
                )

                if self.predictor.segments is None:
                    show_info("TARDIS-em could not find any instances :(")
                    return
                else:
                    show_info(
                        f"TARDIS-em found {int(np.max(self.predictor.segments[:, 0]))} instances :)"
                    )
                    self.predictor.save_instance_PC(self.dir.split("/")[-1])

    def load_directory(self):
        filename_, out_ = _update_checkpoint_dir(filter_=True)

        self.out_ = "/".join(out_)

        self.output.setText(f"...{self.out_[-17:]}/Predictions/")
        self.output_folder = f"...{self.out_}/Predictions/"

        self.directory.setText(filename_[-30:])
        self.dir = filename_

        self.img, self.px = load_image(self.dir)

        if self.correct_px.text() == "None" and self.px >= 0.0 or self.px != 1.0:
            self.correct_px.setText(f"{self.px}")

        create_image_layer(
            self.viewer,
            image=self.img,
            name=self.dir.split("/")[-1],
            range_=(np.min(self.img), np.max(self.img)),
            transparency=False,
        )
        self.img = None

    def load_output(self):
        filename_ = _update_checkpoint_dir()

        self.output.setText(f"...{filename_[-17:]}/Predictions/")
        self.output_folder = f"{filename_}/Predictions/"

    def predict_semantic(self):
        self.output_formats, self.predictor, self.scale_shape, img_dataset = (
            semantic_preprocess(
                self.viewer,
                self.dir,
                self.output_semantic.currentText(),
                self.output_instance.currentText(),
                {
                    "correct_px": self.correct_px.text(),
                    "normalize_px": self.normalize_px.text(),
                    "cnn_threshold": float(self.cnn_threshold.text()),
                    "dist_threshold": float(self.dist_threshold.text()),
                    "model_version": self.model_version.currentText(),
                    "predict_type": "Microtubule",
                    "mask": bool(self.mask.checkState()),
                    "cnn_type": self.cnn_type.currentText(),
                    "checkpoint": [
                        (
                            None
                            if self.checkpoint.text() == "None"
                            else self.checkpoint_dir
                        ),
                        None,
                    ],
                    "patch_size": int(self.patch_size.currentText()),
                    "points_in_patch": int(self.points_in_patch.text()),
                    "rotate": bool(self.rotate.checkState()),
                    "amira_prefix": self.amira_prefix.text(),
                    "filter_by_length": int(self.filter_by_length.text()),
                    "connect_splines": int(self.connect_splines.text()),
                    "connect_cylinder": int(self.connect_cylinder.text()),
                    "amira_compare_distance": int(self.amira_compare_distance.text()),
                    "amira_inter_probability": float(
                        self.amira_inter_probability.text()
                    ),
                    "device": self.device.currentText(),
                    "image_type": None,
                },
            )
        )

        if not bool(self.mask.checkState()):

            @thread_worker(
                start_thread=False,
                progress={"desc": "semantic-segmentation-progress"},
                connect={"finished": self.update_cnn_threshold},
            )
            def predict_dataset(img_dataset_, predictor):
                for j in range(len(img_dataset_)):
                    input_, name = img_dataset_.__getitem__(j)

                    input_ = predictor.predict_cnn_napari(input_, name)
                    update_viewer_prediction(
                        self.viewer,
                        input_,
                        calculate_position(int(self.patch_size.currentText()), name),
                    )

                show_info("Finished Semantic Prediction !")

                self.img = self.predictor.image_stitcher(
                    image_dir=self.predictor.output, mask=False, dtype=np.float32
                )[
                    : self.predictor.scale_shape[0],
                    : self.predictor.scale_shape[1],
                    : self.predictor.scale_shape[2],
                ]
                self.img, _ = scale_image(
                    image=self.img, scale=self.predictor.org_shape
                )
                self.img = (
                    torch.sigmoid(torch.from_numpy(self.img)).cpu().detach().numpy()
                )
                self.predictor.image = self.img

            worker = predict_dataset(img_dataset, self.predictor)
            worker.start()
        else:
            return

    def predict_instance(self):
        if self.predictor is None:
            show_error(f"Please initialize with 'Predict Semantic' button")
            return

        self.output_formats = (
            f"{self.output_semantic.currentText()}_{self.output_instance.currentText()}"
        )

        if not self.output_formats.endswith("None"):
            if self.predictor.dist is None:
                self.predictor.output_format = self.output_formats
                self.predictor.build_NN("Microtubule")

            self.segments = np.zeros((0, 4))

            if (
                not self.predictor.image.min() == 0
                and not self.predictor.image.max() == 1
            ):
                show_error("You need to first select CNN threshold greater then 0.0")
                return

            @thread_worker(
                start_thread=False,
                progress={"desc": "instance-segmentation-progress"},
                connect={"finished": self.update_dist_layer},
            )
            def predict_dist():
                show_info("Started Instance Prediction !")

                self.predictor.preprocess_DIST(self.dir.split("/")[-1])
                if len(self.predictor.pc_ld) > 0:
                    # Build patches dataset
                    (
                        self.predictor.coords_df,
                        _,
                        self.predictor.output_idx,
                        _,
                    ) = self.predictor.patch_pc.patched_dataset(
                        coord=self.predictor.pc_ld
                    )

                    self.predictor.graphs = self.predictor.predict_DIST(
                        id_i=0, id_name=self.dir.split("/")[-1]
                    )
                    self.predictor.postprocess_DIST(id_i=0, i=self.dir.split("/")[-1])

                    if self.predictor.segments is None:
                        show_info("TARDIS-em could not find any instances :(")
                        return
                    else:
                        show_info(
                            f"TARDIS-em found {int(np.max(self.predictor.segments[:, 0]))} instances :)"
                        )
                        self.predictor.save_instance_PC(self.dir.split("/")[-1])
                        clean_up(dir_s=self.dir)
                    show_info("Finished Instance Prediction !")

            worker = predict_dist()
            worker.start()

    def show_command(self):
        ms = "" if not bool(self.mask.checkState()) else "-ms True"

        px = (
            ""
            if self.correct_px.text() == "None"
            else f"-px {float(self.correct_px.text())} "
        )
        if self.px is not None:
            px = (
                ""
                if self.px == float(self.correct_px.text())
                else f"-px {float(self.correct_px.text())} "
            )

        ch = (
            ""
            if self.checkpoint.text() == "None"
            else f"-ch {self.checkpoint_dir}_None "
        )

        mv = (
            ""
            if self.model_version.currentText() == "None"
            else f"-mv {int(self.model_version.currentText())} "
        )

        cnn = (
            ""
            if self.cnn_type.currentText() == "fnet_attn"
            else f"-cnn {self.cnn_type.currentText()} "
        )

        rt = "" if bool(self.rotate.checkState()) else "-rt False "

        ct = (
            "-ct auto "
            if float(self.cnn_threshold.text()) == 1.0
            else f"-ct {self.cnn_threshold.text()} "
        )

        dt = (
            f"-dt {float(self.dist_threshold.text())} "
            if not self.output_formats.endswith("None")
            else ""
        )

        ap = (
            ""
            if self.amira_prefix.text() == ".CorrelationLines"
            else f"-ap {self.amira_prefix.text()} "
        )
        acd = (
            ""
            if self.amira_compare_distance.text() == "175"
            else f"-acd {self.amira_compare_distance.text()} "
        )
        aip = (
            ""
            if self.amira_inter_probability.text() == "0.25"
            else f"-aip {self.amira_inter_probability.text()} "
        )

        fl = (
            ""
            if self.filter_by_length.text() == "1000"
            else f"-fl {int(self.filter_by_length.text())} "
        )
        cs = (
            ""
            if self.connect_splines.text() == "2500"
            else f"-fl {int(self.connect_splines.text())} "
        )
        cc = (
            ""
            if self.connect_cylinder.text() == "250"
            else f"-fl {int(self.connect_cylinder.text())} "
        )

        show_info(
            f"tardis_mt "
            f"-dir {self.out_} "
            f"{ms}"
            f"{px}"
            f"{ch}"
            f"{mv}"
            f"{cnn}"
            f"-out {self.output_formats} "
            f"-ps {int(self.patch_size.currentText())} "
            f"{rt}"
            f"{ct}"
            f"{dt}"
            f"{ap}"
            f"{acd}"
            f"{aip}"
            f"{fl}"
            f"{cs}"
            f"{cc}"
            f"-pv {int(self.points_in_patch.text())} "
            f"-dv {self.device.currentText()}"
        )
