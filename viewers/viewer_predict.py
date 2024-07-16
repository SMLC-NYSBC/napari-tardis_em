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

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator, QDoubleValidator
from PyQt5.QtWidgets import (
    QPushButton,
    QFormLayout,
    QLineEdit,
    QComboBox,
    QLabel,
    QCheckBox,
)
from napari import Viewer

from qtpy.QtWidgets import QWidget

from napari_tardis_em.utils.styles import border_style
from napari_tardis_em.utils.utils import get_list_of_device


class TardisWidget(QWidget):
    """
    Easy to use plugin for CNN prediction.

    Plugin integrates TARDIS-em and allows to easily set up prediction on a pre-trained
    model
    """

    def __init__(self, viewer_predict: Viewer):
        super().__init__()

        self.viewer = viewer_predict

        """""" """""" """
          UI Elements
        """ """""" """"""
        directory = QPushButton(f"...{os.getcwd()[-30:]}")
        directory.setToolTip(
            "Select directory with image or single file you would like to predict. \n "
            "\n"
            "Supported formats:\n"
            "images: *.mrc, *.rec, *.map, *.am"
        )
        output = QPushButton(f"...{os.getcwd()[-17:]}/Predictions/")
        output.setToolTip(
            "Select directory in which plugin will save train model, checkpoints and training logs."
        )

        ##############################
        # Setting user should change #
        ##############################
        label_2 = QLabel("Setting user should change")
        label_2.setStyleSheet(border_style("green"))

        output_semantic = QComboBox()
        output_semantic.addItems(["mrc", "tif", "am"])
        output_semantic.setToolTip("Select semantic output format file.")

        output_instance = QComboBox()
        output_instance.addItems(["None", "csv", "npy", "amSG"])
        output_instance.setToolTip("Select instance output format file.")

        ###########################
        # Setting user may change #
        ###########################
        label_3 = QLabel("Setting user may change")
        label_3.setStyleSheet(border_style("yellow"))

        correct_px = QLineEdit("None")
        correct_px.setToolTip(
            "Set correct pixel size value, if image header \n"
            "do not contain or stores incorrect information."
        )

        cnn_type = QComboBox()
        cnn_type.addItems(["unet", "resnet", "unet3plus", "fnet", "fnet_attn"])
        cnn_type.setCurrentIndex(0)
        cnn_type.setToolTip("Select type of CNN you would like to train.")

        checkpoint = QLineEdit("None")
        checkpoint.setToolTip("Optional, directory to CNN checkpoint.")

        patch_size = QComboBox()
        patch_size.addItems(
            ["32", "64", "96", "128", "160", "192", "256", "512", "1024"]
        )
        patch_size.setCurrentIndex(1)
        patch_size.setToolTip(
            "Select patch size value that will be used to split \n"
            "all images into smaller patches."
        )

        rotate = QCheckBox()
        rotate.setCheckState(Qt.CheckState.Checked)
        rotate.setToolTip(
            "Select if you want to switch on/of rotation during the prediction. \n"
            "If selected, during CNN prediction image is rotate 4x by 90 degrees.\n"
            "This will increase prediction time 4x. \n"
            "However may lead to more cleaner output."
        )

        cnn_threshold = QLineEdit("0.25")
        cnn_threshold.setValidator(QDoubleValidator(0.0, 1.0, 3))
        cnn_threshold.setToolTip(
            "Threshold value for binary prediction. Lower value will increase \n"
            "recall [retrieve more of predicted object] but also may increase \n"
            "false/positives. Higher value will result in cleaner output but may \n"
            "reduce recall."
        )

        dist_threshold = QLineEdit("0.5")
        dist_threshold.setValidator(QDoubleValidator(0.0, 1.0, 3))
        dist_threshold.setToolTip(
            "Threshold value for instance prediction. Lower value will increase \n"
            "recall [retrieve more of predicted object] but also may increase \n"
            "false/positives. Higher value will result in cleaner output but may \n"
            "reduce recall."
        )

        device = QComboBox()
        device.addItems(get_list_of_device())
        device.setCurrentIndex(0)
        device.setToolTip(
            "Select available device on which you want to train your model."
        )

        ########################################
        # Setting user is not advice to change #
        ########################################
        label_4 = QLabel("Setting user is not advice to change")
        label_4.setStyleSheet(border_style("red"))

        points_in_patch = QLineEdit("900")
        points_in_patch.setValidator(QDoubleValidator(100, 10000, 1))
        points_in_patch.setToolTip(
            "Number of point in patch. Higher number will increase how may points \n"
            "DIST model will process at the time. This is usually only the memory GPU constrain."
        )

        predict_button = QPushButton("Predict ...")
        predict_button.setMinimumWidth(225)

        """""" """""" """
           UI Setup
        """ """""" """"""
        layout = QFormLayout()
        layout.addRow("Select Directory", directory)
        layout.addRow("Output Directory", output)

        layout.addRow("---- CNN Options ----", label_2)
        layout.addRow("Semantic output", output_semantic)
        layout.addRow("Instance output", output_instance)

        layout.addRow("----- Extra --------", label_3)
        layout.addRow("Correct pixel size", correct_px)
        layout.addRow("CNN type", cnn_type)
        layout.addRow("Checkpoint", checkpoint)
        layout.addRow("Patch size", patch_size)
        layout.addRow("Rotation", rotate)
        layout.addRow("CNN threshold", cnn_threshold)
        layout.addRow("DIST threshold", dist_threshold)
        layout.addRow("Device", device)

        layout.addRow("---- Advance -------", label_4)
        layout.addRow("No. of points [DIST]", points_in_patch)

        layout.addRow("", predict_button)

        self.setLayout(layout)
