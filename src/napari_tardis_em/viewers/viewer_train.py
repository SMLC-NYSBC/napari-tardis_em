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

from PyQt5.QtGui import QIntValidator, QDoubleValidator
from PyQt5.QtWidgets import QPushButton, QFormLayout, QLineEdit, QComboBox, QLabel
from napari import Viewer

from qtpy.QtWidgets import QWidget

from napari_tardis_em.utils.styles import border_style
from napari_tardis_em.utils.utils import get_list_of_device


class TardisWidget(QWidget):
    """
    Easy to use plugin for CNN training.

    Plugin integrate TARDIS-em and allow to easily set up training. To make it more
    user-friendly, this plugin guid user what to do, and during training display
     results from validation loop.
    """

    def __init__(self, viewer_train: Viewer):
        super().__init__()

        self.viewer = viewer_train

        """""" """""" """
          UI Elements
        """ """""" """"""
        directory = QPushButton(f"...{os.getcwd()[-30:]}")
        directory.setToolTip(
            "Select directory with image and mask files. \n "
            "Image and file files should have the following naming: \n"
            "  - image: name.*\n"
            "  - mask: name_maks.*\n"
            "\n"
            "Supported formats:\n"
            "images: *.mrc, *.rec, *.map, *.am\n"
            "masks: *.CorrelationLines.am, *_mask.am, *_mask.mrc, *_mask.rec, *_mask.csv, *_mask.tif"
        )
        output = QPushButton(f"...{os.getcwd()[-17:]}/tardis-em_training/")
        output.setToolTip(
            "Select directory in which plugin will save train model, checkpoints and training logs."
        )

        ##############################
        # Setting user should change #
        ##############################
        label_2 = QLabel("Setting user should change")
        label_2.setStyleSheet(border_style("green"))

        patch_size = QComboBox()
        patch_size.addItems(
            ["32", "64", "96", "128", "160", "192", "256", "512", "1024"]
        )
        patch_size.setCurrentIndex(1)
        patch_size.setToolTip(
            "Select patch size value that will be used to split \n"
            "all images into smaller patches."
        )

        cnn_type = QComboBox()
        cnn_type.addItems(["unet", "resnet", "unet3plus", "fnet", "fnet_attn"])
        cnn_type.setCurrentIndex(0)
        cnn_type.setToolTip("Select type of CNN you would like to train.")

        image_type = QComboBox()
        image_type.addItems(["2D", "3D"])
        image_type.setCurrentIndex(1)
        image_type.setToolTip(
            "Select type of images you would like to train CNN model on."
        )

        cnn_in_channel = QLineEdit("1")
        cnn_in_channel.setValidator(QIntValidator(1, 100))
        cnn_in_channel.setToolTip(
            "Select how many input channels the CNN network should expect."
        )

        device = QComboBox()
        device.addItems(get_list_of_device())
        device.setCurrentIndex(0)
        device.setToolTip(
            "Select available device on which you want to train your model."
        )

        checkpoint = QLineEdit("None")
        checkpoint.setToolTip(
            "Optional, directory to CNN checkpoint to restart training."
        )

        ###########################
        # Setting user may change #
        ###########################
        label_3 = QLabel("Setting user may change")
        label_3.setStyleSheet(border_style("yellow"))

        pixel_size = QLineEdit("None")
        pixel_size.setValidator(QDoubleValidator(0.1, 50.0, 2))
        pixel_size.setToolTip(
            "Optionally, select pixel size value that will be \n"
            "used to normalize all images fixed resolution."
        )

        mask_size = QLineEdit("150")
        mask_size.setValidator(QIntValidator(5, 250))
        mask_size.setToolTip(
            "Select mask size in Angstrom. The mask size is used \n"
            "to draw mask/labels based on coordinates if name_maks.* \n"
            "files is a *.csv file with coordinates."
        )

        batch_size = QLineEdit("24")
        batch_size.setValidator(QIntValidator(5, 50))
        batch_size.setToolTip(
            "Select number of batches. The batch  refers to a set of multiple data \n"
            "samples processed together. This setting will heavy imply how much GPU memory \n"
            "CNN training will require. Reduce this number if needed."
        )

        cnn_layers = QLineEdit("5")
        cnn_layers.setValidator(QIntValidator(2, 6))
        cnn_layers.setToolTip("Select number of convolution layer for CNN.")

        cnn_scaler = QComboBox()
        cnn_scaler.addItems(["16", "32", "64"])
        cnn_scaler.setCurrentIndex(1)
        cnn_scaler.setToolTip(
            "Convolution multiplayer for CNN layers. This mean what is the CNN layer \n"
            "multiplayer at each layer.\n"
            "For example:\n"
            "If we have 5 layers and 32 multiplayer at the last layer model will have 512 channels."
        )

        loss_function = QComboBox()
        loss_function.addItems(
            [
                "AdaptiveDiceLoss",
                "BCELoss",
                "WBCELoss",
                "BCEDiceLoss",
                "CELoss",
                "DiceLoss",
                "ClDiceLoss",
                "ClBCELoss",
                "SigmoidFocalLoss",
                "LaplacianEigenmapsLoss",
                "BCEMSELoss",
            ]
        )
        loss_function.setCurrentIndex(1)
        loss_function.setToolTip("Select one of the pre-build loss functions.")

        learning_rate = QLineEdit("0.0005")
        learning_rate.setValidator(QDoubleValidator(0.0000001, 0.1, 7))
        learning_rate.setToolTip(
            "Select learning rate.\n"
            "The learning rate is a hyperparameter that controls how much to adjust \n"
            "the model’s weights with respect to the loss gradient during training"
        )

        epoch = QLineEdit("1000")
        epoch.setValidator(QIntValidator(10, 100000))
        epoch.setToolTip(
            "Select maximum number of epoches for which CNN model should train."
        )

        early_stop = QLineEdit("100")
        early_stop.setValidator(QIntValidator(5, 10000))
        early_stop.setToolTip(
            "Early stopping in CNN training is a regularization technique that halts \n"
            "training when the model’s performance on a validation set stops improving, \n"
            "preventing overfitting. This ensures the model retains optimal generalization \n"
            "capabilities by terminating training at the point of best validation performance. \n"
            "It's recommended to use 10% value of epoch size."
        )

        dropout_rate = QLineEdit("0.5")
        dropout_rate.setValidator(QDoubleValidator(0.00, 1.00, 3))
        dropout_rate.setToolTip(
            "In machine learning, dropout is a regularization technique that randomly \n"
            "omits a fraction of neurons during training to prevent overfitting, \n"
            "while in education, dropout refers to a student who leaves school \n"
            "before completing their program."
        )

        ########################################
        # Setting user is not advice to change #
        ########################################
        label_4 = QLabel("Setting user is not advice to change")
        label_4.setStyleSheet(border_style("red"))

        cnn_out_channel = QLineEdit("1")
        cnn_out_channel.setValidator(QIntValidator(1, 100))
        cnn_out_channel.setToolTip(
            "Select how many output channels the CNN network should return."
        )

        cnn_structure = QLineEdit("gcl")
        cnn_structure.setToolTip(
            "Define structure order of the convolution block."
            "c - convolution"
            "g - group normalization"
            "b - batch normalization"
            "r - ReLU"
            "l - LeakyReLU"
            "e - GeLu"
            "p - PReLu",
        )

        cnn_kernel = QLineEdit("3")
        cnn_kernel.setToolTip("Select convolution kernel size.")

        cnn_padding = QLineEdit("1")
        cnn_padding.setToolTip("Select convolution padding size.")

        cnn_max_pool = QLineEdit("2")
        cnn_max_pool.setToolTip("Select convolution max pool size.")

        train_button = QPushButton("Train ...")
        train_button.setMinimumWidth(225)

        """""" """""" """
           UI Setup
        """ """""" """"""
        layout = QFormLayout()
        layout.addRow("Select Directory", directory)
        layout.addRow("Output Directory", output)

        layout.addRow("---- CNN Options ----", label_2)
        layout.addRow("Patch Size", patch_size)
        layout.addRow("CNN type", cnn_type)
        layout.addRow("Image type", image_type)
        layout.addRow("No. of input channel", cnn_in_channel)
        layout.addRow("Checkpoint", checkpoint)
        layout.addRow("Device", device)

        layout.addRow("----- Extra --------", label_3)
        layout.addRow("Pixel Size", pixel_size)
        layout.addRow("Mask Size", mask_size)
        layout.addRow("Batch Size", batch_size)
        layout.addRow("No. of CNN layers", cnn_layers)
        layout.addRow("Channel scaler size", cnn_scaler)
        layout.addRow("Loss function", loss_function)
        layout.addRow("Learning rate", learning_rate)
        layout.addRow("No. of Epoches", epoch)
        layout.addRow("Early stop", early_stop)
        layout.addRow("Dropout rate", dropout_rate)

        layout.addRow("---- Advance -------", label_4)
        layout.addRow("No. of output channel", cnn_out_channel)
        layout.addRow("Define CNN structure", cnn_structure)
        layout.addRow("CNN kernel size", cnn_kernel)
        layout.addRow("CNN padding size", cnn_padding)
        layout.addRow("CNN max_pool size", cnn_max_pool)

        layout.addRow("", train_button)

        self.setLayout(layout)
