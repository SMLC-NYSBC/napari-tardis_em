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
import shutil
import sys
from os.path import join

import numpy as np
import torch
from PyQt5.QtGui import QIntValidator, QDoubleValidator
from PyQt5.QtWidgets import (
    QPushButton,
    QFormLayout,
    QLineEdit,
    QComboBox,
    QLabel,
    QFileDialog,
)
from napari import Viewer
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info
from qtpy.QtWidgets import QWidget
from torch import optim
from torch.utils.data import DataLoader

from napari_tardis_em.utils.utils import get_list_of_device
from napari_tardis_em.viewers import loss_functions
from napari_tardis_em.viewers.styles import (
    border_style,
    PlotPopup,
    build_gird_with_masks,
)
from napari_tardis_em.viewers.utils import (
    setup_environment_and_dataset,
    create_image_layer,
)
from tardis_em.cnn.cnn import build_cnn_network
from tardis_em.cnn.datasets.dataloader import CNNDataset
from tardis_em.cnn.utils.utils import check_model_dict
from tardis_em.utils.errors import TardisError
from tardis_em.utils.metrics import calculate_f1


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
        self.training_plot = PlotPopup()

        self.model, self.loss_fn, self.optimizer = None, None, None
        self.train_DL, self.test_DL = None, None

        self.out_, self.output_folder = None, None

        self.dir = os.getcwd()

        """""" """""" """
          UI Elements
        """ """""" """"""
        self.directory = QPushButton(f"...{os.getcwd()[-30:]}")
        self.directory.setToolTip(
            "Select directory with image and mask files. \n "
            "Image and file files should have the following naming: \n"
            "  - image: name.*\n"
            "  - mask: name_maks.*\n"
            "\n"
            "Supported formats:\n"
            "images: *.mrc, *.rec, *.map, *.am\n"
            "masks: *.CorrelationLines.am, *_mask.am, *_mask.mrc, *_mask.rec, *_mask.csv, *_mask.tif"
        )
        self.directory.clicked.connect(self.load_directory)
        self.dir = os.getcwd()

        self.output = QPushButton(f"...{os.getcwd()[-17:]}/tardis-em_training/")
        self.output.setToolTip(
            "Select directory in which plugin will save train model, checkpoints and training logs."
        )
        self.output.clicked.connect(self.load_output)

        ##############################
        # Setting user should change #
        ##############################
        label_2 = QLabel("Setting user should change")
        label_2.setStyleSheet(border_style("green"))

        self.patch_size = QComboBox()
        self.patch_size.addItems(
            ["32", "64", "96", "128", "160", "192", "256", "512", "1024"]
        )
        self.patch_size.setCurrentIndex(1)
        self.patch_size.setToolTip(
            "Select patch size value that will be used to split \n"
            "all images into smaller patches."
        )

        self.cnn_type = QComboBox()
        self.cnn_type.addItems(["unet", "resnet", "unet3plus", "fnet", "fnet_attn"])
        self.cnn_type.setCurrentIndex(0)
        self.cnn_type.setToolTip("Select type of CNN you would like to train.")
        self.output_folder = f"{os.getcwd()}/{self.cnn_type.currentText()}_checkpoints/"

        self.image_type = QComboBox()
        self.image_type.addItems(["2D", "3D"])
        self.image_type.setCurrentIndex(1)
        self.image_type.setToolTip(
            "Select type of images you would like to train CNN model on."
        )

        self.cnn_in_channel = QLineEdit("1")
        self.cnn_in_channel.setValidator(QIntValidator(1, 100))
        self.cnn_in_channel.setToolTip(
            "Select how many input channels the CNN network should expect."
        )

        self.device = QComboBox()
        self.device.addItems(get_list_of_device())
        self.device.setCurrentIndex(0)
        self.device.setToolTip(
            "Select available device on which you want to train your model."
        )

        self.checkpoint = QPushButton("None")
        self.checkpoint.setToolTip(
            "Optional, directory to CNN checkpoint to restart training."
        )
        self.checkpoint.clicked.connect(self.update_checkpoint_dir)
        self.checkpoint_dir = None

        ###########################
        # Setting user may change #
        ###########################
        label_3 = QLabel("Setting user may change")
        label_3.setStyleSheet(border_style("yellow"))

        self.pixel_size = QLineEdit("None")
        self.pixel_size.setValidator(QDoubleValidator(0.1, 1000.0, 3))
        self.pixel_size.setToolTip(
            "Optionally, select pixel size value that will be \n"
            "used to normalize all images fixed resolution."
        )

        self.correct_pixel_size = QLineEdit("None")
        self.correct_pixel_size.setValidator(QDoubleValidator(0.1, 1000.0, 3))
        self.correct_pixel_size.setToolTip(
            "Optionally, select correct pixel size value that will be \n"
            "used for all images. This is used only in case of .tif files when \n"
            "pixel size value is not retrieved from image header.."
        )

        self.mask_size = QLineEdit("150")
        self.mask_size.setValidator(QIntValidator(5, 100000))
        self.mask_size.setToolTip(
            "Select mask size in Angstrom. The mask size is used \n"
            "to draw mask/labels based on coordinates if name_maks.* \n"
            "files is a *.csv file with coordinates."
        )

        self.batch_size = QLineEdit("24")
        self.batch_size.setValidator(QIntValidator(5, 50))
        self.batch_size.setToolTip(
            "Select number of batches. The batch  refers to a set of multiple data \n"
            "samples processed together. This setting will heavy imply how much GPU memory \n"
            "CNN training will require. Reduce this number if needed."
        )

        self.cnn_layers = QLineEdit("5")
        self.cnn_layers.setValidator(QIntValidator(2, 6))
        self.cnn_layers.setToolTip("Select number of convolution layer for CNN.")

        self.cnn_scaler = QComboBox()
        self.cnn_scaler.addItems(["16", "32", "64"])
        self.cnn_scaler.setCurrentIndex(1)
        self.cnn_scaler.setToolTip(
            "Convolution multiplayer for CNN layers. This mean what is the CNN layer \n"
            "multiplayer at each layer.\n"
            "For example:\n"
            "If we have 5 layers and 32 multiplayer at the last layer model will have 512 channels."
        )

        self.loss_function = QComboBox()
        self.loss_function.addItems(
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
        self.loss_function.setCurrentIndex(1)
        self.loss_function.setToolTip("Select one of the pre-build loss functions.")

        self.learning_rate = QLineEdit("0.0005")
        self.learning_rate.setValidator(QDoubleValidator(0.0000001, 0.1, 7))
        self.learning_rate.setToolTip(
            "Select learning rate.\n"
            "The learning rate is a hyperparameter that controls how much to adjust \n"
            "the model’s weights with respect to the loss gradient during training"
        )

        self.epoch = QLineEdit("1000")
        self.epoch.setValidator(QIntValidator(10, 100000))
        self.epoch.setToolTip(
            "Select maximum number of epoches for which CNN model should train."
        )

        self.early_stop = QLineEdit("100")
        self.early_stop.setValidator(QIntValidator(5, 10000))
        self.early_stop.setToolTip(
            "Early stopping in CNN training is a regularization technique that halts \n"
            "training when the model’s performance on a validation set stops improving, \n"
            "preventing overfitting. This ensures the model retains optimal generalization \n"
            "capabilities by terminating training at the point of best validation performance. \n"
            "It's recommended to use 10% value of epoch size."
        )

        self.dropout_rate = QLineEdit("0.5")
        self.dropout_rate.setValidator(QDoubleValidator(0.00, 1.00, 3))
        self.dropout_rate.setToolTip(
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

        self.cnn_out_channel = QLineEdit("1")
        self.cnn_out_channel.setValidator(QIntValidator(1, 100))
        self.cnn_out_channel.setToolTip(
            "Select how many output channels the CNN network should return."
        )

        self.cnn_structure = QLineEdit("gcl")
        self.cnn_structure.setToolTip(
            "Define structure order of the convolution block."
            "c - convolution"
            "g - group normalization"
            "b - batch normalization"
            "r - ReLU"
            "l - LeakyReLU"
            "e - GeLu"
            "p - PReLu",
        )

        self.cnn_kernel = QLineEdit("3")
        self.cnn_kernel.setToolTip("Select convolution kernel size.")

        self.cnn_padding = QLineEdit("1")
        self.cnn_padding.setToolTip("Select convolution padding size.")

        self.cnn_max_pool = QLineEdit("2")
        self.cnn_max_pool.setToolTip("Select convolution max pool size.")

        self.train_button = QPushButton("Train ...")
        self.train_button.setMinimumWidth(225)
        self.train_button.clicked.connect(self.trainer)

        self.stop_button = QPushButton("Stop Training")
        self.stop_button.setMinimumWidth(225)

        self.export_command = QPushButton("Export command for high-throughput")
        self.export_command.setMinimumWidth(225)
        self.export_command.clicked.connect(self.show_command)

        """""" """""" """
           UI Setup
        """ """""" """"""
        layout = QFormLayout()
        layout.addRow("Select Directory", self.directory)
        layout.addRow("Output Directory", self.output)

        layout.addRow("---- CNN Options ----", label_2)
        layout.addRow("Patch Size [px]", self.patch_size)
        layout.addRow("CNN type", self.cnn_type)
        layout.addRow("Image type", self.image_type)
        layout.addRow("No. of input channel", self.cnn_in_channel)
        layout.addRow("Checkpoint", self.checkpoint)
        layout.addRow("Device", self.device)

        layout.addRow("----- Extra --------", label_3)
        layout.addRow("Pixel Size [A]", self.pixel_size)
        layout.addRow("Correct Pixel Size [A]", self.correct_pixel_size)
        layout.addRow("Mask Size [A]", self.mask_size)
        layout.addRow("Batch Size", self.batch_size)
        layout.addRow("No. of CNN layers", self.cnn_layers)
        layout.addRow("Channel scaler size", self.cnn_scaler)
        layout.addRow("Loss function", self.loss_function)
        layout.addRow("Learning rate", self.learning_rate)
        layout.addRow("No. of Epoches", self.epoch)
        layout.addRow("Early stop", self.early_stop)
        layout.addRow("Dropout rate", self.dropout_rate)

        layout.addRow("---- Advance -------", label_4)
        layout.addRow("No. of output channel", self.cnn_out_channel)
        layout.addRow("Define CNN structure", self.cnn_structure)
        layout.addRow("CNN kernel size", self.cnn_kernel)
        layout.addRow("CNN padding size", self.cnn_padding)
        layout.addRow("CNN max_pool size", self.cnn_max_pool)

        layout.addRow("", self.train_button)
        layout.addRow("", self.stop_button)
        layout.addRow("", self.export_command)

        self.setLayout(layout)

    def trainer(self):
        """Set environment"""
        TRAIN_IMAGE_DIR = join(self.dir, "train", "imgs")
        TRAIN_MASK_DIR = join(self.dir, "train", "masks")
        TEST_IMAGE_DIR = join(self.dir, "test", "imgs")
        TEST_MASK_DIR = join(self.dir, "test", "masks")

        pixel_size = self.pixel_size.text()
        if pixel_size == "None":
            pixel_size = None
        else:
            pixel_size = float(pixel_size)

        correct_pixel_size = self.correct_pixel_size.text()
        if correct_pixel_size == "None":
            correct_pixel_size = None
        else:
            correct_pixel_size = float(correct_pixel_size)

        setup_environment_and_dataset(
            self.dir,
            int(self.mask_size.text()),
            pixel_size,
            int(self.patch_size.currentText()),
            correct_pixel_size,
        )

        """Build training and test dataset 2D/3D"""
        self.train_DL = DataLoader(
            dataset=CNNDataset(
                img_dir=TRAIN_IMAGE_DIR,
                mask_dir=TRAIN_MASK_DIR,
                size=int(self.patch_size.currentText()),
                out_channels=int(self.cnn_out_channel.text()),
            ),
            batch_size=int(self.batch_size.text()),
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

        self.test_DL = DataLoader(
            dataset=CNNDataset(
                img_dir=TEST_IMAGE_DIR,
                mask_dir=TEST_MASK_DIR,
                size=int(self.patch_size.currentText()),
                out_channels=int(self.cnn_out_channel.text()),
            ),
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

        model_dict = None
        cnn_type = self.cnn_type.currentText()
        cnn_out_channel = int(self.cnn_out_channel.text())
        patch_size = int(self.patch_size.currentText())
        dropout_rate = float(self.dropout_rate.text())
        cnn_layers = int(self.cnn_layers.text())
        cnn_scaler = int(self.cnn_scaler.currentText())
        conv_kernel = int(self.cnn_kernel.text())
        conv_padding = int(self.cnn_padding.text())
        pool_kernel = int(self.cnn_max_pool.text())

        if self.image_type.currentText() == "2D":
            cnn_structure = f"2{self.cnn_structure.text()}"
        else:
            cnn_structure = f"2{self.cnn_structure.text()}"

        if self.checkpoint_dir is not None:
            save_train = torch.load(self.checkpoint_dir, map_location="cpu")

            if "model_struct_dict" in save_train.keys():
                model_dict = save_train["model_struct_dict"]
                model_dict = check_model_dict(model_dict)
                globals().update(model_dict)
        else:
            save_train = None

        if model_dict is None:
            model_dict = {
                "cnn_type": cnn_type,
                "classification": False,
                "in_channel": 1,
                "out_channel": cnn_out_channel,
                "img_size": patch_size,
                "dropout": dropout_rate,
                "num_conv_layers": cnn_layers,
                "conv_scaler": cnn_scaler,
                "conv_kernel": conv_kernel,
                "conv_padding": conv_padding,
                "maxpool_kernel": pool_kernel,
                "layer_components": cnn_structure,
                "attn_features": True if cnn_type == "fnet_attn" else False,
                "num_group": 8,
                "prediction": False,
            }
        else:
            model_dict["img_size"] = patch_size

        self.structure = model_dict
        """Build CNN model"""
        try:
            self.model = build_cnn_network(
                network_type=model_dict["cnn_type"],
                structure=model_dict,
                img_size=model_dict["img_size"],
                prediction=False,
            )
        except:
            TardisError(
                "14",
                "tardis_em/cnn/train.py",
                f"CNNModelError: Model type: {type} was not build correctly!",
            )
            sys.exit()

        print_setting = [
            f"Training is started for {model_dict['cnn_type']}:",
            f"Local dir: {os.getcwd()}",
            f"Training for {model_dict['cnn_type']} with "
            f"No. of Layers: {model_dict['num_conv_layers']} with "
            f"{model_dict['in_channel']} input and "
            f"{model_dict['out_channel']} output channel",
            f"Layers are build of {model_dict['layer_components']} modules, "
            f"train on {model_dict['img_size']} pixel images, "
            f"with {model_dict['conv_scaler']} up/down sampling "
            "channel scaler.",
        ]

        """Optionally: Load checkpoint for retraining"""
        if self.checkpoint.text() != "None":
            self.model.load_state_dict(save_train["model_state_dict"])
        self.model = self.model.to(self.device.currentText())

        """Losses"""
        loss_function = {f.__name__: f() for f in loss_functions}
        self.loss_fn = loss_function["BCELoss"]
        if self.loss_function.currentText() in loss_function:
            self.loss_fn = loss_function[self.loss_function.currentText()]

        self.optimizer = optim.NAdam(
            params=self.model.parameters(),
            lr=float(self.learning_rate.text()),
            betas=(0.9, 0.999),
            eps=1e-8,
            momentum_decay=4e-3,
        )

        """Optionally: Checkpoint model"""
        if self.checkpoint.text() != "None":
            self.optimizer.load_state_dict(save_train["optimizer_state_dict"])
        del save_train
        show_info(f"Finished building model and dataset!")

        self.training_loss, self.validation_loss, self.scores = [], [], []

        @thread_worker(
            start_thread=False,
            progress={"desc": f"Training-{self.cnn_type.currentText()}"},
        )
        def train():
            def _save_metric():
                if len(self.training_loss) > 0:
                    np.savetxt(
                        join(
                            self.output_folder,
                            "training_losses.csv",
                        ),
                        self.training_loss,
                        delimiter=",",
                    )
                if len(self.validation_loss) > 0:
                    np.savetxt(
                        join(
                            self.output_folder,
                            "validation_losses.csv",
                        ),
                        self.validation_loss,
                        delimiter=",",
                    )
                if len(self.scores) > 0:
                    np.savetxt(
                        join(
                            self.output_folder,
                            "validation_f1.csv",
                        ),
                        self.validation_loss,
                        delimiter=",",
                    )

                """ Save current model weights"""
                # If mean evaluation loss is higher than save checkpoint
                if all(self.scores[-1:][0] >= j for j in self.scores[:-1]):
                    torch.save(
                        {
                            "model_struct_dict": self.structure,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                        },
                        join(
                            self.output_folder,
                            f"{self.cnn_type.currentText()}_checkpoint.pth",
                        ),
                    )

                torch.save(
                    {
                        "model_struct_dict": self.structure,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                    },
                    join(
                        self.output_folder,
                        "model_weights.pth",
                    ),
                )

            @thread_worker(
                start_thread=False,
                progress={"desc": f"Validation-{self.cnn_type.currentText()}"},
                connect={"finished": self.show_validation},
            )
            def _validate(model):
                self.first_25_m = []
                self.img_indexes = []
                model.eval()

                loss_ = []
                f1_s = []
                for idx, (img, mask) in enumerate(self.test_DL):
                    if idx < 25:
                        self.img_indexes.append(img.cpu().detach().numpy()[0, 0, :])

                    img, mask = img.to(self.device.currentText()), mask.to(
                        self.device.currentText()
                    )
                    with torch.no_grad():
                        img = model(img)
                        loss = self.loss_fn(img, mask)
                        loss_.append(loss.item())

                        img = torch.sigmoid(img)[0, 0, :]
                        mask = mask[0, 0, :]

                        img = np.where(img.cpu().detach().numpy() >= 0.5, 1, 0)
                        mask = mask.cpu().detach().numpy()
                        acc, prec, recall, f1 = calculate_f1(
                            logits=img, targets=mask, best_f1=False
                        )
                        f1_s.append(f1)

                    if idx < 25:
                        self.first_25_m.append(img)
                yield

                self.scores.append(np.mean(f1_s))
                self.validation_loss.append(np.mean(loss_))

            def _mid_training_eval(idx):
                if idx % (len(self.train_DL) // 4) == 0:
                    if (
                        idx != 0 or self.id_ != 0
                    ):  # do not compute at trainer initialization
                        # Do not validate at first idx and last 10%
                        if idx != 0 and idx <= int(len(self.train_DL) * 0.75):
                            self.model.eval()  # Enter Validation
                            worker = _validate(self.model)
                            worker.start()
                            _save_metric()

                            self.model.train()  # Move back to training

            if os.path.isdir(self.output_folder):
                shutil.rmtree(self.output_folder)
            os.mkdir(self.output_folder)

            # Run Epoch loop
            for id_ in range(int(self.epoch.text())):
                show_info(f"Training progress: {id_}/{int(self.epoch.text())}")
                self.id_ = id_

                # ToDo build validation set to display (random <=25 data)
                if self.test_DL is not None and self.id_ != 0:
                    self.model.eval()

                    worker = _validate(self.model)
                    worker.start()

                self.model.train()
                for idx, (i, m) in enumerate(self.train_DL):
                    """Validate"""
                    _mid_training_eval(idx=idx)

                    """Training"""
                    i, m = i.to(self.device.currentText()), m.to(
                        self.device.currentText()
                    )

                    self.optimizer.zero_grad()
                    i = self.model(i)  # one forward pass

                    loss = self.loss_fn(i, m)
                    loss.backward()  # one backward pass
                    self.optimizer.step()  # update the parameters
                    loss_value = loss.item()

                    loss_value = loss.item()
                    self.training_loss.append(loss_value)

                _save_metric()
                yield

        self.img_indexes = []
        self.first_25_m = []
        for idx, (i, m) in enumerate(self.test_DL):
            self.img_indexes.append(i.cpu().detach().numpy()[0, 0, :])
            self.first_25_m.append(m.cpu().detach().numpy()[0, 0, :])
            if len(self.img_indexes) == 25:
                break

        self.show_validation(init_=True)
        worker = train()
        self.stop_button.clicked.connect(worker.quit)
        worker.start()

    def load_directory(self):
        filename = QFileDialog.getExistingDirectory(
            caption="Open File",
            directory=os.getcwd(),
        )

        self.out_ = filename

        self.output.setText(
            f"...{self.out_[-17:]}/{self.cnn_type.currentText()}_checkpoints/"
        )

        self.directory.setText(filename[-30:])

        self.dir = filename
        self.output_folder = f"{filename}/{self.cnn_type.currentText()}_checkpoints/"

    def load_output(self):
        filename = QFileDialog.getExistingDirectory(
            caption="Open File",
            directory=os.getcwd(),
        )

        self.output.setText(
            f"...{filename[-17:]}/{self.cnn_type.currentText()}_checkpoints/"
        )
        self.output_folder = f"{filename}/{self.cnn_type.currentText()}_checkpoints/"

    def update_checkpoint_dir(self):
        filename, _ = QFileDialog.getOpenFileName(
            caption="Open File",
            directory=os.getcwd(),
        )
        self.checkpoint.setText(filename[-30:])
        self.checkpoint_dir = filename

    def show_validation(self, init_=False):
        """
        Receive output from the validation loop prediction and display it.

        Returns:
            napari.Viewer.addImage
        """
        if init_:
            self.training_plot.show()
        else:
            try:
                self.viewer.layers.remove("GT")
            except Exception as e:
                pass

        self.training_plot.update_plot(
            self.training_loss, self.validation_loss, self.scores
        )

        img, predictions = build_gird_with_masks(
            self.img_indexes, self.first_25_m, int(self.patch_size.currentText())
        )
        create_image_layer(
            self.viewer,
            img,
            "Validation_dataset_sample",
            transparency=False,
            visibility=True,
            range_=(-1, 1),
        )

        if predictions is not None:
            create_image_layer(
                self.viewer,
                predictions,
                "GT" if init_ else "Validation_prediction",
                transparency=True,
                visibility=True,
                range_=(0, 1),
            )

    def show_command(self):
        ps = (
            ""
            if self.patch_size.currentText() == "64"
            else f"-ps {int(self.patch_size.currentText())} "
        )

        px = (
            ""
            if self.pixel_size.text() == "None"
            else f"-px {float(self.pixel_size.text())} "
        )

        ms = (
            ""
            if not int(self.mask_size.text()) == 150
            else f"-ms {int(self.mask_size.text())} "
        )

        cnn = (
            ""
            if self.cnn_type.currentText() == "unet"
            else f"-cnn {self.cnn_type.currentText()} "
        )

        co = (
            ""
            if int(self.cnn_out_channel.text()) == 1
            else f"-co {int(self.cnn_out_channel.text())} "
        )

        b = (
            ""
            if int(self.batch_size.text()) == 25
            else f"-b {int(self.batch_size.text())} "
        )

        cl = (
            ""
            if int(self.cnn_layers.text()) == 5
            else f"-cl {int(self.cnn_layers.text())} "
        )

        cm = (
            ""
            if int(self.cnn_scaler.currentText()) == 32
            else f"-cm {int(self.cnn_scaler.currentText())} "
        )

        if self.image_type.currentText() == "2D":
            cs = (
                ""
                if self.cnn_structure.text() == "gcl"
                else f"-cs 2{self.cnn_structure.text()} "
            )
        else:
            cs = (
                ""
                if self.cnn_structure.text() == "gcl"
                else f"-cs 3{self.cnn_structure.text()} "
            )

        ck = (
            ""
            if int(self.cnn_kernel.text()) == 3
            else f"-ck {int(self.cnn_kernel.text())} "
        )

        cp = (
            ""
            if int(self.cnn_padding.text()) == 1
            else f"-cp {int(self.cnn_padding.text())} "
        )

        cmpk = (
            ""
            if int(self.cnn_max_pool.text()) == 2
            else f"-cmpk {int(self.cnn_max_pool.text())} "
        )

        l = (
            ""
            if self.loss_function.currentText() == "BCELoss"
            else f"-l {self.loss_function.currentText()} "
        )

        lr = (
            ""
            if float(self.learning_rate.text()) == 0.0005
            else f"-lr {float(self.learning_rate.text())} "
        )

        e = "" if int(self.epoch.text()) == 10000 else f"-e {int(self.epoch.text())} "

        es = (
            ""
            if int(self.early_stop.text()) == 10000
            else f"-es {int(self.early_stop.text())} "
        )

        dp = (
            ""
            if float(self.dropout_rate.text()) == 0.5
            else f"-dp {float(self.dropout_rate.text())} "
        )

        cch = "" if self.checkpoint_dir is None else f"-cch {self.checkpoint_dir} "

        show_info(
            f"tardis_cnn_train "
            f"-dir {self.out_} "
            f"{ps}"
            f"{px}"
            f"{ms}"
            f"{cnn}"
            f"{co}"
            f"{b}"
            f"{cl}"
            f"{ck}"
            f"{cm}"
            f"{cs}"
            f"{cp}"
            f"{cmpk}"
            f"{l}"
            f"{lr}"
            f"{e}"
            f"{es}"
            f"{dp}"
            f"{cch}"
            f"-dv {self.device.currentText()}"
        )
