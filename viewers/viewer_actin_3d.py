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
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
)
from napari import Viewer
from napari.utils.progress import progress
from napari.qt.threading import thread_worker

from qtpy.QtWidgets import QWidget
from PyQt5.QtCore import Qt

from tardis_em.utils.predictor import GeneralPredictor
from tardis_em.utils.load_data import load_image
from tardis_em.cnn.data_processing.trim import trim_with_stride
from tardis_em.utils.aws import get_all_version_aws
from tardis_em.cnn.datasets.dataloader import PredictionDataset
from tardis_em.cnn.data_processing.scaling import scale_image
from tardis_em.utils.normalization import adaptive_threshold

from napari.utils.notifications import show_info, show_error
from napari_tardis_em.utils.styles import border_style
from napari_tardis_em.utils.utils import get_list_of_device
from napari_tardis_em.viewers.utils import create_image_layer, update_viewer_prediction


class TardisWidget(QWidget):
    """
    Easy to use plugin for general Actin prediction.

    Plugin integrate TARDIS-em and allow to easily set up training. To make it more
    user-friendly, this plugin guid user what to do, and during training display
     results from validation loop.
    """

    def __init__(self, viewer_actin_3d: Viewer):
        super().__init__()
        self.viewer = viewer_actin_3d
