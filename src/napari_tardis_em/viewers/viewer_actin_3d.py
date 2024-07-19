#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2024                                                   #
#######################################################################

from napari import Viewer

from qtpy.QtWidgets import QWidget


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
