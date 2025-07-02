#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2024                                                   #
#######################################################################
from typing import Union

import numpy as np
from PyQt5.QtCore import QPropertyAnimation
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QWidget,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QFrame,
)
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


def border_style(color="blue", direction="bottom", border=3, padding=4):
    assert direction in ["top", "bottom", "right", "left"]
    style = (
        f"border-{direction}: {border}px solid {color};"
        f"padding-{direction}: {padding}px;"
        "background-color:none;"
    )

    return style


class PlotPopup(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Napari Tardis-em training progress")
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        plt.style.use(["dark_background"])

        self.figure = plt.figure(figsize=(18, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.setLayout(layout)
        self.canvas.draw()

    def update_plot(self, y_values1, y_values2, y_values3):
        if len(y_values1) > 0:
            y_values1 = [round(value, 6) for value in y_values1]
            y_values2 = [round(value, 6) for value in y_values2]
            y_values3 = [round(value, 3) for value in y_values3]

            x_values1 = np.arange(len(y_values1))
            x_values2 = np.arange(len(y_values2))
            x_values3 = np.arange(len(y_values3))

            self.figure.clear()

            # Create two subplots side by side
            ax1 = self.figure.add_subplot(131)  # 1 row, 3 columns, 1st subplot
            ax1.plot(x_values1, y_values1, "-b", label="Training Loss")
            ax1.set_xlabel("Training step")
            ax1.set_ylabel("Loss Score")
            ax1.set_title("Training loss")

            ax2 = self.figure.add_subplot(132)  # 1 row, 3 columns, 2nd subplot
            ax2.plot(x_values2, y_values2, "-b", label="Validation Loss")
            ax2.set_xlabel("Validation step")
            ax2.set_ylabel("Loss Score")
            ax2.set_title("Validation loss")

            ax3 = self.figure.add_subplot(133)  # 1 row, 3 columns, 2nd subplot
            ax3.plot(x_values3, y_values3, "-r", label="Validation F1 score")
            ax3.set_xlabel("Validation step")
            ax3.set_ylabel("F1 Score")
            ax3.set_title("Validation F1 score")

            self.canvas.draw()


def build_gird_with_masks(
    imgs: list[np.ndarray],
    predictions: Union[list[np.ndarray], None],
    box_size: int,
):
    patch_size = box_size // 2
    crop_size = box_size
    gap_size = 5
    grid = 5

    if imgs[0].ndim == 3:
        crop_grid_img = np.zeros(
            (crop_size, crop_size, 5 * crop_size + (5 - 1) * gap_size),
            dtype=imgs[0].dtype,
        )

        if predictions is not None:
            crop_grid_scores = np.zeros(
                (crop_size, crop_size, 5 * crop_size + (5 - 1) * gap_size),
                dtype=predictions[0].dtype,
            )
        else:
            crop_grid_scores = None
    else:
        crop_grid_img = np.zeros(
            (5 * crop_size + (5 - 1) * gap_size, 5 * crop_size + (5 - 1) * gap_size),
            dtype=imgs[0].dtype,
        )

        if predictions is not None:
            crop_grid_scores = np.zeros(
                (
                    5 * crop_size + (5 - 1) * gap_size,
                    5 * crop_size + (5 - 1) * gap_size,
                ),
                dtype=predictions[0].dtype,
            )
        else:
            crop_grid_scores = None

    # Build and display particle grid
    x_min = 0
    y_min = 0
    if imgs[0].ndim == 3:
        if crop_grid_scores is not None:
            for idx, (i, j) in enumerate(zip(imgs, predictions)):
                # Add crops
                crop_grid_img[
                    0:crop_size, y_min : y_min + crop_size, x_min : x_min + crop_size
                ] = i
                crop_grid_scores[
                    0:crop_size, y_min : y_min + crop_size, x_min : x_min + crop_size
                ] = j

                if (idx + 1) % grid == 0 and x_min != 0:
                    x_min = 0
                    y_min += crop_size + gap_size
                else:
                    x_min += crop_size + gap_size
        else:
            for idx, i in enumerate(imgs):
                # Add crops
                crop_grid_img[
                    0:crop_size, y_min : y_min + crop_size, x_min : x_min + crop_size
                ] = i

                if (idx + 1) % grid == 0 and x_min != 0:
                    x_min = 0
                    y_min += crop_size + gap_size
                else:
                    x_min += crop_size + gap_size
    else:
        if crop_grid_scores is not None:
            for idx, (i, j) in enumerate(zip(imgs, predictions)):
                # Add crops
                crop_grid_img[y_min : y_min + crop_size, x_min : x_min + crop_size] = i
                crop_grid_scores[
                    y_min : y_min + crop_size, x_min : x_min + crop_size
                ] = j

                if (idx + 1) % grid == 0 and x_min != 0:
                    x_min = 0
                    y_min += crop_size + gap_size
                else:
                    x_min += crop_size + gap_size
        else:
            for idx, i in enumerate(imgs):
                # Add crops
                crop_grid_img[y_min : y_min + crop_size, x_min : x_min + crop_size] = i

                if (idx + 1) % grid == 0 and x_min != 0:
                    x_min = 0
                    y_min += crop_size + gap_size
                else:
                    x_min += crop_size + gap_size
    return crop_grid_img, crop_grid_scores


class CollapsibleBox(QWidget):
    def __init__(self, title="", parent=None):
        super().__init__(parent)

        self.toggle_button = QPushButton(title)
        self.toggle_button.setStyleSheet("text-align: center;")
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(True)
        self.toggle_button.clicked.connect(self.on_toggle)

        self.content_area = QScrollArea()
        self.content_area.setStyleSheet("border: none;")
        self.content_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.content_area.setMaximumHeight(0)
        self.content_area.setMinimumHeight(0)

        self.content_frame = QFrame()
        self.content_layout = QVBoxLayout()
        self.content_frame.setLayout(self.content_layout)

        self.content_area.setWidget(self.content_frame)
        self.content_area.setWidgetResizable(True)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.addWidget(self.toggle_button)
        lay.addWidget(self.content_area)

        self.toggle_animation = QPropertyAnimation(self.content_area, b"maximumHeight")
        self.toggle_animation.setDuration(200)

    def on_toggle(self, checked):
        content_height = self.content_frame.sizeHint().height()

        if not checked:
            self.toggle_animation.setStartValue(0)
            self.toggle_animation.setEndValue(content_height)
        else:
            self.toggle_animation.setStartValue(content_height)
            self.toggle_animation.setEndValue(0)

        self.toggle_animation.start()

    def setContentLayout(self, layout):
        # Clear existing widgets
        for i in reversed(range(self.content_layout.count())):
            item = self.content_layout.itemAt(i)
            widget = item.widget()
            if widget:
                widget.setParent(None)
            else:
                self.content_layout.removeItem(item)

        self.content_layout.addLayout(layout)
        self.on_toggle(True)
