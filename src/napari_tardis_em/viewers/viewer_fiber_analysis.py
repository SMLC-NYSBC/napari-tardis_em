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
import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QDialogButtonBox
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QFormLayout,
    QComboBox,
    QLabel,
    QFileDialog,
    QLineEdit,
)
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from napari import Viewer
from napari.utils.notifications import show_info
from qtpy.QtWidgets import QWidget
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
from superqt import QRangeSlider

from napari_tardis_em.viewers.styles import border_style, CollapsibleBox
from napari_tardis_em.viewers.utils import create_point_layer
from tardis_em.analysis.mt_classification.utils import assign_filaments_to_n_poles
from tardis_em.analysis.filament_utils import (
    reorder_segments_id,
    sort_by_length,
    sort_segments,
    resample_filament,
)
from tardis_em.analysis.geometry_metrics import (
    length_list,
    curvature_list,
    tortuosity_list,
    group_points_by_distance,
)
from tardis_em.utils.export_data import NumpyToAmira


class TardisWidget(QWidget):
    """
    Easy to use plugin for general Microtubule prediction.

    Plugin integrate TARDIS-em and allow to easily set up training. To make it more
    user-friendly, this plugin guid user what to do, and during training display
     results from validation loop.
    """

    def __init__(self, viewer_fiber_edit: Viewer):
        super().__init__()
        self.plot_universal = PlotPopup()

        self.predicted = False
        self.viewer = viewer_fiber_edit

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
        ##############################
        # Setting user should change #
        ##############################
        label_2 = QLabel("                                      ")
        label_2.setStyleSheet(border_style("green"))

        self.pixel_size_bt = QLineEdit("1.0")
        self.pixel_size_bt.setToolTip("Select correct pixel size for analysis")
        self.pixel_size_bt.setMaximumWidth(60)

        self.resample_bt = QPushButton("Resample")
        self.resample_bt.clicked.connect(self.resample)

        self.pixel_resize_box = QHBoxLayout()
        self.pixel_resize_box.addWidget(self.pixel_size_bt)
        self.pixel_resize_box.addWidget(self.resample_bt)

        self.edite_mode_bt_1 = QPushButton("Points")
        self.edite_mode_bt_1.clicked.connect(self.point_layer)

        self.edite_mode_bt_2 = QPushButton("Filament")
        self.edite_mode_bt_2.clicked.connect(self.track_layer)

        self.edite_mode_box = QHBoxLayout()
        self.edite_mode_box.addWidget(self.edite_mode_bt_1)
        self.edite_mode_box.addWidget(self.edite_mode_bt_2)

        self.cluster_ends_th = QLineEdit("auto")
        self.cluster_ends_th.setMaximumWidth(60)
        self.cluster_ends_1_bt = QPushButton("Cluster fibers")
        self.cluster_ends_1_bt.clicked.connect(self.cluster_ends)
        self.cluster_ends_box = QHBoxLayout()
        self.cluster_ends_box.addWidget(self.cluster_ends_th)
        self.cluster_ends_box.addWidget(self.cluster_ends_1_bt)

        label_anal = QLabel("Analysis                 ")
        label_anal.setStyleSheet(border_style("blue"))

        self.add_centrosome_1_bt = QPushButton("Manually")
        self.add_centrosome_1_bt.clicked.connect(self.add_new_point)

        self.add_centrosome_auto_select = QComboBox()
        self.add_centrosome_auto_select.addItems(["1", "2", "3", "4"])
        self.add_centrosome_auto_select.setEditable(True)
        self.add_centrosome_auto_select.setCurrentIndex(1)
        self.add_centrosome_auto_bt = QPushButton("Auto")
        self.add_centrosome_auto_bt.clicked.connect(self.add_new_point_auto)
        self.add_centrosome_auto_box = QHBoxLayout()
        self.add_centrosome_auto_box.addWidget(self.add_centrosome_auto_select)
        self.add_centrosome_auto_box.addWidget(self.add_centrosome_auto_bt)
        self.add_centrosome_auto_box.addWidget(self.add_centrosome_1_bt)

        self.save_bt = QPushButton("Save as Amira/CSV")
        self.save_bt.clicked.connect(self.save_edited_instances)

        self.hist_bins_bt = QComboBox()
        self.hist_bins_bt.addItems(
            [
                "auto",
                "fd",
                "doane",
                "scott",
                "stone",
                "rice",
                "sturges",
                "sqrt",
                "knuth",
                "blocks",
            ]
        )
        self.hist_bins_bt.setEditable(True)
        self.hist_bins_bt.setCurrentIndex(5)

        """
        Filter
        """
        """Filter by Distance"""

        # Filter by distance to track
        # Filter by distance to a selected point

        # Filter length
        self.filter_length = QRangeSlider(Qt.Horizontal)
        self.filter_length.setRange(0, 100)
        self.filter_length.setValue((0, 100))
        self.filter_length.sliderReleased.connect(self.filter_value_changed)

        self.filter_length_min_label = QLineEdit("0.0")
        self.filter_length_min_label.setMaximumWidth(40)
        self.filter_length_min_label.editingFinished.connect(self.filter_value_changed)
        self.filter_length_max_label = QLineEdit("100.0")
        self.filter_length_max_label.setMaximumWidth(40)
        self.filter_length_max_label.editingFinished.connect(self.filter_value_changed)

        self.filter_length_box = QHBoxLayout()
        self.filter_length_box.addWidget(self.filter_length_min_label)
        self.filter_length_box.addWidget(self.filter_length)
        self.filter_length_box.addWidget(self.filter_length_max_label)

        # Filter by Curv
        self.filter_curv = QRangeSlider(Qt.Horizontal)
        self.filter_curv.setRange(0, 100)
        self.filter_curv.setValue((0, 100))
        self.filter_curv.sliderReleased.connect(self.filter_value_changed)

        self.filter_curv_min_label = QLineEdit("0.0")
        self.filter_curv_min_label.setMaximumWidth(40)
        self.filter_curv_min_label.editingFinished.connect(self.filter_value_changed)
        self.filter_curv_max_label = QLineEdit("100.0")
        self.filter_curv_max_label.setMaximumWidth(40)
        self.filter_curv_max_label.editingFinished.connect(self.filter_value_changed)

        self.filter_curv_box = QHBoxLayout()
        self.filter_curv_box.addWidget(self.filter_curv_min_label)
        self.filter_curv_box.addWidget(self.filter_curv)
        self.filter_curv_box.addWidget(self.filter_curv_max_label)

        # Filter by Tortuosity
        self.filter_tort = QRangeSlider(Qt.Horizontal)
        self.filter_tort.setRange(0, 100)
        self.filter_tort.setValue((0, 100))
        self.filter_tort.sliderReleased.connect(self.filter_value_changed)

        self.filter_tort_min_label = QLineEdit("0.0")
        self.filter_tort_min_label.setMaximumWidth(40)
        self.filter_tort_min_label.editingFinished.connect(self.filter_value_changed)
        self.filter_tort_max_label = QLineEdit("100.0")
        self.filter_tort_max_label.setMaximumWidth(40)
        self.filter_tort_min_label.editingFinished.connect(self.filter_value_changed)

        self.filter_tort_box = QHBoxLayout()
        self.filter_tort_box.addWidget(self.filter_tort_min_label)
        self.filter_tort_box.addWidget(self.filter_tort)
        self.filter_tort_box.addWidget(self.filter_tort_max_label)

        # Filter by End Interaction dist
        self.filter_end_inter_dist = QRangeSlider(Qt.Horizontal)
        self.filter_end_inter_dist.setRange(0, 100)
        self.filter_end_inter_dist.setValue((0, 100))
        self.filter_end_inter_dist.sliderReleased.connect(self.filter_value_changed)

        self.filter_end_inter_dist_min_label = QLineEdit("0.0")
        self.filter_end_inter_dist_min_label.setMaximumWidth(40)
        self.filter_end_inter_dist_min_label.editingFinished.connect(
            self.filter_value_changed
        )
        self.filter_end_inter_dist_max_label = QLineEdit("100.0")
        self.filter_end_inter_dist_max_label.setMaximumWidth(40)
        self.filter_end_inter_dist_max_label.editingFinished.connect(
            self.filter_value_changed
        )

        self.filter_end_inter_dist_box = QHBoxLayout()
        self.filter_end_inter_dist_box.addWidget(self.filter_end_inter_dist_min_label)
        self.filter_end_inter_dist_box.addWidget(self.filter_end_inter_dist)
        self.filter_end_inter_dist_box.addWidget(self.filter_end_inter_dist_max_label)

        # Filter by End Interaction angle
        self.filter_end_inter_angle = QRangeSlider(Qt.Horizontal)
        self.filter_end_inter_angle.setRange(0, 100)
        self.filter_end_inter_angle.setValue((0, 100))
        self.filter_end_inter_angle.sliderReleased.connect(self.filter_value_changed)

        self.filter_end_inter_angle_min_label = QLineEdit("0.0")
        self.filter_end_inter_angle_min_label.setMaximumWidth(40)
        self.filter_end_inter_angle_min_label.editingFinished.connect(
            self.filter_value_changed
        )
        self.filter_end_inter_angle_max_label = QLineEdit("100.0")
        self.filter_end_inter_angle_max_label.setMaximumWidth(40)
        self.filter_end_inter_angle_max_label.editingFinished.connect(
            self.filter_value_changed
        )

        self.filter_end_inter_angle_box = QHBoxLayout()
        self.filter_end_inter_angle_box.addWidget(self.filter_end_inter_angle_min_label)
        self.filter_end_inter_angle_box.addWidget(self.filter_end_inter_angle)
        self.filter_end_inter_angle_box.addWidget(self.filter_end_inter_angle_max_label)

        self.last_selected_obj = QLineEdit("None")
        self.last_selected_obj.setReadOnly(False)
        self.last_selected_obj.setMaximumWidth(60)

        self.filter_to_selected_point_dist = QLineEdit("0")
        self.filter_to_selected_point_dist.setMaximumWidth(60)

        self.filter_to_selected_point_nearest = QPushButton("Ends")
        self.filter_to_selected_point_nearest.clicked.connect(
            self.filter_to_selected_point_nearest_
        )
        self.filter_to_selected_obj = QPushButton("Filament")
        # self.filter_to_selected_obj.clicked.connect(self.filter_to_selected_obj_)

        self.filter_to_selected_point_box = QHBoxLayout()
        self.filter_to_selected_point_box.addWidget(self.filter_to_selected_point_dist)
        self.filter_to_selected_point_box.addWidget(
            self.filter_to_selected_point_nearest
        )
        self.filter_to_selected_point_box.addWidget(self.filter_to_selected_obj)

        """
        Analysis
        """
        # Lenght
        self.length_compute = QPushButton("Compute")
        self.length_compute.clicked.connect(self.calc_length)
        self.length_plot = QPushButton("Plot")
        self.length_plot.clicked.connect(self.plot_length)
        self.length_save = QPushButton("Save")
        self.length_save.clicked.connect(self.save_length)

        self.lenght_box = QHBoxLayout()
        self.lenght_box.addWidget(self.length_compute)
        self.lenght_box.addWidget(self.length_plot)
        self.lenght_box.addWidget(self.length_save)

        # End distance
        self.end_dist_compute = QPushButton("Compute")
        self.end_dist_compute.clicked.connect(self.calc_end_dist)
        self.end_dist_plot = QPushButton("Plot")
        self.end_dist_plot.clicked.connect(self.plot_end_dist)
        self.end_dist_save = QPushButton("Save")
        self.end_dist_save.clicked.connect(self.save_end_dist)

        self.end_dist_box = QHBoxLayout()
        self.end_dist_box.addWidget(self.end_dist_compute)
        self.end_dist_box.addWidget(self.end_dist_plot)
        self.end_dist_box.addWidget(self.end_dist_save)

        # End interactions
        self.inter_ends_compute = QPushButton("Compute")
        self.inter_ends_compute.clicked.connect(self.calc_inter_ends)
        self.inter_ends_plot = QPushButton("Plot")
        self.inter_ends_plot.clicked.connect(self.plot_inter_ends)
        self.inter_ends_save = QPushButton("Save")
        self.inter_ends_save.clicked.connect(self.save_inter_ends)

        self.interaction_end_box = QHBoxLayout()
        self.interaction_end_box.addWidget(self.inter_ends_compute)
        self.interaction_end_box.addWidget(self.inter_ends_plot)
        self.interaction_end_box.addWidget(self.inter_ends_save)

        self.interaction_end_dict = {}

        # Filament interactions
        self.inter_filament_compute = QPushButton("Compute")
        self.inter_filament_compute.clicked.connect(self.calc_inter_filament)
        self.inter_filament_plot = QPushButton("Plot")
        self.inter_filament_plot.clicked.connect(self.plot_inter_filament)
        self.inter_filament_save = QPushButton("Save")
        self.inter_filament_save.clicked.connect(self.save_inter_filament)

        self.interaction_filament_box = QHBoxLayout()
        self.interaction_filament_box.addWidget(self.inter_filament_compute)
        self.interaction_filament_box.addWidget(self.inter_filament_plot)
        self.interaction_filament_box.addWidget(self.inter_filament_save)

        # Curvature
        self.curv_compute = QPushButton("Compute")
        self.curv_compute.clicked.connect(self.calc_curv)
        self.curv_plot = QPushButton("Plot")
        self.curv_plot.clicked.connect(self.plot_curv)
        self.curv_save = QPushButton("Save")
        self.curv_save.clicked.connect(self.save_curv)

        self.curv_box = QHBoxLayout()
        self.curv_box.addWidget(self.curv_compute)
        self.curv_box.addWidget(self.curv_plot)
        self.curv_box.addWidget(self.curv_save)

        # Tortuosity
        self.tortuosity_compute = QPushButton("Compute")
        self.tortuosity_compute.clicked.connect(self.calc_tortuosity)
        self.tortuosity_plot = QPushButton("Plot")
        self.tortuosity_plot.clicked.connect(self.plot_tortuosity)
        self.tortuosity_save = QPushButton("Save")
        self.tortuosity_save.clicked.connect(self.save_tortuosity)

        self.tortuosity_box = QHBoxLayout()
        self.tortuosity_box.addWidget(self.tortuosity_compute)
        self.tortuosity_box.addWidget(self.tortuosity_plot)
        self.tortuosity_box.addWidget(self.tortuosity_save)

        """
        Initialized UI
        """
        layout = QVBoxLayout()

        # ---- Pre-setting ----
        pre_box = CollapsibleBox("---- Pre-setting ----")
        pre_layout = QFormLayout()
        pre_layout.addRow("Pixel size", self.pixel_resize_box)
        pre_layout.addRow("Layer type", self.edite_mode_box)
        pre_box.setContentLayout(pre_layout)

        # ---- Parameters -----
        param_box = CollapsibleBox("---- Parameters -----")
        param_layout = QFormLayout()
        param_layout.addRow("Add point [auto]", self.add_centrosome_auto_box)
        param_layout.addRow("Cluster ends", self.cluster_ends_box)
        param_layout.addRow("Hist. Bins", self.hist_bins_bt)
        param_box.setContentLayout(param_layout)

        # ---- Filter ----
        filter_box = CollapsibleBox("---- Filter ----")
        filter_layout = QFormLayout()
        filter_layout.addRow("By Length", self.filter_length_box)
        filter_layout.addRow("By Curv.", self.filter_curv_box)
        filter_layout.addRow("By Tort.", self.filter_tort_box)
        filter_layout.addRow("Last selected", self.last_selected_obj)
        filter_layout.addRow("Dist. to obj.", self.filter_to_selected_point_box)
        filter_layout.addRow("End Inter. dist.", self.filter_end_inter_dist_box)
        filter_layout.addRow("End Inter. angle", self.filter_end_inter_angle_box)
        filter_box.setContentLayout(filter_layout)

        # ---- Analysis ----
        analysis_box = CollapsibleBox("---- Analysis ----")
        analysis_layout = QFormLayout()
        analysis_layout.addRow("Length", self.lenght_box)
        analysis_layout.addRow("End Distance", self.end_dist_box)
        analysis_layout.addRow("Inter. Lattices", self.interaction_filament_box)
        analysis_layout.addRow("Inter. Ends", self.interaction_end_box)
        analysis_layout.addRow("Curvature", self.curv_box)
        analysis_layout.addRow("Tortuosity", self.tortuosity_box)
        analysis_box.setContentLayout(analysis_layout)

        # Add collapsible sections to the main layout
        layout.addWidget(pre_box)
        layout.addWidget(param_box)
        layout.addWidget(filter_box)
        layout.addWidget(analysis_box)
        layout.addWidget(self.save_bt)
        self.setLayout(layout)

        self.viewer.mouse_double_click_callbacks.pop(0)
        self.viewer.mouse_double_click_callbacks.append(self._on_double_click)

    def _on_double_click(self, layer, event):
        mouse_position = np.asarray(event.position).reshape(1, -1)

        if not self.viewer.layers.selection.active:
            return  # No active layer selected

        data, name = self.get_selected_data(name=True)
        if not len(data) > 0:
            return

        kdtree = KDTree(data[:, 1:])
        try:
            distance, closest_point = kdtree.query(mouse_position, k=1)
        except ValueError:
            distance, closest_point = kdtree.query(mouse_position[:, 1:], k=1)

        data_index = data[closest_point[0][0]]
        self.last_selected_obj.setText(f"{name[:10]}; {data_index[0]}")

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

    def get_data_by_name(self, name, type_=False, properties=False):
        data = self.viewer.layers[name].data

        # Convert IDxTimexZxYxX to IDxXxYxZ
        if data.shape[-1] == 5:
            data = np.array((data[:, 0], data[:, 4], data[:, 3], data[:, 2])).T
            type_layer = "tracks"
        else:
            try:
                ids = self.viewer.layers[name].properties["ID"]
                data = np.array((ids, data[:, 2], data[:, 1], data[:, 0])).T
            except KeyError:
                data = np.zeros((0, 4))

            type_layer = "points"
        properties_v = self.viewer.layers[name].properties

        if type_:
            if properties:
                return data, type_layer, properties_v
            return data, type_layer
        if properties:
            return data, properties_v
        return data

    def get_selected_data(self, name=False, type_=False, properties=False):
        active_layer = self.viewer.layers.selection.active.name
        data = self.viewer.layers[active_layer].data

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
        properties_v = self.viewer.layers[active_layer].properties

        if name:
            if type_:
                if properties:
                    return data, active_layer, type_layer, properties_v
                return data, active_layer, type_layer
            if properties:
                return data, active_layer, properties_v
            return data, active_layer

        if type_:
            if properties:
                return data, type_layer, properties_v
            return data, type_layer
        return data

    def save_edited_instances(self):
        selected = self.viewer.layers.selection
        if len(selected) == 0:
            return
        elif len(selected) == 1:
            data, name, properties = self.get_selected_data(name=True, properties=True)
        else:
            data, name, properties = [], [], []
            for i in selected:
                d, p = self.get_data_by_name(i.name, properties=True)
                data.append(d)
                name.append(i.name)
                properties.append(p)

        ids, first_indices = np.unique(data[:, 0], return_index=True)
        properties = {
            k: v[first_indices] for k, v in properties.items() if k != "track_id"
        }

        if any(key.startswith("Label_") for key in properties):
            labels = [
                [k for k, v in properties.items() if k.startswith("Label_")],
                [v for k, v in properties.items() if k.startswith("Label_")],
            ]
            labels = group_indices_by_value(labels)
        else:
            labels = None

        x = pd.DataFrame(np.array(list(properties.values())).T)
        header = list(properties.keys())

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        f_dir, f_format = QFileDialog.getSaveFileName(
            caption="Save Files",
            directory=join(getcwd()),
            filter="Amira Files (*.am);;CSV File (*.csv)",
            options=options,
        )

        if f_dir == "":
            return

        if not f_dir.endswith((".csv", ".am")):
            if f_format == "CSV File (*.csv)":
                f_dir = f_dir + ".csv"
            else:
                f_dir = f_dir + ".am"

        if f_dir.endswith(".csv"):
            self._save_csv(x, header, f_dir=f_dir)
        elif f_dir.endswith(".am"):
            amira = NumpyToAmira()
            amira.export_amiraV2(
                file_dir=f_dir,
                coords=data,
                labels=labels,
                scores=[
                    [
                        k
                        for k, v in properties.items()
                        if k != "ID" and not k.startswith("Label_")
                    ],
                    [
                        v
                        for k, v in properties.items()
                        if k != "ID" and not k.startswith("Label_")
                    ],
                ],
            )

    def resample(self):
        data, name, type_ = self.get_selected_data(name=True, type_=True)

        # px = np.ceil(25 / float(self.pixel_size_bt.text()))
        # data = sort_segments(data)
        data = resample_filament(data, "auto")

        create_point_layer(
            viewer=self.viewer,
            points=data,
            name=name,
            visibility=True,
            as_filament=True if type_ == "tracks" else False,
        )

    def add_new_point_auto(self):
        data, name = self.get_selected_data(name=True)
        k = int(self.add_centrosome_auto_select.currentText())

        unique_ids, first_indices, count = np.unique(
            data[:, 0], return_index=True, return_counts=True
        )
        last_indices = first_indices + count - 1

        data = np.vstack([data[first_indices, 1:], data[last_indices, 1:]])

        kmeans = KMeans(n_clusters=k, n_init=100)
        kmeans.fit(data)
        centers = kmeans.cluster_centers_

        centers = np.hstack([np.arange(len(centers))[:, None], centers])

        create_point_layer(
            viewer=self.viewer,
            points=centers,
            name="Centers",
            visibility=True,
            as_filament=False,
            size_=100 / float(self.pixel_size_bt.text()),
        )

    def add_new_point(self):
        name = "Centers"
        xyz = show_coordinate_dialog()

        try:
            data = self.viewer.layers[name].data
            data = np.array(
                (
                    self.viewer.layers[name].properties["ID"],
                    data[:, 0],
                    data[:, 1],
                    data[:, 2],
                )
            ).T
            ids = np.max(data[:, 0]).item() + 1
        except KeyError:
            data = np.zeros((0, 4))
            ids = 0

        try:
            data = np.vstack([data, xyz])
        except:
            data = np.vstack([data, np.zeros((1, 4))])
        data[-1, 0] = ids

        create_point_layer(
            viewer=self.viewer,
            points=data,
            name=name,
            visibility=True,
            as_filament=False,
            size_=100 / float(self.pixel_size_bt.text()),
        )

    def cluster_ends(self):
        th = self.cluster_ends_th.text()
        if th != "auto":
            th = float(th)

        data, name, properties = self.get_selected_data(name=True, properties=True)

        layers = [layer.name for layer in self.viewer.layers]
        types_ = [layer.__class__.__name__ for layer in self.viewer.layers]

        last_obj = self.last_selected_obj.text().split(";")

        if last_obj[0] == "None":
            try:
                idx = types_.index("Points")
            except ValueError:
                return

            layer = [self.viewer.layers[idx].name]
        else:
            layer = [n for n in layers if n.startswith(last_obj[0])]

        poles = self.get_data_by_name(layer[0])

        # Label 1
        filaments = assign_filaments_to_n_poles(data, poles[:, 1:])

        filaments_label = np.zeros(len(np.unique(data[:, 0])))
        id_ = 1
        for i in filaments:
            for j in np.unique(i[:, 0]):
                filaments_label[int(j)] = id_
            id_ += 1

        unique_ids, first_indices = np.unique(data[:, 0], return_index=True)
        point_no = np.array(list(first_indices[1:]) + [len(data)]) - first_indices
        filaments_label = np.repeat(filaments_label, point_no)

        filaments_labels = {"Label_Grouped_Centers": filaments_label}

        grouped_filaments = []
        for i in filaments:
            # get last index of ID
            _, first_idx_filament = np.unique(i[:, 0], return_index=True)

            # get points with indexes and groupped and add to list
            grouped_filaments.extend(
                group_points_by_distance(i[first_idx_filament, :], eps=th)
            )

        grouped_label = np.zeros(len(np.unique(data[:, 0])))
        id_ = 1
        for i in grouped_filaments:
            for j in np.unique(i):
                grouped_label[int(j)] = id_
            id_ += 1

        grouped_label = np.repeat(grouped_label, point_no)
        filaments_labels["Label_Grouped_Ends"] = grouped_label

        create_point_layer(
            viewer=self.viewer,
            points=data,
            name=name,
            add_properties=filaments_labels,
            visibility=True,
            as_filament=True,
            color_="viridis",
        )

    def filter_value_changed(self):
        sender = self.sender()
        filter_ = False

        try:
            data, name, properties = self.get_selected_data(name=True, properties=True)
        except:
            return

        if len(properties) == 0 or properties is None:
            return

        # Filter by % given as value = min_val + x * (max_val - min_val)
        filter_ = False
        if "Length" in properties:
            filter_ = True
            length = properties["Length"]
            _min = min(length)
            _max = max(length)

            if sender in [self.filter_length_min_label, self.filter_length_max_label]:
                filter_length_min, filter_length_max = float(
                    self.filter_length_min_label.text()
                ), float(self.filter_length_max_label.text())
                filter_length_bool = np.logical_and(
                    filter_length_min <= length, length <= filter_length_max
                )

                filter_length_value_min = int(
                    (filter_length_min - _min) / (_max - _min) * 100
                )
                filter_length_value_max = int(
                    (filter_length_max - _min) / (_max - _min) * 100
                )
                self.filter_length.setValue(
                    (filter_length_value_min, filter_length_value_max)
                )
            else:
                filter_length_min, filter_length_max = self.filter_length.value()
                filter_length_min /= 100
                filter_length_max /= 100

                filter_length_value_min = _min + (filter_length_min * (_max - _min))
                filter_length_value_max = _min + (filter_length_max * (_max - _min))
                self.filter_length_min_label.setText(
                    f"{float(filter_length_value_min):.2f}"
                )
                self.filter_length_max_label.setText(
                    f"{float(filter_length_value_max):.2f}"
                )

                filter_length_bool = np.logical_and(
                    filter_length_value_min <= length, length <= filter_length_value_max
                )

        else:
            filter_length_bool = np.ones(len(data), dtype=bool)

        if "Curvature" in properties:
            filter_ = True
            curvature = properties["Curvature"]
            _min = min(curvature)
            _max = max(curvature)

            if sender in [self.filter_curv_min_label, self.filter_curv_max_label]:
                filter_curv_min, filter_curv_max = float(
                    self.filter_curv_min_label.text()
                ), float(self.filter_curv_max_label.text())
                filter_curve_bool = np.logical_and(
                    filter_curv_min <= curvature, curvature <= filter_curv_max
                )

                filter_curv_min_value = int(
                    (filter_curv_min - _min) / (_max - _min) * 100
                )
                filter_curv_max_value = int(
                    (filter_curv_max - _min) / (_max - _min) * 100
                )
                self.filter_curv.setValue(
                    (filter_curv_min_value, filter_curv_max_value)
                )
            else:
                filter_curv_min, filter_curv_max = self.filter_curv.value()
                filter_curv_min /= 100
                filter_curv_max /= 100

                filter_curv_value_min = _min + (filter_curv_min * (_max - _min))
                filter_curv_value_max = _min + (filter_curv_max * (_max - _min))
                self.filter_curv_min_label.setText(
                    f"{float(filter_curv_value_min):.4f}"
                )
                self.filter_curv_max_label.setText(
                    f"{float(filter_curv_value_max):.4f}"
                )

                filter_curve_bool = np.logical_and(
                    filter_curv_value_min <= curvature,
                    curvature <= filter_curv_value_max,
                )
        else:
            filter_curve_bool = np.ones(len(data), dtype=bool)

        if "Tortuosity" in properties:
            filter_ = True
            tortuosity = properties["Tortuosity"]
            _min = min(tortuosity)
            _max = max(tortuosity)

            if sender in [self.filter_tort_min_label, self.filter_tort_max_label]:
                filter_tort_min, filter_tort_max = float(
                    self.filter_tort_min_label.text()
                ), float(self.filter_tort_max_label.text())
                filter_tort_bool = np.logical_and(
                    filter_tort_min <= tortuosity, tortuosity <= filter_tort_max
                )

                filter_tort_min_value = int(
                    (filter_tort_min - _min) / (_max - _min) * 100
                )
                filter_tort_max_value = int(
                    (filter_tort_max - _min) / (_max - _min) * 100
                )
                self.filter_tort.setValue(
                    (filter_tort_min_value, filter_tort_max_value)
                )
            else:
                filter_tort_min, filter_tort_max = self.filter_curv.value()
                filter_tort_min /= 100
                filter_tort_max /= 100

                filter_tort_value_min = _min + (filter_tort_min * (_max - _min))
                filter_tort_value_max = _min + (filter_tort_max * (_max - _min))
                self.filter_tort_min_label.setText(
                    f"{float(filter_tort_value_min):.2f}"
                )
                self.filter_tort_max_label.setText(
                    f"{float(filter_tort_value_max):.2f}"
                )

                filter_tort_bool = np.logical_and(
                    filter_tort_value_min <= tortuosity,
                    tortuosity <= filter_tort_value_max,
                )
        else:
            filter_tort_bool = np.ones(len(data), dtype=bool)

        if "Branching_Distance" in properties:
            filter_ = True
            end_inter_dist = properties["Branching_Distance"]
            _min = min(end_inter_dist)
            _max = max(end_inter_dist)

            if sender in [
                self.filter_end_inter_dist_min_label,
                self.filter_end_inter_dist_max_label,
            ]:
                filter_dist_min, filter_dist_max = float(
                    self.filter_end_inter_dist_min_label.text()
                ), float(self.filter_end_inter_dist_max_label.text())
                filter_end_dist_inter_bool = np.logical_and(
                    filter_dist_min <= end_inter_dist, end_inter_dist <= filter_dist_max
                )

                filter_dist_min_value = int(
                    (filter_dist_min - _min) / (_max - _min) * 100
                )
                filter_dist_max_value = int(
                    (filter_dist_max - _min) / (_max - _min) * 100
                )
                self.filter_end_inter_dist.setValue(
                    (filter_dist_min_value, filter_dist_max_value)
                )
            else:
                filter_dist_min, filter_dist_max = self.filter_end_inter_dist.value()
                filter_dist_min /= 100
                filter_dist_max /= 100

                filter_dist_min_value = _min + (filter_dist_min * (_max - _min))
                filter_dist_max_value = _min + (filter_dist_max * (_max - _min))
                self.filter_end_inter_dist_min_label.setText(
                    f"{float(filter_dist_min_value):.2f}"
                )
                self.filter_end_inter_dist_max_label.setText(
                    f"{float(filter_dist_max_value):.2f}"
                )

                filter_end_dist_inter_bool = np.logical_and(
                    filter_dist_min_value <= end_inter_dist,
                    end_inter_dist <= filter_dist_max_value,
                )
        else:
            filter_end_dist_inter_bool = np.ones(len(data), dtype=bool)

        if "Branching_Angle" in properties:
            filter_ = True
            end_inter_angle = properties["Branching_Angle"]
            _min = min(end_inter_angle)
            _max = max(end_inter_angle)

            if sender in [
                self.filter_end_inter_angle_min_label,
                self.filter_end_inter_angle_max_label,
            ]:
                filter_angle_min, filter_angle_max = float(
                    self.filter_end_inter_angle_min_label.text()
                ), float(self.filter_end_inter_angle_max_label.text())
                filter_end_angle_inter_bool = np.logical_and(
                    filter_angle_min <= end_inter_angle,
                    end_inter_angle <= filter_angle_max,
                )

                filter_angle_min_value = int(
                    (filter_angle_min - _min) / (_max - _min) * 100
                )
                filter_angle_max_value = int(
                    (filter_angle_max - _min) / (_max - _min) * 100
                )
                self.filter_end_inter_angle.setValue(
                    (filter_angle_min_value, filter_angle_max_value)
                )
            else:
                filter_angle_min, filter_angle_max = self.filter_end_inter_angle.value()
                filter_angle_min /= 100
                filter_angle_max /= 100

                filter_angle_min_value = _min + (filter_angle_min * (_max - _min))
                filter_angle_max_value = _min + (filter_angle_max * (_max - _min))
                self.filter_end_inter_angle_min_label.setText(
                    f"{float(filter_angle_min_value):.2f}"
                )
                self.filter_end_inter_angle_max_label.setText(
                    f"{float(filter_angle_max_value):.2f}"
                )

                filter_end_angle_inter_bool = np.logical_and(
                    filter_angle_min_value <= end_inter_angle,
                    end_inter_angle <= filter_angle_max_value,
                )
        else:
            filter_end_angle_inter_bool = np.ones(len(data), dtype=bool)

        filter_anals_bool = np.logical_and.reduce(
            [
                filter_length_bool,
                filter_curve_bool,
                filter_tort_bool,
                filter_end_dist_inter_bool,
                filter_end_angle_inter_bool,
            ],
            axis=0,
        )

        if np.sum(filter_anals_bool) == 0:
            show_info("Filter-out all filaments, reduce your filters!")
            return

        if filter_:
            data = data[filter_anals_bool]
            for k, v in properties.items():
                properties[k] = v[filter_anals_bool]

            create_point_layer(
                viewer=self.viewer,
                points=data,
                name=name + "_filtered",
                add_properties=properties,
                visibility=True,
                as_filament=True,
                select_layer=name,
                keep_view_by=True,
            )

    def filter_to_selected_point_nearest_(self):
        self.resample()
        data, name, properties = self.get_selected_data(name=True, properties=True)

        layers = [layer.name for layer in self.viewer.layers]
        last_obj = self.last_selected_obj.text().split(";")
        dist_th = float(self.filter_to_selected_point_dist.text())

        if dist_th == 0.0:
            return

        layer = [n for n in layers if n.startswith(last_obj[0])]
        if len(layer) != 0:
            filter_to = self.get_data_by_name(layer[0])[int(float(last_obj[1]))][1:]

            # Get tracks ends
            _, first_indices, count = np.unique(
                data[:, 0], return_index=True, return_counts=True
            )
            last_indices = first_indices + count - 1

            track_ends_idx = np.hstack([first_indices, last_indices])
            track_ends = data[track_ends_idx]

            # cdist
            distances = np.linalg.norm(track_ends[:, 1:] - filter_to, axis=1)

            # find points withing a threshold
            dist_indices_th = np.where(distances <= dist_th)[0]
            dist_indices_th = np.unique(track_ends[:, 0][dist_indices_th]).astype(
                np.uint16
            )

            if len(dist_indices_th) == 0:
                return

            # Filtered layer
            filter_ = np.isin(data[:, 0], dist_indices_th)
            data = data[filter_]

            for k, v in properties.items():
                properties[k] = v[filter_]

            create_point_layer(
                viewer=self.viewer,
                points=data,
                name=name + "_filtered",
                add_properties=properties,
                visibility=True,
                as_filament=True,
                select_layer=name,
            )
        else:
            return

    @staticmethod
    def _save_csv(x: pd.DataFrame, header: list, f_dir=None):

        if f_dir is None:
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            f_dir, _ = QFileDialog.getSaveFileName(
                caption="Save Files",
                directory=join(getcwd()),
                filter="CSV File (*.csv)",
                options=options,
            )

            if not f_dir.endswith(".csv"):
                f_dir = f_dir + ".csv"

        x.to_csv(
            f_dir,
            header=header,
            index=False,
            sep=",",
        )

    def calc_length(self):
        data, name = self.get_selected_data(name=True)

        lengths = np.array(length_list(data))
        unique_ids, first_indices = np.unique(data[:, 0], return_index=True)
        point_no = np.array(list(first_indices[1:]) + [len(data)]) - first_indices
        lengths = np.repeat(lengths, point_no)

        properties = {"Length": lengths}

        create_point_layer(
            viewer=self.viewer,
            points=data,
            name=name,
            add_properties=properties,
            visibility=True,
            as_filament=True,
        )

    def plot_length(self, viewer):
        data, name, properties = self.get_selected_data(name=True, properties=True)

        try:
            _, first_indices = np.unique(data[:, 0], return_index=True)
            lengths = properties["Length"][first_indices]
        except KeyError:
            return

        print(
            self.hist_bins_bt.currentText(),
        )
        self.plot_universal.show()
        self.plot_universal.update_hist(
            lengths,
            title="Length Distribution",
            y_label="Counts",
            x_label="Length [U]",
            bins_=self.hist_bins_bt.currentText(),
        )

    def save_length(self):
        data, name, properties = self.get_selected_data(name=True, properties=True)

        ids, first_indices = np.unique(data[:, 0], return_index=True)
        lengths = properties["Length"][first_indices]

        x = pd.DataFrame(np.vstack([ids, lengths]).T)
        header = ["IDs", "Length [U]"]
        self._save_csv(x, header)

    def calc_end_dist(self):
        data, name, properties = self.get_selected_data(name=True, properties=True)

        layers = [layer.name for layer in self.viewer.layers]
        types_ = [layer.__class__.__name__ for layer in self.viewer.layers]

        last_obj = self.last_selected_obj.text().split(";")

        if last_obj[0] == "None":
            try:
                idx = types_.index("Points")
            except ValueError:
                return

            layer = [self.viewer.layers[idx].name]
        else:
            layer = [n for n in layers if n.startswith(last_obj[0])]

        if len(layer) != 0:
            filter_to = self.get_data_by_name(layer[0])

            # Get tracks ends
            unique_id, first_indices, count = np.unique(
                data[:, 0], return_index=True, return_counts=True
            )
            last_indices = first_indices + count - 1

            track_ends_idx = np.hstack([first_indices, last_indices])
            track_ends = data[track_ends_idx]

            # cdist
            distances = [
                np.linalg.norm(track_ends[:, 1:] - i[1:], axis=1) for i in filter_to
            ]

            distances = np.vstack(
                [np.vstack([track_ends[:, 0], i]).T for i in distances]
            )

            # Combine distances and track_ends select closest end idx and distance
            dist_idx = np.array(
                [np.argmin(distances[distances[:, 0] == uid, 1]) for uid in unique_id]
            )

            dist_idx = np.floor_divide(dist_idx, 2) + 1
            distances = np.array(
                [np.min(distances[distances[:, 0] == uid, 1]) for uid in unique_id]
            )

            point_no = np.array(list(first_indices[1:]) + [len(data)]) - first_indices
            distances = np.repeat(distances, point_no)
            dist_idx = np.repeat(dist_idx, point_no)

            create_point_layer(
                viewer=self.viewer,
                points=data,
                name=name,
                add_properties={
                    "nearest_end_idx": dist_idx,
                    "nearest_end_distance": distances,
                },
                visibility=True,
                as_filament=True,
            )
        else:
            return

    def plot_end_dist(self):
        data, name, properties = self.get_selected_data(name=True, properties=True)

        try:
            unique_ids, first_indices = np.unique(data[:, 0], return_index=True)
            ends = properties["nearest_end_distance"][first_indices]
        except KeyError:
            return

        self.plot_universal.show()
        self.plot_universal.update_hist(
            ends,
            title="End Distance Distribution",
            y_label="Counts",
            x_label="End Distance [U]",
            FWHM=True,
            bins_=self.hist_bins_bt.currentText(),
        )

    def save_end_dist(self):
        data, name, properties = self.get_selected_data(name=True, properties=True)

        ids, first_indices = np.unique(data[:, 0], return_index=True)
        lengths = properties["nearest_end_distance"][first_indices]
        end_idx = properties["nearest_end_idx"][first_indices]

        x = pd.DataFrame(np.vstack([ids, lengths, end_idx]).T)
        header = ["IDs", "End Distance [U]", "End Index"]
        self._save_csv(x, header)

    def calc_inter_ends(self):
        data, name = self.get_selected_data(name=True, properties=False)
        properties = {}

        layers = [layer.name for layer in self.viewer.layers]
        types_ = [layer.__class__.__name__ for layer in self.viewer.layers]

        last_obj = self.last_selected_obj.text().split(";")

        if last_obj[0] == "None":
            try:
                idx = types_.index("Points")
            except ValueError:
                return

            layer = [self.viewer.layers[idx].name]
        else:
            layer = [n for n in layers if n.startswith(last_obj[0])]

        if len(layer) == 0:
            show_info(f"No point layer found. Add points to continue.")
            return

        # Sort filaments with minus ends
        poles = self.get_data_by_name(layer[0])
        filaments = assign_filaments_to_n_poles(data, poles[:, 1:])
        filaments = np.vstack(filaments).astype(np.float32)

        _, point_no = np.unique(filaments[:, 0], return_counts=True)
        _, first_indices, count = np.unique(
            filaments[:, 0], return_index=True, return_counts=True
        )
        last_indices = first_indices + count - 1

        ends = filaments[first_indices].astype(np.float32)

        properties["Branching_Distance"] = cdist(ends[:, 1:], filaments[:, 1:])
        for i, (f, l) in enumerate(zip(first_indices, last_indices)):
            properties["Branching_Distance"][i, f:l] = np.nan

        child_id = np.nanargmin(properties["Branching_Distance"], axis=1)

        properties["Branching_Distance"] = properties["Branching_Distance"][
            np.arange(len(ends)), child_id
        ]
        properties["Branching_Parent_F_ID"] = ends[:, 0]
        properties["Branching_Child_F_ID"] = filaments[child_id, 0]
        properties["Branching_Child_P_ID"] = child_id

        angles = np.zeros(len(ends[:, 0]))
        for i, interaction in enumerate(
            zip(ends[:, 0], filaments[child_id, 0], child_id)
        ):
            parent_id, child_id, child_point_id = interaction

            # Get parent spline points (first 3 points)
            parent_mask = filaments[:, 0] == parent_id
            parent_points = filaments[parent_mask, 1:4][:3]

            # Get child spline points, handling edge cases
            child_mask = filaments[:, 0] == child_id
            child_indices = np.where(child_mask)[0]
            # child_points = filaments[child_mask, 1:4]

            # Determine the slice for child points
            if child_point_id == child_indices[0]:
                # At the start of spline, take the first 3 points
                child_points = filaments[child_point_id : child_point_id + 3, 1:4]
            elif child_point_id == child_indices[-1]:
                # At end of spline, take the last 3 points
                child_points = filaments[child_point_id - 3 : child_point_id, 1:4]
            else:
                child_points = filaments[child_point_id - 1 : child_point_id + 1, 1:4]

            # Compute vectors
            parent_vec = parent_points[-1] - parent_points[0]
            child_vec = child_points[-1] - child_points[0]

            # Compute angle
            cos_angle = np.dot(parent_vec, child_vec) / (
                np.linalg.norm(parent_vec) * np.linalg.norm(child_vec)
            )

            # Handle numerical precision
            cos_angle = np.clip(cos_angle, -1.0, 1.0)

            # Convert to degrees and ensure 0-90 range
            angle = np.degrees(np.arccos(np.abs(cos_angle)))
            angles[i] = angle
        # B = filaments[properties["Branching_Child_F_ID"].astype(np.int32), 1:]
        # C = properties["Branching_Child_F_ID"] + 2
        # if np.any(C > len(filaments)):
        #     C[C > len(filaments)] -= 4
        # C = filaments[C.astype(np.int32), 1:]

        # # A is and - B is filament - C is extension of filament
        # BA = B - filaments[first_indices + 2, 1:]
        # BC = (B - C)[None, ...]
        # dot_products = np.sum(BA * BC, axis=-1)

        properties["Branching_Angle"] = angles
        b_idx = properties["Branching_Parent_F_ID"].astype(np.int32)
        for k, v in properties.items():
            properties[k] = properties[k][b_idx]
            properties[k] = np.repeat(properties[k], point_no)

        create_point_layer(
            viewer=self.viewer,
            points=data,
            name=name,
            add_properties=properties,
            visibility=True,
            as_filament=True,
        )

    def plot_inter_ends(self):
        data, name, properties = self.get_selected_data(name=True, properties=True)

        try:
            unique_ids, first_indices = np.unique(data[:, 0], return_index=True)
            dist = properties["Branching_Distance"][first_indices]
            angle = properties["Branching_Angle"][first_indices]
        except KeyError:
            return

        self.plot_universal.show()
        self.plot_universal.update_hist_list(
            [dist, angle],
            titles=[
                "Branching End Distances Distribution",
                "Branching End Angle Distribution",
            ],
            y_label=["Counts", "Counts"],
            x_label=["Distance", "Angle"],
            bins_=self.hist_bins_bt.currentText(),
        )

    def save_inter_ends(self):
        data, name, properties = self.get_selected_data(name=True, properties=True)

        ids, first_indices = np.unique(data[:, 0], return_index=True)
        bpf_ID = properties["Branching_Parent_F_ID"][first_indices]
        bcf_ID = properties["Branching_Child_F_ID"][first_indices]
        bcp_ID = properties["Branching_Child_P_ID"][first_indices]
        dist = properties["Branching_Distance"][first_indices]
        angle = properties["Branching_Angle"][first_indices]

        x = pd.DataFrame(np.vstack([bpf_ID, bcf_ID, bcp_ID, dist, angle]).T)
        header = [
            "Branching Parent ID",
            "Branching Child ID",
            "Branching Child point ID",
            "Branching Distance [U]",
            "Branching Angle [deg]",
        ]
        self._save_csv(x, header)

    def calc_inter_filament(self):
        pass

    def plot_inter_filament(self):
        pass

    def save_inter_filament(self):
        pass

    def calc_curv(self):
        data, name = self.get_selected_data(name=True)

        curvature = np.array(curvature_list(data, mean_b=True))
        _, first_indices = np.unique(data[:, 0], return_index=True)
        point_no = np.array(list(first_indices[1:]) + [len(data)]) - first_indices
        curvature = np.repeat(curvature, point_no)

        properties = {"Curvature": curvature}

        create_point_layer(
            viewer=self.viewer,
            points=data,
            name=name,
            add_properties=properties,
            visibility=True,
            as_filament=True,
        )

    def plot_curv(self, viewer):
        data, name, properties = self.get_selected_data(name=True, properties=True)

        try:
            unique_ids, first_indices = np.unique(data[:, 0], return_index=True)
            curv = properties["Curvature"][first_indices]
        except KeyError:
            return

        self.plot_universal.show()
        self.plot_universal.update_hist(
            curv,
            title="Curvature Distribution",
            y_label="Counts",
            x_label="Curvature",
            bins_=self.hist_bins_bt.currentText(),
        )

    def save_curv(self):
        data, name, properties = self.get_selected_data(name=True, properties=True)

        ids, first_indices = np.unique(data[:, 0], return_index=True)
        curv = properties["Curvature"][first_indices]

        x = pd.DataFrame(np.vstack([ids, curv]).T)
        header = ["IDs", "Curvature"]
        self._save_csv(x, header)

    def calc_tortuosity(self):
        data, name = self.get_selected_data(name=True)

        tortuosity = np.array(tortuosity_list(data))
        unique_ids, first_indices = np.unique(data[:, 0], return_index=True)
        point_no = np.array(list(first_indices[1:]) + [len(data)]) - first_indices
        tortuosity = np.repeat(tortuosity, point_no)

        properties = {"Tortuosity": tortuosity}

        create_point_layer(
            viewer=self.viewer,
            points=data,
            name=name,
            add_properties=properties,
            visibility=True,
            as_filament=True,
        )

    def plot_tortuosity(self, viewer):
        data, name, properties = self.get_selected_data(name=True, properties=True)

        try:
            unique_ids, first_indices = np.unique(data[:, 0], return_index=True)
            tortuosity = properties["Tortuosity"][first_indices]
        except KeyError:
            return

        self.plot_universal.show()
        self.plot_universal.update_hist(
            tortuosity,
            title="Tortuosity Distribution",
            y_label="Tortuosity",
            x_label="Distribution",
            bins_=self.hist_bins_bt.currentText(),
        )

    def save_tortuosity(self):
        data, name, properties = self.get_selected_data(name=True, properties=True)

        ids, first_indices = np.unique(data[:, 0], return_index=True)
        tort = properties["Tortuosity"][first_indices]

        x = pd.DataFrame(np.vstack([ids, tort]).T)
        header = ["IDs", "Tortuosity"]
        self._save_csv(x, header)


class PlotPopup(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = None
        self.canvas = None

        self.setWindowTitle("TARDIS Analysis Plot")
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        plt.style.use(["dark_background"])

        self.figure = plt.figure(figsize=(18, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.setLayout(layout)
        self.canvas.draw()

    def update_plot(
        self, y, title, y_label="Length", x_label="Distribution", label=None
    ):
        if len(y) > 0:
            y = [round(value, 6) for value in y]
            x_values1 = np.arange(len(y))

            self.figure.clear()

            # Create two subplots side by side
            ax1 = self.figure.add_subplot(111)  # 1 row, 1 columns, 1st subplot
            ax1.hist(x_values1, y, "-b", label=label)
            ax1.set_xlabel(x_label)
            ax1.set_ylabel(y_label)
            ax1.set_title(title)
            self.canvas.draw()

    def update_hist(
        self,
        y,
        title,
        y_label="Length",
        x_label="Distribution",
        FWHM=False,
        bins_="rice",
    ):
        if len(y) > 0:
            y = [round(value, 6) for value in y]

            self.figure.clear()

            # Create two subplots side by side
            ax1 = self.figure.add_subplot(111)  # 1 row, 1 columns, 1st subplot

            try:
                bins = int(bins_)
            except ValueError:
                bins = np.histogram_bin_edges(y, bins=bins_)

            if FWHM:
                counts, bins, _ = ax1.hist(
                    y,
                    bins=bins,
                    density=True,
                    color="skyblue",
                    edgecolor="black",
                )
                ax1.hist(
                    y,
                    bins=bins,
                    density=True,
                    color="skyblue",
                    edgecolor="black",
                    histtype="step",
                )

                peak_idx = np.argmax(counts)
                peak_y = counts[peak_idx]
                half_max = peak_y / 2

                bin_centers = (bins[:-1] + bins[1:]) / 2
                peak_x = bin_centers[peak_idx]

                fwhm_left = peak_x // 2
                fwhm_right = peak_x + fwhm_left

                # # Left side
                # left_idx = np.where(counts[:peak_idx] <= half_max)[0]
                # fwhm_left = bin_centers[left_idx[-1]] if len(left_idx) > 0 else bin_centers[0]
                #
                # # Right side
                # right_idx = np.where(counts[peak_idx:] <= half_max)[0]
                # fwhm_right = bin_centers[peak_idx + right_idx[0]] if len(right_idx) > 0 else bin_centers[-1]

                # Draw vertical lines at FWHM boundaries
                ax1.axvline(fwhm_left, color="green", linestyle="--", linewidth=2)
                ax1.axvline(fwhm_right, color="green", linestyle="--", linewidth=2)

                ax1.axvline(
                    peak_x,
                    color="white",
                    linestyle="--",
                    linewidth=2,
                    label="Distribution peak",
                )
                ax1.axhline(
                    peak_y, color="orange", linestyle="--", linewidth=2, label="F_max"
                )
                ax1.axhline(
                    half_max, color="yellow", linestyle="--", linewidth=2, label="FWHM"
                )

                # Optionally shade the FWHM region
                ax1.fill_betweenx(
                    [0, np.max(counts).item()],
                    fwhm_left,
                    fwhm_right,
                    color="green",
                    alpha=0.2,
                )

                # Annotate FWHM width
                fwhm_width = fwhm_right - fwhm_left
                ax1.text(
                    fwhm_right * 1.25,
                    np.max(counts).item() / 1.25,
                    f"FWHM: {fwhm_width:.2f}\n"
                    f"FWHM_Left: {fwhm_left:.2f}\n"
                    f"FWHM_Right: {fwhm_right:.2f}",
                    color="green",
                    ha="left",
                )
                ax1.legend()
            else:
                ax1.hist(
                    y, bins=bins_, density=True, color="skyblue", edgecolor="black"
                )

            ax1.set_xlabel(x_label)
            ax1.set_ylabel(y_label)
            ax1.set_title(title)
            self.canvas.draw()

    def update_hist_list(
        self, y, titles, y_label=["Length"], x_label=["Distribution"], bins_="rice"
    ):
        if not isinstance(y, list):
            return

        for idx, i in enumerate(y):
            y[idx] = [round(value, 6) for value in i]

        self.figure.clear()
        for idx, values in enumerate(y):
            try:
                bins = int(bins_)
            except ValueError:
                bins = np.histogram_bin_edges(y, bins=bins_)

            ax = self.figure.add_subplot(len(y), 1, idx + 1)
            ax.hist(values, bins=bins, density=True, color="skyblue", edgecolor="black")

            ax.set_xlabel(x_label[idx])
            ax.set_ylabel(y_label[idx])
            ax.set_title(titles[idx])

        self.figure.tight_layout()
        self.canvas.draw()


def show_coordinate_dialog():
    # Create dialog
    dialog = QDialog()
    dialog.setWindowTitle("Enter Coordinates")
    layout = QFormLayout()

    # Create input fields
    z_input = QLineEdit()
    y_input = QLineEdit()
    x_input = QLineEdit()

    # Add input fields to layout
    layout.addRow("X:", x_input)
    layout.addRow("Y:", y_input)
    layout.addRow("Z:", z_input)

    # Add OK/Cancel buttons
    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)

    dialog.setLayout(layout)

    # Show dialog and process input
    if dialog.exec_():
        try:
            # Get and convert inputs to floats
            z = float(z_input.text())
            y = float(y_input.text())
            x = float(x_input.text())

            return np.array([[0, x, y, z]])
        except ValueError:
            show_info("Invalid input: Please enter valid numbers for coordinates.")


def group_indices_by_value(data):
    # Initialize output lists
    new_names = []
    new_indices = []

    # Process each array and its corresponding name
    for name, arr in zip(data[0], data[1]):
        # Get unique values
        unique_vals = np.unique(arr)
        # For each unique value, find all indices where it appears
        for val in unique_vals:
            # Create name with the value as suffix
            new_names.append(f"{name}_{int(val)}")
            # Find indices where this value appears
            indices = np.where(arr == val)[0]
            new_indices.append(indices)

    # Create the output list
    output = [new_names, new_indices]

    return output
