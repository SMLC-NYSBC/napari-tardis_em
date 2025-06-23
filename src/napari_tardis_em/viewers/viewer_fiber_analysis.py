#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2024                                                   #
#######################################################################
from collections import defaultdict
from os import getcwd
from os.path import join

from scipy.stats import norm

import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QDialogButtonBox
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QSlider,
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
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree

from napari_tardis_em.viewers.styles import border_style
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

        self.resample_bt = QPushButton("Resample")
        self.resample_bt.clicked.connect(self.resample)

        self.pixel_resize_box = QHBoxLayout()
        self.pixel_resize_box.addWidget(self.pixel_size_bt)
        self.pixel_resize_box.addWidget(self.resample_bt)

        self.edite_mode_bt_1 = QPushButton("Selected layer to points")
        self.edite_mode_bt_1.setMinimumWidth(225)
        self.edite_mode_bt_1.clicked.connect(self.point_layer)

        self.edite_mode_bt_2 = QPushButton("Selected layer to filament")
        self.edite_mode_bt_2.setMinimumWidth(225)
        self.edite_mode_bt_2.clicked.connect(self.track_layer)

        self.cluster_ends_th = QLineEdit("auto")
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
        self.add_centrosome_auto_select.setCurrentIndex(1)
        self.add_centrosome_auto_bt = QPushButton("Detect centers")
        self.add_centrosome_auto_bt.clicked.connect(self.add_new_point_auto)
        self.add_centrosome_auto_box = QHBoxLayout()
        self.add_centrosome_auto_box.addWidget(self.add_centrosome_auto_select)
        self.add_centrosome_auto_box.addWidget(self.add_centrosome_auto_bt)
        self.add_centrosome_auto_box.addWidget(self.add_centrosome_1_bt)

        self.save_bt = QPushButton("Save selected instance file")
        self.save_bt.clicked.connect(self.save_edited_instances)

        """
        Filter
        """
        """Filter by Distance"""

        # Filter by distance to track
        # Filter by distance to a selected point

        # Filter length
        self.filter_length = QSlider(Qt.Horizontal)
        self.filter_length.valueChanged.connect(self.filter_value_changed)
        self.filter_length.setMinimum(0)
        self.filter_length.setMaximum(100)
        self.filter_length.setValue(100)
        self.filter_length_label = QLabel("0.0")
        self.filter_length_box = QHBoxLayout()
        self.filter_length_box.addWidget(self.filter_length_label)
        self.filter_length_box.addWidget(self.filter_length)

        # Filter by Curv
        self.filter_curv = QSlider(Qt.Horizontal)
        self.filter_curv.valueChanged.connect(self.filter_value_changed)
        self.filter_curv.setMinimum(0)
        self.filter_curv.setMaximum(100)
        self.filter_curv.setValue(100)
        self.filter_curv_label = QLabel("0.0")
        self.filter_curv_box = QHBoxLayout()
        self.filter_curv_box.addWidget(self.filter_curv_label)
        self.filter_curv_box.addWidget(self.filter_curv)

        # Filter by Tortuosity
        self.filter_tort = QSlider(Qt.Horizontal)
        self.filter_tort.valueChanged.connect(self.filter_value_changed)
        self.filter_tort.setMinimum(0)
        self.filter_tort.setMaximum(100)
        self.filter_tort.setValue(100)
        self.filter_tort_label = QLabel("0.0")
        self.filter_tort_box = QHBoxLayout()
        self.filter_tort_box.addWidget(self.filter_tort_label)
        self.filter_tort_box.addWidget(self.filter_tort)

        self.last_selected_obj = QLineEdit("None")
        self.last_selected_obj.setReadOnly(False)

        self.filter_to_selected_point_dist = QLineEdit("0")

        self.filter_to_selected_point_nearest = QPushButton("End Filter")
        self.filter_to_selected_point_nearest.clicked.connect(
            self.filter_to_selected_point_nearest_
        )
        self.filter_to_selected_obj = QPushButton("Dist. Filter")
        # self.filter_to_selected_obj.clicked.connect(self.filter_to_selected_obj_)

        self.filter_to_selected_point_box = QHBoxLayout()
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
        layout = QFormLayout()

        layout.addRow("---- Pre-setting ----", label_2)
        layout.addRow("Pixel size", self.pixel_resize_box)

        layout.addRow("", self.edite_mode_bt_1)
        layout.addRow("", self.edite_mode_bt_2)

        layout.addRow("---- Parameters -----", label_anal)
        layout.addRow("Add centrosome [auto]", self.add_centrosome_auto_box)
        layout.addRow("Cluster ends", self.cluster_ends_box)

        layout.addRow("---- Filter ----", label_2)
        layout.addRow("By Length", self.filter_length_box)
        layout.addRow("By Curv.", self.filter_curv_box)
        layout.addRow("By Tort.", self.filter_tort_box)

        layout.addRow("Last selected", self.last_selected_obj)
        layout.addRow("Distance to object", self.filter_to_selected_point_dist)
        layout.addRow("", self.filter_to_selected_point_box)

        layout.addRow("---- Analysis ----", label_2)
        layout.addRow("Length", self.lenght_box)
        layout.addRow("End Distance", self.end_dist_box)
        layout.addRow("Curvature", self.curv_box)
        layout.addRow("Tortuosity", self.tortuosity_box)

        layout.addRow("", label_2)
        layout.addRow("Save Amira/CSV", self.save_bt)
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

        if f_dir == '':
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
                    [k for k, v in properties.items() if k != "ID" and not k.startswith("Label_")],
                    [v for k, v in properties.items() if k != "ID" and not k.startswith("Label_")],
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

        unique_ids, first_indices = np.unique(data[:, 0], return_index=True)
        last_indices = np.array(list(first_indices[1:]) + [len(data)]) - first_indices

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
        zyx = show_coordinate_dialog()

        try:
            data = self.viewer.layers[name].data
            data = np.array(
                (
                    self.viewer.layers[name].properties["ID"],
                    data[:, 2],
                    data[:, 1],
                    data[:, 0],
                )
            ).T
            ids = np.max(data[:, 0]).item() + 1
        except KeyError:
            data = np.zeros((0, 4))
            ids = 0

        try:
            data = np.vstack([data, zyx])
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
        if th != 'auto':
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
            grouped_filaments.extend(group_points_by_distance(i[first_idx_filament, :], eps=th))

        grouped_label = np.zeros(len(np.unique(data[:, 0])))
        id_ = 1
        for i in grouped_filaments:
            for j in np.unique(i):
                grouped_label[int(j)] = id_
            id_ += 1

        grouped_label = np.repeat(grouped_label, point_no)
        filaments_labels['Label_Grouped_Ends'] = grouped_label

        create_point_layer(
            viewer=self.viewer,
            points=data,
            name=name,
            add_properties=filaments_labels,
            visibility=True,
            as_filament=True,
            color_='viridis'
        )

    def filter_value_changed(self):
        try:
            data, name, properties = self.get_selected_data(name=True, properties=True)
        except:
            return

        if len(properties) == 0 or properties is None:
            return

        # Filter by % given as value = min_val + x * (max_val - min_val)
        filter_ = False
        if "Length" in properties:
            filter_length = self.filter_length.value()
            filter_length /= 100

            filter_ = True
            length = properties["Length"]
            l_min = min(length)
            l_max = max(length)

            filter_length_value = l_min + filter_length * (l_max - l_min)
            self.filter_length_label.setText(f"{int(filter_length_value)}")

            filter_length_bool = length <= filter_length_value
        else:
            filter_length_bool = np.ones(len(data), dtype=bool)

        if "Curvature" in properties:
            filter_curv = self.filter_curv.value()
            filter_curv /= 100

            filter_ = True
            curvature = properties["Curvature"]
            c_min = min(curvature)
            c_max = max(curvature)
            filter_curv_value = c_min + filter_curv * (c_max - c_min)

            self.filter_curv_label.setText(f"{int(filter_curv_value):.3f}")

            filter_curv_bool = curvature <= filter_curv_value
        else:
            filter_curv_bool = np.ones(len(data), dtype=bool)

        if "Tortuosity" in properties:
            filter_tort = self.filter_tort.value()
            filter_tort /= 100

            filter_ = True
            tortuosity = properties["Tortuosity"]
            t_min = min(tortuosity)
            t_max = max(tortuosity)
            filter_tort_value = t_min + filter_tort * (t_max - t_min)

            self.filter_tort_label.setText(f"{float(filter_tort_value):.2f}")

            filter_tort_bool = tortuosity <= filter_tort_value
        else:
            filter_tort_bool = np.ones(len(data), dtype=bool)

        filter_anals_bool = np.logical_and.reduce(
            [filter_length_bool, filter_curv_bool, filter_tort_bool], axis=0
        )

        if filter_ and not np.all(filter_anals_bool):
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
            _, first_indices = np.unique(data[:, 0], return_index=True)
            last_indices = (
                np.array(list(first_indices[1:]) + [len(data)]) - first_indices
            )

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
            lengths = np.array(length_list(data))

        self.plot_universal.show()
        self.plot_universal.update_hist(
            lengths,
            title="Length Distribution",
            y_label="Counts",
            x_label="Length [U]",
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
            unique_id, first_indices = np.unique(data[:, 0], return_index=True)
            last_indices = (
                np.array(list(first_indices[1:]) + [len(data)]) - first_indices
            )

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
            distances = np.array(
                [np.min(distances[distances[:, 0] == uid, 1]) for uid in unique_id]
            )

            point_no = np.array(list(first_indices[1:]) + [len(data)]) - first_indices
            distances = np.repeat(distances, point_no)

            create_point_layer(
                viewer=self.viewer,
                points=data,
                name=name,
                add_properties={"nearest_end_distance": distances},
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
            ends = np.array(curvature_list(data, mean_b=True))

        self.plot_universal.show()
        self.plot_universal.update_hist(
            ends,
            title="End Distance Distribution",
            y_label="Counts",
            x_label="End Distance [U]",
            FWHM=True,
        )

    def save_end_dist(self):
        data, name, properties = self.get_selected_data(name=True, properties=True)

        ids, first_indices = np.unique(data[:, 0], return_index=True)
        lengths = properties["nearest_end_distance"][first_indices]

        x = pd.DataFrame(np.vstack([ids, lengths]).T)
        header = ["IDs", "End Distance [U]"]
        self._save_csv(x, header)

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
            curv = np.array(curvature_list(data, mean_b=True))

        self.plot_universal.show()
        self.plot_universal.update_hist(
            curv,
            title="Curvature Distribution",
            y_label="Counts",
            x_label="Curvature",
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
            tortuosity = np.array(curvature_list(data, mean_b=True))

        self.plot_universal.show()
        self.plot_universal.update_hist(
            tortuosity,
            title="Tortuosity Distribution",
            y_label="Tortuosity",
            x_label="Distribution",
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
        self, y, title, y_label="Length", x_label="Distribution", FWHM=False
    ):
        if len(y) > 0:
            y = [round(value, 6) for value in y]

            self.figure.clear()

            # Create two subplots side by side
            ax1 = self.figure.add_subplot(111)  # 1 row, 1 columns, 1st subplot

            bins = np.histogram_bin_edges(y, bins='rice')
            if FWHM:
                counts, bins, _ = ax1.hist(
                    y,
                    bins=int(len(y) / 100) if len(y) > 1000 else 'fd',
                    density=True,
                    color="skyblue",
                    edgecolor="black",
                )
                ax1.hist(
                    y,
                    bins=int(len(y) / 100) if len(y) > 1000 else 'fd',
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
                ax1.hist(y, bins="fd", density=True, color="skyblue", edgecolor="black")

            ax1.set_xlabel(x_label)
            ax1.set_ylabel(y_label)
            ax1.set_title(title)
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

            return np.array([[0, z, y, x]])
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
