# -*- coding: utf-8 -*-
"""ICA GUI for labeling components."""

# Authors: Adam Li <adam2392@gmail.com>
#
# License: BSD (3-clause)


import platform

from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from mne.preprocessing import ICA
from mne.viz.utils import safe_event
from qtpy import QtGui
from qtpy.QtCore import Slot
from qtpy.QtWidgets import (
    QAbstractItemView,
    QGridLayout,
    QHBoxLayout,
    QListView,
    QMainWindow,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

# _IMG_LABELS = [['I', 'P'], ['I', 'L'], ['P', 'L']]
# _CH_PLOT_SIZE = 1024
# _ZOOM_STEP_SIZE = 5
# _RADIUS_SCALAR = 0.4
# _TUBE_SCALAR = 0.1
# _BOLT_SCALAR = 30  # mm
_CH_MENU_WIDTH = 30 if platform.system() == "Windows" else 10


# TODO: remove
def _make_topo_plot(width=4, height=4, dpi=300):
    """Make subplot for the topomap."""
    fig = Figure(figsize=(width, height), dpi=dpi)
    canvas = FigureCanvas(fig)
    ax = fig.subplots()
    fig.subplots_adjust(bottom=0, left=0, right=1, top=1, wspace=0, hspace=0)
    ax.set_facecolor("k")
    # clean up excess plot text, invert
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    return canvas, fig


# TODO: remove
def _make_ts_plot(width=4, height=4, dpi=300):
    """Make subplot for the component time-series."""
    fig = Figure(figsize=(width, height), dpi=dpi)
    canvas = FigureCanvas(fig)
    ax = fig.subplots()
    fig.subplots_adjust(bottom=0, left=0, right=1, top=1, wspace=0, hspace=0)
    ax.set_facecolor("k")
    # clean up excess plot text
    ax.set_xticks([])
    ax.set_yticks([])
    return canvas, fig


# TODO: remove
def _make_spectrum_plot(width=4, height=4, dpi=300):
    """Make subplot for the spectrum."""
    fig = Figure(figsize=(width, height), dpi=dpi)
    canvas = FigureCanvas(fig)
    ax = fig.subplots()
    fig.subplots_adjust(bottom=0, left=0, right=1, top=1, wspace=0, hspace=0)
    ax.set_facecolor("k")
    # clean up excess plot text
    ax.set_xticks([])
    ax.set_yticks([])
    return canvas, fig


class TimeSeriesFig(FigureCanvas):
    """Spectrum map widget."""

    def __init__(self, width=4, height=4, dpi=300):
        """Make subplot for the spectrum."""
        fig = Figure(figsize=(width, height), dpi=dpi)
        canvas = FigureCanvas(fig)
        ax = fig.subplots()
        fig.subplots_adjust(bottom=0, left=0, right=1, top=1, wspace=0, hspace=0)
        ax.set_facecolor("k")
        # clean up excess plot text
        ax.set_xticks([])
        ax.set_yticks([])
        super().__init__(fig)


class SpectrumFig(FigureCanvas):
    """Spectrum map widget."""

    def __init__(self, width=4, height=4, dpi=300):
        """Make subplot for the spectrum."""
        fig = Figure(figsize=(width, height), dpi=dpi)
        canvas = FigureCanvas(fig)
        ax = fig.subplots()
        fig.subplots_adjust(bottom=0, left=0, right=1, top=1, wspace=0, hspace=0)
        ax.set_facecolor("k")
        # clean up excess plot text
        ax.set_xticks([])
        ax.set_yticks([])
        super().__init__(fig)


class TopomapFig(FigureCanvas):
    """Topographic map widget."""

    def __init__(self, width=4, height=4, dpi=300):
        fig = Figure(figsize=(width, height), dpi=dpi)
        ax = fig.subplots()
        fig.subplots_adjust(bottom=0, left=0, right=1, top=1, wspace=0, hspace=0)
        ax.set_facecolor("k")
        # clean up excess plot text
        ax.set_xticks([])
        ax.set_yticks([])
        super().__init__(fig)


# TODO:
# ? - plot_properties plot - topoplot, ICA time-series
# ? - update ICA components
# ? - menu with save, load
class ICAComponentLabeler(QMainWindow):
    def __init__(self, inst, ica: ICA, verbose: bool = False) -> None:
        # initialize QMainWindow class
        super().__init__()

        self.verbose = verbose

        # keep an internal pointer to the ICA and Raw
        self._ica = ica
        self._inst = inst

        # GUI design to add widgets into a Layout
        # Main plots: make one plot for each view: topographic, time-series, power-spectrum
        plt_grid = QGridLayout()
        plts = [_make_topo_plot(), _make_ts_plot(), _make_spectrum_plot()]
        self._figs = [plts[0][1], plts[1][1], plts[2][1]]
        plt_grid.addWidget(plts[0][0], 0, 0)
        plt_grid.addWidget(plts[1][0], 0, 1)
        plt_grid.addWidget(plts[2][0], 1, 0)

        # TODO: is this the correct function to use to render? or nah... since we don't have 3D?
        # self._renderer = _get_renderer(name="ICA Component Labeler", size=(400, 400), bgcolor="w")
        # plt_grid.addWidget(self._renderer.plotter)

        # initialize channel data
        self._component_index = 0

        # component names are just a list of numbers from 0 to n_components
        self._component_names = [f"ICA-{idx}" for idx in range(ica.n_components_)]

        # Component selector in a clickable selection list
        self._component_list = QListView()
        self._component_list.setSelectionMode(QAbstractItemView.SingleSelection)
        max_comp_name_len = max([len(name) for name in self._component_names])
        self._component_list.setMinimumWidth(max_comp_name_len * _CH_MENU_WIDTH)
        self._component_list.setMaximumWidth(max_comp_name_len * _CH_MENU_WIDTH)
        self._set_component_names()

        # Plots
        self._plot_images()

        # TODO: Menus for user interface
        # button_hbox = self._get_button_bar()
        # slider_hbox = self._get_slider_bar()
        # bottom_hbox = self._get_bottom_bar()

        # Put everything together
        plot_component_hbox = QHBoxLayout()
        plot_component_hbox.addLayout(plt_grid)
        plot_component_hbox.addWidget(self._component_list)

        # TODO: add the rest of the button and other widgets/menus
        main_vbox = QVBoxLayout()
        main_vbox.addLayout(plot_component_hbox)
        # main_vbox.addLayout(button_hbox)
        # main_vbox.addLayout(slider_hbox)
        # main_vbox.addLayout(bottom_hbox)

        central_widget = QWidget()
        central_widget.setLayout(main_vbox)
        self.setCentralWidget(central_widget)

        # ready for user
        self._component_list.setFocus()  # always focus on list

    def _set_component_names(self):
        """Add the component names to the selector."""
        self._component_list_model = QtGui.QStandardItemModel(self._component_list)
        for name in self._component_names:
            self._component_list_model.appendRow(QtGui.QStandardItem(name))
            # TODO: can add a method to color code the list of items
            # self._color_list_item(name=name)
        self._component_list.setModel(self._component_list_model)
        self._component_list.clicked.connect(self._go_to_component)
        self._component_list.setCurrentIndex(
            self._component_list_model.index(self._component_index, 0)
        )
        self._component_list.keyPressEvent = self._key_press_event

    def _go_to_component(self, index):
        """Change current channel to the item selected."""
        self._component_index = index.row()
        self._update_component_selection()

    def _update_component_selection(self):
        """Update which channel is selected."""
        name = self._component_names[self._component_index]
        self._component_list.setCurrentIndex(
            self._component_list_model.index(self._component_index, 0)
        )

    def _plot_images(self):
        # TODO: embed the matplotlib figure in each FigureCanvas
        pass

    def _save_component_labels(self):
        pass

    @Slot()
    def _mark_component(self):
        pass

    @safe_event
    def closeEvent(self, event):
        """Clean up upon closing the window."""
        self._renderer.plotter.close()
        self.close()

    def _key_press_event(self, event):
        pass

    def _show_help(self):
        """Show the help menu."""
        QMessageBox.information(
            self,
            "Help",
            "Help:\n'g': mark component as good (brain)\n"
            "up/down arrow: move up/down the list of components\n",
        )
