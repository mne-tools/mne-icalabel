# -*- coding: utf-8 -*-
"""ICA GUI for labeling components."""

# Authors: Adam Li <adam2392@gmail.com>
#
# License: BSD (3-clause)


import platform

from qtpy import QtCore, QtGui
from qtpy.QtCore import Slot, Signal
from qtpy.QtWidgets import (QMainWindow, QGridLayout,
                            QVBoxLayout, QHBoxLayout, QLabel,
                            QMessageBox, QWidget, QAbstractItemView,
                            QListView, QSlider, QPushButton,
                            QComboBox, QPlainTextEdit)


# _IMG_LABELS = [['I', 'P'], ['I', 'L'], ['P', 'L']]
# _CH_PLOT_SIZE = 1024
# _ZOOM_STEP_SIZE = 5
# _RADIUS_SCALAR = 0.4
# _TUBE_SCALAR = 0.1
# _BOLT_SCALAR = 30  # mm
_CH_MENU_WIDTH = 30 if platform.system() == 'Windows' else 10


# TODO:
#? - plot_properties plot
#? - update ICA components
#? - menu with save, load
class ICAComponentLabeler(QMainWindow):
    def __init__(self, ica, raw, ) -> None:
        # initialize QMainWindow class
        super().__init__()

        self._ica = ica
        self._raw = raw

        # GUI design

        # Main plots: make one plot for each view; sagittal, coronal, axial
        plt_grid = QGridLayout()
        plts = [_make_slice_plot(), _make_slice_plot(), _make_slice_plot()]
        self._figs = [plts[0][1], plts[1][1], plts[2][1]]
        plt_grid.addWidget(plts[0][0], 0, 0)
        plt_grid.addWidget(plts[1][0], 0, 1)
        plt_grid.addWidget(plts[2][0], 1, 0)
        self._renderer = _get_renderer(
            name='IEEG Locator', size=(400, 400), bgcolor='w')
        plt_grid.addWidget(self._renderer.plotter)

        # Channel selector
        self._ch_list = QListView()
        self._ch_list.setSelectionMode(QAbstractItemView.SingleSelection)
        max_ch_name_len = max([len(name) for name in self._chs])
        self._ch_list.setMinimumWidth(max_ch_name_len * _CH_MENU_WIDTH)
        self._ch_list.setMaximumWidth(max_ch_name_len * _CH_MENU_WIDTH)
        self._set_ch_names()

        # Plots
        self._plot_images()

        # Menus
        button_hbox = self._get_button_bar()
        slider_hbox = self._get_slider_bar()
        bottom_hbox = self._get_bottom_bar()

        # Add lines
        self._lines = dict()
        self._lines_2D = dict()
        for group in set(self._groups.values()):
            self._update_lines(group)

        # Put everything together
        plot_ch_hbox = QHBoxLayout()
        plot_ch_hbox.addLayout(plt_grid)
        plot_ch_hbox.addWidget(self._ch_list)

        main_vbox = QVBoxLayout()
        main_vbox.addLayout(button_hbox)
        main_vbox.addLayout(slider_hbox)
        main_vbox.addLayout(plot_ch_hbox)
        main_vbox.addLayout(bottom_hbox)

        central_widget = QWidget()
        central_widget.setLayout(main_vbox)
        self.setCentralWidget(central_widget)

        # ready for user
        self._move_cursors_to_pos()
        self._ch_list.setFocus()  # always focus on list

    def _plot_images(self):
        pass

    def _save_component_labels(self):
        pass

    def _update_grouop(self):
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
            self, 'Help',
            "Help:\n'm': mark channel location\n"
            "'r': remove channel location\n"
            "'b': toggle viewing of brain in T1\n"
            "'+'/'-': zoom\nleft/right arrow: left/right\n"
            "up/down arrow: superior/inferior\n"
            "left angle bracket/right angle bracket: anterior/posterior")
