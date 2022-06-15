import platform

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtCore import Qt, Slot
from qtpy.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QGridLayout,
    QListWidget,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

_CH_MENU_WIDTH = 30 if platform.system() == "Windows" else 10


class TopomapFig(FigureCanvasQTAgg):
    """Topographic map figure widget."""

    def __init__(self, width=4, height=4, dpi=300):
        fig = Figure(figsize=(width, height), dpi=dpi)
        ax = fig.subplots()
        fig.subplots_adjust(bottom=0, left=0, right=1, top=1, wspace=0, hspace=0)
        # clean up excess plot text, invert
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        super().__init__(fig)

    def change_ic(self, ica, idx):
        pass


class PowerSpectralDensityFig(FigureCanvasQTAgg):
    """PSD figure widget."""

    def __init__(self, width=4, height=4, dpi=300):
        fig = Figure(figsize=(width, height), dpi=dpi)
        ax = fig.subplots()
        fig.subplots_adjust(bottom=0, left=0, right=1, top=1, wspace=0, hspace=0)
        # clean up excess plot text, invert
        ax.set_xticks([])
        ax.set_yticks([])
        super().__init__(fig)

    def change_ic(self, ica, idx):
        pass


class TimeSeriesFig(FigureCanvasQTAgg):
    """Time-series figure widget."""

    def __init__(self, width=4, height=4, dpi=300):
        fig = Figure(figsize=(width, height), dpi=dpi)
        ax = fig.subplots()
        fig.subplots_adjust(bottom=0, left=0, right=1, top=1, wspace=0, hspace=0)
        # clean up excess plot text, invert
        ax.set_xticks([])
        ax.set_yticks([])
        super().__init__(fig)

    def change_ic(self, ica, inst, idx):
        pass


# TODO: Maybe that should inherit from a QGroupBox?
class Labels(QWidget):
    """Widget with labels as push buttons.

    Only one of the labels can be selected at once.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        labels = [
            "Brain",
            "Eye",
            "Heart",
            "Muscle",
            "Channel Noise",
            "Line Noise",
            "Other",
        ]

        self.buttonGroup = QButtonGroup()
        self.buttonGroup.setExclusive(True)
        layout = QVBoxLayout()
        for k, label in enumerate(labels):
            pushButton = Labels.create_pushButton(label)
            self.buttonGroup.addButton(pushButton, k)
            layout.addWidget(pushButton)
        self.setLayout(layout)
        # connect signal to slots
        self.buttonGroup.buttonClicked.connect(self.pushButton_clicked)

    @staticmethod
    def create_pushButton(label):
        pushButton = QPushButton()
        pushButton.setObjectName(f"pushButton_{label.lower().replace(' ', '_')}")
        pushButton.setText(label)
        pushButton.setCheckable(True)
        pushButton.setChecked(False)
        pushButton.setEnabled(False)
        return pushButton

    @Slot()
    def pushButton_clicked(self):
        # retrieve label from object name
        pass


class ICAComponentLabeler(QMainWindow):
    def __init__(self, inst, ica) -> None:
        super().__init__()
        self.setWindowTitle("ICA Component Labeler")
        self.setContextMenuPolicy(Qt.NoContextMenu)

        # keep an internal pointer to the ICA and Raw
        self._ica = ica
        self._inst = inst

        # create viewbox to select components
        self.list_components = ICAComponentLabeler.list_components(self._ica)
        # create buttons to select label
        self.buttonGroup_labels = Labels()
        # create figure widgets
        self.widget_topo = TopomapFig()
        self.widget_psd = PowerSpectralDensityFig()
        self.widget_timeSeries = TimeSeriesFig()

        # add central widget and layout
        self.central_widget = QWidget(self)
        self.central_widget.setObjectName("central_widget")
        layout = QGridLayout()
        layout.addWidget(self.list_components, 0, 0, 2, 1)
        layout.addWidget(self.buttonGroup_labels, 0, 1, 2, 1)
        layout.addWidget(self.widget_topo, 0, 2)
        layout.addWidget(self.widget_psd, 0, 3)
        layout.addWidget(self.widget_timeSeries, 1, 2, 1, 2)
        self.central_widget.setLayout(layout)
        self.setCentralWidget(self.central_widget)
        self.resize(800, 350)

        # connect signal and slots
        self.connect_signals_to_slots()

    @staticmethod
    def list_components(ica):
        """List the components in a QListView."""
        list_components = QListWidget()
        list_components.setSelectionMode(QAbstractItemView.SingleSelection)
        list_components.setMinimumWidth(6 * _CH_MENU_WIDTH)
        list_components.setMaximumWidth(6 * _CH_MENU_WIDTH)
        list_components.addItems([f"ICA{str(k).zfill(3)}" for k in range(ica.n_components_)])
        return list_components

    def connect_signals_to_slots(self):  # noqa: D102
        self.list_components.clicked.connect(self.list_component_clicked)

    @Slot()
    def list_component_clicked(self):
        """Jump to the selected component and draw the plots."""
        checked_idx = self.buttonGroup_labels.buttonGroup.checkedId()
        # checked_idx: the value range from 0 to n_buttons-1
        # -1 is returned if no button is checked
        for button in self.buttonGroup_labels.buttonGroup.buttons():
            button.setChecked(False)
            button.setEnabled(True)

        idx = self.list_components.currentRow()
        # idx: the value range from 0 to n_components-1
        self.widget_topo.change_ic(self._ica, idx)
        self.widget_psd.change_ic(self._ica, idx)
        self.widget_timeSeries.change_ic(self._ica, self._inst, idx)

    def closeEvent(self, event):
        """Clean up upon closing the window.

        Check if any IC is not labelled and ask the user to confirm if this is
        the case. Save all labels in BIDS format.
        """
        event.accept()
