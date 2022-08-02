from typing import Dict, List, Union

from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from mne import BaseEpochs
from mne.io import BaseRaw
from mne.preprocessing import ICA
from mne.utils import _validate_type
from mne.viz import set_browser_backend
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

from mne_icalabel.config import ICLABEL_LABELS_TO_MNE


class ICAComponentLabeler(QMainWindow):
    """Qt GUI to annotate components.

    Parameters
    ----------
    inst : Raw | Epochs
    ica : ICA
    show : bool
    """

    def __init__(self, inst: Union[BaseRaw, BaseEpochs], ica: ICA, show: bool = True) -> None:
        ICAComponentLabeler._check_inst_ica(inst, ica)
        super().__init__()  # initialize the QMainwindow
        set_browser_backend("qt")  # force MNE to use the QT Browser

        # keep an internal pointer to the instance and to the ICA
        self._inst = inst
        self._ica = ica
        # define valid labels
        self._labels = list(ICLABEL_LABELS_TO_MNE.keys())
        # prepare the GUI
        self._load_ui()

        # dictionary to remember selected labels, with the key as the 'indice'
        # of the component and the value as the 'label'.
        self.selected_labels: Dict[int, str] = dict()

        # connect signal to slots
        self._connect_signals_to_slots()

        # select first IC
        self._selected_component = 0
        self._components_listWidget.setCurrentRow(0)  # emit signal

        if show:
            self.show()

    def _save_labels(self) -> None:
        """Save the selected labels to the ICA instance."""
        # convert the dict[int, str] to dict[str, List[int]] with the key as
        # 'label' and value as a list of component indices.
        labels2save: Dict[str, List[int]] = {key: [] for key in self.labels}
        for component, label in self.selected_labels.items():
            labels2save[label].append(component)
        # sanity-check: uniqueness
        assert all(len(elt) == len(set(elt)) for elt in labels2save.values())

        for label, comp_list in labels2save.items():
            mne_label = ICLABEL_LABELS_TO_MNE[label]
            if mne_label not in self._ica.labels_:
                self._ica.labels_[mne_label] = comp_list
                continue
            for comp in comp_list:
                if comp not in self._ica.labels_[mne_label]:
                    self._ica.labels_[mne_label].append(comp)

    # - UI --------------------------------------------------------------------
    def _load_ui(self) -> None:
        """Prepare the GUI.

        Widgets
        -------
        self._components_listWidget
        self._labels_buttonGroup
        self._mpl_widgets (dict)
            - topomap
            - psd
        self._timeSeries_widget

        Matplotlib figures
        ------------------
        self._mpl_figures (dict)
            - topomap
            - psd
        """
        self.setWindowTitle("ICA Component Labeler")
        self.setContextMenuPolicy(Qt.NoContextMenu)

        # create central widget and main layout
        self._central_widget = QWidget(self)
        self._central_widget.setObjectName("central_widget")
        grid_layout = QGridLayout()
        self._central_widget.setLayout(grid_layout)
        self.setCentralWidget(self._central_widget)

        # QListWidget with the components' names.
        self._components_listWidget = QListWidget(self._central_widget)
        self._components_listWidget.setSelectionMode(QAbstractItemView.SingleSelection)
        self._components_listWidget.addItems(
            [f"ICA{str(k).zfill(3)}" for k in range(self.n_components_)]
        )
        grid_layout.addWidget(self._components_listWidget, 0, 0, 2, 1)

        # buttons to select labels
        self._labels_buttonGroup = QButtonGroup(self._central_widget)
        buttonGroup_layout = QVBoxLayout()
        self._labels_buttonGroup.setExclusive(True)
        for k, label in enumerate(self.labels + ["Reset"]):
            pushButton = QPushButton(self._central_widget)
            pushButton.setObjectName(f"pushButton_{label.lower().replace(' ', '_')}")
            pushButton.setText(label)
            pushButton.setCheckable(True)
            pushButton.setChecked(False)
            pushButton.setEnabled(False)
            # buttons are ordered in the same order as labels
            self._labels_buttonGroup.addButton(pushButton, k)
            buttonGroup_layout.addWidget(pushButton)
        grid_layout.addLayout(buttonGroup_layout, 0, 1, 2, 1)

        # matplotlib figures
        self._mpl_figures = dict()
        self._mpl_widgets = dict()

        # topographic map
        fig, _ = plt.subplots(1, 1, figsize=(4, 4), dpi=100)
        fig.subplots_adjust(bottom=0, left=0, right=1, top=1, wspace=0, hspace=0)
        self._mpl_figures["topomap"] = fig
        self._mpl_widgets["topomap"] = FigureCanvasQTAgg(fig)
        grid_layout.addWidget(self._mpl_widgets["topomap"], 0, 2)

        # PSD
        fig, _ = plt.subplots(1, 1, figsize=(4, 4), dpi=100)
        fig.subplots_adjust(bottom=0, left=0, right=1, top=1, wspace=0, hspace=0)
        self._mpl_figures["psd"] = fig
        self._mpl_widgets["psd"] = FigureCanvasQTAgg(fig)
        grid_layout.addWidget(self._mpl_widgets["psd"], 0, 3)

        # time-series, initialized with an empty widget.
        # TODO: When the browser supports changing the instance displayed, this
        # should be initialized to a browser with the first IC.
        self._timeSeries_widget = QWidget()
        grid_layout.addWidget(self._timeSeries_widget, 1, 2, 1, 2)

    # - Checkers --------------------------------------------------------------
    @staticmethod
    def _check_inst_ica(inst: Union[BaseRaw, BaseEpochs], ica: ICA) -> None:
        """Check if the ICA was fitted."""
        _validate_type(inst, (BaseRaw, BaseEpochs), "inst", "raw or epochs")
        _validate_type(ica, ICA, "ica", "ICA")
        if ica.current_fit == "unfitted":
            raise ValueError(
                "ICA instance should be fitted on raw/epochs data before "
                "running the ICA labeling GUI. Run 'ica.fit(inst)'."
            )

    # - Properties ------------------------------------------------------------
    @property
    def inst(self) -> Union[BaseRaw, BaseEpochs]:
        """Instance on which the ICA has been fitted."""
        return self._inst

    @property
    def ica(self) -> ICA:
        """Fitted ICA decomposition."""
        return self._ica

    @property
    def n_components_(self) -> int:
        """The number of fitted components."""
        return self._ica.n_components_

    @property
    def labels(self) -> List[str]:
        """List of valid labels."""
        return self._labels

    @property
    def selected_component(self) -> int:
        """IC selected and displayed."""
        return self._selected_component

    # - Slots -----------------------------------------------------------------
    def _connect_signals_to_slots(self) -> None:
        """Connect all the signals and slots of the GUI."""
        self._components_listWidget.currentRowChanged.connect(self._components_listWidget_clicked)
        self._labels_buttonGroup.buttons()[-1].clicked.connect(self._reset)

    @Slot()
    def _components_listWidget_clicked(self) -> None:
        """Update the plots and the saved labels accordingly."""
        self._update_selected_labels()
        self._reset_buttons()

        # update selected IC
        self._selected_component = self._components_listWidget.currentRow()

        # reset matplotlib figures
        for fig in self._mpl_figures.values():
            fig.axes[0].clear()
        # create dummy figure and axes to hold the unused plots from plot_properties
        dummy_fig, dummy_axes = plt.subplots(3)
        # create axes argument provided to plot_properties
        axes = [
            self._mpl_figures["topomap"].axes[0],
            dummy_axes[0],
            dummy_axes[1],
            self._mpl_figures["psd"].axes[0],
            dummy_axes[2],
        ]
        # update matplotlib plots with plot_properties
        self.ica.plot_properties(self.inst, axes=axes, picks=self.selected_component, show=False)
        del dummy_fig
        # remove title from topomap axes
        self._mpl_figures["topomap"].axes[0].set_title("")
        # update the matplotlib canvas
        for fig in self._mpl_figures.values():
            fig.tight_layout()
            fig.canvas.draw()
            fig.canvas.flush_events()

        # swap timeSeries widget
        timeSeries_widget = self.ica.plot_sources(self.inst, picks=[self.selected_component])
        self._central_widget.layout().replaceWidget(self._timeSeries_widget, timeSeries_widget)
        self._timeSeries_widget.setParent(None)
        self._timeSeries_widget = timeSeries_widget

        # select buttons that were previously selected for this IC
        if self.selected_component in self.selected_labels:
            idx = self.labels.index(self.selected_labels[self.selected_component])
            self._labels_buttonGroup.button(idx).setChecked(True)

    def _update_selected_labels(self) -> None:
        """Update the labels saved."""
        selected = self._labels_buttonGroup.checkedButton()
        if selected is not None:
            self.selected_labels[self.selected_component] = selected.text()
        self._save_labels()  # updates the ICA instance every time

    @Slot()
    def _reset(self) -> None:  # noqa: D401
        """Action of the reset button."""
        self._reset_buttons()
        if self.selected_component in self.selected_labels:
            del self.selected_labels[self.selected_component]

    def _reset_buttons(self) -> None:
        """Reset all buttons."""
        self._labels_buttonGroup.setExclusive(False)
        for button in self._labels_buttonGroup.buttons():
            button.setEnabled(True)
            button.setChecked(False)
        self._labels_buttonGroup.setExclusive(True)

    def closeEvent(self, event) -> None:
        """Clean up upon closing the window.

        Update the labels since the user might have selected one for the
        currently being displayed IC.
        """
        self._update_selected_labels()
        event.accept()
