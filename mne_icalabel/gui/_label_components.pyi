from typing import Union

from _typeshed import Incomplete
from mne import BaseEpochs
from mne.io import BaseRaw
from mne.preprocessing import ICA
from qtpy.QtWidgets import QMainWindow

from mne_icalabel.config import ICA_LABELS_TO_MNE as ICA_LABELS_TO_MNE

class ICAComponentLabeler(QMainWindow):
    """Qt GUI to annotate components.

    Parameters
    ----------
    inst : Raw | Epochs
    ica : ICA
    show : bool
    """

    _inst: Incomplete
    _ica: Incomplete
    _labels: Incomplete
    selected_labels: Incomplete
    _selected_component: int

    def __init__(
        self, inst: Union[BaseRaw, BaseEpochs], ica: ICA, show: bool = True
    ) -> None: ...
    def _save_labels(self) -> None:
        """Save the selected labels to the ICA instance."""
    _central_widget: Incomplete
    _components_listWidget: Incomplete
    _labels_buttonGroup: Incomplete
    _mpl_figures: Incomplete
    _mpl_widgets: Incomplete
    _timeSeries_widget: Incomplete

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

    @staticmethod
    def _check_inst_ica(inst: Union[BaseRaw, BaseEpochs], ica: ICA) -> None:
        """Check if the ICA was fitted."""

    @property
    def inst(self) -> Union[BaseRaw, BaseEpochs]:
        """Instance on which the ICA has been fitted."""

    @property
    def ica(self) -> ICA:
        """Fitted ICA decomposition."""

    @property
    def n_components_(self) -> int:
        """The number of fitted components."""

    @property
    def labels(self) -> list[str]:
        """List of valid labels."""

    @property
    def selected_component(self) -> int:
        """IC selected and displayed."""

    def _connect_signals_to_slots(self) -> None:
        """Connect all the signals and slots of the GUI."""

    def _components_listWidget_clicked(self) -> None:
        """Update the plots and the saved labels accordingly."""

    def _update_selected_labels(self) -> None:
        """Update the labels saved."""

    def _reset(self) -> None:
        """Action of the reset button."""

    def _reset_buttons(self) -> None:
        """Reset all buttons."""

    def closeEvent(self, event) -> None:
        """Clean up upon closing the window.

        Update the labels since the user might have selected one for the
        currently being displayed IC.
        """
