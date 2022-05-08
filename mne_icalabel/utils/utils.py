from typing import Union, Optional

from mne import BaseEpochs
from mne.io import BaseRaw
from mne.io.pick import _picks_to_idx
from mne.preprocessing import ICA
from mne.utils import _validate_type

from .imports import import_optional_dependency


def _validate_inst_and_ica(inst: Union[BaseRaw, BaseEpochs], ica: Optional[ICA]):
    """Make sure that the provided instance and ICA are valid."""
    # check instance
    _validate_type(inst, (BaseRaw, BaseEpochs), "inst", "Raw or Epochs")

    # fit an ICA decomposition if None is provided
    if ica is None:
        # check if python-picard is available
        import_optional_dependency("picard")
        # retrieve data channels and determine the number of components
        picks = _picks_to_idx(inst.info, picks='data', exclude='bads')
        n_components = picks.size
        # if CAR is applied, look for one less component
        # TODO: 'custom_ref_applied' does not necessarily correspond to a CAR reference.
        # At the moment, the reference of the EEG data is not stored in the info.
        # c.f. https://github.com/mne-tools/mne-python/issues/8962
        if inst.info["custom_ref_applied"] == 1:
            n_components -= 1
        assert 0 < n_components  # sanity-check
        ica = ICA(n_components=n_components, method='picard')
        ica.fit(inst, picks=picks)

    # check ica
    _validate_type(ica, ICA, "ica")
    if ica.current_fit == "unfitted":
        raise RuntimeError(
            "The provided ICA instance was not fitted. Please use the '.fit()' method to "
            "determine the independent components before trying to label them."
        )
