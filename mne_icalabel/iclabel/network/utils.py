from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _format_input(topo: ArrayLike, psd: ArrayLike, autocorr: ArrayLike) -> Tuple[NDArray, NDArray, NDArray]:
    """Replicate the input formatting in EEGLAB -ICLabel.

    .. code-block:: matlab

       images = cat(4, images, -images, images(:, end:-1:1, :, :), ...
                    -images(:, end:-1:1, :, :));
       psds = repmat(psds, [1 1 1 4]);
       autocorrs = repmat(autocorrs, [1 1 1 4]);
    """
    formatted_topo = np.concatenate(
        (topo, -1 * topo, np.flip(topo, axis=1), np.flip(-1 * topo, axis=1)),
        axis=3,
    )
    formatted_psd = np.tile(psd, (1, 1, 1, 4))
    formatted_autocorr = np.tile(autocorr, (1, 1, 1, 4))
    return formatted_topo, formatted_psd, formatted_autocorr
