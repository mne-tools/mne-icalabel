from numpy.typing import ArrayLike
from numpy.typing import NDArray as NDArray

def _format_input(topo: ArrayLike, psd: ArrayLike, autocorr: ArrayLike) -> tuple[NDArray, NDArray, NDArray]:
    """Replicate the input formatting in EEGLAB -ICLabel.

    .. code-block:: matlab

       images = cat(4, images, -images, images(:, end:-1:1, :, :), ...
                    -images(:, end:-1:1, :, :));
       psds = repmat(psds, [1 1 1 4]);
       autocorrs = repmat(autocorrs, [1 1 1 4]);
    """