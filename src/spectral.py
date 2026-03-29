"""FFT spectral analysis and frequency filtering utilities."""

import numpy as np


def filtering_norm(
    freq: np.ndarray,
    data: np.ndarray,
    cen_f: float,
    cutoff: float,
    ttt: int,
    vz: float = 0.0,
) -> np.ndarray:
    """Bandpass frequency filter.

    Parameters
    ----------
    freq : np.ndarray
        Frequency array.
    data : np.ndarray
        FFT data (complex).
    cen_f : float
        Center frequency of the bandpass.
    cutoff : float
        Full bandwidth of the filter.
    ttt : int
        Filter mode:
        0 = standard bandpass around cen_f,
        2 = sideband bandpass around cen_f +/- vz.
    vz : float, optional
        Synchrotron tune offset for sideband mode.

    Returns
    -------
    np.ndarray
        Filtered data (same shape as input).
    """
    lowband = cen_f - cutoff / 2.0
    highband = cen_f + cutoff / 2.0
    tmp = np.zeros_like(data)

    for i in range(len(freq)):
        f_abs = np.abs(freq[i])
        if ttt == 2:
            if (lowband - vz < f_abs < highband - vz) or (lowband + vz < f_abs < highband + vz):
                tmp[i] = data[i]
        else:
            if lowband < f_abs < highband:
                tmp[i] = data[i]
    return tmp
