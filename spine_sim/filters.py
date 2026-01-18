from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def cfc_filter(x: list[float], sample_rate_hz: float, cfc: float) -> list[float]:
    """
    CFC filter (SAE J211/1 style): 2nd-order Butterworth lowpass per pass,
    applied forward+backward (zero-phase).
    """
    f_design_hz = 2.0775 * float(cfc)
    x_arr = np.asarray(x, dtype=float)
    sos = butter(N=2, Wn=f_design_hz, btype="low", fs=sample_rate_hz, output="sos")
    y = sosfiltfilt(sos, x_arr, padtype=None, padlen=0)
    return y.tolist()
