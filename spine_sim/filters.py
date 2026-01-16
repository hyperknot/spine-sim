from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def cfc_filter(x: list[float], sample_rate_hz: float, cfc: float) -> list[float]:
    """
    CFC filter (SAE-style mapping): 2nd-order Butterworth lowpass per pass,
    applied forward+backward (zero-phase) => "phaseless 4-pole" magnitude.
    """
    f_design_hz = 2.0775 * float(cfc)
    x = np.asarray(x, dtype=float)
    sos = butter(N=2, Wn=f_design_hz, btype='low', fs=sample_rate_hz, output='sos')
    y = sosfiltfilt(sos, x, padtype=None, padlen=0)
    return y.tolist()
