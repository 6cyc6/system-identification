"""Signal filtering, perturbation, and discontinuity checks."""

import numpy as np
from scipy.signal import savgol_filter


def savitzy_filter(x, freq, window, poly_order=3, d_order=1, mode="nearest"):
    """Differentiate a sampled signal using a Savitzky-Golay filter."""
    return savgol_filter(
        x.T,
        window,
        poly_order,
        d_order,
        1.0 / freq,
        mode=mode,
    ).T


def perturbed_array(x, scale=1):
    return x + np.random.randn(x.shape[0], x.shape[1]) * scale


def check_jumps(x, shrehold=100):
    """Return whether adjacent samples exceed the relative jump threshold."""
    x_next = x[1:, :]
    x_previous = x[:-1, :]
    relative_difference = abs(x_next - x_previous) / abs(x_next)
    return np.any(relative_difference >= shrehold)
