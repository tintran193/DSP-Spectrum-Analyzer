import numpy as np

from src.dsp.dft import dft
from src.dsp.fft import fft


def _max_absolute_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Return the maximum absolute element-wise difference."""
    if reference.shape != candidate.shape:
        raise ValueError("The two transform outputs must have the same shape.")

    if reference.size == 0:
        raise ValueError("Transform outputs must not be empty.")

    return float(np.max(np.abs(reference - candidate)))


def compare_dft_fft(x: np.ndarray) -> float:
    """Return max(abs(Direct DFT(x) - Custom FFT(x))).

    The input length must satisfy the Radix-2 FFT requirement.
    """
    reference = dft(x)
    candidate = fft(x)
    return _max_absolute_error(reference, candidate)


def compare_fft_numpy(x: np.ndarray) -> float:
    """Return max(abs(Custom FFT(x) - numpy.fft.fft(x)))."""
    candidate = fft(x)
    reference = np.fft.fft(np.asarray(x))
    return _max_absolute_error(reference, candidate)