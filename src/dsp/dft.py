import numpy as np


def dft(x: np.ndarray) -> np.ndarray:
    """Compute the N-point DFT directly from its mathematical definition.

    Parameters
    ----------
    x:
        One-dimensional input signal with N samples. Real and complex input
        arrays are supported.

    Returns
    -------
    np.ndarray
        One-dimensional complex array containing all N DFT coefficients.

    Notes
    -----
    This implementation intentionally does not call ``numpy.fft``. It is a
    straightforward reference implementation for validating the custom FFT.
    Its computational complexity is O(N**2).
    """
    signal = np.asarray(x, dtype=np.complex128)

    if signal.ndim != 1:
        raise ValueError("Input signal must be a 1-D array.")

    n_samples = signal.size
    if n_samples == 0:
        raise ValueError("Input signal must not be empty.")

    sample_indices = np.arange(n_samples)
    result = np.empty(n_samples, dtype=np.complex128)

    for k in range(n_samples):
        twiddle = np.exp(-2j * np.pi * k * sample_indices / n_samples)
        result[k] = np.sum(signal * twiddle)

    return result