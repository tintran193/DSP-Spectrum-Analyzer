import numpy as np


def frequency_axis(fs: float, N: int) -> np.ndarray:
    """
    Generate a one-sided frequency axis.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.

    N : int
        FFT size.

    Returns
    -------
    np.ndarray
        Frequency bins from 0 Hz to Nyquist frequency.
    """

    # Validate sampling frequency.
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(
            "Sampling frequency must be positive."
        )

    # Validate FFT size.
    if not isinstance(N, (int, np.integer)) or N <= 0:
        raise ValueError(
            "FFT size must be a positive integer."
        )

    # Generate positive-frequency bins.
    frequency = np.arange(N // 2 + 1) * fs / N

    return frequency


def one_sided_magnitude(
    X: np.ndarray,
    N: int
) -> np.ndarray:
    """
    Calculate normalized one-sided magnitude spectrum.

    Parameters
    ----------
    X : np.ndarray
        Full complex FFT output.

    N : int
        FFT size.

    Returns
    -------
    np.ndarray
        One-sided magnitude spectrum.
    """

    X = np.asarray(X, dtype=np.complex128)

    # Validate input dimension.
    if X.ndim != 1:
        raise ValueError(
            "FFT output must be a 1-D array."
        )

    # Validate FFT size.
    if not isinstance(N, (int, np.integer)) or N <= 0:
        raise ValueError(
            "FFT size must be a positive integer."
        )

    # Validate FFT output length.
    if len(X) != N:
        raise ValueError(
            "FFT output length must equal N."
        )

    # Extract positive-frequency components.
    magnitude = np.abs(X[:N // 2 + 1])

    # Normalize by the number of FFT samples.
    magnitude = magnitude / N

    # Convert two-sided amplitude into one-sided amplitude.
    if N % 2 == 0:
        magnitude[1:-1] *= 2.0
    else:
        magnitude[1:] *= 2.0

    return magnitude


def magnitude_to_db(
    magnitude: np.ndarray,
    eps: float = 1e-12
) -> np.ndarray:
    """
    Convert magnitude spectrum to decibels.

    Parameters
    ----------
    magnitude : np.ndarray
        Non-negative magnitude spectrum.

    eps : float
        Small positive number to avoid log10(0).

    Returns
    -------
    np.ndarray
        Magnitude spectrum in dB.
    """

    magnitude = np.asarray(magnitude, dtype=float)

    # Validate input dimension.
    if magnitude.ndim != 1:
        raise ValueError(
            "Magnitude must be a 1-D array."
        )

    # Validate values.
    if not np.all(np.isfinite(magnitude)):
        raise ValueError(
            "Magnitude contains invalid values."
        )

    if np.any(magnitude < 0):
        raise ValueError(
            "Magnitude must be non-negative."
        )

    # Validate epsilon.
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError(
            "eps must be positive."
        )

    return 20.0 * np.log10(magnitude + eps)