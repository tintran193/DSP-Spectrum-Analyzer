import numpy as np

from scipy.signal import find_peaks

from src.dsp.spectrum import magnitude_to_db


def detect_peaks(
    frequency: np.ndarray,
    magnitude: np.ndarray,
    threshold: float,
    min_distance: int
) -> list[dict]:
    """
    Detect local peaks in a one-sided magnitude spectrum.

    Parameters
    ----------
    frequency : np.ndarray
        Frequency axis in Hz.

    magnitude : np.ndarray
        One-sided magnitude spectrum.

    threshold : float
        Minimum peak height in linear magnitude.

    min_distance : int
        Minimum distance between peaks in FFT bins.

    Returns
    -------
    list[dict]
        Detected peaks sorted by descending magnitude.
    """

    frequency = np.asarray(frequency, dtype=float)
    magnitude = np.asarray(magnitude, dtype=float)

    # Validate dimensions.
    if frequency.ndim != 1 or magnitude.ndim != 1:
        raise ValueError(
            "Inputs must be 1-D arrays."
        )

    # Validate lengths.
    if len(frequency) != len(magnitude):
        raise ValueError(
            "Frequency and magnitude must have the same length."
        )

    if len(magnitude) == 0:
        raise ValueError(
            "Magnitude spectrum must not be empty."
        )

    # Validate values.
    if not np.all(np.isfinite(frequency)):
        raise ValueError(
            "Frequency contains invalid values."
        )

    if not np.all(np.isfinite(magnitude)):
        raise ValueError(
            "Magnitude contains invalid values."
        )

    if np.any(magnitude < 0):
        raise ValueError(
            "Magnitude must be non-negative."
        )

    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError(
            "Threshold must be non-negative."
        )

    if min_distance < 1:
        raise ValueError(
            "min_distance must be at least 1."
        )

    # Find local peaks.
    peak_indices, properties = find_peaks(
        magnitude,
        height=threshold,
        distance=min_distance
    )

    # Convert magnitude to dB.
    db = magnitude_to_db(magnitude)

    # Build peak list.
    peaks = []

    for index in peak_indices:

        peaks.append({
            "frequency": float(frequency[index]),
            "magnitude": float(magnitude[index]),
            "db": float(db[index]),
            "index": int(index)
        })

    # Sort by magnitude, descending.
    peaks.sort(
        key=lambda peak: peak["magnitude"],
        reverse=True
    )

    return peaks