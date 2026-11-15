import numpy as np


def plot_time_domain(
    ax,
    x: np.ndarray,
    fs: float
):
    """
    Plot time-domain waveform.
    """

    x = np.asarray(x)

    if x.ndim != 1:
        raise ValueError(
            "Signal must be a 1-D array."
        )

    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(
            "Sampling frequency must be positive."
        )

    # Generate time axis.
    time = np.arange(len(x)) / fs

    # Clear previous plot.
    ax.clear()

    # Plot waveform.
    ax.plot(
        time,
        x,
        linewidth=1
    )

    ax.set_title("Time Domain")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")

    ax.grid(True)


def plot_spectrum(
    ax,
    frequency: np.ndarray,
    magnitude: np.ndarray,
    peaks: list[dict] | None = None
):
    """
    Plot one-sided magnitude spectrum.
    """

    frequency = np.asarray(frequency)
    magnitude = np.asarray(magnitude)

    if len(frequency) != len(magnitude):
        raise ValueError(
            "Frequency and magnitude lengths must match."
        )

    ax.clear()

    ax.plot(
        frequency,
        magnitude,
        linewidth=1
    )

    # Mark detected peaks.
    if peaks:

        peak_frequencies = [
            peak["frequency"] for peak in peaks
        ]

        peak_magnitudes = [
            peak["magnitude"] for peak in peaks
        ]

        ax.scatter(
            peak_frequencies,
            peak_magnitudes,
            marker="x",
            label="Detected Peaks"
        )

        ax.legend()

    ax.set_title("Magnitude Spectrum")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")

    ax.grid(True)


def plot_db_spectrum(
    ax,
    frequency: np.ndarray,
    db: np.ndarray,
    peaks: list[dict] | None = None
):
    """
    Plot one-sided dB spectrum.
    """

    frequency = np.asarray(frequency)
    db = np.asarray(db)

    if len(frequency) != len(db):
        raise ValueError(
            "Frequency and dB lengths must match."
        )

    ax.clear()

    ax.plot(
        frequency,
        db,
        linewidth=1
    )

    # Mark detected peaks.
    if peaks:

        peak_frequencies = [
            peak["frequency"] for peak in peaks
        ]

        peak_db = [
            peak["db"] for peak in peaks
        ]

        ax.scatter(
            peak_frequencies,
            peak_db,
            marker="x",
            label="Detected Peaks"
        )

        ax.legend()

    ax.set_title("Magnitude Spectrum (dB)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")

    ax.grid(True)