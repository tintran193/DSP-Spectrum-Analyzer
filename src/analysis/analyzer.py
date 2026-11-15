import numpy as np

from src.dsp.preprocessing import extract_frame

from src.dsp.window import (
    get_window,
    apply_window
)

from src.dsp.fft import fft

from src.dsp.spectrum import (
    frequency_axis,
    one_sided_magnitude,
    magnitude_to_db
)

from src.dsp.peak_detector import detect_peaks


class SpectrumAnalyzer:
    """
    Main DSP pipeline for the Digital Spectrum Analyzer.
    """

    def __init__(
        self,
        peak_threshold: float = 0.05,
        min_peak_distance: int = 3
    ):

        self.peak_threshold = peak_threshold
        self.min_peak_distance = min_peak_distance

    def analyze(
        self,
        x: np.ndarray,
        fs: float,
        fft_size: int,
        window_name: str
    ) -> dict:
        """
        Analyze an input signal.

        Parameters
        ----------
        x : np.ndarray
            Input signal.

        fs : float
            Sampling frequency.

        fft_size : int
            FFT size.

        window_name : str
            rectangular, hamming, or hann.

        Returns
        -------
        dict
            Complete spectrum analysis results.
        """

        # ---------------------------------------
        # Step 1: Validate input
        # ---------------------------------------

        x = np.asarray(x, dtype=float)

        if x.ndim != 1:
            raise ValueError(
                "Input signal must be 1-D."
            )

        if x.size == 0:
            raise ValueError(
                "Input signal must not be empty."
            )

        if not np.all(np.isfinite(x)):
            raise ValueError(
                "Input signal contains invalid values."
            )

        if not np.isfinite(fs) or fs <= 0:
            raise ValueError(
                "Sampling frequency must be positive."
            )

        if (
            not isinstance(fft_size, (int, np.integer))
            or fft_size <= 0
            or (fft_size & (fft_size - 1)) != 0
        ):
            raise ValueError(
                "FFT size must be a positive power of 2."
            )

        # ---------------------------------------
        # Step 2: Extract frame
        # ---------------------------------------

        frame = extract_frame(
            x,
            start=0,
            size=fft_size
        )

        frame = np.asarray(frame, dtype=float)

        if frame.shape != (fft_size,):
            raise ValueError(
                "extract_frame must return exactly fft_size samples. "
                "Check the zero-padding policy in preprocessing.py."
            )

        # ---------------------------------------
        # Step 3: Generate window
        # ---------------------------------------

        window_name = window_name.strip().lower()

        window = get_window(
            window_name,
            fft_size
        )

        window = np.asarray(window, dtype=float)

        if window.shape != (fft_size,):
            raise ValueError(
                "Window length must equal fft_size."
            )

        # ---------------------------------------
        # Step 4: Apply window
        # ---------------------------------------

        windowed = apply_window(
            frame,
            window
        )

        # ---------------------------------------
        # Step 5: Calculate FFT
        # ---------------------------------------

        X = fft(windowed)

        # ---------------------------------------
        # Step 6: Frequency axis
        # ---------------------------------------

        frequency = frequency_axis(
            fs,
            fft_size
        )

        # ---------------------------------------
        # Step 7: Magnitude spectrum
        # ---------------------------------------

        magnitude = one_sided_magnitude(
            X,
            fft_size
        )

        # ---------------------------------------
        # Step 8: Convert to dB
        # ---------------------------------------

        db = magnitude_to_db(magnitude)

        # ---------------------------------------
        # Step 9: Peak detection
        # ---------------------------------------

        peaks = detect_peaks(
            frequency,
            magnitude,
            threshold=self.peak_threshold,
            min_distance=self.min_peak_distance
        )

        # ---------------------------------------
        # Step 10: Return analysis results
        # ---------------------------------------

        return {
            "frame": frame,
            "windowed": windowed,
            "fft": X,
            "frequency": frequency,
            "magnitude": magnitude,
            "db": db,
            "peaks": peaks,
            "fs": fs,
            "fft_size": fft_size,
            "window": window_name,
            "frequency_resolution": fs / fft_size
        }