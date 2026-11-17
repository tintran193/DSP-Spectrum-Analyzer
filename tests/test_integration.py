"""Integration Tests T01-T07 for Digital Spectrum Analyzer."""

from pathlib import Path

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.input.signal_generator import (
    generate_sine,
    generate_multi_sine,
    generate_square,
)

from src.input.wav_loader import load_wav
from src.analysis.analyzer import SpectrumAnalyzer
from src.dsp.peak_detector import detect_peaks

from src.visualization.plot import (
    plot_spectrum,
    plot_db_spectrum,
)


# ---------------------------------------------------------
# Common configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

analyzer = SpectrumAnalyzer()


def run_analysis(x, fs, fft_size, window_name="rectangular"):
    """
    Run the complete signal analysis pipeline.

    Input:
        x           : Input signal
        fs          : Sampling frequency
        fft_size    : FFT size
        window_name : Window type

    Output:
        Dictionary returned by SpectrumAnalyzer.
    """

    result = analyzer.analyze(
        x=x,
        fs=fs,
        fft_size=fft_size,
        window_name=window_name,
    )

    # Check output structure
    required_keys = [
        "frame",
        "windowed",
        "fft",
        "frequency",
        "magnitude",
        "db",
        "peaks",
        "fs",
        "fft_size",
        "window",
        "frequency_resolution",
    ]

    for key in required_keys:
        assert key in result, f"Missing result key: {key}"

    # Check output dimensions
    assert len(result["frequency"]) == fft_size // 2 + 1
    assert len(result["magnitude"]) == fft_size // 2 + 1
    assert len(result["db"]) == fft_size // 2 + 1

    assert np.all(np.isfinite(result["magnitude"]))
    assert np.all(np.isfinite(result["db"]))

    return result



def find_peaks_from_result(
    result,
    threshold_ratio=0.05,
    min_distance=2,
):
    """
    Run Peak Detection using the spectrum returned by Analyzer.

    Parameters
    ----------
    result : dict
        Result returned by SpectrumAnalyzer.

    threshold_ratio : float
        Minimum magnitude threshold relative to maximum magnitude.

    min_distance : int
        Minimum distance between detected peaks, in FFT bins.
    """

    magnitude = result["magnitude"]

    if not 0 <= threshold_ratio <= 1:
        raise ValueError("threshold_ratio must be between 0 and 1")

    threshold = float(np.max(magnitude) * threshold_ratio)

    peaks = detect_peaks(
        frequency=result["frequency"],
        magnitude=magnitude,
        threshold=threshold,
        min_distance=min_distance,
    )

    return peaks

def assert_peak_near(peaks, expected_frequency, tolerance):
    """Check whether at least one detected peak is near the expected frequency."""

    peak_frequencies = [
        peak["frequency"] for peak in peaks
    ]

    assert any(
        abs(f - expected_frequency) <= tolerance
        for f in peak_frequencies
    ), (
        f"Expected peak near {expected_frequency} Hz, "
        f"but detected peaks were {peak_frequencies}"
    )

def test_t01_sine_1khz():
    """T01: Generate sine -> Analyzer -> Spectrum -> Peak Detection."""

    fs = 8000
    fft_size = 1024
    frequency = 1000
    amplitude = 1.0

    # Step 1: Generate sine signal
    x, fs = generate_sine(
        frequency=frequency,
        amplitude=amplitude,
        phase=0.0,
        fs=fs,
        duration=fft_size / fs,
    )

    # Step 2: Run complete analysis
    result = run_analysis(
        x,
        fs,
        fft_size,
        window_name="rectangular",
    )

    # Step 3: Find dominant frequency
    peak_index = np.argmax(result["magnitude"])

    dominant_frequency = result["frequency"][peak_index]

    # Frequency resolution
    df = fs / fft_size

    # Step 4: Check expected frequency
    assert dominant_frequency == pytest.approx(
        frequency,
        abs=df / 2,
    )

    # Step 5: Check Peak Detection
    peaks = find_peaks_from_result(result)

    assert_peak_near(
        peaks,
        frequency,
        tolerance=df,
    )
        # -------------------------------------------------
    # Step 6: Print test results
    # -------------------------------------------------

    dominant_magnitude = result["magnitude"][peak_index]
    dominant_db = result["db"][peak_index]

    print("\n========== T01: SINE WAVE ANALYSIS ==========")

    print(f"Signal Frequency : {frequency} Hz")
    print(f"Signal Amplitude : {amplitude}")
    print(f"Sampling Frequency: {fs} Hz")
    print(f"FFT Size          : {fft_size}")
    print(f"Frequency Resolution: {df:.4f} Hz")

    print("\nAnalysis Results:")
    print(f"Dominant Frequency: {dominant_frequency:.2f} Hz")
    print(f"Peak Magnitude    : {dominant_magnitude:.6f}")
    print(f"Peak dB           : {dominant_db:.2f} dB")

    print("\nDetected Peaks:")

    for peak in peaks:
        print(
            f"Frequency: {peak['frequency']:.2f} Hz | "
            f"Magnitude: {peak['magnitude']:.6f} | "
            f"dB: {peak['db']:.2f}"
        )

    print("=============================================")

def test_t02_multi_sine():
    """
    T02: Analyze a multi-sine signal containing
    1 kHz and 3 kHz frequency components.

    Expected:
        - Peak at 1000 Hz
        - Peak at 3000 Hz
        - Amplitude ratio approximately 2:1
    """

    # Configuration
    fs = 8000
    fft_size = 1024

    frequencies = [1000, 3000]
    amplitudes = [1.0, 0.5]
    phases = [0.0, 0.0]

    # Step 1: Generate multi-sine signal
    x, fs = generate_multi_sine(
        frequencies=frequencies,
        amplitudes=amplitudes,
        phases=phases,
        fs=fs,
        duration=fft_size / fs,
    )

    # Check generated signal
    assert x.ndim == 1
    assert len(x) == fft_size
    assert fs == 8000

    # Step 2: Run complete analysis pipeline
    result = run_analysis(
        x=x,
        fs=fs,
        fft_size=fft_size,
        window_name="rectangular",
    )

    # Step 3: Get frequency and magnitude spectrum
    frequency = result["frequency"]
    magnitude = result["magnitude"]

    # Step 4: Find FFT bins corresponding to 1 kHz and 3 kHz
    index_1k = np.argmin(np.abs(frequency - 1000))
    index_3k = np.argmin(np.abs(frequency - 3000))

    magnitude_1k = magnitude[index_1k]
    magnitude_3k = magnitude[index_3k]

    # Step 5: Check frequency positions
    assert frequency[index_1k] == pytest.approx(1000)
    assert frequency[index_3k] == pytest.approx(3000)

    # Step 6: Check both frequency components exist
    assert magnitude_1k > 0
    assert magnitude_3k > 0

    # Step 7: Check relative amplitudes
    amplitude_ratio = magnitude_1k / magnitude_3k

    assert amplitude_ratio == pytest.approx(
        2.0,
        rel=0.02,
    )

    # Step 8: Run Peak Detection
    peaks = find_peaks_from_result(
        result,
        threshold_ratio=0.05,
    )

    # Step 9: Verify both peaks are detected
    df = fs / fft_size

    assert_peak_near(
        peaks,
        expected_frequency=1000,
        tolerance=df,
    )

    assert_peak_near(
        peaks,
        expected_frequency=3000,
        tolerance=df,
    )

    # -------------------------------------------------
    # Step 10: Print test results
    # -------------------------------------------------

    print("\n========== T02: MULTI-SINE ANALYSIS ==========")

    print("Input Signal:")
    print(f"Frequency Components: {frequencies}")
    print(f"Amplitude Components: {amplitudes}")
    print(f"Sampling Frequency  : {fs} Hz")
    print(f"FFT Size            : {fft_size}")
    print(f"Frequency Resolution: {df:.4f} Hz")

    print("\nSpectrum Results:")
    print(
        f"1000 Hz -> Bin {index_1k:4d} | "
        f"Magnitude: {magnitude_1k:.6f} | "
        f"dB: {result['db'][index_1k]:.2f}"
    )

    print(
        f"3000 Hz -> Bin {index_3k:4d} | "
        f"Magnitude: {magnitude_3k:.6f} | "
        f"dB: {result['db'][index_3k]:.2f}"
    )

    print(f"\nAmplitude Ratio (1kHz / 3kHz): {amplitude_ratio:.4f}")

    print("\nDetected Peaks:")

    for peak in peaks:
        print(
            f"Frequency: {peak['frequency']:.2f} Hz | "
            f"Magnitude: {peak['magnitude']:.6f} | "
            f"dB: {peak['db']:.2f}"
        )

    print("==============================================")



# def test_t03_frequency_resolution():
#     """
#     T03: Test frequency resolution using two nearby
#     frequency components: 1000 Hz and 1010 Hz.

#     Expected:
#         - Frequency resolution decreases as FFT size increases.
#         - Both frequency components can be detected
#           with FFT size = 4096.
#     """

#     # -------------------------------------------------
#     # Configuration
#     # -------------------------------------------------

#     fs = 8000

#     fft_sizes = [512, 1024, 2048, 4096]

#     f1 = 1000
#     f2 = 1010

#     # -------------------------------------------------
#     # Step 1: Generate multi-sine signal
#     # -------------------------------------------------

#     x, fs = generate_multi_sine(
#         frequencies=[f1, f2],
#         amplitudes=[1.0, 1.0],
#         phases=[0.0, 0.0],
#         fs=fs,
#         duration=max(fft_sizes) / fs,
#     )

#     # Check generated signal
#     assert x.ndim == 1
#     assert len(x) == 4096
#     assert fs == 8000

#     # -------------------------------------------------
#     # Step 2: Analyze with different FFT sizes
#     # -------------------------------------------------

#     resolutions = {}
#     result_4096 = None

#     for fft_size in fft_sizes:

#         result = run_analysis(
#             x=x,
#             fs=fs,
#             fft_size=fft_size,
#             window_name="rectangular",
#         )

#         # Expected frequency resolution
#         expected_df = fs / fft_size

#         # Get frequency resolution from Analyzer
#         actual_df = result["frequency_resolution"]

#         # Check frequency resolution
#         assert actual_df == pytest.approx(expected_df)

#         resolutions[fft_size] = actual_df

#         # Save the largest FFT result
#         if fft_size == 4096:
#             result_4096 = result

#     # -------------------------------------------------
#     # Step 3: Check resolution improvement
#     # -------------------------------------------------

#     assert resolutions[512] > resolutions[1024]
#     assert resolutions[1024] > resolutions[2048]
#     assert resolutions[2048] > resolutions[4096]

#     # -------------------------------------------------
#     # Step 4: Check spectrum with FFT size = 4096
#     # -------------------------------------------------

#     assert result_4096 is not None

#     frequency = result_4096["frequency"]
#     magnitude = result_4096["magnitude"]

#     assert len(frequency) == 4096 // 2 + 1
#     assert len(magnitude) == 4096 // 2 + 1

#     # -------------------------------------------------
#     # Step 5: Detect frequency peaks
#     # -------------------------------------------------

#     peaks = find_peaks_from_result(
#         result_4096,
#         threshold_ratio=0.05,
#     )

#     df = fs / 4096

#     # Check 1000 Hz component
#     assert_peak_near(
#         peaks,
#         expected_frequency=f1,
#         tolerance=df,
#     )

#     # Check 1010 Hz component
#     assert_peak_near(
#         peaks,
#         expected_frequency=f2,
#         tolerance=df,
#     )

#     # -------------------------------------------------
#     # Step 6: Check frequency separation
#     # -------------------------------------------------

#     frequency_separation = abs(f2 - f1)

#     bin_separation = frequency_separation / df

#     assert bin_separation >= 2

#     # -------------------------------------------------
#     # Step 7: Print test results
#     # -------------------------------------------------

#     print("\n========== T03: FREQUENCY RESOLUTION ==========")

#     print(f"Input Frequencies: {f1} Hz and {f2} Hz")
#     print(f"Sampling Frequency: {fs} Hz")

#     print("\nFrequency Resolution Comparison:")
#     print(f"{'FFT Size':>12} | {'Resolution (Hz)':>18}")
#     print("-" * 35)

#     for size in fft_sizes:
#         print(
#             f"{size:12d} | "
#             f"{resolutions[size]:18.4f}"
#         )

#     print("\nPeak Detection Results (FFT Size = 4096):")

#     for peak in peaks:
#         print(
#             f"Frequency: {peak['frequency']:.2f} Hz | "
#             f"Magnitude: {peak['magnitude']:.6f} | "
#             f"dB: {peak['db']:.2f}"
#         )

#     print(f"\nFrequency Separation: {frequency_separation:.2f} Hz")
#     print(f"Bin Separation      : {bin_separation:.2f} bins")

#     print("\nExpected Frequencies:")
#     print(f"Component 1: {f1} Hz")
#     print(f"Component 2: {f2} Hz")

#     print("================================================")


def _calculate_leakage_ratio(magnitude, guard_bins=5):
    """
    Calculate the spectral energy ratio outside
    the main peak neighborhood.

    Parameters
    ----------
    magnitude : np.ndarray
        One-sided magnitude spectrum.

    guard_bins : int
        Number of bins excluded on each side
        of the dominant peak.

    Returns
    -------
    float
        Leakage ratio.
    """

    magnitude = np.asarray(magnitude, dtype=np.float64)

    if magnitude.ndim != 1 or magnitude.size == 0:
        raise ValueError("Magnitude must be a non-empty 1-D array")

    if not np.all(np.isfinite(magnitude)):
        raise ValueError("Magnitude contains NaN or Inf")

    if np.any(magnitude < 0):
        raise ValueError("Magnitude cannot be negative")

    if guard_bins < 0:
        raise ValueError("guard_bins must be non-negative")

    # Find dominant frequency bin
    peak_index = int(np.argmax(magnitude))

    # Define the main peak neighborhood
    start = max(0, peak_index - guard_bins)
    end = min(magnitude.size, peak_index + guard_bins + 1)

    # Create mask for bins outside the main peak
    mask = np.ones(magnitude.size, dtype=bool)
    mask[start:end] = False

    # Calculate total spectral energy
    total_energy = np.sum(magnitude ** 2)

    if total_energy <= 0:
        raise ValueError("Spectrum has zero energy")

    # Calculate leakage energy
    leakage_energy = np.sum(magnitude[mask] ** 2)

    return float(leakage_energy / total_energy)


def test_t03_frequency_resolution():
    """
    T03: Evaluate frequency resolution using two nearby
    frequency components: 1000 Hz and 1010 Hz.

    Expected:
        N = 512:  Only one frequency component is resolved.
        N = 1024: Only one frequency component is resolved.
        N = 2048: Both frequency components are resolved.
        N = 4096: Both frequency components are resolved.
    """

    # -------------------------------------------------
    # Configuration
    # -------------------------------------------------

    fs = 8000

    fft_sizes = [512, 1024, 2048, 4096]

    f1 = 1000
    f2 = 1010

    threshold_ratio = 0.05
    min_distance = 1

    # Two distinct peaks must be closer than this tolerance.
    # The tolerance is smaller than half the frequency separation.
    detection_tolerance = 2.0

    # Expected detection results
    expected_detection = {
        512: (True, False),
        1024: (True, False),
        2048: (True, True),
        4096: (True, True),
    }

    # -------------------------------------------------
    # Step 1: Generate multi-sine signal
    # -------------------------------------------------

    x, fs = generate_multi_sine(
        frequencies=[f1, f2],
        amplitudes=[1.0, 1.0],
        phases=[0.0, 0.0],
        fs=fs,
        duration=max(fft_sizes) / fs,
    )

    assert x.ndim == 1
    assert len(x) == 4096
    assert fs == 8000

    # -------------------------------------------------
    # Step 2: Analyze using different FFT sizes
    # -------------------------------------------------

    results = {}

    for fft_size in fft_sizes:

        result = run_analysis(
            x=x,
            fs=fs,
            fft_size=fft_size,
            window_name="rectangular",
        )

        results[fft_size] = result

        # Check frequency resolution
        expected_df = fs / fft_size

        assert result["frequency_resolution"] == pytest.approx(
            expected_df
        )

        # Check one-sided spectrum dimensions
        assert len(result["frequency"]) == fft_size // 2 + 1
        assert len(result["magnitude"]) == fft_size // 2 + 1

    # -------------------------------------------------
    # Step 3: Run Peak Detection for every FFT size
    # -------------------------------------------------

    detection_results = {}

    for fft_size in fft_sizes:

        result = results[fft_size]

        peaks = find_peaks_from_result(
            result,
            threshold_ratio=threshold_ratio,
            min_distance=min_distance,
        )

        # Check if 1000 Hz is detected
        detected_f1 = any(
            abs(peak["frequency"] - f1) <= detection_tolerance
            for peak in peaks
        )

        # Check if 1010 Hz is detected
        detected_f2 = any(
            abs(peak["frequency"] - f2) <= detection_tolerance
            for peak in peaks
        )

        detection_results[fft_size] = {
            "peaks": peaks,
            "detected_f1": detected_f1,
            "detected_f2": detected_f2,
        }

    # -------------------------------------------------
    # Step 4: Verify detection results
    # -------------------------------------------------

    for fft_size in fft_sizes:

        detected_f1 = detection_results[fft_size]["detected_f1"]
        detected_f2 = detection_results[fft_size]["detected_f2"]

        expected_f1, expected_f2 = expected_detection[fft_size]

        assert detected_f1 == expected_f1, (
            f"FFT Size {fft_size}: "
            f"Unexpected detection result for {f1} Hz"
        )

        assert detected_f2 == expected_f2, (
            f"FFT Size {fft_size}: "
            f"Unexpected detection result for {f2} Hz"
        )

    # -------------------------------------------------
    # Step 5: Check that frequency resolution improves
    # -------------------------------------------------

    resolutions = {
        fft_size: results[fft_size]["frequency_resolution"]
        for fft_size in fft_sizes
    }

    assert resolutions[512] > resolutions[1024]
    assert resolutions[1024] > resolutions[2048]
    assert resolutions[2048] > resolutions[4096]

    # -------------------------------------------------
    # Step 6: Print frequency resolution results
    # -------------------------------------------------

    print("\n========== T03: FREQUENCY RESOLUTION ==========")

    print(f"Input Frequencies: {f1} Hz and {f2} Hz")
    print(f"Sampling Frequency: {fs} Hz")

    print("\nResolution and Peak Detection:")
    print(
        f"{'FFT Size':>10} | "
        f"{'Resolution':>12} | "
        f"{'1000 Hz':>10} | "
        f"{'1010 Hz':>10} | "
        f"{'Peak Count':>10}"
    )

    print("-" * 65)

    for fft_size in fft_sizes:

        df = resolutions[fft_size]

        detected_f1 = detection_results[fft_size]["detected_f1"]
        detected_f2 = detection_results[fft_size]["detected_f2"]

        peaks = detection_results[fft_size]["peaks"]

        # Print the table
        print(
            f"{fft_size:10d} | "
            f"{df:9.4f} Hz | "
            f"{'YES' if detected_f1 else 'NO':>10} | "
            f"{'YES' if detected_f2 else 'NO':>10} | "
            f"{len(peaks):10d}"
        )

        # Print detected peak frequencies
        print("    Detected frequencies:", [
            round(peak["frequency"], 2)
            for peak in peaks
        ])

    print("\n===============================================")

def test_t04_spectral_leakage():
    """
    T04: Compare spectral leakage using
    Rectangular, Hann and Hamming windows.

    Expected:
        - All windows generate valid spectra.
        - Dominant frequency is near 1003 Hz.
        - Hann and Hamming reduce sidelobe energy
          compared with Rectangular Window.
    """

    # -------------------------------------------------
    # Configuration
    # -------------------------------------------------

    fs = 8000
    fft_size = 1024

    signal_frequency = 1003
    amplitude = 1.0

    window_types = [
        "rectangular",
        "hann",
        "hamming",
    ]

    # -------------------------------------------------
    # Step 1: Generate sine signal
    # -------------------------------------------------

    x, fs = generate_sine(
        frequency=signal_frequency,
        amplitude=amplitude,
        phase=0.0,
        fs=fs,
        duration=fft_size / fs,
    )

    assert x.ndim == 1
    assert len(x) == fft_size
    assert fs == 8000

    # -------------------------------------------------
    # Step 2: Analyze using different windows
    # -------------------------------------------------

    results = {}
    leakage_ratios = {}

    for window_name in window_types:

        result = run_analysis(
            x=x,
            fs=fs,
            fft_size=fft_size,
            window_name=window_name,
        )

        results[window_name] = result

        magnitude = result["magnitude"]
        frequency = result["frequency"]

        # -------------------------------------------------
        # Step 3: Check spectrum validity
        # -------------------------------------------------

        assert len(magnitude) == fft_size // 2 + 1
        assert np.all(np.isfinite(magnitude))
        assert np.all(magnitude >= 0)

        # -------------------------------------------------
        # Step 4: Check dominant frequency
        # -------------------------------------------------

        peak_index = int(np.argmax(magnitude))
        dominant_frequency = frequency[peak_index]

        df = fs / fft_size

        assert abs(
            dominant_frequency - signal_frequency
        ) <= df

        # -------------------------------------------------
        # Step 5: Calculate leakage ratio
        # -------------------------------------------------

        leakage_ratio = _calculate_leakage_ratio(
            magnitude,
            guard_bins=5,
        )

        leakage_ratios[window_name] = leakage_ratio

        assert 0 <= leakage_ratio <= 1

    # -------------------------------------------------
    # Step 6: Verify that windows are applied
    # -------------------------------------------------

    rectangular_signal = results["rectangular"]["windowed"]
    hann_signal = results["hann"]["windowed"]
    hamming_signal = results["hamming"]["windowed"]

    assert not np.allclose(
        rectangular_signal,
        hann_signal,
    )

    assert not np.allclose(
        rectangular_signal,
        hamming_signal,
    )

    # -------------------------------------------------
    # Step 7: Compare leakage ratios
    # -------------------------------------------------

    rectangular_leakage = leakage_ratios["rectangular"]
    hann_leakage = leakage_ratios["hann"]
    hamming_leakage = leakage_ratios["hamming"]

    assert hann_leakage < rectangular_leakage

    assert hamming_leakage < rectangular_leakage

    # -------------------------------------------------
    # Step 8: Print results
    # -------------------------------------------------

    print("\n========== T04: SPECTRAL LEAKAGE ==========")

    for window_name in window_types:
        print(
            f"{window_name:12s}: "
            f"Leakage Ratio = "
            f"{leakage_ratios[window_name]:.6f}"
        )

    print("===========================================")

def test_t05_square_wave_harmonics():
    """
    T05: Verify harmonic components of a 1 kHz square wave.

    Expected:
        - Fundamental frequency at 1000 Hz.
        - Third harmonic at 3000 Hz.
        - Even harmonic at 2000 Hz is suppressed.
        - Peak Detection identifies the fundamental
          and third harmonic.
    """

    # -------------------------------------------------
    # Configuration
    # -------------------------------------------------

    fs = 8000
    fft_size = 1024

    fundamental = 1000
    amplitude = 1.0

    # -------------------------------------------------
    # Step 1: Generate square wave
    # -------------------------------------------------

    x, fs = generate_square(
        frequency=fundamental,
        amplitude=amplitude,
        fs=fs,
        duration=fft_size / fs,
    )

    # Check generated signal
    assert x.ndim == 1
    assert len(x) == fft_size
    assert fs == 8000

    # Check square wave amplitude levels
    np.testing.assert_array_equal(
        np.unique(x),
        np.array([-amplitude, amplitude]),
    )

    # -------------------------------------------------
    # Step 2: Run complete analysis pipeline
    # -------------------------------------------------

    result = run_analysis(
        x=x,
        fs=fs,
        fft_size=fft_size,
        window_name="rectangular",
    )

    # -------------------------------------------------
    # Step 3: Get frequency and magnitude spectrum
    # -------------------------------------------------

    frequency = result["frequency"]
    magnitude = result["magnitude"]

    assert len(frequency) == fft_size // 2 + 1
    assert len(magnitude) == fft_size // 2 + 1

    # -------------------------------------------------
    # Step 4: Find harmonic frequency bins
    # -------------------------------------------------

    def get_magnitude_at(target_frequency):
        index = np.argmin(
            np.abs(frequency - target_frequency)
        )
        return magnitude[index]

    fundamental_mag = get_magnitude_at(1000)
    second_harmonic_mag = get_magnitude_at(2000)
    third_harmonic_mag = get_magnitude_at(3000)

    # -------------------------------------------------
    # Step 5: Check harmonic components
    # -------------------------------------------------

    # Fundamental component must exist
    assert fundamental_mag > 0

    # Third harmonic must exist
    assert third_harmonic_mag > 0

    # Second harmonic should be suppressed
    assert second_harmonic_mag < fundamental_mag * 1e-8

    # -------------------------------------------------
    # Step 6: Check frequency resolution and bin positions
    # -------------------------------------------------

    df = fs / fft_size

    expected_frequencies = [1000, 2000, 3000]

    for expected_frequency in expected_frequencies:

        index = np.argmin(
            np.abs(frequency - expected_frequency)
        )

        assert frequency[index] == pytest.approx(
            expected_frequency,
            abs=1e-10,
        )

    # -------------------------------------------------
    # Step 7: Run Peak Detection
    # -------------------------------------------------

    peaks = find_peaks_from_result(
        result,
        threshold_ratio=0.05,
    )

    # Check fundamental frequency
    assert_peak_near(
        peaks,
        expected_frequency=1000,
        tolerance=df,
    )

    # Check third harmonic
    assert_peak_near(
        peaks,
        expected_frequency=3000,
        tolerance=df,
    )

    # Check that the second harmonic is not detected
    assert not any(
        abs(peak["frequency"] - 2000) <= df
        for peak in peaks
    )

    # -------------------------------------------------
    # Step 8: Print results
    # -------------------------------------------------

    print("\n========== T05: HARMONIC ANALYSIS ==========")

    print(f"Fundamental (1000 Hz): {fundamental_mag:.8f}")
    print(f"Second Harmonic (2000 Hz): {second_harmonic_mag:.8e}")
    print(f"Third Harmonic (3000 Hz): {third_harmonic_mag:.8f}")

    print("\nDetected Peaks:")

    for peak in peaks:
        print(
            f"Frequency: {peak['frequency']:.2f} Hz | "
            f"Magnitude: {peak['magnitude']:.6f} | "
            f"dB: {peak['db']:.2f}"
        )

    print("============================================")

def test_t06_gaussian_noise():
    """
    T06: Analyze a sine signal with Gaussian noise.

    Expected:
        - Generated signal has a valid shape.
        - Actual SNR is close to 10 dB.
        - Spectrum contains the main frequency at 1000 Hz.
        - Peak Detection identifies the main frequency.
        - Spectrum outputs contain finite values.
    """

    # -------------------------------------------------
    # Configuration
    # -------------------------------------------------

    fs = 8000
    fft_size = 1024

    signal_frequency = 1000
    amplitude = 1.0
    target_snr_db = 10.0

    # -------------------------------------------------
    # Step 1: Generate clean sine signal
    # -------------------------------------------------

    x_clean, fs = generate_sine(
        frequency=signal_frequency,
        amplitude=amplitude,
        phase=0.0,
        fs=fs,
        duration=fft_size / fs,
    )

    # Check generated signal
    assert x_clean.ndim == 1
    assert len(x_clean) == fft_size
    assert fs == 8000

    # -------------------------------------------------
    # Step 2: Calculate clean signal power
    # -------------------------------------------------

    signal_power = np.mean(x_clean ** 2)

    assert signal_power > 0

    # -------------------------------------------------
    # Step 3: Calculate required noise power
    # -------------------------------------------------

    noise_power = signal_power / (
        10 ** (target_snr_db / 10)
    )

    noise_std = np.sqrt(noise_power)

    # -------------------------------------------------
    # Step 4: Generate reproducible Gaussian noise
    # -------------------------------------------------

    rng = np.random.default_rng(seed=2026)

    noise = rng.normal(
        loc=0.0,
        scale=noise_std,
        size=len(x_clean),
    )

    assert noise.shape == x_clean.shape
    assert np.all(np.isfinite(noise))

    # -------------------------------------------------
    # Step 5: Calculate actual SNR
    # -------------------------------------------------

    actual_noise_power = np.mean(noise ** 2)

    actual_snr_db = 10 * np.log10(
        signal_power / actual_noise_power
    )

    # Allow small variation due to finite noise samples
    assert actual_snr_db == pytest.approx(
        target_snr_db,
        abs=1.0,
    )

    # -------------------------------------------------
    # Step 6: Create noisy signal
    # -------------------------------------------------

    x_noisy = x_clean + noise

    assert x_noisy.shape == x_clean.shape
    assert np.all(np.isfinite(x_noisy))
    assert not np.array_equal(x_noisy, x_clean)

    # -------------------------------------------------
    # Step 7: Run complete analysis pipeline
    # -------------------------------------------------

    result = run_analysis(
        x=x_noisy,
        fs=fs,
        fft_size=fft_size,
        window_name="rectangular",
    )

    # -------------------------------------------------
    # Step 8: Check spectrum validity
    # -------------------------------------------------

    frequency = result["frequency"]
    magnitude = result["magnitude"]
    db = result["db"]

    assert np.all(np.isfinite(magnitude))
    assert np.all(np.isfinite(db))

    # -------------------------------------------------
    # Step 9: Check dominant frequency
    # -------------------------------------------------

    peak_index = int(np.argmax(magnitude))

    dominant_frequency = frequency[peak_index]

    df = fs / fft_size

    assert dominant_frequency == pytest.approx(
        signal_frequency,
        abs=df,
    )

    # -------------------------------------------------
    # Step 10: Run Peak Detection
    # -------------------------------------------------

    peaks = find_peaks_from_result(
        result,
        threshold_ratio=0.15,
    )

    # Verify that 1000 Hz is detected
    assert_peak_near(
        peaks,
        expected_frequency=signal_frequency,
        tolerance=df,
    )

    # -------------------------------------------------
    # Step 11: Check peak information
    # -------------------------------------------------

    main_peak = min(
        peaks,
        key=lambda peak: abs(
            peak["frequency"] - signal_frequency
        ),
    )

    assert np.isfinite(main_peak["frequency"])
    assert np.isfinite(main_peak["magnitude"])
    assert np.isfinite(main_peak["db"])

    # -------------------------------------------------
    # Step 12: Print test results
    # -------------------------------------------------

    print("\n========== T06: GAUSSIAN NOISE ==========")

    print(f"Target SNR: {target_snr_db:.2f} dB")
    print(f"Actual SNR: {actual_snr_db:.2f} dB")
    print(f"Signal Power: {signal_power:.6f}")
    print(f"Noise Power: {actual_noise_power:.6f}")

    print(f"Dominant Frequency: {dominant_frequency:.2f} Hz")

    print("\nDetected Peaks:")

    for peak in peaks:
        print(
            f"Frequency: {peak['frequency']:.2f} Hz | "
            f"Magnitude: {peak['magnitude']:.6f} | "
            f"dB: {peak['db']:.2f}"
        )

    print("========================================")

def test_t07_wav_integration():
    """
    T07: Test complete integration using real WAV files.

    Flow:
        WAV File
            -> WAV Loader
            -> Spectrum Analyzer
            -> Spectrum Calculation
            -> Peak Detection
            -> Visualization

    Expected:
        - WAV files are loaded successfully.
        - Sampling frequency is preserved.
        - Analyzer generates valid spectrum data.
        - Peak Detection returns valid results.
        - Magnitude and dB plots are rendered successfully.
    """

    # -------------------------------------------------
    # Step 1: Find WAV files in data directory
    # -------------------------------------------------

    wav_files = sorted(
        path
        for path in DATA_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() == ".wav"
    ) if DATA_DIR.exists() else []

    # Skip test if no WAV files are available
    if not wav_files:
        pytest.skip(
            "No WAV files found in data/ directory"
        )

    # -------------------------------------------------
    # Step 2: Process each WAV file
    # -------------------------------------------------

    for wav_path in wav_files:

        print(f"\nTesting WAV file: {wav_path.name}")

        # -------------------------------------------------
        # Step 2.1: Load WAV file
        # -------------------------------------------------

        x, fs = load_wav(str(wav_path))

        # Check loaded signal
        assert isinstance(x, np.ndarray)
        assert x.ndim == 1
        assert x.dtype == np.float64
        assert len(x) > 0
        assert fs > 0

        assert np.all(np.isfinite(x))
        assert np.max(np.abs(x)) > 0

        # -------------------------------------------------
        # Step 2.2: Run complete analysis
        # -------------------------------------------------

        fft_size = 1024

        result = run_analysis(
            x=x,
            fs=fs,
            fft_size=fft_size,
            window_name="hamming",
        )

        # -------------------------------------------------
        # Step 2.3: Check analysis information
        # -------------------------------------------------

        assert result["fs"] == pytest.approx(fs)
        assert result["fft_size"] == fft_size
        assert result["window"] == "hamming"

        assert result["frequency_resolution"] == pytest.approx(
            fs / fft_size
        )

        # Check output dimensions
        assert len(result["fft"]) == fft_size

        assert len(result["frequency"]) == fft_size // 2 + 1
        assert len(result["magnitude"]) == fft_size // 2 + 1
        assert len(result["db"]) == fft_size // 2 + 1

        # Check spectrum values
        assert np.all(np.isfinite(result["magnitude"]))
        assert np.all(np.isfinite(result["db"]))

        # -------------------------------------------------
        # Step 3: Run Peak Detection
        # -------------------------------------------------

        peaks = find_peaks_from_result(
            result,
            threshold_ratio=0.05,
        )

        assert isinstance(peaks, list)

        # Check peak information
        for peak in peaks:

            assert "frequency" in peak
            assert "magnitude" in peak
            assert "db" in peak
            assert "index" in peak

            assert np.isfinite(peak["frequency"])
            assert np.isfinite(peak["magnitude"])
            assert np.isfinite(peak["db"])

            assert 0 <= peak["frequency"] <= fs / 2
            assert peak["magnitude"] >= 0

        # -------------------------------------------------
        # Step 4: Test Spectrum Visualization
        # -------------------------------------------------

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(10, 7),
        )

        try:
            # Magnitude Spectrum
            plot_spectrum(
                axes[0],
                result["frequency"],
                result["magnitude"],
                peaks,
            )

            # dB Spectrum
            plot_db_spectrum(
                axes[1],
                result["frequency"],
                result["db"],
                peaks,
            )

            # Render plots using non-GUI backend
            fig.canvas.draw()

            # Verify that plots contain data
            assert axes[0].has_data()
            assert axes[1].has_data()

        finally:
            plt.close(fig)

        # -------------------------------------------------
        # Step 5: Print test results
        # -------------------------------------------------

        print(f"Sampling Frequency: {fs} Hz")
        print(f"Number of Samples: {len(x)}")
        print(f"Duration: {len(x) / fs:.3f} seconds")
        print(f"FFT Size: {fft_size}")
        print(
            f"Frequency Resolution: "
            f"{result['frequency_resolution']:.4f} Hz"
        )
        print(f"Detected Peaks: {len(peaks)}")

        print("-------------------------------------------")
