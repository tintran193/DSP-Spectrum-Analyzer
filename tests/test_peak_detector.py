import numpy as np
import pytest

from src.dsp.peak_detector import detect_peaks


def test_detect_two_peaks():

    frequency = np.arange(10) * 100.0

    magnitude = np.array([
        0.0,
        0.1,
        0.8,
        0.2,
        0.1,
        0.7,
        0.1,
        0.2,
        0.1,
        0.0
    ])

    peaks = detect_peaks(
        frequency,
        magnitude,
        threshold=0.3,
        min_distance=2
    )

    assert len(peaks) == 2

    assert peaks[0]["frequency"] == 200.0
    assert peaks[1]["frequency"] == 500.0


def test_peak_output_format():

    frequency = np.arange(7, dtype=float)

    magnitude = np.array([
        0.0,
        0.1,
        0.8,
        0.1,
        0.0,
        0.7,
        0.0
    ])

    peaks = detect_peaks(
        frequency,
        magnitude,
        threshold=0.2,
        min_distance=2
    )

    expected_keys = {
        "frequency",
        "magnitude",
        "db",
        "index"
    }

    for peak in peaks:

        assert set(peak.keys()) == expected_keys


def test_peaks_sorted_by_magnitude():

    frequency = np.arange(10, dtype=float)

    magnitude = np.array([
        0.0,
        0.4,
        0.1,
        0.0,
        0.9,
        0.1,
        0.0,
        0.7,
        0.1,
        0.0
    ])

    peaks = detect_peaks(
        frequency,
        magnitude,
        threshold=0.2,
        min_distance=2
    )

    assert (
        peaks[0]["magnitude"]
        >= peaks[1]["magnitude"]
    )


def test_no_peak_detected():

    frequency = np.arange(10, dtype=float)

    magnitude = np.zeros(10)

    peaks = detect_peaks(
        frequency,
        magnitude,
        threshold=0.1,
        min_distance=2
    )

    assert peaks == []


def test_reject_different_lengths():

    with pytest.raises(ValueError):

        detect_peaks(
            np.array([0, 1, 2]),
            np.array([0, 1]),
            threshold=0.1,
            min_distance=1
        )