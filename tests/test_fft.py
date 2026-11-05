import numpy as np
import pytest

from src.dsp.dft import dft
from src.dsp.fft import fft
from src.dsp.validation import compare_dft_fft, compare_fft_numpy


def test_fft_one_point():
    x = np.array([5.0])
    np.testing.assert_allclose(fft(x), np.array([5.0 + 0j]))


def test_fft_four_point_impulse():
    x = np.array([1, 0, 0, 0])
    expected = np.array([1, 1, 1, 1], dtype=np.complex128)
    np.testing.assert_allclose(fft(x), expected, rtol=1e-12, atol=1e-12)


def test_fft_four_point_known_result():
    x = np.array([1, 2, 3, 4])
    expected = np.array([10, -2 + 2j, -2, -2 - 2j], dtype=np.complex128)
    np.testing.assert_allclose(fft(x), expected, rtol=1e-12, atol=1e-12)


def test_fft_eight_point_matches_dft():
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    np.testing.assert_allclose(fft(x), dft(x), rtol=1e-10, atol=1e-10)


def test_fft_matches_numpy_for_multiple_power_of_two_sizes():
    rng = np.random.default_rng(123)
    for length in (2, 4, 8, 16, 64, 128, 1024):
        x = rng.standard_normal(length)
        np.testing.assert_allclose(
            fft(x), np.fft.fft(x), rtol=1e-10, atol=1e-10
        )


def test_fft_supports_complex_input():
    x = np.array([1 + 1j, 2 - 1j, 3 + 2j, 4 - 2j])
    np.testing.assert_allclose(
        fft(x), np.fft.fft(x), rtol=1e-10, atol=1e-10
    )


def test_fft_rejects_non_power_of_two_length():
    with pytest.raises(ValueError, match="power of 2"):
        fft(np.array([1, 2, 3, 4, 5, 6]))


def test_fft_rejects_empty_input():
    with pytest.raises(ValueError, match="power of 2"):
        fft(np.array([]))


def test_fft_rejects_two_dimensional_input():
    with pytest.raises(ValueError, match="1-D"):
        fft(np.array([[1, 2], [3, 4]]))


def test_validation_helpers_return_small_error():
    rng = np.random.default_rng(10)
    x = rng.standard_normal(128)
    assert compare_dft_fft(x) < 1e-9
    assert compare_fft_numpy(x) < 1e-9


def test_sine_one_khz_peak_bin():
    fs = 8000
    length = 1024
    frequency = 1000
    n = np.arange(length)
    x = np.sin(2 * np.pi * frequency * n / fs)

    spectrum = np.abs(fft(x)[: length // 2 + 1])
    peak_bin = int(np.argmax(spectrum))

    assert peak_bin == 128
    assert peak_bin * fs / length == pytest.approx(1000.0)