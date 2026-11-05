import numpy as np
import pytest

from src.dsp.dft import dft


def test_dft_impulse():
    x = np.array([1, 0, 0, 0])
    expected = np.array([1, 1, 1, 1], dtype=np.complex128)
    np.testing.assert_allclose(dft(x), expected, rtol=1e-12, atol=1e-12)


def test_dft_known_four_point_result():
    x = np.array([1, 2, 3, 4])
    expected = np.array([10, -2 + 2j, -2, -2 - 2j], dtype=np.complex128)
    np.testing.assert_allclose(dft(x), expected, rtol=1e-12, atol=1e-12)


def test_dft_matches_numpy_for_real_input():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(17)
    np.testing.assert_allclose(dft(x), np.fft.fft(x), rtol=1e-10, atol=1e-10)


def test_dft_matches_numpy_for_complex_input():
    rng = np.random.default_rng(7)
    x = rng.standard_normal(9) + 1j * rng.standard_normal(9)
    np.testing.assert_allclose(dft(x), np.fft.fft(x), rtol=1e-10, atol=1e-10)


def test_dft_rejects_empty_input():
    with pytest.raises(ValueError, match="must not be empty"):
        dft(np.array([]))


def test_dft_rejects_two_dimensional_input():
    with pytest.raises(ValueError, match="1-D"):
        dft(np.array([[1, 2], [3, 4]]))