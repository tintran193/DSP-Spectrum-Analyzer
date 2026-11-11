"""Test E: window. Chạy: python -m pytest tests -v"""
import numpy as np
import pytest

from src.dsp.window import get_window, apply_window, coherent_gain

NAMES = ["rectangular", "hamming", "hann"]


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("N", [1, 2, 8, 255, 1024])
def test_length_and_dtype(name, N):
    w = get_window(name, N)
    assert w.shape == (N,)
    assert w.dtype == np.float64


def test_rectangular_all_ones():
    assert np.all(get_window("rectangular", 64) == 1.0)


def test_hann_endpoints_zero_and_center_one():
    w = get_window("hann", 1025)          # N lẻ -> có mẫu giữa đúng bằng 1
    assert w[0] == pytest.approx(0.0, abs=1e-12)
    assert w[-1] == pytest.approx(0.0, abs=1e-12)
    assert w[512] == pytest.approx(1.0)
    assert w.max() <= 1.0 + 1e-12


def test_hamming_endpoints_and_center():
    w = get_window("hamming", 1025)
    assert w[0] == pytest.approx(0.08)
    assert w[-1] == pytest.approx(0.08)
    assert w[512] == pytest.approx(1.0)


def test_known_values_n5():
    assert np.allclose(get_window("hann", 5), [0.0, 0.5, 1.0, 0.5, 0.0])
    assert np.allclose(get_window("hamming", 5), [0.08, 0.54, 1.0, 0.54, 0.08])


@pytest.mark.parametrize("name", NAMES)
def test_symmetric(name):
    w = get_window(name, 100)
    assert np.allclose(w, w[::-1])


@pytest.mark.parametrize("name,expected", [("rectangular", 1.0), ("hann", 0.5), ("hamming", 0.54)])
def test_coherent_gain(name, expected):
    assert coherent_gain(get_window(name, 4096)) == pytest.approx(expected, abs=1e-3)


def test_name_case_insensitive():
    assert np.allclose(get_window("HANN", 16), get_window("hann", 16))
    assert np.allclose(get_window("  Hamming ", 16), get_window("hamming", 16))


@pytest.mark.parametrize("bad_name", ["blackman", "", "hanning"])
def test_invalid_name(bad_name):
    with pytest.raises(ValueError):
        get_window(bad_name, 16)


@pytest.mark.parametrize("bad_N", [0, -5, 3.5, True])
def test_invalid_N(bad_N):
    with pytest.raises(ValueError):
        get_window("hann", bad_N)


def test_apply_window_values():
    x = np.ones(16)
    w = get_window("hann", 16)
    assert np.allclose(apply_window(x, w), w)


def test_apply_window_does_not_modify_input():
    x = np.arange(16, dtype=float)
    x_copy = x.copy()
    apply_window(x, get_window("hamming", 16))
    assert np.array_equal(x, x_copy)


def test_apply_window_length_mismatch():
    with pytest.raises(ValueError):
        apply_window(np.ones(10), get_window("hann", 12))


def test_apply_window_rejects_2d():
    with pytest.raises(ValueError):
        apply_window(np.ones((4, 4)), np.ones(4))