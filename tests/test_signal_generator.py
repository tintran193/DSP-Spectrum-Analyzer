"""Test A, B, C. Chạy: python -m pytest tests -v"""
import numpy as np
import pytest

from src.input.signal_generator import generate_sine, generate_multi_sine, generate_square

FS = 8000.0


def amp_spectrum(x):
    """Phổ biên độ một phía, sin biên độ A cho vạch cao A (khi khớp bin, không window)."""
    return 2.0 * np.abs(np.fft.rfft(x)) / len(x)


# ---------------- Test A: sine ----------------
def test_a_shape_type_and_fs():
    x, fs = generate_sine(1000, 1.0, 0.0, FS, 1.0)
    assert fs == FS
    assert x.ndim == 1 and x.dtype == np.float64
    assert len(x) == 8000


def test_a_waveform_values_and_periodicity():
    x, _ = generate_sine(1000, 2.0, 0.0, FS, 1.0)       # chu kỳ = 8 mẫu
    assert np.allclose(x[:8], 2.0 * np.sin(2 * np.pi * np.arange(8) / 8))
    assert np.allclose(x[:8], x[8:16])
    assert x.max() == pytest.approx(2.0)
    assert x.min() == pytest.approx(-2.0)


def test_a_phase():
    x, _ = generate_sine(1000, 1.0, np.pi / 2, FS, 1.0)
    assert x[0] == pytest.approx(1.0)


def test_a_spectrum_peak_at_1khz():
    x, _ = generate_sine(1000, 1.0, 0.0, FS, 1.0)       # N = 8000 -> Δf = 1 Hz
    X = amp_spectrum(x)
    assert np.argmax(X) == 1000
    assert X[1000] == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("kwargs", [
    dict(frequency=4000, fs=8000, duration=1),     # = Nyquist
    dict(frequency=5000, fs=8000, duration=1),     # > Nyquist
    dict(frequency=0, fs=8000, duration=1),
    dict(frequency=-10, fs=8000, duration=1),
    dict(frequency=1000, fs=0, duration=1),
    dict(frequency=1000, fs=8000, duration=0),
    dict(frequency=float("nan"), fs=8000, duration=1),
    dict(frequency=1000, fs=8000, duration=1e-9),  # chưa tới 1 mẫu
])
def test_a_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        generate_sine(amplitude=1.0, phase=0.0, **kwargs)


# ---------------- Test B: multi-sine ----------------
def test_b_two_tones_spectrum():
    x, fs = generate_multi_sine([1000, 3000], [1.0, 0.5], [0.0, 0.0], FS, 1.0)
    assert fs == FS and x.ndim == 1 and len(x) == 8000
    X = amp_spectrum(x)
    assert X[1000] == pytest.approx(1.0, abs=1e-9)
    assert X[3000] == pytest.approx(0.5, abs=1e-9)
    mask = np.ones(len(X), dtype=bool)
    mask[[1000, 3000]] = False
    assert X[mask].max() < 1e-9


def test_b_equals_sum_of_single_sines():
    x, _ = generate_multi_sine([1000, 3000], [1.0, 0.5], [0.0, 0.3], FS, 0.1)
    s1, _ = generate_sine(1000, 1.0, 0.0, FS, 0.1)
    s2, _ = generate_sine(3000, 0.5, 0.3, FS, 0.1)
    assert np.allclose(x, s1 + s2)


def test_b_invalid_arguments():
    with pytest.raises(ValueError):
        generate_multi_sine([1000, 3000], [1.0], [0.0, 0.0], FS, 1.0)   # lệch độ dài
    with pytest.raises(ValueError):
        generate_multi_sine([], [], [], FS, 1.0)                         # rỗng
    with pytest.raises(ValueError):
        generate_multi_sine([1000, 4500], [1, 1], [0, 0], FS, 1.0)       # 4500 > Nyquist


# ---------------- Test C: square ----------------
def test_c_two_levels_and_period():
    x, fs = generate_square(1000, 1.5, FS, 1.0)
    assert fs == FS and x.ndim == 1 and len(x) == 8000
    assert set(np.unique(x)) == {-1.5, 1.5}
    assert np.all(x[:4] == 1.5) and np.all(x[4:8] == -1.5)   # chu kỳ 8 mẫu, duty 50%
    assert abs(np.mean(x)) < 1e-12


def test_c_odd_harmonics_only():
    x, _ = generate_square(1000, 1.0, 48000.0, 1.0)          # N = 48000 -> Δf = 1 Hz
    X = amp_spectrum(x)
    assert X[1000] == pytest.approx(4 / np.pi, rel=0.02)     # cơ bản ≈ 4A/π
    assert X[3000] == pytest.approx(X[1000] / 3, rel=0.05)   # hài 3 ≈ 1/3
    assert X[2000] < 1e-9 and X[4000] < 1e-9                 # hài chẵn = 0


def test_c_invalid_frequency():
    with pytest.raises(ValueError):
        generate_square(4000, 1.0, 8000.0, 1.0)



def test_n_samples_instead_of_duration_sine():
    x, fs = generate_sine(1000, 1.0, 0.0, FS, n_samples=1234)
    assert len(x) == 1234 and fs == FS
    y, _ = generate_sine(1000, 1.0, 0.0, FS, 1.0)
    assert np.allclose(x, y[:1234])                      # cùng tín hiệu, chỉ khác độ dài


def test_n_samples_multi_sine_and_square():
    x, _ = generate_multi_sine([1000, 3000], [1.0, 0.5], [0.0, 0.0], FS, n_samples=1000)
    assert len(x) == 1000
    s, _ = generate_square(1000, 1.0, FS, n_samples=777)
    assert len(s) == 777 and set(np.unique(s)) == {-1.0, 1.0}


def test_duration_positional_still_works():
    x, _ = generate_sine(1000, 1.0, 0.0, FS, 0.25)       # đúng API 5.3
    assert len(x) == 2000


@pytest.mark.parametrize("kwargs", [
    dict(),                                  # không nhập gì
    dict(duration=1.0, n_samples=100),       # nhập cả hai
    dict(n_samples=0),
    dict(n_samples=-5),
    dict(n_samples=2.5),
    dict(n_samples=True),
])
def test_length_arguments_invalid(kwargs):
    with pytest.raises(ValueError):
        generate_sine(1000, 1.0, 0.0, FS, **kwargs)


def test_b_three_components_spectrum():
    
    x, _ = generate_multi_sine([500, 1500, 2500], [1.0, 0.5, 0.25], [0.0, 0.0, 0.0], FS, 1.0)
    X = amp_spectrum(x)
    assert X[500] == pytest.approx(1.0, abs=1e-9)
    assert X[1500] == pytest.approx(0.5, abs=1e-9)
    assert X[2500] == pytest.approx(0.25, abs=1e-9)
    mask = np.ones(len(X), dtype=bool)
    mask[[500, 1500, 2500]] = False
    assert X[mask].max() < 1e-9


def test_b_n_samples_and_phase():
    x, _ = generate_multi_sine([1000, 3000], [1.0, 0.5], [np.pi / 2, 0.0], FS, n_samples=64)
    s1, _ = generate_sine(1000, 1.0, np.pi / 2, FS, n_samples=64)
    s2, _ = generate_sine(3000, 0.5, 0.0, FS, n_samples=64)
    assert len(x) == 64 and np.allclose(x, s1 + s2)