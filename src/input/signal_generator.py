"""
signal_generator.py - Tạo tín hiệu thử: sine, multi-sine, square.

Mọi hàm đều trả về (x, fs):
    x  : np.ndarray 1 chiều, float64
    fs : float, tần số lấy mẫu (Hz)

Quy ước:
    - Độ dài: nhập duration (giây) HOẶC n_samples (số mẫu), đúng một trong hai.
      Với duration: N = round(fs * duration). Mẫu thứ n ứng với thời điểm t = n / fs.
    - Tần số phải thoả 0 < f < fs/2 (Nyquist). Vi phạm sẽ ném ValueError
      để tránh aliasing "âm thầm".
    - Square bắt đầu bằng nửa chu kỳ dương (+A), duty 50%.
"""
from __future__ import annotations

import numpy as np


def _as_finite_float(value, name: str) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} phải là số, nhận được: {value!r}") from None
    if not np.isfinite(v):
        raise ValueError(f"{name} phải hữu hạn, nhận được: {value!r}")
    return v


def _sample_index(fs, duration=None, n_samples=None) -> tuple[np.ndarray, float]:
    """
    Kiểm tra fs và độ dài; trả về (chỉ số mẫu n = 0..N-1, fs dạng float).
    Phải nhập ĐÚNG MỘT trong hai: duration (giây) hoặc n_samples (số mẫu).
    """
    fs = _as_finite_float(fs, "fs")
    if fs <= 0:
        raise ValueError(f"fs phải > 0, nhận được: {fs}")
    if duration is None and n_samples is None:
        raise ValueError("Cần nhập duration (giây) hoặc n_samples (số mẫu)")
    if duration is not None and n_samples is not None:
        raise ValueError("Chỉ nhập một trong hai: duration hoặc n_samples, không nhập cả hai")

    if n_samples is not None:
        if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)):
            raise ValueError(f"n_samples phải là số nguyên, nhận được: {n_samples!r}")
        if n_samples < 1:
            raise ValueError(f"n_samples phải >= 1, nhận được: {n_samples}")
        count = int(n_samples)
    else:
        duration = _as_finite_float(duration, "duration")
        if duration <= 0:
            raise ValueError(f"duration phải > 0, nhận được: {duration}")
        count = int(round(fs * duration))
        if count < 1:
            raise ValueError(f"fs*duration = {fs * duration} cho ra chưa tới 1 mẫu")
    return np.arange(count, dtype=np.float64), fs


def _check_frequency(frequency, fs: float) -> float:
    f = _as_finite_float(frequency, "frequency")
    if not (0.0 < f < fs / 2.0):
        raise ValueError(
            f"frequency phải thoả 0 < f < fs/2 = {fs / 2.0} Hz (Nyquist), nhận được: {f} Hz"
        )
    return f


def generate_sine(
    frequency: float,
    amplitude: float,
    phase: float,
    fs: float,
    duration: float | None = None,
    *,
    n_samples: int | None = None,
) -> tuple[np.ndarray, float]:
    """x[n] = amplitude * sin(2*pi*frequency*n/fs + phase)  (phase tính bằng radian)."""
    n, fs = _sample_index(fs, duration, n_samples)
    f = _check_frequency(frequency, fs)
    a = _as_finite_float(amplitude, "amplitude")
    p = _as_finite_float(phase, "phase")
    x = a * np.sin(2.0 * np.pi * f * n / fs + p)
    return x, fs


def generate_multi_sine(
    frequencies: list[float],
    amplitudes: list[float],
    phases: list[float],
    fs: float,
    duration: float | None = None,
    *,
    n_samples: int | None = None,
) -> tuple[np.ndarray, float]:
    """Tổng của nhiều sine. Ba danh sách phải cùng độ dài và không rỗng."""
    n, fs = _sample_index(fs, duration, n_samples)
    if not (len(frequencies) == len(amplitudes) == len(phases)):
        raise ValueError(
            "frequencies, amplitudes, phases phải cùng độ dài, nhận được "
            f"{len(frequencies)}, {len(amplitudes)}, {len(phases)}"
        )
    if len(frequencies) == 0:
        raise ValueError("Cần ít nhất một thành phần sine")

    x = np.zeros_like(n)
    for f, a, p in zip(frequencies, amplitudes, phases):
        f = _check_frequency(f, fs)
        a = _as_finite_float(a, "amplitude")
        p = _as_finite_float(p, "phase")
        x += a * np.sin(2.0 * np.pi * f * n / fs + p)
    return x, fs


def generate_square(
    frequency: float,
    amplitude: float,
    fs: float,
    duration: float | None = None,
    *,
    n_samples: int | None = None,
) -> tuple[np.ndarray, float]:
    """
    Sóng vuông duty 50%: +A trong nửa chu kỳ đầu, -A trong nửa chu kỳ sau.

    Lưu ý: sóng vuông không giới hạn băng tần (có vô số hài lẻ 3f, 5f, ...),
    các hài vượt quá fs/2 sẽ bị aliasing trong phổ. Đó là hiện tượng đúng.
    """
    n, fs = _sample_index(fs, duration, n_samples)
    f = _check_frequency(frequency, fs)
    a = _as_finite_float(amplitude, "amplitude")
    # (n * f) / fs: với f nguyên, n*f chính xác nên ranh giới nửa chu kỳ không bị lệch do số học.
    phase_frac = np.mod(n * f / fs, 1.0)
    x = np.where(phase_frac < 0.5, a, -a).astype(np.float64)
    return x, fs