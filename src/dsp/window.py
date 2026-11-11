"""
window.py - Tạo và áp dụng cửa sổ (window) cho frame trước khi FFT.

Tên chuẩn: "rectangular", "hamming", "hann".
Quy ước: window đối xứng, w[n] = a - (1 - a) * cos(2*pi*n / (N - 1)), n = 0..N-1
    - hann   : a = 0.5   (hai đầu = 0)
    - hamming: a = 0.54  (hai đầu = 0.08)
    - rectangular: w[n] = 1
"""
from __future__ import annotations

import numpy as np

VALID_WINDOWS = ("rectangular", "hamming", "hann")


def get_window(name: str, N: int) -> np.ndarray:
    """
    Trả về window độ dài N (mảng float64 1 chiều).

    Raises:
        ValueError: nếu tên window không hợp lệ hoặc N không phải số nguyên dương.
    """
    if isinstance(N, bool) or not isinstance(N, (int, np.integer)) or N < 1:
        raise ValueError(f"N phải là số nguyên dương, nhận được: {N!r}")
    if not isinstance(name, str):
        raise ValueError(f"name phải là chuỗi, nhận được: {name!r}")

    key = name.strip().lower()
    if key not in VALID_WINDOWS:
        raise ValueError(f"Window '{name}' không hỗ trợ. Chọn một trong {VALID_WINDOWS}")

    if key == "rectangular" or N == 1:
        return np.ones(int(N), dtype=np.float64)

    n = np.arange(int(N), dtype=np.float64)
    a = 0.5 if key == "hann" else 0.54
    return a - (1.0 - a) * np.cos(2.0 * np.pi * n / (N - 1))


def apply_window(x: np.ndarray, window: np.ndarray) -> np.ndarray:
    """
    Nhân từng phần tử frame x với window (cùng độ dài). Không sửa x gốc.

    Raises:
        ValueError: nếu x hoặc window không phải mảng 1 chiều, hoặc độ dài khác nhau.
    """
    x = np.asarray(x, dtype=np.float64)
    window = np.asarray(window, dtype=np.float64)
    if x.ndim != 1 or window.ndim != 1:
        raise ValueError("x và window phải là mảng 1 chiều")
    if len(x) != len(window):
        raise ValueError(f"Độ dài không khớp: x có {len(x)} mẫu, window có {len(window)} mẫu")
    return x * window


def coherent_gain(window: np.ndarray) -> float:
    """
    Hệ số giảm biên độ trung bình = sum(w)/N (Rect = 1, Hann ~ 0.5, Hamming ~ 0.54).
    Dùng để hiệu chỉnh biên độ phổ: biên_độ = 2*|X[k]| / sum(w).
    """
    window = np.asarray(window, dtype=np.float64)
    return float(np.sum(window) / len(window))


if __name__ == "__main__":
    for nm in VALID_WINDOWS:
        w = get_window(nm, 8)
        print(f"{nm:12s} len={len(w)}  w[0]={w[0]:.3f}  max={w.max():.3f}  gain={coherent_gain(w):.3f}")