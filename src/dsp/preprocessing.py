"""
preprocessing.py - Chuẩn hoá, cắt frame, zero padding.

QUY ƯỚC CHUNG CỦA NHÓM (ghi vào README):
    - Mọi hàm nhận/trả mảng NumPy 1 chiều, float64.
    - Zero padding luôn thêm số 0 vào CUỐI frame (không thêm vào đầu hay giữa).
    - Normalize theo đỉnh: chia cho max|x| để đỉnh về đúng 1 (không bỏ DC, không đổi hình dạng).
    - Các hàm không sửa mảng đầu vào; luôn trả về mảng mới.
"""
from __future__ import annotations

import numpy as np


def _as_1d_float(x, name: str = "x") -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} phải là mảng 1 chiều, nhận được shape {arr.shape}")
    return arr


def _as_int(value, name: str, min_value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} phải là số nguyên, nhận được: {value!r}")
    if value < min_value:
        raise ValueError(f"{name} phải >= {min_value}, nhận được: {value}")
    return int(value)


def normalize_signal(x: np.ndarray) -> np.ndarray:
    """
    Chuẩn hoá theo đỉnh: y = x / max|x|  =>  max|y| = 1.
    Tín hiệu toàn 0 được trả về nguyên (không chia cho 0).

    Raises:
        ValueError: x rỗng, không phải 1 chiều, hoặc chứa NaN/Inf.
    """
    arr = _as_1d_float(x)
    if arr.size == 0:
        raise ValueError("x rỗng")
    if not np.all(np.isfinite(arr)):
        raise ValueError("x chứa NaN hoặc Inf")
    peak = np.max(np.abs(arr))
    if peak == 0.0:
        return arr.copy()
    return arr / peak


def zero_pad(x: np.ndarray, target_size: int) -> np.ndarray:
    """
    Thêm số 0 vào cuối x cho đủ target_size mẫu.
    Nếu x đã đủ độ dài thì trả về bản sao. Nếu x dài hơn target_size thì báo lỗi
    (không cắt âm thầm; muốn cắt hãy dùng extract_frame).

    Raises:
        ValueError: target_size không hợp lệ hoặc nhỏ hơn len(x).
    """
    arr = _as_1d_float(x)
    target = _as_int(target_size, "target_size", 1)
    if len(arr) > target:
        raise ValueError(f"x có {len(arr)} mẫu, dài hơn target_size = {target}")
    out = np.zeros(target, dtype=np.float64)
    out[: len(arr)] = arr
    return out


def extract_frame(x: np.ndarray, start: int, size: int) -> np.ndarray:
    """
    Lấy đúng `size` mẫu bắt đầu từ chỉ số `start`: x[start : start + size].
    Nếu gần cuối tín hiệu không đủ mẫu thì phần thiếu được zero-pad vào CUỐI.
    Luôn trả về mảng mới độ dài `size`.

    Raises:
        ValueError: start < 0, start >= len(x) (frame rỗng), hoặc size < 1.
    """
    arr = _as_1d_float(x)
    start = _as_int(start, "start", 0)
    size = _as_int(size, "size", 1)
    if start >= len(arr):
        raise ValueError(f"start = {start} vượt quá độ dài tín hiệu ({len(arr)} mẫu)")
    return zero_pad(arr[start : start + size], size)