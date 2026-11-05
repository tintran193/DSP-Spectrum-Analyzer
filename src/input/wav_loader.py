"""
wav_loader.py - Đọc file âm thanh (WAV, FLAC, OGG...) về mảng mono float64.

load_wav(path) -> (x, fs)
    x  : np.ndarray 1 chiều, float64, nằm trong [-1, 1] với file PCM
    fs : float, tần số lấy mẫu đọc từ file (Hz)
Stereo / đa kênh được đưa về mono bằng trung bình các kênh.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def load_wav(path: str) -> tuple[np.ndarray, float]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Không tìm thấy file: {p}")

    try:
        # always_2d=True -> luôn có shape (số mẫu, số kênh), kể cả file mono
        data, fs = sf.read(str(p), dtype="float64", always_2d=True)
    except RuntimeError as e:  # soundfile.LibsndfileError là lớp con của RuntimeError
        raise ValueError(f"Không đọc được file âm thanh '{p.name}': {e}") from e

    if data.shape[0] == 0:
        raise ValueError(f"File không có mẫu nào: {p.name}")

    # (N, C) -> (N,): trung bình các kênh; file mono chỉ lấy kênh duy nhất
    x = np.array(data.mean(axis=1), dtype=np.float64, copy=True)
    return x, float(fs)