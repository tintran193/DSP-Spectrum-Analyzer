"""
wav_loader.py - Đọc file âm thanh (WAV, FLAC, OGG...) về mảng mono float64.

load_wav(path)     -> (x, fs)      # dữ liệu mono để phân tích (đúng API 5.4)
get_wav_info(path) -> dict         # thông tin file: sample_rate, channels, n_samples, duration
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


def get_wav_info(path: str) -> dict:
    """
    Đọc thông tin của file KHÔNG cần nạp dữ liệu:
        sample_rate : float, Hz
        channels    : int, số kênh của file gốc (1 = mono, 2 = stereo)
        n_samples   : int, số mẫu MỖI KÊNH
        duration    : float, giây = n_samples / sample_rate
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Không tìm thấy file: {p}")
    try:
        info = sf.info(str(p))
    except RuntimeError as e:
        raise ValueError(f"Không đọc được file âm thanh '{p.name}': {e}") from e
    if info.samplerate <= 0:
        raise ValueError(f"Sample rate không hợp lệ trong file: {p.name}")
    return {
        "sample_rate": float(info.samplerate),
        "channels": int(info.channels),
        "n_samples": int(info.frames),
        "duration": float(info.frames) / float(info.samplerate),
    }