"""Test D: đọc WAV, stereo -> mono. Chạy: python -m pytest tests -v"""
import numpy as np
import pytest
import soundfile as sf

from src.input.wav_loader import load_wav

FS = 8000


def _tones():
    n = np.arange(FS) / FS
    return 0.5 * np.sin(2 * np.pi * 1000 * n), 0.25 * np.sin(2 * np.pi * 3000 * n)


def test_d_stereo_to_mono_exact(tmp_path):
    left, right = _tones()
    p = tmp_path / "stereo.wav"
    sf.write(p, np.column_stack([left, right]), FS, subtype="FLOAT")
    x, fs = load_wav(str(p))
    assert fs == float(FS)
    assert x.ndim == 1 and x.dtype == np.float64 and len(x) == FS
    assert np.allclose(x, (left + right) / 2, atol=1e-6)


def test_d_stereo_16bit_within_quantization(tmp_path):
    left, right = _tones()
    p = tmp_path / "stereo16.wav"
    sf.write(p, np.column_stack([left, right]), FS, subtype="PCM_16")
    x, _ = load_wav(str(p))
    assert x.ndim == 1
    assert np.allclose(x, (left + right) / 2, atol=2e-5)   # sai số ~ 1 mức lượng tử (1/32768)


def test_d_mono_file(tmp_path):
    left, _ = _tones()
    p = tmp_path / "mono.wav"
    sf.write(p, left, FS, subtype="FLOAT")
    x, fs = load_wav(str(p))
    assert fs == float(FS) and x.ndim == 1 and len(x) == FS
    assert np.allclose(x, left, atol=1e-6)


def test_d_values_in_range_for_pcm(tmp_path):
    left, right = _tones()
    p = tmp_path / "range.wav"
    sf.write(p, np.column_stack([left, right]), FS, subtype="PCM_16")
    x, _ = load_wav(str(p))
    assert np.max(np.abs(x)) <= 1.0


def test_d_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_wav(str(tmp_path / "khong_ton_tai.wav"))


def test_d_not_audio_file(tmp_path):
    p = tmp_path / "fake.wav"
    p.write_bytes(b"day khong phai file am thanh")
    with pytest.raises(ValueError):
        load_wav(str(p))