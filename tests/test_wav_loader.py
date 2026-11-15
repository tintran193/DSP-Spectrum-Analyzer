"""Test D: đọc WAV, stereo -> mono. Chạy: python -m pytest tests -v"""
import numpy as np
import pytest
import soundfile as sf
import matplotlib.pyplot as plt
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

def test_read_and_analyze_real_wav():
    # Nhập đường dẫn file WAV
    path = input("\nNhap duong dan file WAV: ").strip().strip('"')

    # Đọc file
    x, fs = load_wav(path)

    # Thông tin cơ bản
    duration = len(x) / fs

    print("\n========== WAV INFORMATION ==========")
    print(f"Sample rate       : {fs} Hz")
    print(f"Number of samples: {len(x)}")
    print(f"Duration          : {duration:.3f} seconds")
    print(f"Channels after load: Mono")
    print(f"Data type         : {x.dtype}")
    print(f"Amplitude min     : {np.min(x):.4f}")
    print(f"Amplitude max     : {np.max(x):.4f}")
    print(f"RMS               : {np.sqrt(np.mean(x**2)):.4f}")

    # Phân tích phổ FFT
    N = len(x)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, d=1/fs)

    magnitude = 2 * np.abs(X) / N

    # Không nhân đôi thành phần DC và Nyquist
    magnitude[0] /= 2
    if N % 2 == 0:
        magnitude[-1] /= 2

    peak_index = np.argmax(magnitude)
    peak_frequency = freqs[peak_index]

    print("\n========== FREQUENCY ANALYSIS ==========")
    print(f"Dominant frequency: {peak_frequency:.2f} Hz")
    print(f"Peak amplitude    : {magnitude[peak_index]:.4f}")

#này vẽ tín hiệu theo tgian
    t = np.arange(N) / fs

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(t, x)
    plt.title("Time-domain Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

#Này vẽ phổ nha 
    plt.subplot(2, 1, 2)
    plt.plot(freqs, magnitude)
    plt.title("Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)

    plt.tight_layout()
    plt.show()