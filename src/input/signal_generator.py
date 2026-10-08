import numpy as np

def generate_signal(
        freq: float,
        amp: float,
        fs: float,
        duration: float
):
    n = np.arange(int(fs*duration))
    x = amp * np.sin(2 * np.pi * freq * n / fs)
    return x, fs
