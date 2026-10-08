from src.input.signal_generator import generate_signal
from src.visualization.plot import plot_time_signal

x, fs = generate_signal(
    freq=1000,
    amp=1.0,
    fs=8000,
    duration=0.01
)

plot_time_signal(x, fs)