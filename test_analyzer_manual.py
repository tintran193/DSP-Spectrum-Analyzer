import numpy as np

from src.input.signal_generator import generate_sine

from src.analysis.analyzer import SpectrumAnalyzer


# Signal configuration.
fs = 8000
frequency = 1000
amplitude = 1.0
phase = 0
duration = 0.128

fft_size = 1024

# Generate input signal.
x, fs = generate_sine(
    frequency=frequency,
    amplitude=amplitude,
    phase=phase,
    fs=fs,
    duration=duration
)

# Create analyzer.
analyzer = SpectrumAnalyzer(
    peak_threshold=0.05,
    min_peak_distance=3
)

# Run analysis.
result = analyzer.analyze(
    x=x,
    fs=fs,
    fft_size=fft_size,
    window_name="rectangular"
)

# Display information.
print("Sampling frequency:", result["fs"])

print("FFT size:", result["fft_size"])

print(
    "Frequency resolution:",
    result["frequency_resolution"]
)

print(
    "FFT output shape:",
    result["fft"].shape
)

print(
    "Magnitude output shape:",
    result["magnitude"].shape
)

print("\nDetected peaks:")

for peak in result["peaks"]:

    print(
        f"Frequency: {peak['frequency']:.2f} Hz, "
        f"Magnitude: {peak['magnitude']:.4f}, "
        f"dB: {peak['db']:.2f}"
    )