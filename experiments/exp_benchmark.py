import csv
import time
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from src.dsp.dft import dft
from src.dsp.fft import fft
from src.dsp.validation import compare_dft_fft, compare_fft_numpy


FFT_SIZES = (256, 512, 1024, 2048, 4096)
REPEAT = 3
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"


def measure_time_ms(
    function: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    repeat: int = REPEAT,
) -> float:
    """Return the median execution time in milliseconds.

    One warm-up call is performed before measurements. Median is used to
    reduce the influence of occasional scheduling or background-process noise.
    """
    if repeat < 1:
        raise ValueError("repeat must be at least 1")

    function(x)  # Warm-up; excluded from the measurement.

    elapsed_ms = []
    for _ in range(repeat):
        start = time.perf_counter()
        function(x)
        elapsed_ms.append((time.perf_counter() - start) * 1000.0)

    return float(np.median(elapsed_ms))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    """Write a list of dictionaries to a CSV file."""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    benchmark_rows: list[dict] = []
    accuracy_rows: list[dict] = []

    for length in FFT_SIZES:
        print(f"Running N={length} ...")
        rng = np.random.default_rng(2026 + length)
        x = rng.standard_normal(length)

        # Measure each transform using exactly the same input array.
        dft_time = measure_time_ms(dft, x)
        custom_fft_time = measure_time_ms(fft, x)
        numpy_fft_time = measure_time_ms(np.fft.fft, x)

        error_dft_fft = compare_dft_fft(x)
        error_fft_numpy = compare_fft_numpy(x)

        benchmark_rows.append(
            {
                "N": length,
                "DFT_time_ms": dft_time,
                "Custom_FFT_time_ms": custom_fft_time,
                "NumPy_FFT_time_ms": numpy_fft_time,
            }
        )
        accuracy_rows.append(
            {
                "N": length,
                "DFT_FFT_max_error": error_dft_fft,
                "Custom_NumPy_max_error": error_fft_numpy,
            }
        )

        print(
            f"  time (ms): DFT={dft_time:.6f}, "
            f"Custom FFT={custom_fft_time:.6f}, NumPy FFT={numpy_fft_time:.6f}"
        )
        print(
            f"  max error: DFT/FFT={error_dft_fft:.3e}, "
            f"Custom/NumPy={error_fft_numpy:.3e}"
        )

    write_csv(
        TABLE_DIR / "benchmark.csv",
        benchmark_rows,
        ["N", "DFT_time_ms", "Custom_FFT_time_ms", "NumPy_FFT_time_ms"],
    )
    write_csv(
        TABLE_DIR / "accuracy.csv",
        accuracy_rows,
        ["N", "DFT_FFT_max_error", "Custom_NumPy_max_error"],
    )

    sizes = np.array([row["N"] for row in benchmark_rows])

    # Execution-time chart. Log scales make the three curves easier to compare.
    plt.figure(figsize=(9, 6))
    plt.plot(
        sizes,
        [row["DFT_time_ms"] for row in benchmark_rows],
        marker="o",
        label="Direct DFT",
    )
    plt.plot(
        sizes,
        [row["Custom_FFT_time_ms"] for row in benchmark_rows],
        marker="o",
        label="Custom Radix-2 FFT",
    )
    plt.plot(
        sizes,
        [row["NumPy_FFT_time_ms"] for row in benchmark_rows],
        marker="o",
        label="NumPy FFT",
    )
    plt.xlabel("Number of samples (N)")
    plt.ylabel("Execution time (ms)")
    plt.title("Execution Time: Direct DFT vs FFT")
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.grid(True, which="both", linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "benchmark.png", dpi=200)
    plt.close()

    # Accuracy chart. The floor is only for plotting on a logarithmic axis;
    # CSV files retain the actual measured error values.
    dft_fft_errors = np.array(
        [row["DFT_FFT_max_error"] for row in accuracy_rows], dtype=float
    )
    numpy_errors = np.array(
        [row["Custom_NumPy_max_error"] for row in accuracy_rows], dtype=float
    )
    plot_floor = np.finfo(float).tiny

    plt.figure(figsize=(9, 6))
    plt.plot(
        sizes,
        np.maximum(dft_fft_errors, plot_floor),
        marker="o",
        label="Direct DFT vs Custom FFT",
    )
    plt.plot(
        sizes,
        np.maximum(numpy_errors, plot_floor),
        marker="o",
        label="Custom FFT vs NumPy FFT",
    )
    plt.xlabel("Number of samples (N)")
    plt.ylabel("Maximum absolute error")
    plt.title("Numerical Accuracy of Custom Radix-2 FFT")
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.grid(True, which="both", linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "accuracy.png", dpi=200)
    plt.close()

    print("\nBenchmark completed.")
    print(f"CSV files: {TABLE_DIR}")
    print(f"Figure files: {FIGURE_DIR}")
    
if __name__ == "__main__":
    main()