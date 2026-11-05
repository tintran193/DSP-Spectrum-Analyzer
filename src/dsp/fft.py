import numpy as np


def _bit_reversal_indices(length: int) -> np.ndarray:
    """Return indices that place an array in bit-reversed order."""
    number_of_bits = length.bit_length() - 1
    indices = np.empty(length, dtype=np.int64)

    for original_index in range(length):
        value = original_index
        reversed_index = 0

        for _ in range(number_of_bits):
            reversed_index = (reversed_index << 1) | (value & 1)
            value >>= 1

        indices[original_index] = reversed_index

    return indices


def fft(x: np.ndarray) -> np.ndarray:
    """Compute the DFT using an iterative Radix-2 DIT FFT.

    Parameters
    ----------
    x:
        One-dimensional real or complex input. Its length must be a positive
        power of two (1, 2, 4, 8, ...).

    Returns
    -------
    np.ndarray
        Complex FFT output with the same length as the input.

    Raises
    ------
    ValueError
        If the input is not one-dimensional, is empty, or its length is not a
        power of two.

    Notes
    -----
    The transform uses the same convention as ``numpy.fft.fft``: a negative
    exponent in the forward transform and no normalization by N.
    """
    # Convert input to complex array
    signal = np.asarray(x, dtype=np.complex128)

    # Check input dimension
    if signal.ndim != 1:
        raise ValueError("Input signal must be a 1-D array.")
    
    # Get number of samples
    length = signal.size

    # Check whether N is a positive power of 2
    if length == 0 or (length & (length - 1)) != 0:
        raise ValueError(
            "FFT input length must be a positive power of 2."
        )

    # Step 1: reorder the input using bit-reversal indices.
    indices = _bit_reversal_indices(length)
    result = signal[indices].copy()

    # Step 2: perform the butterfly operations, stage by stage.
    stage_size = 2
    while stage_size <= length:
        half_size = stage_size // 2

        # W_stage_size^j = exp(-j * 2*pi*j / stage_size)
        twiddles = np.exp(
            -2j * np.pi * np.arange(half_size) / stage_size
        )

        # Process each group
        for group_start in range(0, length, stage_size):
            for j in range(half_size):
                upper_index = group_start + j
                lower_index = upper_index + half_size

                even_value = result[upper_index]
                odd_value = result[lower_index] * twiddles[j]

                # Butterfly operation
                result[upper_index] = even_value + odd_value
                result[lower_index] = even_value - odd_value

        # Move to the next stage
        stage_size *= 2

    return result