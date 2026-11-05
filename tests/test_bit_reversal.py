import numpy as np

from src.dsp.fft import _bit_reversal_indices


def test_bit_reversal_2():
    expected = np.array([0, 1])

    result = _bit_reversal_indices(2)

    np.testing.assert_array_equal(result, expected)


def test_bit_reversal_4():
    expected = np.array([0, 2, 1, 3])

    result = _bit_reversal_indices(4)

    np.testing.assert_array_equal(result, expected)


def test_bit_reversal_8():
    expected = np.array([0, 4, 2, 6, 1, 5, 3, 7])

    result = _bit_reversal_indices(8)

    np.testing.assert_array_equal(result, expected)


def test_bit_reversal_16():
    expected = np.array([
        0, 8, 4, 12,
        2, 10, 6, 14,
        1, 9, 5, 13,
        3, 11, 7, 15
    ])

    result = _bit_reversal_indices(16)

    np.testing.assert_array_equal(result, expected)