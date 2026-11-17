import numpy as np
import pytest

from src.dsp.spectrum import (
    frequency_axis,
    one_sided_magnitude,
    magnitude_to_db
)

from src.dsp.fft import fft


def test_frequency_axis():

    result = frequency_axis(
        fs=8000,
        N=8
    )

    expected = np.array([
        0, 1000, 2000, 3000, 4000
    ])

    np.testing.assert_allclose(
        result,
        expected
    )


def test_one_sided_magnitude_sine():

    fs = 8000
    N = 8
    f = 1000

    n = np.arange(N)

    x = np.sin(
        2 * np.pi * f * n / fs
    )

    X = fft(x)

    magnitude = one_sided_magnitude(
        X,
        N
    )

    assert magnitude[1] == pytest.approx(1.0)


def test_one_sided_magnitude_dc():

    x = np.ones(8)

    X = fft(x)

    magnitude = one_sided_magnitude(
        X,
        8
    )

    assert magnitude[0] == pytest.approx(1.0)


def test_magnitude_to_db():

    magnitude = np.array([
        1.0,
        0.5,
        0.1
    ])

    db = magnitude_to_db(magnitude)

    expected = np.array([
        0.0,
        -6.0206,
        -20.0
    ])

    np.testing.assert_allclose(
        db,
        expected,
        atol=1e-3
    )


def test_frequency_axis_invalid_fs():

    with pytest.raises(ValueError):
        frequency_axis(0, 8)


def test_magnitude_wrong_length():

    with pytest.raises(ValueError):
        one_sided_magnitude(
            np.zeros(4),
            8
        )


def test_magnitude_to_db_negative():

    with pytest.raises(ValueError):
        magnitude_to_db(
            np.array([1.0, -0.5])
        )