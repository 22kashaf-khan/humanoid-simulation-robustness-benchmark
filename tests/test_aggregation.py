import pytest

from benchmark.aggregation import (
    calculate_statistics,
)


def test_statistics_mean():
    stats = calculate_statistics(
        [1.0, 2.0, 3.0]
    )

    assert stats["mean"] == pytest.approx(2.0)


def test_statistics_min_and_max():
    stats = calculate_statistics(
        [0.8, 1.0, 0.9, 0.7]
    )

    assert stats["min"] == pytest.approx(0.7)
    assert stats["max"] == pytest.approx(1.0)


def test_statistics_uses_sample_standard_deviation():
    stats = calculate_statistics(
        [1.0, 2.0, 3.0]
    )

    # Sample std for [1, 2, 3] is exactly 1.0.
    assert stats["std"] == pytest.approx(1.0)


def test_statistics_single_seed_std_is_zero():
    stats = calculate_statistics(
        [0.95]
    )

    assert stats["mean"] == pytest.approx(0.95)
    assert stats["std"] == pytest.approx(0.0)
    assert stats["min"] == pytest.approx(0.95)
    assert stats["max"] == pytest.approx(0.95)


def test_statistics_identical_values_have_zero_std():
    stats = calculate_statistics(
        [0.5, 0.5, 0.5, 0.5, 0.5]
    )

    assert stats["mean"] == pytest.approx(0.5)
    assert stats["std"] == pytest.approx(0.0)


def test_statistics_empty_values_fail():
    with pytest.raises(
        ValueError,
        match="At least one value is required",
    ):
        calculate_statistics([])