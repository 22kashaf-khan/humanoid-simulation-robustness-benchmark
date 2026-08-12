import pytest

from benchmark.regression import (
    check_regression_thresholds,
)


def nominal_metrics():
    return {
        "success_rate": 1.0,
        "fall_rate": 0.0,
        "linear_velocity_rmse_mps": 0.0807,
        "yaw_velocity_rmse_radps": 0.1395,
        "mean_base_tilt_degrees": 2.99,
        "joint_limit_violation_rate": 0.000979,
    }


def nominal_thresholds():
    return {
        "success_rate": {
            "min": 0.95,
        },
        "fall_rate": {
            "max": 0.05,
        },
        "linear_velocity_rmse_mps": {
            "max": 0.12,
        },
        "yaw_velocity_rmse_radps": {
            "max": 0.20,
        },
        "mean_base_tilt_degrees": {
            "max": 5.0,
        },
        "joint_limit_violation_rate": {
            "max": 0.01,
        },
    }


def test_nominal_metrics_pass():
    failures = check_regression_thresholds(
        nominal_metrics(),
        nominal_thresholds(),
    )

    assert failures == []


def test_low_survival_fails():
    metrics = nominal_metrics()
    metrics["success_rate"] = 0.80

    failures = check_regression_thresholds(
        metrics,
        nominal_thresholds(),
    )

    assert len(failures) == 1

    assert (
        "success_rate=0.8 is below minimum 0.95"
        in failures
    )


def test_high_rmse_fails():
    metrics = nominal_metrics()
    metrics[
        "linear_velocity_rmse_mps"
    ] = 0.25

    failures = check_regression_thresholds(
        metrics,
        nominal_thresholds(),
    )

    assert len(failures) == 1

    assert (
        "linear_velocity_rmse_mps=0.25 "
        "is above maximum 0.12"
        in failures
    )


def test_multiple_regressions_are_reported():
    metrics = nominal_metrics()

    metrics["success_rate"] = 0.70
    metrics["fall_rate"] = 0.30
    metrics[
        "yaw_velocity_rmse_radps"
    ] = 0.5

    failures = check_regression_thresholds(
        metrics,
        nominal_thresholds(),
    )

    assert len(failures) == 3


def test_missing_metric_is_reported():
    metrics = nominal_metrics()

    del metrics[
        "mean_base_tilt_degrees"
    ]

    failures = check_regression_thresholds(
        metrics,
        nominal_thresholds(),
    )

    assert (
        "Missing metric: mean_base_tilt_degrees"
        in failures
    )


def test_threshold_without_min_or_max_fails():
    with pytest.raises(
        ValueError,
        match=(
            "Threshold for success_rate "
            "must define min and/or max"
        ),
    ):
        check_regression_thresholds(
            {
                "success_rate": 1.0,
            },
            {
                "success_rate": {},
            },
        )


def test_metrics_must_be_dictionary():
    with pytest.raises(
        ValueError,
        match="metrics must be a dictionary",
    ):
        check_regression_thresholds(
            [],
            nominal_thresholds(),
        )