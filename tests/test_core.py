import pytest

from benchmark.core import (
    calculate_episode_rates,
    calculate_mean,
    calculate_rmse,
    classify_termination,
    validate_config,
    validate_physics_modifications,
)


def valid_config():
    return {
        "experiment": {
            "name": "test_benchmark",
            "seed": 42,
        },
        "robot": {
            "name": "Unitree H1",
            "task": "Isaac-Velocity-Flat-H1-v0",
        },
        "policy": {
            "checkpoint": "baseline_checkpoints/baseline_v1.pt",
        },
        "evaluation": {
            "num_envs": 64,
            "episodes": 100,
            "headless": True,
        },
        "scenario": {
            "name": "nominal",
            "physics_modifications": "none",
        },
        "metrics": {},
        "output": {
            "directory": "reports/test",
        },
    }


# =========================================================
# Config validation
# =========================================================

def test_valid_config_passes():
    validate_config(valid_config())


def test_missing_config_section_fails():
    cfg = valid_config()
    del cfg["robot"]

    with pytest.raises(
        ValueError,
        match="Missing required config section: robot",
    ):
        validate_config(cfg)


def test_non_dictionary_config_fails():
    with pytest.raises(
        ValueError,
        match="Benchmark config must be a dictionary",
    ):
        validate_config([])


def test_missing_nested_config_value_fails():
    cfg = valid_config()
    del cfg["experiment"]["name"]

    with pytest.raises(
        ValueError,
        match="Missing required config value: experiment.name",
    ):
        validate_config(cfg)


def test_num_envs_must_be_positive():
    cfg = valid_config()
    cfg["evaluation"]["num_envs"] = 0

    with pytest.raises(
        ValueError,
        match="evaluation.num_envs must be a positive integer",
    ):
        validate_config(cfg)


def test_episodes_must_be_positive():
    cfg = valid_config()
    cfg["evaluation"]["episodes"] = -1

    with pytest.raises(
        ValueError,
        match="evaluation.episodes must be a positive integer",
    ):
        validate_config(cfg)


def test_headless_must_be_boolean():
    cfg = valid_config()
    cfg["evaluation"]["headless"] = "true"

    with pytest.raises(
        ValueError,
        match="evaluation.headless must be a boolean",
    ):
        validate_config(cfg)


def test_robot_task_cannot_be_empty():
    cfg = valid_config()
    cfg["robot"]["task"] = ""

    with pytest.raises(
        ValueError,
        match="robot.task must be a non-empty string",
    ):
        validate_config(cfg)


# =========================================================
# Physics validation
# =========================================================

def test_nominal_physics_passes():
    validate_physics_modifications("none")


def test_valid_friction_passes():
    validate_physics_modifications(
        {
            "friction": {
                "static": 0.4,
                "dynamic": 0.3,
            }
        }
    )


def test_valid_mass_scale_passes():
    validate_physics_modifications(
        {
            "mass_scale": 1.4,
        }
    )


def test_invalid_physics_type_fails():
    with pytest.raises(
        ValueError,
        match=(
            "physics_modifications must be "
            "either 'none' or a dictionary"
        ),
    ):
        validate_physics_modifications("invalid")


def test_missing_dynamic_friction_fails():
    with pytest.raises(
        ValueError,
        match="friction.dynamic is required",
    ):
        validate_physics_modifications(
            {
                "friction": {
                    "static": 0.4,
                }
            }
        )


def test_negative_friction_fails():
    with pytest.raises(
        ValueError,
        match="Friction values cannot be negative",
    ):
        validate_physics_modifications(
            {
                "friction": {
                    "static": -0.1,
                    "dynamic": 0.2,
                }
            }
        )


def test_non_numeric_friction_fails():
    with pytest.raises(
        ValueError,
        match="Friction values must be numeric",
    ):
        validate_physics_modifications(
            {
                "friction": {
                    "static": "low",
                    "dynamic": 0.2,
                }
            }
        )


def test_zero_mass_scale_fails():
    with pytest.raises(
        ValueError,
        match="mass_scale must be greater than zero",
    ):
        validate_physics_modifications(
            {
                "mass_scale": 0,
            }
        )


def test_non_numeric_mass_scale_fails():
    with pytest.raises(
        ValueError,
        match="mass_scale must be numeric",
    ):
        validate_physics_modifications(
            {
                "mass_scale": "heavy",
            }
        )


# =========================================================
# Episode termination classification
# =========================================================

def test_timeout_is_success():
    assert classify_termination(True) == "success"


def test_early_termination_is_fall():
    assert classify_termination(False) == "fall"


# =========================================================
# Episode rates
# =========================================================

def test_episode_rates():
    success_rate, fall_rate = calculate_episode_rates(
        successes=90,
        falls=10,
    )

    assert success_rate == pytest.approx(0.9)
    assert fall_rate == pytest.approx(0.1)


def test_episode_rates_require_completed_episode():
    with pytest.raises(
        ValueError,
        match="At least one completed episode is required",
    ):
        calculate_episode_rates(
            successes=0,
            falls=0,
        )


def test_episode_rates_reject_negative_counts():
    with pytest.raises(
        ValueError,
        match="Episode counts cannot be negative",
    ):
        calculate_episode_rates(
            successes=-1,
            falls=10,
        )


# =========================================================
# RMSE
# =========================================================

def test_rmse():
    result = calculate_rmse(
        squared_error_sum=25,
        observation_count=1,
    )

    assert result == pytest.approx(5.0)


def test_rmse_multiple_observations():
    result = calculate_rmse(
        squared_error_sum=16,
        observation_count=4,
    )

    assert result == pytest.approx(2.0)


def test_rmse_requires_observations():
    with pytest.raises(
        ValueError,
        match="observation_count must be greater than zero",
    ):
        calculate_rmse(
            squared_error_sum=10,
            observation_count=0,
        )


def test_rmse_rejects_negative_squared_error():
    with pytest.raises(
        ValueError,
        match="squared_error_sum cannot be negative",
    ):
        calculate_rmse(
            squared_error_sum=-1,
            observation_count=1,
        )


# =========================================================
# Mean
# =========================================================

def test_mean():
    result = calculate_mean(
        total=20,
        count=4,
    )

    assert result == pytest.approx(5.0)


def test_mean_requires_observations():
    with pytest.raises(
        ValueError,
        match="count must be greater than zero",
    ):
        calculate_mean(
            total=10,
            count=0,
        )