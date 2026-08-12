REQUIRED_CONFIG_SECTIONS = (
    "experiment",
    "robot",
    "policy",
    "evaluation",
    "scenario",
    "metrics",
    "output",
)


def _require_key(section, key, section_name):
    if key not in section:
        raise ValueError(
            f"Missing required config value: "
            f"{section_name}.{key}"
        )


def _require_non_empty_string(value, field_name):
    if (
        not isinstance(value, str)
        or not value.strip()
    ):
        raise ValueError(
            f"{field_name} must be a non-empty string"
        )


def validate_physics_modifications(modifications):
    """
    Validate simulator-independent physics parameters.

    Supported values:
    - None
    - "none"
    - dictionary containing friction and/or mass_scale
    """

    if (
        modifications is None
        or modifications == "none"
    ):
        return

    if not isinstance(modifications, dict):
        raise ValueError(
            "physics_modifications must be "
            "either 'none' or a dictionary"
        )

    friction = modifications.get("friction")

    if friction is not None:
        if not isinstance(friction, dict):
            raise ValueError(
                "friction modification must be a dictionary"
            )

        if "static" not in friction:
            raise ValueError(
                "friction.static is required"
            )

        if "dynamic" not in friction:
            raise ValueError(
                "friction.dynamic is required"
            )

        try:
            static_friction = float(
                friction["static"]
            )
            dynamic_friction = float(
                friction["dynamic"]
            )
        except (TypeError, ValueError):
            raise ValueError(
                "Friction values must be numeric"
            )

        if (
            static_friction < 0
            or dynamic_friction < 0
        ):
            raise ValueError(
                "Friction values cannot be negative"
            )

    mass_scale = modifications.get(
        "mass_scale"
    )

    if mass_scale is not None:
        try:
            mass_scale = float(
                mass_scale
            )
        except (TypeError, ValueError):
            raise ValueError(
                "mass_scale must be numeric"
            )

        if mass_scale <= 0:
            raise ValueError(
                "mass_scale must be greater than zero"
            )


def validate_config(cfg):
    """
    Validate a single benchmark-run configuration.

    No Isaac Lab dependencies are used here so this
    function can run in CPU-only unit tests and CI.
    """

    if not isinstance(cfg, dict):
        raise ValueError(
            "Benchmark config must be a dictionary"
        )

    # -----------------------------------------------------
    # Required top-level sections
    # -----------------------------------------------------

    for section_name in REQUIRED_CONFIG_SECTIONS:

        if section_name not in cfg:
            raise ValueError(
                f"Missing required config section: "
                f"{section_name}"
            )

        if not isinstance(
            cfg[section_name],
            dict,
        ):
            raise ValueError(
                f"Config section '{section_name}' "
                f"must be a dictionary"
            )

    # -----------------------------------------------------
    # Experiment
    # -----------------------------------------------------

    experiment = cfg["experiment"]

    _require_key(
        experiment,
        "name",
        "experiment",
    )

    _require_key(
        experiment,
        "seed",
        "experiment",
    )

    _require_non_empty_string(
        experiment["name"],
        "experiment.name",
    )

    if not isinstance(
        experiment["seed"],
        int,
    ):
        raise ValueError(
            "experiment.seed must be an integer"
        )

    # -----------------------------------------------------
    # Robot
    # -----------------------------------------------------

    robot = cfg["robot"]

    _require_key(
        robot,
        "name",
        "robot",
    )

    _require_key(
        robot,
        "task",
        "robot",
    )

    _require_non_empty_string(
        robot["name"],
        "robot.name",
    )

    _require_non_empty_string(
        robot["task"],
        "robot.task",
    )

    # -----------------------------------------------------
    # Policy
    # -----------------------------------------------------

    policy = cfg["policy"]

    _require_key(
        policy,
        "checkpoint",
        "policy",
    )

    _require_non_empty_string(
        policy["checkpoint"],
        "policy.checkpoint",
    )

    # -----------------------------------------------------
    # Evaluation
    # -----------------------------------------------------

    evaluation = cfg["evaluation"]

    for key in (
        "num_envs",
        "episodes",
        "headless",
    ):
        _require_key(
            evaluation,
            key,
            "evaluation",
        )

    if (
        not isinstance(
            evaluation["num_envs"],
            int,
        )
        or evaluation["num_envs"] <= 0
    ):
        raise ValueError(
            "evaluation.num_envs must be "
            "a positive integer"
        )

    if (
        not isinstance(
            evaluation["episodes"],
            int,
        )
        or evaluation["episodes"] <= 0
    ):
        raise ValueError(
            "evaluation.episodes must be "
            "a positive integer"
        )

    if not isinstance(
        evaluation["headless"],
        bool,
    ):
        raise ValueError(
            "evaluation.headless must be a boolean"
        )

    # -----------------------------------------------------
    # Scenario
    # -----------------------------------------------------

    scenario = cfg["scenario"]

    _require_key(
        scenario,
        "name",
        "scenario",
    )

    _require_key(
        scenario,
        "physics_modifications",
        "scenario",
    )

    _require_non_empty_string(
        scenario["name"],
        "scenario.name",
    )

    validate_physics_modifications(
        scenario["physics_modifications"]
    )

    # -----------------------------------------------------
    # Output
    # -----------------------------------------------------

    output = cfg["output"]

    _require_key(
        output,
        "directory",
        "output",
    )

    _require_non_empty_string(
        output["directory"],
        "output.directory",
    )


def classify_termination(is_timeout):
    """
    Classify one completed benchmark episode.

    timeout=True  -> survived to timeout -> success
    timeout=False -> early termination -> fall
    """

    if is_timeout:
        return "success"

    return "fall"


def calculate_episode_rates(successes, falls):
    if successes < 0 or falls < 0:
        raise ValueError(
            "Episode counts cannot be negative"
        )

    completed_episodes = (
        successes + falls
    )

    if completed_episodes <= 0:
        raise ValueError(
            "At least one completed episode is required"
        )

    success_rate = (
        successes / completed_episodes
    )

    fall_rate = (
        falls / completed_episodes
    )

    return success_rate, fall_rate


def calculate_rmse(
    squared_error_sum,
    observation_count,
):
    if observation_count <= 0:
        raise ValueError(
            "observation_count must be greater than zero"
        )

    if squared_error_sum < 0:
        raise ValueError(
            "squared_error_sum cannot be negative"
        )

    return (
        squared_error_sum
        / observation_count
    ) ** 0.5


def calculate_mean(total, count):
    if count <= 0:
        raise ValueError(
            "count must be greater than zero"
        )

    return total / count