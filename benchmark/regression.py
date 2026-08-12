def check_regression_thresholds(
    metrics,
    thresholds,
):
    """
    Compare benchmark metrics against configured
    minimum/maximum acceptance thresholds.

    Returns a list of failure messages.

    An empty list means all checks passed.
    """

    if not isinstance(metrics, dict):
        raise ValueError(
            "metrics must be a dictionary"
        )

    if not isinstance(thresholds, dict):
        raise ValueError(
            "thresholds must be a dictionary"
        )

    failures = []

    for metric_name, limits in thresholds.items():

        if metric_name not in metrics:
            failures.append(
                f"Missing metric: {metric_name}"
            )
            continue

        if not isinstance(limits, dict):
            raise ValueError(
                f"Threshold for {metric_name} "
                f"must be a dictionary"
            )

        value = metrics[metric_name]

        if not isinstance(
            value,
            (int, float),
        ):
            failures.append(
                f"{metric_name} is not numeric"
            )
            continue

        minimum = limits.get("min")
        maximum = limits.get("max")

        if (
            minimum is None
            and maximum is None
        ):
            raise ValueError(
                f"Threshold for {metric_name} "
                f"must define min and/or max"
            )

        if (
            minimum is not None
            and value < minimum
        ):
            failures.append(
                f"{metric_name}={value} "
                f"is below minimum {minimum}"
            )

        if (
            maximum is not None
            and value > maximum
        ):
            failures.append(
                f"{metric_name}={value} "
                f"is above maximum {maximum}"
            )

    return failures