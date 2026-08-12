from statistics import mean, stdev


def calculate_statistics(values):
    """
    Calculate summary statistics across evaluation seeds.

    Standard deviation is the sample standard deviation
    using n - 1 in the denominator.

    For a single value, std is defined as 0.0.
    """

    if not values:
        raise ValueError(
            "At least one value is required"
        )

    return {
        "mean": mean(values),
        "std": (
            stdev(values)
            if len(values) > 1
            else 0.0
        ),
        "min": min(values),
        "max": max(values),
    }