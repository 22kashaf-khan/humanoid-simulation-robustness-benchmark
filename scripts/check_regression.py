import argparse
import json
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from benchmark.regression import (
    check_regression_thresholds,
)


def load_json(path):
    with open(path, "r") as file:
        return json.load(file)


def load_yaml(path):
    with open(path, "r") as file:
        return yaml.safe_load(file)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Check benchmark results against "
            "regression thresholds."
        )
    )

    parser.add_argument(
        "--results",
        default=(
            "reports/baseline_v1_nominal/results.json"
        ),
        help="Benchmark results JSON",
    )

    parser.add_argument(
        "--thresholds",
        default=(
            "configs/regression_thresholds.yaml"
        ),
        help="Regression threshold YAML",
    )

    parser.add_argument(
        "--profile",
        default="nominal",
        help="Threshold profile to use",
    )

    args = parser.parse_args()

    results_path = (
        PROJECT_ROOT
        / args.results
    )

    thresholds_path = (
        PROJECT_ROOT
        / args.thresholds
    )

    if not results_path.exists():
        raise FileNotFoundError(
            f"Results file not found: "
            f"{results_path}"
        )

    if not thresholds_path.exists():
        raise FileNotFoundError(
            f"Threshold file not found: "
            f"{thresholds_path}"
        )

    results = load_json(
        results_path
    )

    threshold_config = load_yaml(
        thresholds_path
    )

    if not isinstance(
        threshold_config,
        dict,
    ):
        raise ValueError(
            "Threshold configuration "
            "must be a dictionary"
        )

    if args.profile not in threshold_config:
        raise ValueError(
            f"Threshold profile not found: "
            f"{args.profile}"
        )

    metrics = results.get(
        "metrics"
    )

    if metrics is None:
        raise ValueError(
            "Results JSON contains no metrics"
        )

    thresholds = threshold_config[
        args.profile
    ]

    failures = (
        check_regression_thresholds(
            metrics,
            thresholds,
        )
    )

    print(
        "\nREGRESSION CHECK"
    )

    print(
        "--------------------------------"
    )

    print(
        f"Results    : {results_path}"
    )

    print(
        f"Thresholds : {thresholds_path}"
    )

    print(
        f"Profile    : {args.profile}"
    )

    print()

    if failures:
        print(
            "STATUS: FAIL"
        )

        print()

        for failure in failures:
            print(
                f"- {failure}"
            )

        return 1

    print(
        "STATUS: PASS"
    )

    print(
        "All configured benchmark "
        "thresholds satisfied."
    )

    return 0


if __name__ == "__main__":
    sys.exit(
        main()
    )