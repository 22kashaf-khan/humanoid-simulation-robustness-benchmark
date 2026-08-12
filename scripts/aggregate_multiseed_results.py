import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from benchmark.aggregation import calculate_statistics
METRICS = [
    "success_rate",
    "fall_rate",
    "mean_episode_length",
    "linear_velocity_rmse_mps",
    "yaw_velocity_rmse_radps",
    "mean_base_tilt_degrees",
    "joint_limit_violation_rate",
]


def load_json(path):
    with open(path, "r") as file:
        return json.load(file)


def save_json(path, data):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(path, "w") as file:
        json.dump(
            data,
            file,
            indent=2,
        )





def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="reports/multiseed/raw_results.json",
        help="Combined raw multi-seed results JSON",
    )

    parser.add_argument(
        "--output",
        default="reports/multiseed/aggregated_results.json",
        help="Output statistics JSON",
    )

    args = parser.parse_args()


    input_path = (
        PROJECT_ROOT
        / args.input
    )

    output_path = (
        PROJECT_ROOT
        / args.output
    )


    raw = load_json(
        input_path
    )


    results = raw.get(
        "results",
        []
    )


    if not results:
        raise ValueError(
            "No benchmark results found."
        )


    expected_runs = raw.get(
        "total_expected_runs"
    )

    if (
        expected_runs is not None
        and len(results) != expected_runs
    ):
        raise ValueError(
            f"Expected {expected_runs} runs "
            f"but found {len(results)}."
        )


    # =============================================
    # Group results by scenario
    # =============================================

    grouped = {}

    for result in results:

        scenario = result[
            "scenario"
        ]

        grouped.setdefault(
            scenario,
            []
        ).append(
            result
        )


    aggregated_scenarios = {}


    # =============================================
    # Aggregate each scenario across seeds
    # =============================================

    for scenario, scenario_results in grouped.items():

        scenario_results = sorted(
            scenario_results,
            key=lambda x: x["seed"],
        )


        seeds = [
            result["seed"]
            for result in scenario_results
        ]


        metric_stats = {}


        for metric_name in METRICS:

            values = [
                result["metrics"][
                    metric_name
                ]
                for result
                in scenario_results
            ]


            metric_stats[
                metric_name
            ] = calculate_statistics(
                values
            )


        aggregated_scenarios[
            scenario
        ] = {

            "physics_modifications": (
                scenario_results[0]
                .get(
                    "physics_modifications"
                )
            ),

            "num_seeds": len(
                scenario_results
            ),

            "seeds": seeds,

            "metrics": metric_stats,
        }


    # =============================================
    # Final output
    # =============================================

    aggregated = {

        "experiment": raw.get(
            "experiment"
        ),

        "statistics": {
            "aggregation": (
                "mean, sample standard deviation, "
                "minimum, maximum across evaluation seeds"
            ),
            "standard_deviation": (
                "sample standard deviation (n-1)"
            ),
        },

        "total_runs": len(
            results
        ),

        "scenario_count": len(
            aggregated_scenarios
        ),

        "scenarios": (
            aggregated_scenarios
        ),
    }


    save_json(
        output_path,
        aggregated,
    )


    # =============================================
    # Terminal summary
    # =============================================

    print(
        "\n"
        + "=" * 100
    )

    print(
        "MULTI-SEED STATISTICAL SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        f"{'Scenario':<24}"
        f"{'Success':>18}"
        f"{'Lin RMSE':>18}"
        f"{'Yaw RMSE':>18}"
        f"{'Tilt':>18}"
    )

    print(
        "-" * 100
    )


    for scenario, data in aggregated_scenarios.items():

        metrics = data[
            "metrics"
        ]


        success = metrics[
            "success_rate"
        ]

        linear = metrics[
            "linear_velocity_rmse_mps"
        ]

        yaw = metrics[
            "yaw_velocity_rmse_radps"
        ]

        tilt = metrics[
            "mean_base_tilt_degrees"
        ]


        print(
            f"{scenario:<24}"
            f"{success['mean'] * 100:>7.2f}"
            f" ± {success['std'] * 100:<7.2f}"
            f"{linear['mean']:>8.4f}"
            f" ± {linear['std']:<7.4f}"
            f"{yaw['mean']:>8.4f}"
            f" ± {yaw['std']:<7.4f}"
            f"{tilt['mean']:>8.2f}"
            f" ± {tilt['std']:<7.2f}"
        )


    print(
        "\nStatistics use sample standard "
        "deviation across evaluation seeds."
    )

    print(
        "\nSaved to:"
    )

    print(
        output_path
    )


if __name__ == "__main__":
    main()