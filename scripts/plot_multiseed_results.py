import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_json(path):
    with open(path, "r") as file:
        return json.load(file)


def get_metric(
    scenarios,
    scenario_names,
    metric_name,
    scale=1.0,
):
    means = []
    stds = []

    for scenario_name in scenario_names:

        stats = (
            scenarios[scenario_name]
            ["metrics"]
            [metric_name]
        )

        means.append(
            stats["mean"] * scale
        )

        stds.append(
            stats["std"] * scale
        )

    return means, stds


def save_errorbar_plot(
    labels,
    means,
    stds,
    title,
    ylabel,
    output_path,
):

    x = list(
        range(len(labels))
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.errorbar(
        x,
        means,
        yerr=stds,
        marker="o",
        capsize=5,
    )

    plt.xticks(
        x,
        labels,
    )

    plt.xlabel(
        "Simulation condition"
    )

    plt.ylabel(
        ylabel
    )

    plt.title(
        title
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=(
            "reports/multiseed/"
            "aggregated_results.json"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "reports/multiseed/plots"
        ),
    )

    args = parser.parse_args()


    input_path = (
        PROJECT_ROOT
        / args.input
    )

    output_dir = (
        PROJECT_ROOT
        / args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    data = load_json(
        input_path
    )

    scenarios = data[
        "scenarios"
    ]


    # =====================================================
    # Friction scenarios
    #
    # Nominal H1 task uses the default contact material.
    # The labels show static / dynamic friction.
    # =====================================================

    friction_scenarios = [
        "nominal",
        "friction_0_4_0_3",
        "friction_0_3_0_2",
        "friction_0_2_0_15",
    ]

    friction_labels = [
        "Nominal\n0.8 / 0.6",
        "0.4 / 0.3",
        "0.3 / 0.2",
        "0.2 / 0.15",
    ]


    # =====================================================
    # Mass scenarios
    # =====================================================

    mass_scenarios = [
        "nominal",
        "mass_1_2x",
        "mass_1_4x",
        "mass_1_6x",
    ]

    mass_labels = [
        "1.0x\nNominal",
        "1.2x",
        "1.4x",
        "1.6x",
    ]


    # =====================================================
    # Metric definitions
    # =====================================================

    metrics = [

        {
            "name": "success_rate",
            "scale": 100.0,
            "ylabel": "Survival rate (%)",
            "friction_title": (
                "H1 Robustness to Contact-Friction Mismatch"
            ),
            "mass_title": (
                "H1 Robustness to Whole-Robot Mass Mismatch"
            ),
            "filename": "success_rate",
        },

        {
            "name": "linear_velocity_rmse_mps",
            "scale": 1.0,
            "ylabel": "Linear velocity RMSE (m/s)",
            "friction_title": (
                "Linear Velocity Tracking vs Contact Friction"
            ),
            "mass_title": (
                "Linear Velocity Tracking vs Robot Mass"
            ),
            "filename": "linear_velocity_rmse",
        },

        {
            "name": "yaw_velocity_rmse_radps",
            "scale": 1.0,
            "ylabel": "Yaw velocity RMSE (rad/s)",
            "friction_title": (
                "Yaw Tracking vs Contact Friction"
            ),
            "mass_title": (
                "Yaw Tracking vs Robot Mass"
            ),
            "filename": "yaw_velocity_rmse",
        },

        {
            "name": "mean_base_tilt_degrees",
            "scale": 1.0,
            "ylabel": "Mean base tilt (degrees)",
            "friction_title": (
                "Base Stability vs Contact Friction"
            ),
            "mass_title": (
                "Base Stability vs Robot Mass"
            ),
            "filename": "base_tilt",
        },

        {
            "name": "joint_limit_violation_rate",
            "scale": 100.0,
            "ylabel": "Joint-limit violation rate (%)",
            "friction_title": (
                "Joint-Limit Violations vs Contact Friction"
            ),
            "mass_title": (
                "Joint-Limit Violations vs Robot Mass"
            ),
            "filename": "joint_limit_violations",
        },
    ]


    # =====================================================
    # Generate friction and mass plots
    # =====================================================

    generated_files = []


    for metric in metrics:

        # -----------------------------
        # Friction
        # -----------------------------

        means, stds = get_metric(
            scenarios,
            friction_scenarios,
            metric["name"],
            metric["scale"],
        )

        friction_path = (
            output_dir
            / (
                "friction_"
                + metric["filename"]
                + ".png"
            )
        )

        save_errorbar_plot(
            friction_labels,
            means,
            stds,
            metric["friction_title"],
            metric["ylabel"],
            friction_path,
        )

        generated_files.append(
            friction_path
        )


        # -----------------------------
        # Mass
        # -----------------------------

        means, stds = get_metric(
            scenarios,
            mass_scenarios,
            metric["name"],
            metric["scale"],
        )

        mass_path = (
            output_dir
            / (
                "mass_"
                + metric["filename"]
                + ".png"
            )
        )

        save_errorbar_plot(
            mass_labels,
            means,
            stds,
            metric["mass_title"],
            metric["ylabel"],
            mass_path,
        )

        generated_files.append(
            mass_path
        )


    print(
        "\nMULTI-SEED PLOTS COMPLETE"
    )

    print(
        "Statistics shown: mean ± sample std "
        "across 5 evaluation seeds"
    )

    print(
        "\nGenerated:"
    )

    for path in generated_files:
        print(
            path
        )


if __name__ == "__main__":
    main()