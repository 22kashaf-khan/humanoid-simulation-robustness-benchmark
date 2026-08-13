import json
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "multiseed"
    / "aggregated_results.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "multiseed"
    / "plots"
)


SCENARIOS = [
    "nominal",
    "actuator_effort_0_5x",
    "actuator_effort_0_4x",
    "actuator_effort_0_375x",
    "actuator_effort_0_35x",
]

LABELS = [
    "1.0x\nNominal",
    "0.50x",
    "0.40x",
    "0.375x",
    "0.35x",
]


METRICS = [
    (
        "success_rate",
        100.0,
        "Survival rate (%)",
        "H1 Robustness to Actuator Effort-Limit Mismatch",
        "actuator_success_rate.png",
    ),
    (
        "linear_velocity_rmse_mps",
        1.0,
        "Linear velocity RMSE (m/s)",
        "Linear Velocity Tracking vs Actuator Effort",
        "actuator_linear_velocity_rmse.png",
    ),
    (
        "yaw_velocity_rmse_radps",
        1.0,
        "Yaw velocity RMSE (rad/s)",
        "Yaw Tracking vs Actuator Effort",
        "actuator_yaw_velocity_rmse.png",
    ),
    (
        "mean_base_tilt_degrees",
        1.0,
        "Mean base tilt (degrees)",
        "Base Stability vs Actuator Effort",
        "actuator_base_tilt.png",
    ),
    (
        "joint_limit_violation_rate",
        100.0,
        "Joint-limit violation rate (%)",
        "Joint-Limit Violations vs Actuator Effort",
        "actuator_joint_limit_violations.png",
    ),
]


def main():

    with open(INPUT_PATH, "r") as file:
        data = json.load(file)

    scenarios = data["scenarios"]

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for (
        metric_name,
        scale,
        ylabel,
        title,
        filename,
    ) in METRICS:

        means = []
        stds = []

        for scenario_name in SCENARIOS:

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

        x = list(range(len(LABELS)))

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
            LABELS,
        )

        plt.xlabel(
            "Actuator effort-limit scale"
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

        output_path = (
            OUTPUT_DIR
            / filename
        )

        plt.savefig(
            output_path,
            dpi=200,
        )

        plt.close()

        print(
            f"Generated: {output_path}"
        )


if __name__ == "__main__":
    main()