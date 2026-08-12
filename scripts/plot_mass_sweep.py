import json
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "mass_sweep"
    / "sweep_results.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "mass_sweep"
    / "plots"
)


def load_results(path):
    with open(path, "r") as file:
        return json.load(file)


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = load_results(
        RESULTS_PATH
    )

    results = data["results"]

    mass_scales = []
    success_rates = []
    linear_rmse = []
    yaw_rmse = []

    for result in results:

        mass_scales.append(
            result["mass_scale"]
        )

        metrics = result["metrics"]

        success_rates.append(
            metrics["success_rate"] * 100
        )

        linear_rmse.append(
            metrics[
                "linear_velocity_rmse_mps"
            ]
        )

        yaw_rmse.append(
            metrics[
                "yaw_velocity_rmse_radps"
            ]
        )


    # =====================================================
    # Plot 1 — Success rate
    # =====================================================

    plt.figure(figsize=(10, 6))

    plt.plot(
        mass_scales,
        success_rates,
        marker="o",
        linewidth=2,
    )

    plt.xlabel(
        "Whole-Robot Mass Scale"
    )

    plt.ylabel(
        "Success Rate (%)"
    )

    plt.title(
        "H1 Locomotion Robustness vs Mass Mismatch"
    )

    plt.ylim(
        0,
        105,
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    for x, value in zip(
        mass_scales,
        success_rates,
    ):

        plt.annotate(
            f"{value:.0f}%",
            (x, value),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
        )


    plt.tight_layout()

    success_path = (
        OUTPUT_DIR
        / "mass_success_rate.png"
    )

    plt.savefig(
        success_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


    # =====================================================
    # Plot 2 — Linear RMSE
    # =====================================================

    plt.figure(figsize=(10, 6))

    plt.plot(
        mass_scales,
        linear_rmse,
        marker="o",
        linewidth=2,
    )

    plt.xlabel(
        "Whole-Robot Mass Scale"
    )

    plt.ylabel(
        "Linear Velocity RMSE (m/s)"
    )

    plt.title(
        "H1 Linear Velocity Tracking vs Mass Mismatch"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    linear_path = (
        OUTPUT_DIR
        / "mass_linear_rmse.png"
    )

    plt.savefig(
        linear_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


    # =====================================================
    # Plot 3 — Yaw RMSE
    # =====================================================

    plt.figure(figsize=(10, 6))

    plt.plot(
        mass_scales,
        yaw_rmse,
        marker="o",
        linewidth=2,
    )

    plt.xlabel(
        "Whole-Robot Mass Scale"
    )

    plt.ylabel(
        "Yaw Velocity RMSE (rad/s)"
    )

    plt.title(
        "H1 Yaw Tracking vs Mass Mismatch"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    yaw_path = (
        OUTPUT_DIR
        / "mass_yaw_rmse.png"
    )

    plt.savefig(
        yaw_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


    # =====================================================
    # Plot 4 — Normalized degradation
    # =====================================================

    nominal_index = mass_scales.index(
        1.0
    )

    nominal_linear = (
        linear_rmse[
            nominal_index
        ]
    )

    nominal_yaw = (
        yaw_rmse[
            nominal_index
        ]
    )


    normalized_linear = [
        value / nominal_linear
        for value in linear_rmse
    ]

    normalized_yaw = [
        value / nominal_yaw
        for value in yaw_rmse
    ]


    plt.figure(figsize=(10, 6))

    plt.plot(
        mass_scales,
        normalized_linear,
        marker="o",
        linewidth=2,
        label="Linear velocity RMSE",
    )

    plt.plot(
        mass_scales,
        normalized_yaw,
        marker="o",
        linewidth=2,
        label="Yaw velocity RMSE",
    )

    plt.axhline(
        y=1.0,
        linestyle="--",
        linewidth=1,
        label="Nominal performance",
    )

    plt.xlabel(
        "Whole-Robot Mass Scale"
    )

    plt.ylabel(
        "RMSE Relative to Nominal (×)"
    )

    plt.title(
        "H1 Tracking Degradation Relative to Nominal Mass"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    normalized_path = (
        OUTPUT_DIR
        / "mass_normalized_degradation.png"
    )

    plt.savefig(
        normalized_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


    print(
        "\nMass plots generated successfully"
    )

    print(
        f"Success plot    : {success_path}"
    )

    print(
        f"Linear RMSE plot: {linear_path}"
    )

    print(
        f"Yaw RMSE plot   : {yaw_path}"
    )

    print(
        f"Normalized plot : {normalized_path}"
    )


if __name__ == "__main__":
    main()