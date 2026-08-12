import json
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "friction_sweep"
    / "sweep_results.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "friction_sweep"
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

    sweep_results = data["results"]


    # -----------------------------------------------------
    # Extract experiment data
    # -----------------------------------------------------

    friction_labels = []
    success_rates = []
    linear_rmse = []
    yaw_rmse = []


    for result in sweep_results:

        static_friction = result[
            "static_friction"
        ]

        dynamic_friction = result[
            "dynamic_friction"
        ]

        metrics = result["metrics"]


        friction_labels.append(
            f"{static_friction}/{dynamic_friction}"
        )

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


    x = list(
        range(len(friction_labels))
    )


    # =====================================================
    # Plot 1 — Success rate
    # =====================================================

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        x,
        success_rates,
        marker="o",
        linewidth=2,
    )

    plt.xticks(
        x,
        friction_labels,
    )

    plt.xlabel(
        "Static / Dynamic Friction"
    )

    plt.ylabel(
        "Success Rate (%)"
    )

    plt.title(
        "H1 Locomotion Robustness vs Friction"
    )

    plt.ylim(
        -5,
        105,
    )

    plt.grid(
        True,
        alpha=0.3,
    )


    for i, value in enumerate(
        success_rates
    ):

        plt.annotate(
            f"{value:.0f}%",
            (
                x[i],
                value,
            ),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
        )


    plt.tight_layout()


    success_path = (
        OUTPUT_DIR
        / "friction_success_rate.png"
    )

    plt.savefig(
        success_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


    # =====================================================
    # Plot 2 — Linear velocity RMSE
    # =====================================================

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        x,
        linear_rmse,
        marker="o",
        linewidth=2,
    )

    plt.xticks(
        x,
        friction_labels,
    )

    plt.xlabel(
        "Static / Dynamic Friction"
    )

    plt.ylabel(
        "Linear Velocity RMSE (m/s)"
    )

    plt.title(
        "H1 Linear Velocity Tracking vs Friction"
    )

    plt.grid(
        True,
        alpha=0.3,
    )


    for i, value in enumerate(
        linear_rmse
    ):

        plt.annotate(
            f"{value:.3f}",
            (
                x[i],
                value,
            ),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
        )


    plt.tight_layout()


    linear_path = (
        OUTPUT_DIR
        / "friction_linear_rmse.png"
    )

    plt.savefig(
        linear_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


    # =====================================================
    # Plot 3 — Yaw velocity RMSE
    # =====================================================

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        x,
        yaw_rmse,
        marker="o",
        linewidth=2,
    )

    plt.xticks(
        x,
        friction_labels,
    )

    plt.xlabel(
        "Static / Dynamic Friction"
    )

    plt.ylabel(
        "Yaw Velocity RMSE (rad/s)"
    )

    plt.title(
        "H1 Yaw Tracking vs Friction"
    )

    plt.grid(
        True,
        alpha=0.3,
    )


    for i, value in enumerate(
        yaw_rmse
    ):

        plt.annotate(
            f"{value:.3f}",
            (
                x[i],
                value,
            ),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
        )


    plt.tight_layout()


    yaw_path = (
        OUTPUT_DIR
        / "friction_yaw_rmse.png"
    )

    plt.savefig(
        yaw_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


    # =====================================================
    # Plot 4 — Normalized tracking degradation
    # =====================================================

    nominal_linear_rmse = linear_rmse[0]
    nominal_yaw_rmse = yaw_rmse[0]


    normalized_linear_rmse = [
        value / nominal_linear_rmse
        for value in linear_rmse
    ]

    normalized_yaw_rmse = [
        value / nominal_yaw_rmse
        for value in yaw_rmse
    ]


    plt.figure(
        figsize=(10, 6)
    )


    plt.plot(
        x,
        normalized_linear_rmse,
        marker="o",
        linewidth=2,
        label="Linear velocity RMSE",
    )


    plt.plot(
        x,
        normalized_yaw_rmse,
        marker="o",
        linewidth=2,
        label="Yaw velocity RMSE",
    )


    plt.xticks(
        x,
        friction_labels,
    )

    plt.xlabel(
        "Static / Dynamic Friction"
    )

    plt.ylabel(
        "RMSE Relative to Nominal (x)"
    )

    plt.title(
        "H1 Tracking Degradation Relative to Nominal Physics"
    )


    # 1x means equal to nominal tracking error
    plt.axhline(
        y=1.0,
        linestyle="--",
        linewidth=1,
        label="Nominal performance",
    )


    plt.legend()

    plt.grid(
        True,
        alpha=0.3,
    )


    plt.tight_layout()


    normalized_path = (
        OUTPUT_DIR
        / "friction_normalized_degradation.png"
    )


    plt.savefig(
        normalized_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print("\nPlots generated successfully")
    print("=" * 70)

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