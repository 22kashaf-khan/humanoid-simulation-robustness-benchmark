import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path):
    with open(path, "r") as file:
        return yaml.safe_load(file)


def save_yaml(path, data):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(path, "w") as file:
        yaml.safe_dump(
            data,
            file,
            sort_keys=False,
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
        help="Mass sweep YAML configuration",
    )

    args = parser.parse_args()


    # -----------------------------------------------------
    # Load sweep configuration
    # -----------------------------------------------------

    sweep_config_path = (
        PROJECT_ROOT
        / args.config
    )

    cfg = load_yaml(
        sweep_config_path
    )


    mass_scales = (
        cfg["sweep"]["mass_scale"]
    )


    print(
        f"\nMass sweep contains "
        f"{len(mass_scales)} conditions"
    )


    # -----------------------------------------------------
    # Directories
    # -----------------------------------------------------

    generated_config_dir = (
        PROJECT_ROOT
        / "configs"
        / "generated"
        / "mass_sweep"
    )

    report_root = (
        PROJECT_ROOT
        / "reports"
        / "mass_sweep"
    )


    generated_config_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_root.mkdir(
        parents=True,
        exist_ok=True,
    )


    combined_results = []


    # =====================================================
    # Run each mass condition
    # =====================================================

    for index, mass_scale in enumerate(
        mass_scales,
        start=1,
    ):

        mass_scale = float(
            mass_scale
        )


        # Example:
        # 1.2 -> mass_1_2x
        scale_label = (
            str(mass_scale)
            .replace(".", "_")
        )

        scenario_name = (
            f"mass_{scale_label}x"
        )


        print(
            "\n"
            + "=" * 70
        )

        print(
            f"Mass condition "
            f"{index}/{len(mass_scales)}"
        )

        print(
            f"Mass scale : "
            f"{mass_scale}x"
        )

        print(
            f"Scenario   : "
            f"{scenario_name}"
        )

        print(
            "=" * 70
        )


        # -------------------------------------------------
        # Generate benchmark config
        # -------------------------------------------------

        generated_config = {

            "experiment": {
                "name": (
                    f"baseline_v1_"
                    f"{scenario_name}"
                ),
                "seed": (
                    cfg["experiment"]["seed"]
                ),
            },

            "robot": (
                cfg["robot"]
            ),

            "policy": (
                cfg["policy"]
            ),

            "evaluation": (
                cfg["evaluation"]
            ),

            "scenario": {
                "name": scenario_name,

                "physics_modifications": {
                    "mass_scale": (
                        mass_scale
                    ),
                },
            },

            "metrics": {
                "success_rate": True,
                "fall_rate": True,
                "velocity_tracking_rmse": True,
                "episode_length": True,
                "base_orientation_error": True,
                "joint_limit_violations": True,
            },

            "output": {
                "directory": (
                    f"reports/"
                    f"mass_sweep/"
                    f"{scenario_name}"
                ),
            },
        }


        generated_config_path = (
            generated_config_dir
            / f"{scenario_name}.yaml"
        )


        save_yaml(
            generated_config_path,
            generated_config,
        )


        # -------------------------------------------------
        # Run benchmark
        # -------------------------------------------------

        relative_config_path = (
            generated_config_path
            .relative_to(
                PROJECT_ROOT
            )
        )


        command = [
            sys.executable,
            str(
                PROJECT_ROOT
                / "benchmark"
                / "run_benchmark.py"
            ),
            "--config",
            str(
                relative_config_path
            ),
        ]


        print(
            "\nRunning:"
        )

        print(
            " ".join(command)
        )


        subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=True,
        )


        # -------------------------------------------------
        # Load produced result
        # -------------------------------------------------

        result_path = (
            report_root
            / scenario_name
            / "results.json"
        )


        if not result_path.exists():
            raise FileNotFoundError(
                f"Expected benchmark result "
                f"not found: {result_path}"
            )


        with open(
            result_path,
            "r",
        ) as file:

            result = json.load(
                file
            )


        combined_results.append(
            {
                "mass_scale": (
                    mass_scale
                ),

                "scenario": (
                    scenario_name
                ),

                "metrics": (
                    result["metrics"]
                ),
            }
        )


    # =====================================================
    # Save combined sweep results
    # =====================================================

    combined_output = {
        "experiment": (
            cfg["experiment"]["name"]
        ),

        "seed": (
            cfg["experiment"]["seed"]
        ),

        "results": (
            combined_results
        ),
    }


    combined_output_path = (
        report_root
        / "sweep_results.json"
    )


    with open(
        combined_output_path,
        "w",
    ) as file:

        json.dump(
            combined_output,
            file,
            indent=2,
        )


    # =====================================================
    # Print summary
    # =====================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "MASS SWEEP COMPLETE"
    )

    print(
        "=" * 70
    )


    for result in combined_results:

        metrics = (
            result["metrics"]
        )


        print(
            f"\nMass scale: "
            f"{result['mass_scale']}x"
        )

        print(
            f"  Success rate : "
            f"{metrics['success_rate']:.2%}"
        )

        print(
            f"  Fall rate    : "
            f"{metrics['fall_rate']:.2%}"
        )

        print(
            f"  Linear RMSE  : "
            f"{metrics['linear_velocity_rmse_mps']:.4f} m/s"
        )

        print(
            f"  Yaw RMSE     : "
            f"{metrics['yaw_velocity_rmse_radps']:.4f} rad/s"
        )


    print(
        f"\nCombined results saved to:"
    )

    print(
        combined_output_path
    )


if __name__ == "__main__":
    main()