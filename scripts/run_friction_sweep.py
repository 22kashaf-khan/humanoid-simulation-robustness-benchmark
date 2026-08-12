import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path):
    with open(path, "r") as file:
        return yaml.safe_load(file)


def save_yaml(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as file:
        yaml.safe_dump(
            data,
            file,
            sort_keys=False,
        )


def load_json(path):
    with open(path, "r") as file:
        return json.load(file)


parser = argparse.ArgumentParser()

parser.add_argument(
    "--config",
    required=True,
    help="Friction sweep YAML relative to project root",
)

args = parser.parse_args()


# ---------------------------------------------------------
# Load sweep definition
# ---------------------------------------------------------

sweep_path = PROJECT_ROOT / args.config
sweep_cfg = load_yaml(sweep_path)

friction_values = sweep_cfg["sweep"]["friction"]

generated_config_dir = (
    PROJECT_ROOT
    / "configs"
    / "generated"
    / "friction_sweep"
)

sweep_output_dir = (
    PROJECT_ROOT
    / "reports"
    / "friction_sweep"
)

generated_config_dir.mkdir(
    parents=True,
    exist_ok=True,
)

sweep_output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------
# Run every friction experiment
# ---------------------------------------------------------

all_results = []

total_runs = len(friction_values)

print("\nFriction sweep")
print("=" * 70)
print(f"Experiments: {total_runs}")
print("=" * 70)


for run_index, friction in enumerate(
    friction_values,
    start=1,
):

    static_friction = float(
        friction["static"]
    )

    dynamic_friction = float(
        friction["dynamic"]
    )


    scenario_name = (
        f"friction_static_{static_friction}"
        f"_dynamic_{dynamic_friction}"
    )


    experiment_name = (
        f"baseline_v1_{scenario_name}"
    )


    report_relative = (
        Path("reports")
        / "friction_sweep"
        / scenario_name
    )


    generated_cfg = {
        "experiment": {
            "name": experiment_name,
            "seed": sweep_cfg["experiment"]["seed"],
        },

        "robot": copy.deepcopy(
            sweep_cfg["robot"]
        ),

        "policy": copy.deepcopy(
            sweep_cfg["policy"]
        ),

        "evaluation": copy.deepcopy(
            sweep_cfg["evaluation"]
        ),

        "scenario": {
            "name": scenario_name,
            "physics_modifications": {
                "friction": {
                    "static": static_friction,
                    "dynamic": dynamic_friction,
                }
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
            "directory": str(
                report_relative
            ).replace("\\", "/")
        },
    }


    generated_config_path = (
        generated_config_dir
        / f"{scenario_name}.yaml"
    )


    save_yaml(
        generated_cfg,
        generated_config_path,
    )


    print()
    print("=" * 70)
    print(
        f"Run {run_index}/{total_runs}"
    )
    print(
        f"Static friction : "
        f"{static_friction}"
    )
    print(
        f"Dynamic friction: "
        f"{dynamic_friction}"
    )
    print("=" * 70)


    relative_generated_config = (
        generated_config_path.relative_to(
            PROJECT_ROOT
        )
    )


    # -----------------------------------------------------
    # Call our existing benchmark runner
    # -----------------------------------------------------

    command = [
        sys.executable,
        str(
            PROJECT_ROOT
            / "benchmark"
            / "run_benchmark.py"
        ),
        "--config",
        str(relative_generated_config),
    ]


    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )


    # -----------------------------------------------------
    # Read result produced by run_benchmark.py
    # -----------------------------------------------------

    result_path = (
        PROJECT_ROOT
        / report_relative
        / "results.json"
    )


    result = load_json(
        result_path
    )


    all_results.append(
        {
            "static_friction": static_friction,
            "dynamic_friction": dynamic_friction,
            "scenario": scenario_name,
            "metrics": result["metrics"],
        }
    )


# ---------------------------------------------------------
# Save combined sweep result
# ---------------------------------------------------------

combined_results = {
    "experiment": sweep_cfg["experiment"]["name"],
    "num_runs": len(all_results),
    "results": all_results,
}


combined_path = (
    sweep_output_dir
    / "sweep_results.json"
)


with open(
    combined_path,
    "w",
) as file:

    json.dump(
        combined_results,
        file,
        indent=2,
    )


# ---------------------------------------------------------
# Final summary
# ---------------------------------------------------------

print("\n")
print("=" * 70)
print("Friction sweep complete")
print("=" * 70)


for result in all_results:

    metrics = result["metrics"]

    print(
        f"static={result['static_friction']:<5} "
        f"dynamic={result['dynamic_friction']:<5} | "
        f"success={metrics['success_rate']:.0%} | "
        f"falls={metrics['fall_rate']:.0%} | "
        f"linear_rmse="
        f"{metrics['linear_velocity_rmse_mps']:.4f} | "
        f"yaw_rmse="
        f"{metrics['yaw_velocity_rmse_radps']:.4f}"
    )


print()
print(
    f"Combined results saved to: "
    f"{combined_path}"
)