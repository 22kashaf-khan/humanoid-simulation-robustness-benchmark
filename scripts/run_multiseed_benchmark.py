import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BENCHMARK_SCRIPT = (
    PROJECT_ROOT
    / "benchmark"
    / "run_benchmark.py"
)

GENERATED_CONFIG_ROOT = (
    PROJECT_ROOT
    / "configs"
    / "generated"
    / "multiseed"
)

REPORT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "multiseed"
)


# =========================================================
# Configuration helpers
# =========================================================

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


# =========================================================
# Validate master configuration
# =========================================================

def validate_master_config(cfg):

    required_sections = [
        "experiment",
        "seeds",
        "robot",
        "policy",
        "evaluation",
        "scenarios",
    ]

    for section in required_sections:

        if section not in cfg:
            raise ValueError(
                f"Missing required section: {section}"
            )


    if not cfg["seeds"]:
        raise ValueError(
            "At least one seed is required"
        )


    if not cfg["scenarios"]:
        raise ValueError(
            "At least one scenario is required"
        )


    scenario_names = [
        scenario["name"]
        for scenario in cfg["scenarios"]
    ]


    if len(scenario_names) != len(
        set(scenario_names)
    ):
        raise ValueError(
            "Scenario names must be unique"
        )


# =========================================================
# Generate one benchmark configuration
# =========================================================

def create_run_config(
    master_cfg,
    scenario,
    seed,
):

    scenario_name = (
        scenario["name"]
    )


    output_directory = (
        Path("reports")
        / "multiseed"
        / master_cfg["experiment"]["name"]
        / scenario_name
        / f"seed_{seed}"
    )   


    run_config = {

        "experiment": {
            "name": (
                f"{master_cfg['experiment']['name']}"
                f"__{scenario_name}"
                f"__seed_{seed}"
            ),

            "seed": seed,
        },

        "robot": {
            "name": (
                master_cfg["robot"]["name"]
            ),

            "task": (
                master_cfg["robot"]["task"]
            ),
        },

        "policy": {
            "checkpoint": (
                master_cfg[
                    "policy"
                ][
                    "checkpoint"
                ]
            ),
        },

        "evaluation": {
            "num_envs": (
                master_cfg[
                    "evaluation"
                ][
                    "num_envs"
                ]
            ),

            "episodes": (
                master_cfg[
                    "evaluation"
                ][
                    "episodes"
                ]
            ),

            "headless": (
                master_cfg[
                    "evaluation"
                ][
                    "headless"
                ]
            ),
        },

        "scenario": {
            "name": scenario_name,

            "physics_modifications": (
                scenario.get(
                    "physics_modifications",
                    "none",
                )
            ),
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
                output_directory.as_posix()
            ),
        },
    }


    return run_config


# =========================================================
# Check whether a completed result can be reused
# =========================================================

def result_is_valid(
    result_path,
    scenario_name,
    seed,
):

    if not result_path.exists():
        return False


    try:
        result = load_json(
            result_path
        )

    except Exception:
        return False


    if result.get("scenario") != scenario_name:
        return False


    if result.get("seed") != seed:
        return False


    if "metrics" not in result:
        return False


    return True


# =========================================================
# Execute one benchmark
# =========================================================

def run_benchmark(
    config_path,
):

    relative_config_path = (
        config_path
        .relative_to(
            PROJECT_ROOT
        )
    )


    command = [
        sys.executable,

        str(
            BENCHMARK_SCRIPT
        ),

        "--config",

        str(
            relative_config_path
        ),
    ]


    print(
        "\nExecuting:"
    )

    print(
        " ".join(command)
    )


    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )


# =========================================================
# Main
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
        help=(
            "Master multi-seed YAML "
            "relative to project root"
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Rerun experiments even if "
            "a valid results.json exists"
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Generate configs and show "
            "planned runs without "
            "starting Isaac Sim"
        ),
    )

    args = parser.parse_args()


    master_config_path = (
        PROJECT_ROOT
        / args.config
    )


    master_cfg = load_yaml(
        master_config_path
    )


    validate_master_config(
        master_cfg
    )


    seeds = (
        master_cfg["seeds"]
    )

    scenarios = (
        master_cfg["scenarios"]
    )


    total_runs = (
        len(seeds)
        * len(scenarios)
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "MULTI-SEED ROBUSTNESS BENCHMARK"
    )

    print(
        "=" * 70
    )

    print(
        f"Seeds      : {seeds}"
    )

    print(
        f"Scenarios  : "
        f"{len(scenarios)}"
    )

    print(
        f"Total runs : "
        f"{total_runs}"
    )

    print(
        f"Force      : "
        f"{args.force}"
    )

    print(
        f"Dry run    : "
        f"{args.dry_run}"
    )


    GENERATED_CONFIG_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


    REPORT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


    combined_results = []

    completed_count = 0
    skipped_count = 0
    planned_count = 0


    run_number = 0


    # =====================================================
    # Scenario × seed matrix
    # =====================================================

    for scenario in scenarios:

        scenario_name = (
            scenario["name"]
        )


        for seed in seeds:

            run_number += 1


            print(
                "\n"
                + "-" * 70
            )

            print(
                f"RUN "
                f"{run_number}/{total_runs}"
            )

            print(
                f"Scenario : "
                f"{scenario_name}"
            )

            print(
                f"Seed     : "
                f"{seed}"
            )

            print(
                "-" * 70
            )


            # =============================================
            # Generated benchmark YAML
            # =============================================

            run_cfg = create_run_config(
                master_cfg,
                scenario,
                seed,
            )


            generated_config_path = (
                GENERATED_CONFIG_ROOT
                / scenario_name
                / f"seed_{seed}.yaml"
            )


            save_yaml(
                generated_config_path,
                run_cfg,
            )


            # =============================================
            # Expected result location
            # =============================================

            result_path = (
                Path(run_cfg["output"]["directory"])
                / "results.json"
            )


            # =============================================
            # Resume / skip logic
            # =============================================

            already_complete = (
                result_is_valid(
                    result_path,
                    scenario_name,
                    seed,
                )
            )


            if (
                already_complete
                and not args.force
            ):

                print(
                    "[SKIP] Valid result "
                    "already exists."
                )

                result = load_json(
                    result_path
                )


                combined_results.append(
                    result
                )


                skipped_count += 1

                continue


            # =============================================
            # Dry-run mode
            # =============================================

            if args.dry_run:

                print(
                    "[DRY RUN] Would execute:"
                )

                print(
                    f"  {generated_config_path}"
                )

                planned_count += 1

                continue


            # =============================================
            # Execute actual benchmark
            # =============================================

            run_benchmark(
                generated_config_path
            )


            # =============================================
            # Verify output
            # =============================================

            if not result_path.exists():

                raise RuntimeError(
                    "Benchmark completed but "
                    "results.json was not found: "
                    f"{result_path}"
                )


            result = load_json(
                result_path
            )


            combined_results.append(
                result
            )


            completed_count += 1


            # =============================================
            # Save progress after every run
            #
            # If the process is interrupted, we still
            # preserve all successfully completed results.
            # =============================================

            progress_path = (
                REPORT_ROOT
                / master_cfg["experiment"]["name"]
                / "raw_results.json"
            )


            save_json(
                progress_path,
                {
                    "experiment": (
                        master_cfg[
                            "experiment"
                        ][
                            "name"
                        ]
                    ),

                    "total_expected_runs": (
                        total_runs
                    ),

                    "results_collected": (
                        len(
                            combined_results
                        )
                    ),

                    "results": (
                        combined_results
                    ),
                },
            )


    # =====================================================
    # Dry-run summary
    # =====================================================

    if args.dry_run:

        print(
            "\n"
            + "=" * 70
        )

        print(
            "DRY RUN COMPLETE"
        )

        print(
            "=" * 70
        )

        print(
            f"Runs planned: "
            f"{planned_count}"
        )

        print(
            f"Runs skipped: "
            f"{skipped_count}"
        )

        print(
            "No simulator runs were started."
        )

        return


    # =====================================================
    # Final combined result
    # =====================================================

    combined_output_path = (
        REPORT_ROOT
        / master_cfg["experiment"]["name"]
        / "raw_results.json"
    )


    save_json(
        combined_output_path,
        {
            "experiment": (
                master_cfg[
                    "experiment"
                ][
                    "name"
                ]
            ),

            "seeds": seeds,

            "scenario_count": (
                len(scenarios)
            ),

            "total_expected_runs": (
                total_runs
            ),

            "results_collected": (
                len(
                    combined_results
                )
            ),

            "new_runs_completed": (
                completed_count
            ),

            "existing_runs_reused": (
                skipped_count
            ),

            "results": (
                combined_results
            ),
        },
    )


    # =====================================================
    # Terminal summary
    # =====================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "MULTI-SEED BENCHMARK COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Expected runs       : "
        f"{total_runs}"
    )

    print(
        f"Results collected   : "
        f"{len(combined_results)}"
    )

    print(
        f"New runs completed  : "
        f"{completed_count}"
    )

    print(
        f"Existing runs reused: "
        f"{skipped_count}"
    )

    print(
        "\nCombined raw results:"
    )

    print(
        combined_output_path
    )


if __name__ == "__main__":
    main()