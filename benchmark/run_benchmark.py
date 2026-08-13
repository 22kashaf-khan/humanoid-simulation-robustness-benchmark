import argparse
import json
from pathlib import Path

import yaml

from core import (
    calculate_episode_rates,
    calculate_mean,
    calculate_rmse,
    classify_termination,
    validate_config,
)

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# =========================================================
# Configuration helpers
# =========================================================

def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


# =========================================================
# Command-line arguments
# =========================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--config",
    required=True,
    help="Benchmark YAML path relative to project root",
)

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()


# =========================================================
# Load benchmark configuration
# =========================================================

config_path = PROJECT_ROOT / args_cli.config

cfg = load_config(config_path)
validate_config(cfg)

checkpoint = (
    PROJECT_ROOT
    / cfg["policy"]["checkpoint"]
)

output_dir = (
    PROJECT_ROOT
    / cfg["output"]["directory"]
)


if not checkpoint.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {checkpoint}"
    )


output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


args_cli.headless = (
    cfg["evaluation"]["headless"]
)


# =========================================================
# Launch Isaac Sim
# =========================================================

app_launcher = AppLauncher(args_cli)

simulation_app = (
    app_launcher.app
)


# =========================================================
# Imports requiring Isaac Sim
# =========================================================

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.managers import (
    EventTermCfg as EventTerm,
    SceneEntityCfg,
)

from isaaclab.utils.math import (
    quat_apply_inverse,
    yaw_quat,
)

from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,
)

import isaaclab_tasks

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from isaaclab_tasks.utils import (
    load_cfg_from_registry,
    parse_env_cfg,
)


# =========================================================
# Physics scenario configuration
# =========================================================

def apply_scenario_physics(
    env_cfg,
    cfg,
):

    modifications = (
        cfg["scenario"]
        .get("physics_modifications")
    )


    # -----------------------------------------------------
    # Nominal scenario
    # -----------------------------------------------------

    if (
        modifications is None
        or modifications == "none"
    ):
        print(
            "[Scenario] Nominal physics"
        )

        return


    if not isinstance(
        modifications,
        dict,
    ):
        raise ValueError(
            "physics_modifications must "
            "be either 'none' or a dictionary"
        )


    # =====================================================
    # Contact-friction modification
    # =====================================================

    friction = modifications.get(
        "friction"
    )


    if friction is not None:

        static_friction = float(
            friction["static"]
        )

        dynamic_friction = float(
            friction["dynamic"]
        )


        if (
            static_friction < 0
            or dynamic_friction < 0
        ):
            raise ValueError(
                "Friction values cannot be negative"
            )


        if (
            env_cfg.events.physics_material
            is None
        ):
            raise RuntimeError(
                "Task has no physics_material event"
            )


        env_cfg.events.physics_material.params[
            "static_friction_range"
        ] = (
            static_friction,
            static_friction,
        )


        env_cfg.events.physics_material.params[
            "dynamic_friction_range"
        ] = (
            dynamic_friction,
            dynamic_friction,
        )


        print(
            "[Scenario] Rigid-body contact friction overridden: "
            f"static={static_friction}, "
            f"dynamic={dynamic_friction}"
        )


    # =====================================================
    # Whole-robot mass scaling
    # =====================================================

    mass_scale = modifications.get(
        "mass_scale"
    )


    if mass_scale is not None:

        mass_scale = float(
            mass_scale
        )


        if mass_scale <= 0:
            raise ValueError(
                "mass_scale must be greater than zero"
            )


        env_cfg.events.add_base_mass = (
            EventTerm(
                func=(
                    mdp.randomize_rigid_body_mass
                ),
                mode="startup",
                params={
                    "asset_cfg": (
                        SceneEntityCfg(
                            "robot",
                            body_names=".*",
                        )
                    ),

                    "mass_distribution_params": (
                        mass_scale,
                        mass_scale,
                    ),

                    "operation": "scale",

                    "recompute_inertia": True,
                },
            )
        )


        print(
            "[Scenario] Whole-robot "
            "mass scale applied: "
            f"{mass_scale}x"
        )


    # =====================================================
    # Whole-robot actuator effort-limit scaling
    # =====================================================

    actuator_effort_scale = modifications.get(
        "actuator_effort_scale"
    )

    if actuator_effort_scale is not None:

        actuator_effort_scale = float(
            actuator_effort_scale
        )

        for actuator_name, actuator_cfg in (
            env_cfg.scene.robot.actuators.items()
        ):
            original_limit = (
                actuator_cfg.effort_limit_sim
            )

            if original_limit is None:
                continue

            actuator_cfg.effort_limit_sim = (
                original_limit
                * actuator_effort_scale
            )

            print(
                f"[Scenario] Actuator '{actuator_name}' "
                f"effort limit: "
                f"{original_limit} -> "
                f"{actuator_cfg.effort_limit_sim}"
            )

        print(
            "[Scenario] Whole-robot actuator "
            "effort-limit scale applied: "
            f"{actuator_effort_scale}x"
        )


# =========================================================
# Main benchmark
# =========================================================

def main():

    task = (
        cfg["robot"]["task"]
    )

    num_envs = (
        cfg["evaluation"]["num_envs"]
    )

    target_episodes = (
        cfg["evaluation"]["episodes"]
    )

    seed = (
        cfg["experiment"]["seed"]
    )


    # =====================================================
    # Benchmark information
    # =====================================================

    print(
        "\nBenchmark configuration valid"
    )

    print(
        "--------------------------------"
    )

    print(
        f"Experiment : "
        f"{cfg['experiment']['name']}"
    )

    print(
        f"Robot      : "
        f"{cfg['robot']['name']}"
    )

    print(
        f"Task       : "
        f"{task}"
    )

    print(
        f"Scenario   : "
        f"{cfg['scenario']['name']}"
    )

    print(
        f"Envs       : "
        f"{num_envs}"
    )

    print(
        f"Episodes   : "
        f"{target_episodes}"
    )

    print(
        f"Checkpoint : "
        f"{checkpoint}"
    )

    print(
        f"Output     : "
        f"{output_dir}"
    )


    # =====================================================
    # Isaac Lab configuration
    # =====================================================

    env_cfg = parse_env_cfg(
        task,
        device="cuda:0",
        num_envs=num_envs,
    )


    apply_scenario_physics(
        env_cfg,
        cfg,
    )


    agent_cfg = (
        load_cfg_from_registry(
            task,
            "rsl_rl_cfg_entry_point",
        )
    )


    env_cfg.seed = seed
    agent_cfg.seed = seed


    # =====================================================
    # Create environment
    # =====================================================

    print(
        "\n[Benchmark] "
        "Creating H1 environments..."
    )


    env = gym.make(
        task,
        cfg=env_cfg,
    )

    runtime_robot = env.unwrapped.scene["robot"]

    print("\n[Runtime actuator verification]")

    for actuator_name, actuator in runtime_robot.actuators.items():
        unique_limits = torch.unique(
            actuator.effort_limit_sim
        ).detach().cpu().tolist()

        print(
            f"{actuator_name}: "
            f"effort_limit_sim={unique_limits}"
        )


    env = RslRlVecEnvWrapper(
        env,
        clip_actions=(
            agent_cfg.clip_actions
        ),
    )


    # =====================================================
    # Load frozen PPO policy
    # =====================================================

    print(
        f"[Benchmark] "
        f"Loading checkpoint: "
        f"{checkpoint}"
    )


    runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=None,
        device=agent_cfg.device,
    )


    runner.load(
        str(checkpoint)
    )


    policy = (
        runner.get_inference_policy(
            device=(
                env.unwrapped.device
            )
        )
    )


    obs = (
        env.get_observations()
    )


    robot = (
        env.unwrapped.scene["robot"]
    )


    device = (
        env.unwrapped.device
    )


    # =====================================================
    # Episode counters
    # =====================================================

    completed_episodes = 0

    successes = 0
    falls = 0

    vector_steps = 0


    episode_steps = torch.zeros(
        num_envs,
        device=device,
        dtype=torch.long,
    )


    completed_episode_lengths = []


    # =====================================================
    # Final metric accumulators
    # =====================================================

    linear_sq_error_sum = 0.0
    linear_error_count = 0

    yaw_sq_error_sum = 0.0
    yaw_error_count = 0

    base_tilt_sum_degrees = 0.0
    base_tilt_count = 0

    joint_limit_violation_count = 0
    joint_limit_observation_count = 0


    # =====================================================
    # Per-environment episode accumulators
    # =====================================================

    episode_linear_sq_error = torch.zeros(
        num_envs,
        device=device,
        dtype=torch.float32,
    )


    episode_linear_count = torch.zeros(
        num_envs,
        device=device,
        dtype=torch.long,
    )


    episode_yaw_sq_error = torch.zeros(
        num_envs,
        device=device,
        dtype=torch.float32,
    )


    episode_yaw_count = torch.zeros(
        num_envs,
        device=device,
        dtype=torch.long,
    )


    episode_base_tilt_sum_degrees = torch.zeros(
        num_envs,
        device=device,
        dtype=torch.float32,
    )


    episode_base_tilt_count = torch.zeros(
        num_envs,
        device=device,
        dtype=torch.long,
    )


    episode_joint_limit_violations = torch.zeros(
        num_envs,
        device=device,
        dtype=torch.long,
    )


    episode_joint_limit_observations = torch.zeros(
        num_envs,
        device=device,
        dtype=torch.long,
    )


    print(
        f"\n[Benchmark] Running until "
        f"{target_episodes} "
        f"episodes complete..."
    )


    # =====================================================
    # Evaluation loop
    # =====================================================

    while (
        completed_episodes
        < target_episodes
    ):

        command = (
            env.unwrapped
            .command_manager
            .get_command(
                "base_velocity"
            )
        )


        # =================================================
        # Linear velocity tracking
        # =================================================

        velocity_yaw_frame = (
            quat_apply_inverse(
                yaw_quat(
                    robot.data.root_quat_w
                ),
                robot.data.root_lin_vel_w[
                    :, :3
                ],
            )
        )


        actual_linear_velocity = (
            velocity_yaw_frame[
                :, :2
            ]
        )


        commanded_linear_velocity = (
            command[
                :, :2
            ]
        )


        linear_error = (
            actual_linear_velocity
            - commanded_linear_velocity
        )


        episode_linear_sq_error += (
            torch.sum(
                linear_error ** 2,
                dim=1,
            )
        )


        episode_linear_count += 2


        # =================================================
        # Yaw velocity tracking
        # =================================================

        actual_yaw_velocity = (
            robot.data
            .root_ang_vel_w[
                :, 2
            ]
        )


        commanded_yaw_velocity = (
            command[
                :, 2
            ]
        )


        yaw_error = (
            actual_yaw_velocity
            - commanded_yaw_velocity
        )


        episode_yaw_sq_error += (
            yaw_error ** 2
        )


        episode_yaw_count += 1


        # =================================================
        # Base tilt
        # =================================================

        projected_gravity = (
            robot.data.projected_gravity_b
        )


        horizontal_gravity = (
            torch.linalg.vector_norm(
                projected_gravity[
                    :, :2
                ],
                dim=1,
            )
        )


        base_tilt_radians = (
            torch.atan2(
                horizontal_gravity,
                -projected_gravity[
                    :, 2
                ],
            )
        )


        base_tilt_degrees = (
            torch.rad2deg(
                base_tilt_radians
            )
        )


        episode_base_tilt_sum_degrees += (
            base_tilt_degrees
        )


        episode_base_tilt_count += 1


        # =================================================
        # Joint-limit violations
        # =================================================

        joint_positions = (
            robot.data.joint_pos
        )


        soft_joint_limits = (
            robot.data.soft_joint_pos_limits
        )


        lower_limits = (
            soft_joint_limits[
                :, :, 0
            ]
        )


        upper_limits = (
            soft_joint_limits[
                :, :, 1
            ]
        )


        joint_limit_violations = (
            (joint_positions < lower_limits)
            | (joint_positions > upper_limits)
        )


        violations_per_env = (
            joint_limit_violations
            .sum(
                dim=1
            )
        )


        episode_joint_limit_violations += (
            violations_per_env
        )


        num_joints = (
            joint_positions.shape[1]
        )


        episode_joint_limit_observations += (
            num_joints
        )


        # =================================================
        # Policy and simulation step
        # =================================================

        with torch.inference_mode():

            actions = policy(
                obs
            )


            (
                obs,
                rewards,
                dones,
                extras,
            ) = env.step(
                actions
            )


        vector_steps += 1

        episode_steps += 1


        # =================================================
        # Episode termination
        # =================================================

        time_outs = extras.get(
            "time_outs",
            torch.zeros_like(
                dones
            ),
        )


        done_indices = (
            torch.nonzero(
                dones,
                as_tuple=False,
            )
            .flatten()
        )


        # =================================================
        # Completed episodes
        # =================================================

        if len(done_indices) > 0:

            remaining = (
                target_episodes
                - completed_episodes
            )


            counted_indices = (
                done_indices[
                    :remaining
                ]
            )


            # -------------------------------------------------
            # Commit velocity metrics
            # -------------------------------------------------

            linear_sq_error_sum += (
                episode_linear_sq_error[
                    counted_indices
                ]
                .sum()
                .item()
            )


            linear_error_count += int(
                episode_linear_count[
                    counted_indices
                ]
                .sum()
                .item()
            )


            yaw_sq_error_sum += (
                episode_yaw_sq_error[
                    counted_indices
                ]
                .sum()
                .item()
            )


            yaw_error_count += int(
                episode_yaw_count[
                    counted_indices
                ]
                .sum()
                .item()
            )


            # -------------------------------------------------
            # Commit tilt metrics
            # -------------------------------------------------

            base_tilt_sum_degrees += (
                episode_base_tilt_sum_degrees[
                    counted_indices
                ]
                .sum()
                .item()
            )


            base_tilt_count += int(
                episode_base_tilt_count[
                    counted_indices
                ]
                .sum()
                .item()
            )


            # -------------------------------------------------
            # Commit joint-limit metrics
            # -------------------------------------------------

            joint_limit_violation_count += int(
                episode_joint_limit_violations[
                    counted_indices
                ]
                .sum()
                .item()
            )


            joint_limit_observation_count += int(
                episode_joint_limit_observations[
                    counted_indices
                ]
                .sum()
                .item()
            )


            # -------------------------------------------------
            # Success / fall classification
            # -------------------------------------------------

            for idx in counted_indices:

                idx = int(
                    idx.item()
                )


                completed_episode_lengths.append(
                    int(
                        episode_steps[
                            idx
                        ].item()
                    )
                )


                termination_type = (
                    classify_termination(
                        bool(
                            time_outs[
                                idx
                            ].item()
                        )
                    )
                        )


                if termination_type == "success":

                    successes += 1

                else:

                    falls += 1


                completed_episodes += 1


            # -------------------------------------------------
            # Reset per-environment accumulators
            # -------------------------------------------------

            episode_steps[
                done_indices
            ] = 0


            episode_linear_sq_error[
                done_indices
            ] = 0.0


            episode_linear_count[
                done_indices
            ] = 0


            episode_yaw_sq_error[
                done_indices
            ] = 0.0


            episode_yaw_count[
                done_indices
            ] = 0


            episode_base_tilt_sum_degrees[
                done_indices
            ] = 0.0


            episode_base_tilt_count[
                done_indices
            ] = 0


            episode_joint_limit_violations[
                done_indices
            ] = 0


            episode_joint_limit_observations[
                done_indices
            ] = 0


    # =====================================================
    # Final statistics
    # =====================================================

    success_rate, fall_rate = (
        calculate_episode_rates(
            successes,
            falls,
        )
    )


    mean_episode_length = (
        calculate_mean(
            sum(
                completed_episode_lengths
            ),
            len(
                completed_episode_lengths
            ),
        )
    )


    linear_velocity_rmse = (
        calculate_rmse(
            linear_sq_error_sum,
            linear_error_count,
        )
    )


    yaw_velocity_rmse = (
        calculate_rmse(
            yaw_sq_error_sum,
            yaw_error_count,
        )
    )


    mean_base_tilt_degrees = (
        calculate_mean(
            base_tilt_sum_degrees,
            base_tilt_count,
        )
    )


    joint_limit_violation_rate = (
        joint_limit_violation_count
        / joint_limit_observation_count
    )


    # Number of individual environment transitions
    # executed by the vectorized simulator.
    environment_transitions = (
        vector_steps
        * num_envs
    )


    # =====================================================
    # Machine-readable results
    # =====================================================

    results = {

        "experiment": (
            cfg["experiment"]["name"]
        ),

        "scenario": (
            cfg["scenario"]["name"]
        ),

        "physics_modifications": (
            cfg["scenario"].get(
                "physics_modifications"
            )
        ),

        "robot": (
            cfg["robot"]["name"]
        ),

        "task": task,

        "checkpoint": str(
            checkpoint
        ),

        "seed": seed,

        "num_envs": num_envs,

        "target_episodes": (
            target_episodes
        ),

        "completed_episodes": (
            completed_episodes
        ),

        "evaluation_protocol": {

            "success_criterion": (
                "survival_to_timeout"
            ),

            "success_definition": (
                "Episode reaches the configured "
                "time limit without early termination."
            ),

            "failure_definition": (
                "Episode terminates before the "
                "configured time limit."
            ),
        },

        "execution": {

            "vector_steps": (
                vector_steps
            ),

            "environment_transitions": (
                environment_transitions
            ),

            "environment_transitions_definition": (
                "vector_steps multiplied by num_envs; "
                "includes all simulated environments "
                "executed while collecting the requested "
                "benchmark episodes."
            ),
        },

        "metrics": {

            "successes": (
                successes
            ),

            "falls": (
                falls
            ),

            "success_rate": (
                success_rate
            ),

            "fall_rate": (
                fall_rate
            ),

            "mean_episode_length": (
                mean_episode_length
            ),

            "linear_velocity_rmse_mps": (
                linear_velocity_rmse
            ),

            "yaw_velocity_rmse_radps": (
                yaw_velocity_rmse
            ),

            "mean_base_tilt_degrees": (
                mean_base_tilt_degrees
            ),

            "joint_limit_violation_rate": (
                joint_limit_violation_rate
            ),

            "joint_limit_violation_count": (
                joint_limit_violation_count
            ),

            "joint_limit_observation_count": (
                joint_limit_observation_count
            ),
        },

        "metric_definitions": {

            "success_rate": (
                "Fraction of counted benchmark episodes "
                "that survived until normal timeout."
            ),

            "fall_rate": (
                "Fraction of counted benchmark episodes "
                "that terminated before normal timeout."
            ),

            "linear_velocity_rmse_mps": (
                "XY velocity command tracking RMSE "
                "in the gravity-aligned robot yaw frame."
            ),

            "yaw_velocity_rmse_radps": (
                "Yaw angular velocity command tracking "
                "RMSE using world-frame angular velocity."
            ),

            "mean_base_tilt_degrees": (
                "Mean angular deviation from upright "
                "computed from projected gravity in "
                "the robot base frame."
            ),

            "joint_limit_violation_rate": (
                "Fraction of joint-time observations "
                "whose joint position lies outside the "
                "configured soft joint-position limits."
            ),
        },
    }


    results_path = (
        output_dir
        / "results.json"
    )


    with open(
        results_path,
        "w",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
        )


    # =====================================================
    # Terminal output
    # =====================================================

    print(
        f"\nBenchmark complete: "
        f"{cfg['scenario']['name']}"
    )

    print(
        "------------------------------------"
    )


    print(
        f"Episodes                   : "
        f"{completed_episodes}"
    )


    print(
        f"Successes                  : "
        f"{successes}"
    )


    print(
        f"Falls                      : "
        f"{falls}"
    )


    print(
        f"Success rate               : "
        f"{success_rate:.2%}"
    )


    print(
        f"Fall rate                  : "
        f"{fall_rate:.2%}"
    )


    print(
        f"Mean episode length        : "
        f"{mean_episode_length:.2f}"
    )


    print(
        f"Linear velocity RMSE       : "
        f"{linear_velocity_rmse:.4f} m/s"
    )


    print(
        f"Yaw velocity RMSE          : "
        f"{yaw_velocity_rmse:.4f} rad/s"
    )


    print(
        f"Mean base tilt             : "
        f"{mean_base_tilt_degrees:.2f} deg"
    )


    print(
        f"Joint-limit violation rate : "
        f"{joint_limit_violation_rate:.4%}"
    )


    print(
        f"Vector steps               : "
        f"{vector_steps}"
    )


    print(
        f"Environment transitions    : "
        f"{environment_transitions}"
    )


    print(
        f"Success criterion          : "
        f"survival_to_timeout"
    )


    print(
        f"Results saved to           : "
        f"{results_path}"
    )


    env.close()


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":

    main()

    simulation_app.close()