"""Train Unitree H1 with a custom mechanical-power reward."""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime

from isaaclab.app import AppLauncher


# =========================================================
# Command-line arguments
# =========================================================

parser = argparse.ArgumentParser(
    description="Train H1 with mechanical-power regularization."
)

parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Velocity-Flat-H1-v0",
)

parser.add_argument(
    "--num_envs",
    type=int,
    default=1024,
)

parser.add_argument(
    "--max_iterations",
    type=int,
    default=1500,
)

parser.add_argument(
    "--seed",
    type=int,
    default=42,
)

parser.add_argument(
    "--reward_weight",
    type=float,
    default=-5.0e-4,
)

parser.add_argument(
    "--resume_checkpoint",
    type=str,
    default=None,
    help="Checkpoint path to resume training from.",
)

parser.add_argument(
    "--resume_iterations",
    type=int,
    default=None,
    help="Number of additional PPO iterations after loading the checkpoint.",
)

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()


# =========================================================
# Launch Isaac Sim
# =========================================================

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# =========================================================
# Imports requiring Isaac Sim
# =========================================================

import gymnasium as gym

from rsl_rl.runners import OnPolicyRunner

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401

from isaaclab_tasks.utils import (
    load_cfg_from_registry,
    parse_env_cfg,
)

from training.rewards import mechanical_power_l1


# =========================================================
# Main
# =========================================================

def main():
    task = args_cli.task

    print("\nCustom H1 training")
    print("--------------------------------")
    print(f"Task          : {task}")
    print(f"Environments  : {args_cli.num_envs}")
    print(f"Iterations    : {args_cli.max_iterations}")
    print(f"Seed          : {args_cli.seed}")
    print(
        f"Power weight  : "
        f"{args_cli.reward_weight}"
    )

    # -----------------------------------------------------
    # Environment configuration
    # -----------------------------------------------------

    env_cfg = parse_env_cfg(
        task,
        device="cuda:0",
        num_envs=args_cli.num_envs,
    )

    env_cfg.seed = args_cli.seed

    # -----------------------------------------------------
    # Inject custom mechanical-power reward
    # -----------------------------------------------------

    env_cfg.rewards.mechanical_power = RewTerm(
        func=mechanical_power_l1,
        weight=args_cli.reward_weight,
    )

    # -----------------------------------------------------
    # RSL-RL configuration
    # -----------------------------------------------------

    agent_cfg = load_cfg_from_registry(
        task,
        "rsl_rl_cfg_entry_point",
    )

    agent_cfg.seed = args_cli.seed

    agent_cfg.max_iterations = (
        args_cli.max_iterations
    )

    agent_cfg.experiment_name = (
        "h1_flat_power_regularized"
    )

    # -----------------------------------------------------
    # Logging
    # -----------------------------------------------------

    log_root_path = os.path.abspath(
        os.path.join(
            "logs",
            "rsl_rl",
            agent_cfg.experiment_name,
        )
    )

    log_dir_name = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    log_dir = os.path.join(
        log_root_path,
        log_dir_name,
    )

    os.makedirs(
        log_dir,
        exist_ok=True,
    )

    env_cfg.log_dir = log_dir

    print(
        f"Log directory : {log_dir}"
    )

    # -----------------------------------------------------
    # Create environment
    # -----------------------------------------------------

    env = gym.make(
        task,
        cfg=env_cfg,
    )

    # -----------------------------------------------------
    # Verify custom reward registration
    # -----------------------------------------------------

    reward_manager = (
        env.unwrapped.reward_manager
    )

    print("\nActive reward verification")
    print("--------------------------------")

    if (
        "mechanical_power"
        not in reward_manager.active_terms
    ):
        raise RuntimeError(
            "mechanical_power reward was not registered"
        )

    power_cfg = (
        reward_manager.get_term_cfg(
            "mechanical_power"
        )
    )

    print(
        "mechanical_power registered"
    )

    print(
        f"weight = {power_cfg.weight}"
    )

    # -----------------------------------------------------
    # RSL-RL wrapper
    # -----------------------------------------------------

    env = RslRlVecEnvWrapper(
        env,
        clip_actions=agent_cfg.clip_actions,
    )

    # -----------------------------------------------------
    # PPO runner
    # -----------------------------------------------------

    runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=log_dir,
        device=agent_cfg.device,
    )

    # -----------------------------------------------------
    # Resume training from checkpoint
    # -----------------------------------------------------

    if args_cli.resume_checkpoint is not None:
        resume_checkpoint = os.path.abspath(
            args_cli.resume_checkpoint
        )

        if not os.path.exists(
            resume_checkpoint
        ):
            raise FileNotFoundError(
                f"Resume checkpoint not found: "
                f"{resume_checkpoint}"
            )

        print(
            f"\nResuming from checkpoint: "
            f"{resume_checkpoint}"
        )

        runner.load(
            resume_checkpoint,
            load_optimizer=True,
        )

        print(
            f"Loaded learning iteration: "
            f"{runner.current_learning_iteration}"
        )

    # -----------------------------------------------------
    # Save exact experiment configuration
    # -----------------------------------------------------

    params_dir = os.path.join(
        log_dir,
        "params",
    )

    os.makedirs(
        params_dir,
        exist_ok=True,
    )

    dump_yaml(
        os.path.join(
            params_dir,
            "env.yaml",
        ),
        env_cfg,
    )

    dump_yaml(
        os.path.join(
            params_dir,
            "agent.yaml",
        ),
        agent_cfg,
    )

    # -----------------------------------------------------
    # Determine number of iterations for this session
    # -----------------------------------------------------

    learning_iterations = (
        args_cli.resume_iterations
        if args_cli.resume_iterations is not None
        else agent_cfg.max_iterations
    )

    print(
        f"Learning iterations this session: "
        f"{learning_iterations}"
    )

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    print("\nStarting PPO training...")
    print("--------------------------------")

    start_time = time.time()

    runner.learn(
        num_learning_iterations=learning_iterations,
        init_at_random_ep_len=True,
    )

    training_time = (
        time.time()
        - start_time
    )

    print(
        f"\nTraining time: "
        f"{training_time:.2f} seconds"
    )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()