"""Verify that the custom H1 mechanical-power reward is registered."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym

from isaaclab.managers import RewardTermCfg as RewTerm

import isaaclab_tasks

from isaaclab_tasks.utils import parse_env_cfg

from training.rewards import mechanical_power_l1


TASK = "Isaac-Velocity-Flat-H1-v0"
WEIGHT = -5.0e-4


def main():
    env_cfg = parse_env_cfg(
        TASK,
        device="cuda:0",
        num_envs=4,
    )

    # Add our custom reward without editing Isaac Lab's H1 source.
    env_cfg.rewards.mechanical_power = RewTerm(
        func=mechanical_power_l1,
        weight=WEIGHT,
    )

    env = gym.make(
        TASK,
        cfg=env_cfg,
    )

    reward_manager = env.unwrapped.reward_manager

    print("\nActive reward terms:")
    print("--------------------")

    for term_name in reward_manager.active_terms:
        print(term_name)

    print("\nMechanical-power configuration:")
    print("--------------------------------")
    print(
        reward_manager.get_term_cfg(
            "mechanical_power"
        )
    )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
    