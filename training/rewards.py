"""Custom reward terms for H1 locomotion experiments."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def mechanical_power_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return whole-robot absolute mechanical actuator power.

    For every selected joint:

        power_j = |torque_j * velocity_j|

    The values are summed across joints to produce one value per
    vectorized environment.

    This represents simulated mechanical actuator power. It does not
    model electrical power, motor efficiency, gearbox losses, battery
    losses, or thermal effects.
    """

    asset: Articulation = env.scene[asset_cfg.name]

    applied_torque = asset.data.applied_torque[
        :,
        asset_cfg.joint_ids,
    ]

    joint_velocity = asset.data.joint_vel[
        :,
        asset_cfg.joint_ids,
    ]

    mechanical_power = torch.abs(
        applied_torque * joint_velocity
    )

    return torch.sum(
        mechanical_power,
        dim=1,
    )