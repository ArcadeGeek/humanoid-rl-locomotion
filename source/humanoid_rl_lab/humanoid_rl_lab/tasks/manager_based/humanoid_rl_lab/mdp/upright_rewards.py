# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms used only by the project-owned G1 upright task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def knee_flexion_above_threshold_l2(
    env: ManagerBasedRLEnv,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize positive knee flexion only above ``threshold``.

    The official G1 asset uses positive knee angles for flexion. Angles at or
    below the threshold produce zero penalty. The squared excess is summed over
    the selected knee joints for each environment.
    """

    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos
    if hasattr(joint_pos, "torch"):
        joint_pos = joint_pos.torch

    selected_joint_pos = joint_pos[:, asset_cfg.joint_ids]
    excess_flexion = torch.clamp(selected_joint_pos - threshold, min=0.0)
    return torch.sum(torch.square(excess_flexion), dim=1)


def _as_torch_tensor(value):
    """Return an Isaac data value as a torch tensor."""

    return value.torch if hasattr(value, "torch") else value


def track_heading_target_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    yaw_rate_threshold: float,
    min_planar_speed: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track the command term heading target near straight motion.

    The reference is the current ``heading_target`` sampled by the
    velocity command term. The reward is active only when the
    generated yaw-rate command is close to zero and the planar
    command is non-trivial. This avoids rewarding a fixed world
    heading and avoids applying the term to standing environments.

    Args:
        env: Manager-based RL environment.
        command_name: Velocity command term name.
        std: Heading-error exponential scale in radians.
        yaw_rate_threshold: Maximum absolute generated yaw-rate
            command for activating the reward.
        min_planar_speed: Minimum commanded planar speed for
            activating the reward.
        asset_cfg: Robot scene entity configuration.

    Returns:
        Per-environment reward in ``[0, 1]``.
    """

    if std <= 0.0:
        raise ValueError(f"std must be positive, got {std}.")
    if yaw_rate_threshold < 0.0:
        raise ValueError(
            "yaw_rate_threshold must be non-negative, "
            f"got {yaw_rate_threshold}."
        )
    if min_planar_speed < 0.0:
        raise ValueError(
            "min_planar_speed must be non-negative, "
            f"got {min_planar_speed}."
        )

    asset: Articulation = env.scene[asset_cfg.name]
    heading_w = _as_torch_tensor(asset.data.heading_w)
    command = _as_torch_tensor(
        env.command_manager.get_command(command_name)
    )
    command_term = env.command_manager.get_term(command_name)

    if not hasattr(command_term, "heading_target"):
        raise AttributeError(
            f"Command term {command_name!r} does not expose "
            "heading_target."
        )

    heading_target = _as_torch_tensor(
        command_term.heading_target
    )
    heading_error = wrap_to_pi(heading_target - heading_w)
    heading_reward = torch.exp(
        -torch.square(heading_error / std)
    )

    near_zero_yaw_rate = (
        torch.abs(command[:, 2]) < yaw_rate_threshold
    )
    planar_speed = torch.linalg.vector_norm(
        command[:, :2],
        dim=1,
    )
    moving_command = planar_speed > min_planar_speed

    return (
        heading_reward
        * near_zero_yaw_rate.to(dtype=heading_reward.dtype)
        * moving_command.to(dtype=heading_reward.dtype)
    )
