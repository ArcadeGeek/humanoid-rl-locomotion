# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab.managers import SceneEntityCfg


def zero_command_motion_l2(
    env,
    command_name: str,
    command_threshold: float,
    base_planar_scale: float,
    foot_xy_scale: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    'Penalize base and foot planar motion only when the full velocity command is near zero.'
    if command_threshold < 0.0:
        raise ValueError(f"command_threshold must be >= 0, got {command_threshold}")
    if base_planar_scale < 0.0:
        raise ValueError(f"base_planar_scale must be >= 0, got {base_planar_scale}")
    if foot_xy_scale < 0.0:
        raise ValueError(f"foot_xy_scale must be >= 0, got {foot_xy_scale}")

    asset = env.scene[asset_cfg.name]

    root_lin_vel_b = asset.data.root_lin_vel_b
    if hasattr(root_lin_vel_b, "torch"):
        root_lin_vel_b = root_lin_vel_b.torch

    body_lin_vel_w = asset.data.body_lin_vel_w
    if hasattr(body_lin_vel_w, "torch"):
        body_lin_vel_w = body_lin_vel_w.torch

    base_planar_sq = torch.sum(torch.square(root_lin_vel_b[:, :2]), dim=1)
    foot_xy_sq = torch.sum(
        torch.square(body_lin_vel_w[:, asset_cfg.body_ids, :2]),
        dim=-1,
    )
    mean_foot_xy_sq = torch.mean(foot_xy_sq, dim=1)

    motion_cost = (
        base_planar_scale * base_planar_sq
        + foot_xy_scale * mean_foot_xy_sq
    )

    command = env.command_manager.get_command(command_name)
    full_command_norm = torch.linalg.norm(command[:, :3], dim=1)
    active = full_command_norm < command_threshold
    return motion_cost * active.to(dtype=motion_cost.dtype)

# ==================================================================================================
# EXPERIMENTAL v4-F FULL-COMMAND STAND-STILL JOINT-DEVIATION COST
# Returns a positive cost; the RewTerm weight is negative.
# Active only when ||[vx, vy, wz]|| < command_threshold.
# ==================================================================================================
def stand_still_joint_deviation_full_command_l1(
    env,
    command_name: str,
    command_threshold: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    'Penalize leg deviation from default pose only when the full velocity command is near zero.'
    command = env.command_manager.get_command(command_name)
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default_joint_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    deviation_l1 = torch.sum(torch.abs(joint_pos - default_joint_pos), dim=1)
    stand_still_gate = torch.linalg.norm(command[:, :3], dim=1) < command_threshold
    return deviation_l1 * stand_still_gate
