# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg


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

# ==================================================================================================
# EXPERIMENTAL v4-G ZERO-COMMAND FULL-MOTION COST
# Positive physical-motion cost; use with a negative RewTerm weight.
# Active only when ||[vx_cmd, vy_cmd, wz_cmd]|| < command_threshold.
# ==================================================================================================
def zero_command_full_motion_l2(
    env,
    command_name: str,
    command_threshold: float,
    base_planar_scale: float,
    yaw_rate_scale: float,
    foot_xy_scale: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    'Penalize base planar, base yaw-rate, and foot planar motion near zero full command.'
    if command_threshold < 0.0:
        raise ValueError(f"command_threshold must be >= 0, got {command_threshold}")
    if base_planar_scale < 0.0:
        raise ValueError(f"base_planar_scale must be >= 0, got {base_planar_scale}")
    if yaw_rate_scale < 0.0:
        raise ValueError(f"yaw_rate_scale must be >= 0, got {yaw_rate_scale}")
    if foot_xy_scale < 0.0:
        raise ValueError(f"foot_xy_scale must be >= 0, got {foot_xy_scale}")

    asset = env.scene[asset_cfg.name]

    root_lin_vel_b = asset.data.root_lin_vel_b
    if hasattr(root_lin_vel_b, "torch"):
        root_lin_vel_b = root_lin_vel_b.torch

    root_ang_vel_b = asset.data.root_ang_vel_b
    if hasattr(root_ang_vel_b, "torch"):
        root_ang_vel_b = root_ang_vel_b.torch

    body_lin_vel_w = asset.data.body_lin_vel_w
    if hasattr(body_lin_vel_w, "torch"):
        body_lin_vel_w = body_lin_vel_w.torch

    base_planar_sq = torch.sum(torch.square(root_lin_vel_b[:, :2]), dim=1)
    yaw_rate_sq = torch.square(root_ang_vel_b[:, 2])
    foot_xy_sq = torch.sum(
        torch.square(body_lin_vel_w[:, asset_cfg.body_ids, :2]),
        dim=-1,
    )
    mean_foot_xy_sq = torch.mean(foot_xy_sq, dim=1)

    motion_cost = (
        base_planar_scale * base_planar_sq
        + yaw_rate_scale * yaw_rate_sq
        + foot_xy_scale * mean_foot_xy_sq
    )

    command = env.command_manager.get_command(command_name)
    full_command_norm = torch.linalg.norm(command[:, :3], dim=1)
    active = full_command_norm < command_threshold
    return motion_cost * active.to(dtype=motion_cost.dtype)

# ==================================================================================================
# EXPERIMENTAL v4-H BOUNDED ZERO-COMMAND FULL-MOTION COST
# Positive physical-motion cost; use with a negative RewTerm weight.
#
# Scientific parent: v4-B Direct-Yaw.
# Active only when ||[vx_cmd, vy_cmd, wz_cmd]|| < command_threshold.
#
# Yaw contribution:
#   1 - exp(-(wz / yaw_sigma)^2)
#
# This keeps the yaw contribution in [0, 1], preventing the unbounded wz^2
# early-scratch outliers diagnosed in v4-G.
# ==================================================================================================
def zero_command_bounded_full_motion(
    env,
    command_name: str,
    command_threshold: float,
    base_planar_scale: float,
    yaw_sigma: float,
    foot_xy_scale: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    'Penalize zero-command physical motion with a bounded yaw-rate contribution.'
    if command_threshold < 0.0:
        raise ValueError(f"command_threshold must be >= 0, got {command_threshold}")
    if base_planar_scale < 0.0:
        raise ValueError(f"base_planar_scale must be >= 0, got {base_planar_scale}")
    if yaw_sigma <= 0.0:
        raise ValueError(f"yaw_sigma must be > 0, got {yaw_sigma}")
    if foot_xy_scale < 0.0:
        raise ValueError(f"foot_xy_scale must be >= 0, got {foot_xy_scale}")

    asset = env.scene[asset_cfg.name]

    root_lin_vel_b = asset.data.root_lin_vel_b
    if hasattr(root_lin_vel_b, "torch"):
        root_lin_vel_b = root_lin_vel_b.torch

    root_ang_vel_b = asset.data.root_ang_vel_b
    if hasattr(root_ang_vel_b, "torch"):
        root_ang_vel_b = root_ang_vel_b.torch

    body_lin_vel_w = asset.data.body_lin_vel_w
    if hasattr(body_lin_vel_w, "torch"):
        body_lin_vel_w = body_lin_vel_w.torch

    base_planar_sq = torch.sum(torch.square(root_lin_vel_b[:, :2]), dim=1)

    yaw_rate = root_ang_vel_b[:, 2]
    bounded_yaw = 1.0 - torch.exp(-torch.square(yaw_rate / yaw_sigma))

    foot_xy_sq = torch.sum(
        torch.square(body_lin_vel_w[:, asset_cfg.body_ids, :2]),
        dim=-1,
    )
    mean_foot_xy_sq = torch.mean(foot_xy_sq, dim=1)

    motion_cost = (
        base_planar_scale * base_planar_sq
        + bounded_yaw
        + foot_xy_scale * mean_foot_xy_sq
    )

    command = env.command_manager.get_command(command_name)
    full_command_norm = torch.linalg.norm(command[:, :3], dim=1)
    active = full_command_norm < command_threshold
    return motion_cost * active.to(dtype=motion_cost.dtype)

# ==================================================================================================
# EXPERIMENTAL v4-J LOW-FREQUENCY STRAIGHT-LINE LATERAL-DRIFT COST
# Positive bounded cost; use with a negative RewTerm weight.
#
# Scientific parent: v4-B Direct-Yaw.
#
# The instantaneous body-frame lateral velocity error is low-pass filtered independently per
# environment. This targets persistent lateral drift without directly penalizing normal high-frequency
# gait-cycle sway. The EMA is reset both at episode reset and whenever the command leaves the
# straight-forward gate.
# ==================================================================================================
class straight_line_lateral_velocity_ema_bounded(ManagerTermBase):
    'Bounded cost on low-frequency body-frame lateral velocity error during straight locomotion.'

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)

        tau_s = float(cfg.params["tau_s"])
        sigma = float(cfg.params["sigma"])
        min_forward_command = float(cfg.params["min_forward_command"])
        lateral_command_threshold = float(cfg.params["lateral_command_threshold"])
        yaw_command_threshold = float(cfg.params["yaw_command_threshold"])

        if tau_s <= 0.0:
            raise ValueError(f"tau_s must be > 0, got {tau_s}")
        if sigma <= 0.0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        if min_forward_command < 0.0:
            raise ValueError(
                f"min_forward_command must be >= 0, got {min_forward_command}"
            )
        if lateral_command_threshold < 0.0:
            raise ValueError(
                "lateral_command_threshold must be >= 0, "
                f"got {lateral_command_threshold}"
            )
        if yaw_command_threshold < 0.0:
            raise ValueError(
                f"yaw_command_threshold must be >= 0, got {yaw_command_threshold}"
            )

        if not hasattr(env, "step_dt"):
            raise AttributeError("environment does not expose step_dt required by v4-J EMA reward")
        step_dt = float(env.step_dt)
        if step_dt <= 0.0:
            raise ValueError(f"env.step_dt must be > 0, got {step_dt}")

        self._alpha = 1.0 - torch.exp(
            torch.tensor(-step_dt / tau_s, device=env.device, dtype=torch.float32)
        )
        self._ema_error = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._ema_error[env_ids] = 0.0

    def __call__(
        self,
        env,
        command_name: str,
        tau_s: float,
        sigma: float,
        min_forward_command: float,
        lateral_command_threshold: float,
        yaw_command_threshold: float,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        del tau_s

        command = env.command_manager.get_command(command_name)
        asset = env.scene[asset_cfg.name]

        root_lin_vel_b = asset.data.root_lin_vel_b
        if hasattr(root_lin_vel_b, "torch"):
            root_lin_vel_b = root_lin_vel_b.torch

        lateral_error = root_lin_vel_b[:, 1] - command[:, 1]

        active = (
            (command[:, 0] > min_forward_command)
            & (torch.abs(command[:, 1]) < lateral_command_threshold)
            & (torch.abs(command[:, 2]) < yaw_command_threshold)
        )

        updated = self._ema_error + self._alpha * (lateral_error - self._ema_error)
        self._ema_error[:] = torch.where(active, updated, torch.zeros_like(updated))

        cost = 1.0 - torch.exp(-torch.square(self._ema_error / sigma))
        return cost * active.to(dtype=cost.dtype)

