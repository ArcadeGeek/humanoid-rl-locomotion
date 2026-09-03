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

# ==================================================================================================
# EXPERIMENTAL v4-K GEOMETRIC COMMAND-PATH CROSS-TRACK COST
# Positive bounded cost; use with a negative RewTerm weight.
#
# Scientific parent: v4-B Direct-Yaw.
#
# For near-straight commands, anchor the robot planar pose when the command segment begins and
# construct the geometric path implied by the constant body-frame command [vx, vy, wz].
# Penalize only lateral cross-track error to that path; along-track speed remains governed by
# the Official velocity-tracking rewards.
#
# At [0.40, 0, 0] the reference path is exactly a straight line in the initial heading direction.
# Small nonzero vy/wz inside the gate are handled by the command-consistent analytic path, so the
# reward does not incorrectly force a turning command onto a fixed line.
#
# State resets:
#   - episode reset
#   - gate exit/re-entry
#   - command change/resample while remaining inside the gate
# ==================================================================================================
class geometric_command_path_cross_track_bounded(ManagerTermBase):
    'Bounded geometric cross-track cost for near-straight command segments.'

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)

        sigma_m = float(cfg.params["sigma_m"])
        min_forward_command = float(cfg.params["min_forward_command"])
        lateral_command_threshold = float(cfg.params["lateral_command_threshold"])
        yaw_command_threshold = float(cfg.params["yaw_command_threshold"])
        command_change_threshold = float(cfg.params["command_change_threshold"])
        yaw_rate_epsilon = float(cfg.params["yaw_rate_epsilon"])

        if sigma_m <= 0.0:
            raise ValueError(f"sigma_m must be > 0, got {sigma_m}")
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
        if command_change_threshold <= 0.0:
            raise ValueError(
                f"command_change_threshold must be > 0, got {command_change_threshold}"
            )
        if yaw_rate_epsilon <= 0.0:
            raise ValueError(
                f"yaw_rate_epsilon must be > 0, got {yaw_rate_epsilon}"
            )
        if not hasattr(env, "step_dt"):
            raise AttributeError(
                "environment does not expose step_dt required by v4-K geometric reward"
            )

        step_dt = float(env.step_dt)
        if step_dt <= 0.0:
            raise ValueError(f"env.step_dt must be > 0, got {step_dt}")

        self._sigma_m = sigma_m
        self._min_forward_command = min_forward_command
        self._lateral_command_threshold = lateral_command_threshold
        self._yaw_command_threshold = yaw_command_threshold
        self._command_change_threshold = command_change_threshold
        self._yaw_rate_epsilon = yaw_rate_epsilon
        self._step_dt = step_dt

        self._anchor_pos_xy = torch.zeros(
            (env.num_envs, 2), device=env.device, dtype=torch.float32
        )
        self._anchor_heading = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.float32
        )
        self._stored_command = torch.zeros(
            (env.num_envs, 3), device=env.device, dtype=torch.float32
        )
        self._elapsed_s = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.float32
        )
        self._was_active = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.bool
        )

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)

        self._anchor_pos_xy[env_ids] = 0.0
        self._anchor_heading[env_ids] = 0.0
        self._stored_command[env_ids] = 0.0
        self._elapsed_s[env_ids] = 0.0
        self._was_active[env_ids] = False

    @staticmethod
    def _heading_from_wxyz(quat_wxyz: torch.Tensor) -> torch.Tensor:
        qw = quat_wxyz[:, 0]
        qx = quat_wxyz[:, 1]
        qy = quat_wxyz[:, 2]
        qz = quat_wxyz[:, 3]

        sin_yaw = 2.0 * (qw * qz + qx * qy)
        cos_yaw = 1.0 - 2.0 * (qy * qy + qz * qz)
        return torch.atan2(sin_yaw, cos_yaw)

    def __call__(
        self,
        env,
        command_name: str,
        sigma_m: float,
        min_forward_command: float,
        lateral_command_threshold: float,
        yaw_command_threshold: float,
        command_change_threshold: float,
        yaw_rate_epsilon: float,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        del (
            sigma_m,
            min_forward_command,
            lateral_command_threshold,
            yaw_command_threshold,
            command_change_threshold,
            yaw_rate_epsilon,
        )

        command = env.command_manager.get_command(command_name)[:, :3]
        asset = env.scene[asset_cfg.name]

        root_pos_w = asset.data.root_pos_w
        root_quat_w = asset.data.root_quat_w

        if hasattr(root_pos_w, "torch"):
            root_pos_w = root_pos_w.torch
        if hasattr(root_quat_w, "torch"):
            root_quat_w = root_quat_w.torch

        pos_xy = root_pos_w[:, :2]
        heading = self._heading_from_wxyz(root_quat_w)

        active = (
            (command[:, 0] > self._min_forward_command)
            & (torch.abs(command[:, 1]) < self._lateral_command_threshold)
            & (torch.abs(command[:, 2]) < self._yaw_command_threshold)
        )

        command_delta = torch.max(
            torch.abs(command - self._stored_command), dim=1
        ).values
        command_changed = command_delta > self._command_change_threshold

        segment_start = active & ((~self._was_active) | command_changed)

        if torch.any(segment_start):
            self._anchor_pos_xy[segment_start] = pos_xy[segment_start]
            self._anchor_heading[segment_start] = heading[segment_start]
            self._stored_command[segment_start] = command[segment_start]
            self._elapsed_s[segment_start] = 0.0

        inactive = ~active
        if torch.any(inactive):
            self._elapsed_s[inactive] = 0.0

        t = self._elapsed_s

        vx = self._stored_command[:, 0]
        vy = self._stored_command[:, 1]
        wz = self._stored_command[:, 2]

        wt = wz * t
        turning = torch.abs(wz) >= self._yaw_rate_epsilon
        safe_wz = torch.where(turning, wz, torch.ones_like(wz))

        sin_wt = torch.sin(wt)
        cos_wt = torch.cos(wt)

        ref_x_turn = (
            vx * sin_wt
            + vy * (cos_wt - 1.0)
        ) / safe_wz
        ref_y_turn = (
            vx * (1.0 - cos_wt)
            + vy * sin_wt
        ) / safe_wz

        ref_x_straight = vx * t
        ref_y_straight = vy * t

        ref_x_body0 = torch.where(
            turning, ref_x_turn, ref_x_straight
        )
        ref_y_body0 = torch.where(
            turning, ref_y_turn, ref_y_straight
        )

        c0 = torch.cos(self._anchor_heading)
        s0 = torch.sin(self._anchor_heading)

        ref_dx_world = (
            c0 * ref_x_body0
            - s0 * ref_y_body0
        )
        ref_dy_world = (
            s0 * ref_x_body0
            + c0 * ref_y_body0
        )

        ref_pos_x = self._anchor_pos_xy[:, 0] + ref_dx_world
        ref_pos_y = self._anchor_pos_xy[:, 1] + ref_dy_world

        command_velocity_angle = torch.atan2(vy, vx)
        tangent_heading = (
            self._anchor_heading
            + wt
            + command_velocity_angle
        )

        lateral_axis_x = -torch.sin(tangent_heading)
        lateral_axis_y = torch.cos(tangent_heading)

        error_x = pos_xy[:, 0] - ref_pos_x
        error_y = pos_xy[:, 1] - ref_pos_y

        cross_track = (
            error_x * lateral_axis_x
            + error_y * lateral_axis_y
        )

        cost = 1.0 - torch.exp(
            -torch.square(cross_track / self._sigma_m)
        )
        cost = cost * active.to(dtype=cost.dtype)

        self._elapsed_s[active] += self._step_dt
        self._was_active[:] = active

        return cost

# ==================================================================================================
# EXPERIMENTAL v4-L GEOMETRIC COMMAND PATH-POSE COST
# Positive bounded cost; use with a negative RewTerm weight.
#
# Scientific parent: v4-B Direct-Yaw.
#
# Single composite intervention for near-straight commands:
#   1) lateral cross-track error to the commanded planar path
#   2) body-heading error relative to the commanded integrated yaw
#
# cost = 1 - exp(
#     -(cross_track / sigma_cross_m)^2
#     -(heading_error / sigma_heading_rad)^2
# )
#
# At [0.40, 0, 0]:
#   - reference path is a straight line through the segment-start position
#   - reference heading is the segment-start heading
#
# For small nonzero vy/wz inside the gate:
#   - reference path follows the constant-body-twist command
#   - reference body heading is anchor_heading + wz * t
#
# Lateral command vy changes path tangent but NOT body heading. Therefore body-heading error is
# intentionally measured against integrated yaw, not against the path-tangent angle.
#
# State resets:
#   - episode reset
#   - gate exit/re-entry
#   - command change/resample while remaining inside the gate
# ==================================================================================================
class geometric_command_path_pose_bounded(ManagerTermBase):
    'Bounded composite command-path cross-track and heading cost.'

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)

        sigma_cross_m = float(cfg.params["sigma_cross_m"])
        sigma_heading_rad = float(cfg.params["sigma_heading_rad"])
        min_forward_command = float(cfg.params["min_forward_command"])
        lateral_command_threshold = float(cfg.params["lateral_command_threshold"])
        yaw_command_threshold = float(cfg.params["yaw_command_threshold"])
        command_change_threshold = float(cfg.params["command_change_threshold"])
        yaw_rate_epsilon = float(cfg.params["yaw_rate_epsilon"])

        if sigma_cross_m <= 0.0:
            raise ValueError(
                f"sigma_cross_m must be > 0, got {sigma_cross_m}"
            )
        if sigma_heading_rad <= 0.0:
            raise ValueError(
                f"sigma_heading_rad must be > 0, got {sigma_heading_rad}"
            )
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
        if command_change_threshold <= 0.0:
            raise ValueError(
                f"command_change_threshold must be > 0, got {command_change_threshold}"
            )
        if yaw_rate_epsilon <= 0.0:
            raise ValueError(
                f"yaw_rate_epsilon must be > 0, got {yaw_rate_epsilon}"
            )
        if not hasattr(env, "step_dt"):
            raise AttributeError(
                "environment does not expose step_dt required by v4-L path-pose reward"
            )

        step_dt = float(env.step_dt)
        if step_dt <= 0.0:
            raise ValueError(
                f"env.step_dt must be > 0, got {step_dt}"
            )

        self._sigma_cross_m = sigma_cross_m
        self._sigma_heading_rad = sigma_heading_rad
        self._min_forward_command = min_forward_command
        self._lateral_command_threshold = lateral_command_threshold
        self._yaw_command_threshold = yaw_command_threshold
        self._command_change_threshold = command_change_threshold
        self._yaw_rate_epsilon = yaw_rate_epsilon
        self._step_dt = step_dt

        self._anchor_pos_xy = torch.zeros(
            (env.num_envs, 2),
            device=env.device,
            dtype=torch.float32,
        )
        self._anchor_heading = torch.zeros(
            env.num_envs,
            device=env.device,
            dtype=torch.float32,
        )
        self._stored_command = torch.zeros(
            (env.num_envs, 3),
            device=env.device,
            dtype=torch.float32,
        )
        self._elapsed_s = torch.zeros(
            env.num_envs,
            device=env.device,
            dtype=torch.float32,
        )
        self._was_active = torch.zeros(
            env.num_envs,
            device=env.device,
            dtype=torch.bool,
        )

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)

        self._anchor_pos_xy[env_ids] = 0.0
        self._anchor_heading[env_ids] = 0.0
        self._stored_command[env_ids] = 0.0
        self._elapsed_s[env_ids] = 0.0
        self._was_active[env_ids] = False

    @staticmethod
    def _heading_from_wxyz(
        quat_wxyz: torch.Tensor,
    ) -> torch.Tensor:
        qw = quat_wxyz[:, 0]
        qx = quat_wxyz[:, 1]
        qy = quat_wxyz[:, 2]
        qz = quat_wxyz[:, 3]

        sin_yaw = 2.0 * (qw * qz + qx * qy)
        cos_yaw = 1.0 - 2.0 * (qy * qy + qz * qz)
        return torch.atan2(sin_yaw, cos_yaw)

    @staticmethod
    def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
        return torch.atan2(
            torch.sin(angle),
            torch.cos(angle),
        )

    def __call__(
        self,
        env,
        command_name: str,
        sigma_cross_m: float,
        sigma_heading_rad: float,
        min_forward_command: float,
        lateral_command_threshold: float,
        yaw_command_threshold: float,
        command_change_threshold: float,
        yaw_rate_epsilon: float,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        del (
            sigma_cross_m,
            sigma_heading_rad,
            min_forward_command,
            lateral_command_threshold,
            yaw_command_threshold,
            command_change_threshold,
            yaw_rate_epsilon,
        )

        command = env.command_manager.get_command(command_name)[:, :3]
        asset = env.scene[asset_cfg.name]

        root_pos_w = asset.data.root_pos_w
        root_quat_w = asset.data.root_quat_w

        if hasattr(root_pos_w, "torch"):
            root_pos_w = root_pos_w.torch
        if hasattr(root_quat_w, "torch"):
            root_quat_w = root_quat_w.torch

        pos_xy = root_pos_w[:, :2]
        heading = self._heading_from_wxyz(root_quat_w)

        active = (
            (command[:, 0] > self._min_forward_command)
            & (torch.abs(command[:, 1]) < self._lateral_command_threshold)
            & (torch.abs(command[:, 2]) < self._yaw_command_threshold)
        )

        command_delta = torch.max(
            torch.abs(command - self._stored_command),
            dim=1,
        ).values
        command_changed = (
            command_delta > self._command_change_threshold
        )
        segment_start = active & (
            (~self._was_active) | command_changed
        )

        if torch.any(segment_start):
            self._anchor_pos_xy[segment_start] = pos_xy[segment_start]
            self._anchor_heading[segment_start] = heading[segment_start]
            self._stored_command[segment_start] = command[segment_start]
            self._elapsed_s[segment_start] = 0.0

        inactive = ~active
        if torch.any(inactive):
            self._elapsed_s[inactive] = 0.0

        t = self._elapsed_s

        vx = self._stored_command[:, 0]
        vy = self._stored_command[:, 1]
        wz = self._stored_command[:, 2]

        wt = wz * t
        turning = (
            torch.abs(wz) >= self._yaw_rate_epsilon
        )
        safe_wz = torch.where(
            turning,
            wz,
            torch.ones_like(wz),
        )

        sin_wt = torch.sin(wt)
        cos_wt = torch.cos(wt)

        ref_x_turn = (
            vx * sin_wt
            + vy * (cos_wt - 1.0)
        ) / safe_wz
        ref_y_turn = (
            vx * (1.0 - cos_wt)
            + vy * sin_wt
        ) / safe_wz

        ref_x_straight = vx * t
        ref_y_straight = vy * t

        ref_x_body0 = torch.where(
            turning,
            ref_x_turn,
            ref_x_straight,
        )
        ref_y_body0 = torch.where(
            turning,
            ref_y_turn,
            ref_y_straight,
        )

        c0 = torch.cos(self._anchor_heading)
        s0 = torch.sin(self._anchor_heading)

        ref_dx_world = (
            c0 * ref_x_body0
            - s0 * ref_y_body0
        )
        ref_dy_world = (
            s0 * ref_x_body0
            + c0 * ref_y_body0
        )

        ref_pos_x = (
            self._anchor_pos_xy[:, 0]
            + ref_dx_world
        )
        ref_pos_y = (
            self._anchor_pos_xy[:, 1]
            + ref_dy_world
        )

        command_velocity_angle = torch.atan2(vy, vx)
        path_tangent_heading = (
            self._anchor_heading
            + wt
            + command_velocity_angle
        )

        lateral_axis_x = -torch.sin(path_tangent_heading)
        lateral_axis_y = torch.cos(path_tangent_heading)

        error_x = pos_xy[:, 0] - ref_pos_x
        error_y = pos_xy[:, 1] - ref_pos_y

        cross_track = (
            error_x * lateral_axis_x
            + error_y * lateral_axis_y
        )

        # Body-heading target depends on commanded yaw only.
        reference_body_heading = (
            self._anchor_heading
            + wt
        )
        heading_error = self._wrap_to_pi(
            heading - reference_body_heading
        )

        exponent = (
            torch.square(
                cross_track / self._sigma_cross_m
            )
            + torch.square(
                heading_error / self._sigma_heading_rad
            )
        )

        cost = 1.0 - torch.exp(-exponent)
        cost = cost * active.to(dtype=cost.dtype)

        self._elapsed_s[active] += self._step_dt
        self._was_active[:] = active

        return cost
