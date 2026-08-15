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
from isaaclab.managers import ManagerTermBase

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


def track_ang_vel_z_pseudo_huber(
    env: ManagerBasedRLEnv,
    command_name: str,
    delta: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Dense pseudo-Huber penalty for commanded world-frame yaw-rate error.

    The returned quantity is non-negative and dimensionless:
    ``sqrt(1 + (error / delta)^2) - 1``.
    Use a negative reward weight to penalize yaw-rate tracking error.
    """

    if delta <= 0.0:
        raise ValueError(f"delta must be positive, got {delta}.")

    asset: Articulation = env.scene[asset_cfg.name]
    actual_wz = _as_torch_tensor(asset.data.root_ang_vel_w[:, 2])
    command = _as_torch_tensor(
        env.command_manager.get_command(command_name)
    )

    normalized_error = (command[:, 2] - actual_wz) / delta
    return torch.sqrt(1.0 + torch.square(normalized_error)) - 1.0


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

def track_heading_target_deadband_exp(env: ManagerBasedRLEnv, command_name: str, std: float, yaw_rate_threshold: float, min_planar_speed: float, asset_cfg: SceneEntityCfg=SceneEntityCfg('robot'), deadband: float=0.05) -> torch.Tensor:
    """Track commanded heading with an exponential reward after a symmetric angular deadband."""
    if std <= 0.0:
        raise ValueError(f'std must be positive, got {std}.')
    if yaw_rate_threshold < 0.0:
        raise ValueError(f'yaw_rate_threshold must be non-negative, got {yaw_rate_threshold}.')
    if min_planar_speed < 0.0:
        raise ValueError(f'min_planar_speed must be non-negative, got {min_planar_speed}.')
    asset: Articulation = env.scene[asset_cfg.name]
    heading_w = _as_torch_tensor(asset.data.heading_w)
    command = _as_torch_tensor(env.command_manager.get_command(command_name))
    command_term = env.command_manager.get_term(command_name)
    if not hasattr(command_term, 'heading_target'):
        raise AttributeError(f'Command term {command_name!r} does not expose heading_target.')
    heading_target = _as_torch_tensor(command_term.heading_target)
    heading_error = wrap_to_pi(heading_target - heading_w)
    effective_heading_error = torch.clamp(torch.abs(heading_error) - deadband, min=0.0)
    heading_reward = torch.exp(-torch.square(effective_heading_error / std))
    near_zero_yaw_rate = torch.abs(command[:, 2]) < yaw_rate_threshold
    planar_speed = torch.linalg.vector_norm(command[:, :2], dim=1)
    moving_command = planar_speed > min_planar_speed
    return heading_reward * near_zero_yaw_rate.to(dtype=heading_reward.dtype) * moving_command.to(dtype=heading_reward.dtype)

def yaw_rate_near_heading_target_l2(env: ManagerBasedRLEnv, command_name: str, std: float, yaw_rate_threshold: float, min_planar_speed: float, asset_cfg: SceneEntityCfg=SceneEntityCfg('robot'), deadband: float=0.05, near_target: float=0.1) -> torch.Tensor:
    """Penalize yaw rate only near the heading target while reusing all validated heading reward gates."""
    heading_gate = track_heading_target_deadband_exp(env, command_name=command_name, std=std, yaw_rate_threshold=yaw_rate_threshold, min_planar_speed=min_planar_speed, asset_cfg=asset_cfg, deadband=deadband) > 0.0
    command_term = env.command_manager.get_term(command_name)
    heading_target = _as_torch_tensor(command_term.heading_target)
    heading_w = _as_torch_tensor(env.scene['robot'].data.heading_w)
    heading_error = wrap_to_pi(heading_target - heading_w)
    near_target_gate = torch.abs(heading_error) <= near_target
    base_wz = _as_torch_tensor(env.scene['robot'].data.root_ang_vel_w[:, 2])
    return torch.square(base_wz) * heading_gate.to(dtype=base_wz.dtype) * near_target_gate.to(dtype=base_wz.dtype)

class yaw_rate_change_near_heading_target_l2(ManagerTermBase):
    # Penalize step-to-step yaw-rate changes near the heading target.

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._previous_wz = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.float32,
        )
        self._initialized = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.bool,
        )

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self._initialized.zero_()
            self._previous_wz.zero_()
        else:
            self._initialized[env_ids] = False
            self._previous_wz[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        yaw_rate_threshold: float,
        min_planar_speed: float,
        near_target: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        if yaw_rate_threshold < 0.0:
            raise ValueError(
                f"yaw_rate_threshold must be non-negative, got {yaw_rate_threshold}."
            )
        if min_planar_speed < 0.0:
            raise ValueError(
                f"min_planar_speed must be non-negative, got {min_planar_speed}."
            )
        if near_target < 0.0:
            raise ValueError(
                f"near_target must be non-negative, got {near_target}."
            )

        asset: Articulation = env.scene[asset_cfg.name]
        current_wz = _as_torch_tensor(
            asset.data.root_ang_vel_w[:, 2]
        )
        heading_w = _as_torch_tensor(asset.data.heading_w)

        command = _as_torch_tensor(
            env.command_manager.get_command(command_name)
        )
        command_term = env.command_manager.get_term(command_name)
        if not hasattr(command_term, "heading_target"):
            raise AttributeError(
                f"Command term {command_name!r} does not expose heading_target."
            )
        heading_target = _as_torch_tensor(
            command_term.heading_target
        )

        heading_error = wrap_to_pi(heading_target - heading_w)
        near_target_gate = torch.abs(heading_error) <= near_target
        near_zero_yaw_rate = (
            torch.abs(command[:, 2]) < yaw_rate_threshold
        )
        planar_speed = torch.linalg.vector_norm(
            command[:, :2],
            dim=1,
        )
        moving_command = planar_speed > min_planar_speed

        delta_wz = current_wz - self._previous_wz
        valid_history = self._initialized.clone()

        raw_penalty = torch.square(delta_wz)
        gated_penalty = (
            raw_penalty
            * valid_history.to(dtype=raw_penalty.dtype)
            * near_target_gate.to(dtype=raw_penalty.dtype)
            * near_zero_yaw_rate.to(dtype=raw_penalty.dtype)
            * moving_command.to(dtype=raw_penalty.dtype)
        )

        self._previous_wz.copy_(current_wz)
        self._initialized.fill_(True)

        return gated_penalty.clone()

def yaw_rate_deadzone_near_heading_target_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    yaw_rate_threshold: float,
    min_planar_speed: float,
    near_target: float,
    free_yaw_rate: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    # Penalize only actual yaw rate above a small free zone near target.
    if yaw_rate_threshold < 0.0:
        raise ValueError(
            f"yaw_rate_threshold must be non-negative, got {yaw_rate_threshold}."
        )
    if min_planar_speed < 0.0:
        raise ValueError(
            f"min_planar_speed must be non-negative, got {min_planar_speed}."
        )
    if near_target < 0.0:
        raise ValueError(
            f"near_target must be non-negative, got {near_target}."
        )
    if free_yaw_rate < 0.0:
        raise ValueError(
            f"free_yaw_rate must be non-negative, got {free_yaw_rate}."
        )

    asset: Articulation = env.scene[asset_cfg.name]
    actual_wz = _as_torch_tensor(
        asset.data.root_ang_vel_w[:, 2]
    )
    heading_w = _as_torch_tensor(asset.data.heading_w)

    command = _as_torch_tensor(
        env.command_manager.get_command(command_name)
    )
    command_term = env.command_manager.get_term(command_name)
    if not hasattr(command_term, "heading_target"):
        raise AttributeError(
            f"Command term {command_name!r} does not expose heading_target."
        )
    heading_target = _as_torch_tensor(
        command_term.heading_target
    )

    heading_error = wrap_to_pi(heading_target - heading_w)
    near_target_gate = torch.abs(heading_error) <= near_target
    near_zero_yaw_rate = (
        torch.abs(command[:, 2]) < yaw_rate_threshold
    )
    planar_speed = torch.linalg.vector_norm(
        command[:, :2],
        dim=1,
    )
    moving_command = planar_speed > min_planar_speed

    excess_wz = torch.clamp(
        torch.abs(actual_wz) - free_yaw_rate,
        min=0.0,
    )
    raw_penalty = torch.square(excess_wz)

    return (
        raw_penalty
        * near_target_gate.to(dtype=raw_penalty.dtype)
        * near_zero_yaw_rate.to(dtype=raw_penalty.dtype)
        * moving_command.to(dtype=raw_penalty.dtype)
    )

class track_heading_target_filtered_deadband_exp(ManagerTermBase):
    # Track persistent commanded-heading error using a wrapped angular EMA.

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._filtered_heading_error = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.float32,
        )
        self._initialized = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.bool,
        )

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self._filtered_heading_error.zero_()
            self._initialized.zero_()
        else:
            self._filtered_heading_error[env_ids] = 0.0
            self._initialized[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        std: float,
        yaw_rate_threshold: float,
        min_planar_speed: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        deadband: float = 0.05,
        ema_alpha: float = 0.960789439152323,
    ) -> torch.Tensor:
        if std <= 0.0:
            raise ValueError(f"std must be positive, got {std}.")
        if yaw_rate_threshold < 0.0:
            raise ValueError(
                f"yaw_rate_threshold must be non-negative, got {yaw_rate_threshold}."
            )
        if min_planar_speed < 0.0:
            raise ValueError(
                f"min_planar_speed must be non-negative, got {min_planar_speed}."
            )
        if deadband < 0.0:
            raise ValueError(f"deadband must be non-negative, got {deadband}.")
        if not 0.0 <= ema_alpha < 1.0:
            raise ValueError(
                f"ema_alpha must satisfy 0 <= alpha < 1, got {ema_alpha}."
            )

        asset: Articulation = env.scene[asset_cfg.name]
        heading_w = _as_torch_tensor(asset.data.heading_w)
        command = _as_torch_tensor(
            env.command_manager.get_command(command_name)
        )
        command_term = env.command_manager.get_term(command_name)
        if not hasattr(command_term, "heading_target"):
            raise AttributeError(
                f"Command term {command_name!r} does not expose heading_target."
            )
        heading_target = _as_torch_tensor(command_term.heading_target)

        heading_error = wrap_to_pi(heading_target - heading_w)

        residual = wrap_to_pi(
            heading_error - self._filtered_heading_error
        )
        updated_error = wrap_to_pi(
            self._filtered_heading_error
            + (1.0 - ema_alpha) * residual
        )
        filtered_error = torch.where(
            self._initialized,
            updated_error,
            heading_error,
        )

        self._filtered_heading_error.copy_(filtered_error)
        self._initialized.fill_(True)

        effective_heading_error = torch.clamp(
            torch.abs(filtered_error) - deadband,
            min=0.0,
        )
        heading_reward = torch.exp(
            -torch.square(effective_heading_error / std)
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

class track_heading_target_progress_deadband(ManagerTermBase):
    # Signed progress rate of persistent heading error outside a deadband.

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._previous_effective_error = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )
        self._previous_heading_target = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )
        self._initialized = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self._previous_effective_error.zero_()
            self._previous_heading_target.zero_()
            self._initialized.zero_()
        else:
            self._previous_effective_error[env_ids] = 0.0
            self._previous_heading_target[env_ids] = 0.0
            self._initialized[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        std: float,
        yaw_rate_threshold: float,
        min_planar_speed: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        deadband: float = 0.05,
        policy_dt: float = 0.02,
        target_change_tolerance: float = 1.0e-4,
    ) -> torch.Tensor:
        if std <= 0.0:
            raise ValueError(f"std must be positive, got {std}.")
        if yaw_rate_threshold < 0.0:
            raise ValueError(
                f"yaw_rate_threshold must be non-negative, got {yaw_rate_threshold}."
            )
        if min_planar_speed < 0.0:
            raise ValueError(
                f"min_planar_speed must be non-negative, got {min_planar_speed}."
            )
        if deadband < 0.0:
            raise ValueError(f"deadband must be non-negative, got {deadband}.")
        if policy_dt <= 0.0:
            raise ValueError(f"policy_dt must be positive, got {policy_dt}.")
        if target_change_tolerance < 0.0:
            raise ValueError(
                "target_change_tolerance must be non-negative, "
                f"got {target_change_tolerance}."
            )

        asset: Articulation = env.scene[asset_cfg.name]
        heading_w = _as_torch_tensor(asset.data.heading_w)
        command = _as_torch_tensor(
            env.command_manager.get_command(command_name)
        )
        command_term = env.command_manager.get_term(command_name)
        if not hasattr(command_term, "heading_target"):
            raise AttributeError(
                f"Command term {command_name!r} does not expose heading_target."
            )
        heading_target = _as_torch_tensor(command_term.heading_target)

        heading_error = wrap_to_pi(heading_target - heading_w)
        effective_error = torch.clamp(
            torch.abs(heading_error) - deadband,
            min=0.0,
        )

        target_delta = wrap_to_pi(
            heading_target - self._previous_heading_target
        )
        target_changed = (
            torch.abs(target_delta) > target_change_tolerance
        )
        comparable = self._initialized & (~target_changed)

        progress_rate = torch.where(
            comparable,
            (
                self._previous_effective_error - effective_error
            ) / (std * policy_dt),
            torch.zeros_like(effective_error),
        )

        self._previous_effective_error.copy_(effective_error)
        self._previous_heading_target.copy_(heading_target)
        self._initialized.fill_(True)

        near_zero_yaw_rate = (
            torch.abs(command[:, 2]) < yaw_rate_threshold
        )
        planar_speed = torch.linalg.vector_norm(command[:, :2], dim=1)
        moving_command = planar_speed > min_planar_speed

        active = near_zero_yaw_rate & moving_command
        return progress_rate * active.to(dtype=progress_rate.dtype)
