# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Project-owned Unitree G1 flat-terrain locomotion configurations.

The training and play configurations initially inherit the official Isaac Lab
G1 task. Evaluation configurations keep the same policy interface while using
deterministic forward commands and deterministic reset states.
"""

from isaaclab.utils.configclass import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import (
    G1FlatEnvCfg,
    G1FlatEnvCfg_PLAY,
)


def _set_deterministic_reset(cfg) -> None:
    """Remove base pose and velocity randomness for repeatable evaluation."""

    pose_range = cfg.events.reset_base.params["pose_range"]
    pose_range["x"] = (0.0, 0.0)
    pose_range["y"] = (0.0, 0.0)
    pose_range["yaw"] = (0.0, 0.0)

    velocity_range = cfg.events.reset_base.params["velocity_range"]
    for key in ("x", "y", "z", "roll", "pitch", "yaw"):
        velocity_range[key] = (0.0, 0.0)


@configclass
class HumanoidRlLabEnvCfg(G1FlatEnvCfg):
    """Training configuration initially identical to the official G1 flat task."""

    pass


@configclass
class HumanoidRlLabEnvCfg_PLAY(G1FlatEnvCfg_PLAY):
    """Play configuration initially identical to the official G1 flat play task."""

    pass


@configclass
class HumanoidRlLabEnvCfg_EVAL(HumanoidRlLabEnvCfg_PLAY):
    """Deterministic evaluation at 0.4 m/s with zero yaw-rate command."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 1
        _set_deterministic_reset(self)

        command = self.commands.base_velocity
        command.heading_command = False
        command.rel_heading_envs = 0.0
        command.rel_standing_envs = 0.0
        command.ranges.lin_vel_x = (0.4, 0.4)
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = (0.0, 0.0)
        command.ranges.heading = (0.0, 0.0)


@configclass
class HumanoidRlLabEnvCfg_HEADING_EVAL(HumanoidRlLabEnvCfg_PLAY):
    """Deterministic evaluation at 0.4 m/s while holding world heading at zero."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 1
        _set_deterministic_reset(self)

        command = self.commands.base_velocity
        command.heading_command = True
        command.heading_control_stiffness = 0.5
        command.rel_heading_envs = 1.0
        command.rel_standing_envs = 0.0
        command.ranges.lin_vel_x = (0.4, 0.4)
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = (-0.5, 0.5)
        command.ranges.heading = (0.0, 0.0)

@configclass
class HumanoidRlLabEnvCfg_ROBUST_EVAL(HumanoidRlLabEnvCfg_EVAL):
    """Seeded evaluation with conservative initial base-velocity perturbations."""

    def __post_init__(self) -> None:
        super().__post_init__()

        velocity_range = self.events.reset_base.params["velocity_range"]
        velocity_range["x"] = (-0.15, 0.15)
        velocity_range["y"] = (-0.15, 0.15)
        velocity_range["z"] = (-0.05, 0.05)
        velocity_range["roll"] = (-0.10, 0.10)
        velocity_range["pitch"] = (-0.10, 0.10)
        velocity_range["yaw"] = (-0.20, 0.20)
