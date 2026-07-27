# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Project-owned Unitree G1 flat-terrain locomotion configurations.

The training and play configurations initially inherit the official Isaac Lab
G1 task. The evaluation configuration keeps the same policy interface while
using a deterministic forward velocity command.
"""

from isaaclab.utils.configclass import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import (
    G1FlatEnvCfg,
    G1FlatEnvCfg_PLAY,
)


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
    """Deterministic evaluation at 0.4 m/s forward velocity."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 1

        command = self.commands.base_velocity
        command.heading_command = False
        command.rel_heading_envs = 0.0
        command.rel_standing_envs = 0.0
        command.ranges.lin_vel_x = (0.4, 0.4)
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = (0.0, 0.0)
        command.ranges.heading = (0.0, 0.0)