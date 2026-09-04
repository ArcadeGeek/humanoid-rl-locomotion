# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Final public v4 locomotion lineage for Unitree G1.

v4-A reproduces the Official Isaac Lab G1 Flat formulation for genuine-scratch training.
v4-B is its direct-yaw causal ablation.
v4-N reuses the v4-B environment and adds native RSL-RL left/right symmetry augmentation
in its PPO configuration. No imitation, reference trajectory, or gait clock is used.
"""

from isaaclab.utils.configclass import configclass

from .humanoid_rl_lab_env_cfg import HumanoidRlLabEnvCfg


@configclass
class HumanoidRlLabG1V4AOfficialBaselineEnvCfg(HumanoidRlLabEnvCfg):
    """Official G1 Flat semantics, isolated under the v4 scientific campaign."""

    pass


@configclass
class HumanoidRlLabG1V4BDirectYawEnvCfg(HumanoidRlLabG1V4AOfficialBaselineEnvCfg):
    """v4-B: v4-A Official baseline with direct yaw-rate commands."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.base_velocity.heading_command = False
