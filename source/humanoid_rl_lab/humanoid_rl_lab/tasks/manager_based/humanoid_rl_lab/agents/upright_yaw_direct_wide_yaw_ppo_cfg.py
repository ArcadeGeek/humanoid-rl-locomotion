# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from .upright_yaw_direct_ppo_cfg import UprightYawDirectPPORunnerCfg


@configclass
class UprightYawDirectWideYawPPORunnerCfg(UprightYawDirectPPORunnerCfg):
    """PPO config for genuine-scratch v2-F wide-yaw command experiment."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "humanoid_rl_lab_g1_upright_yaw_direct_wide_yaw"
