# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from .upright_yaw_direct_ppo_cfg import UprightYawDirectPPORunnerCfg


@configclass
class UprightYawDirectTightYawPPORunnerCfg(UprightYawDirectPPORunnerCfg):
    """PPO config for genuine-scratch v2-B tight yaw-tracking experiment."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "humanoid_rl_lab_g1_upright_yaw_direct_tight_yaw"
