# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from .upright_yaw_direct_wide_yaw_ppo_cfg import UprightYawDirectWideYawPPORunnerCfg


@configclass
class UprightYawDirectWideYawFlatOrientationPPORunnerCfg(UprightYawDirectWideYawPPORunnerCfg):
    """PPO config for genuine-scratch v2-G posture-only refinement."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "humanoid_rl_lab_g1_upright_yaw_direct_wide_yaw_flat_orientation"
