# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO provenance config for G1 Upright Yaw-Direct v1."""

from isaaclab.utils.configclass import configclass

from .upright_ppo_cfg import UprightPPORunnerCfg


@configclass
class UprightYawDirectPPORunnerCfg(UprightPPORunnerCfg):
    """Upright PPO unchanged except for an isolated experiment name."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "humanoid_rl_lab_g1_upright_yaw_direct"
