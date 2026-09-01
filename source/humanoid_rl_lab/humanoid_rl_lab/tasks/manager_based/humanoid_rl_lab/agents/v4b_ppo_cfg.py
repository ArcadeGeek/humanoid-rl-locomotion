# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

'PPO configuration for v4-B direct-yaw genuine-scratch training.'

from isaaclab.utils.configclass import configclass

from .v4a_ppo_cfg import V4AOfficialBaselinePPORunnerCfg


@configclass
class V4BDirectYawPPORunnerCfg(V4AOfficialBaselinePPORunnerCfg):
    'Official G1 Flat PPO semantics; only experiment metadata is separated for v4-B.'

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "humanoid_rl_lab_g1_v4b_direct_yaw_scratch"
