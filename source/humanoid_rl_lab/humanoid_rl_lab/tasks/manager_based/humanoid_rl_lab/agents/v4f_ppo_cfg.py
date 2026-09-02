# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from .v4b_ppo_cfg import V4BDirectYawPPORunnerCfg


@configclass
class V4FStandStillJointDeviationPPORunnerCfg(V4BDirectYawPPORunnerCfg):
    'v4-F keeps every v4-B / Official PPO hyperparameter unchanged.'

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "humanoid_rl_lab_g1_v4f_stand_still_joint_deviation_scratch"
