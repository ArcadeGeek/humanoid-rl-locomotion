# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from .v4b_ppo_cfg import V4BDirectYawPPORunnerCfg


@configclass
class V4DStandingExposurePPORunnerCfg(V4BDirectYawPPORunnerCfg):
    'v4-D keeps every v4-B / Official PPO hyperparameter unchanged.'

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "humanoid_rl_lab_g1_v4d_standing_exposure_scratch"
