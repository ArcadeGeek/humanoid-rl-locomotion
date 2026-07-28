# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Conservative PPO settings for G1 upright speed recovery."""

from isaaclab.utils.configclass import configclass

from .upright_ppo_cfg import UprightPPORunnerCfg


@configclass
class UprightSpeedPPORunnerCfg(UprightPPORunnerCfg):
    """Fine-tune Upright Candidate v1 with a smaller fixed learning rate."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.experiment_name = "humanoid_rl_lab_g1_upright_speed"
        self.algorithm.learning_rate = 5.0e-5
        self.algorithm.entropy_coef = 0.002
        self.algorithm.schedule = "fixed"
        self.save_interval = 5