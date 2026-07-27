# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL configuration for conservative G1 upright fine-tuning."""

from isaaclab.utils.configclass import configclass

from .rsl_rl_ppo_cfg import PPORunnerCfg


@configclass
class UprightPPORunnerCfg(PPORunnerCfg):
    """Preserve the pretrained gait while adapting gradually to the upright reward."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.experiment_name = "humanoid_rl_lab_g1_upright"

        # Conservative warm-start settings. The fixed schedule prevents RSL-RL
        # from automatically increasing the learning rate during early fine-tuning.
        self.algorithm.learning_rate = 1.0e-4
        self.algorithm.entropy_coef = 0.002
        self.algorithm.schedule = "fixed"

        # Save intermediate checkpoints for quantitative comparisons.
        self.save_interval = 10
