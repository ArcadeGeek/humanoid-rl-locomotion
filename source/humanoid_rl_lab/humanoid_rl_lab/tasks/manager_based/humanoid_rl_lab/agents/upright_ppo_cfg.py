# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL configuration for the project-owned G1 upright experiment."""

from isaaclab.utils.configclass import configclass

from .rsl_rl_ppo_cfg import PPORunnerCfg


@configclass
class UprightPPORunnerCfg(PPORunnerCfg):
    """Keep the validated PPO settings while separating upright experiment logs."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "humanoid_rl_lab_g1_upright"
