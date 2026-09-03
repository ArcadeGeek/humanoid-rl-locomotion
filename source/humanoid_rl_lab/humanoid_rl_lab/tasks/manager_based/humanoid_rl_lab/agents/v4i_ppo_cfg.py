# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause

"PPO configuration for v4-I delayed bounded zero-command full-motion genuine-scratch training."

from isaaclab.utils.configclass import configclass

from .v4b_ppo_cfg import V4BDirectYawPPORunnerCfg


@configclass
class V4IDelayedBoundedZeroCommandFullMotionPPORunnerCfg(V4BDirectYawPPORunnerCfg):
    "v4-I keeps every v4-B / Official PPO hyperparameter unchanged."

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "humanoid_rl_lab_g1_v4i_delayed_bounded_zero_command_full_motion_scratch"
