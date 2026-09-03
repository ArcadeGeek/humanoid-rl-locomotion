# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from .v4b_ppo_cfg import V4BDirectYawPPORunnerCfg


@configclass
class V4KGeometricPathPrecisionPPORunnerCfg(V4BDirectYawPPORunnerCfg):
    'v4-K PPO: v4-B PPO unchanged except for experiment_name.'

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "humanoid_rl_lab_g1_v4k_geometric_path_precision_scratch"
