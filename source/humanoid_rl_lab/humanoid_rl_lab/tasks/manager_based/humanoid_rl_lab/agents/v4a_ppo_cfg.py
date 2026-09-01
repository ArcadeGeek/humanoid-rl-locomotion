# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO configuration for v4-A genuine-scratch official-baseline reproduction."""

from isaaclab.utils.configclass import configclass

from .rsl_rl_ppo_cfg import PPORunnerCfg


@configclass
class V4AOfficialBaselinePPORunnerCfg(PPORunnerCfg):
    """Official G1 Flat PPO semantics with isolated v4-A experiment provenance."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "humanoid_rl_lab_g1_v4a_official_baseline_scratch"
