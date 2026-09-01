# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""v4-A genuine-scratch control task.

Scientific purpose:
- reproduce the official Isaac Lab G1 Flat locomotion formulation inside humanoid_rl_lab;
- preserve official reward/action/observation/event semantics;
- provide a clean scratch baseline before adding project-specific refinements.

No imitation, reference trajectory, gait clock, custom knee penalty, or v2/v3 inheritance is used.
"""

from isaaclab.utils.configclass import configclass

from .humanoid_rl_lab_env_cfg import HumanoidRlLabEnvCfg


@configclass
class HumanoidRlLabG1V4AOfficialBaselineEnvCfg(HumanoidRlLabEnvCfg):
    """Official G1 Flat semantics, isolated under the v4 scientific campaign."""

    pass

# ==================================================================================================
# v4-B DIRECT-YAW CAUSAL ABLATION
# Genuine-scratch training task derived from v4-A.
#
# Behavioral change versus v4-A:
#   heading_command: True -> False
#
# Everything else is inherited unchanged from the Official G1 Flat semantics:
# command ranges, rewards, observations, perturbation events, actions, scene and termination logic.
# No v2/v3/Upright configuration is inherited.
# ==================================================================================================
@configclass
class HumanoidRlLabG1V4BDirectYawEnvCfg(HumanoidRlLabG1V4AOfficialBaselineEnvCfg):
    'v4-B: v4-A Official baseline with direct yaw-rate commands.'

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.base_velocity.heading_command = False

