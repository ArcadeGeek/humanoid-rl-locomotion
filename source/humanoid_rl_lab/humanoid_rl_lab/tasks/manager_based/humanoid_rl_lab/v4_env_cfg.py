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
