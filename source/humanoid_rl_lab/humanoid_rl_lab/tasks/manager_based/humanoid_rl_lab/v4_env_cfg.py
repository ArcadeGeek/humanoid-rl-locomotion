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

from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils.configclass import configclass

from .mdp.v4_rewards import zero_command_motion_l2

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

# ==================================================================================================
# EXPERIMENTAL v4-C PRECISION REFINEMENT — GENUINE SCRATCH
# Scientific parent: v4-B Direct-Yaw.
# ONLY scientific delta:
#   track_lin_vel_xy_exp.params["std"]: 0.50 -> 0.25
# Everything else inherits unchanged from v4-B / Official G1 Flat.
# ==================================================================================================
@configclass
class HumanoidRlLabG1V4CPrecisionEnvCfg(HumanoidRlLabG1V4BDirectYawEnvCfg):
    'v4-C: v4-B plus a narrower Official XY velocity-tracking reward std.'

    def __post_init__(self) -> None:
        super().__post_init__()
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.25

# ==================================================================================================
# EXPERIMENTAL v4-D STANDING-EXPOSURE ABLATION - GENUINE SCRATCH
# Scientific parent: v4-B Direct-Yaw.
# ONLY behavioral/scientific delta:
#   rel_standing_envs: 0.02 -> 0.05
# Everything else inherits unchanged from v4-B / Official G1 Flat.
# No custom standing reward, imitation, reference motion, or gait clock is added.
# ==================================================================================================
@configclass
class HumanoidRlLabG1V4DStandingExposureEnvCfg(HumanoidRlLabG1V4BDirectYawEnvCfg):
    'v4-D: v4-B plus increased exposure to zero-command standing environments.'

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.base_velocity.rel_standing_envs = 0.05

# ==================================================================================================
# EXPERIMENTAL v4-E ZERO-COMMAND PHYSICAL-MOTION ABLATION - GENUINE SCRATCH
# Scientific parent: v4-B Direct-Yaw.
# ONLY scientific intervention:
#   add zero_command_motion_l2, active only when ||[vx, vy, wz]|| < 0.06
#   weight=-5.0, base_planar_scale=1.0, foot_xy_scale=0.25
# rel_standing_envs remains the v4-B / Official value 0.02.
# No v4-C precision change, no v4-D standing-exposure change, no joint-pose standing reward.
# No imitation, reference motion, or gait clock.
# ==================================================================================================
@configclass
class HumanoidRlLabG1V4EZeroCommandMotionEnvCfg(HumanoidRlLabG1V4BDirectYawEnvCfg):
    'v4-E: v4-B plus a full-command-gated zero-command physical-motion cost.'

    def __post_init__(self) -> None:
        super().__post_init__()
        self.rewards.zero_command_motion = RewTerm(
            func=zero_command_motion_l2,
            weight=-5.0,
            params={
                "command_name": "base_velocity",
                "command_threshold": 0.06,
                "base_planar_scale": 1.0,
                "foot_xy_scale": 0.25,
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                    preserve_order=True,
                ),
            },
        )
