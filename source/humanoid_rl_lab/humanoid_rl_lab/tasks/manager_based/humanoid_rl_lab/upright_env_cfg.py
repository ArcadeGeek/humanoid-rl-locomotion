# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree G1 flat-terrain fine-tuning task for a less crouched gait."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import G1Rewards

from .humanoid_rl_lab_env_cfg import HumanoidRlLabEnvCfg
from .mdp.upright_rewards import (
    knee_flexion_above_threshold_l2,
    track_heading_target_exp,
)
from .mdp.upright_rewards import track_heading_target_deadband_exp


@configclass
class HumanoidRlLabG1UprightRewardsCfg(G1Rewards):
    """Official G1 rewards plus a mild upper-knee-flexion penalty."""

    knee_flexion_above_threshold = RewTerm(
        func=knee_flexion_above_threshold_l2,
        weight=-0.5,
        params={
            "threshold": 0.85,
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["left_knee_joint", "right_knee_joint"],
                preserve_order=True,
            ),
        },
    )


@configclass
class HumanoidRlLabG1UprightEnvCfg(HumanoidRlLabEnvCfg):
    """Fine-tuning task with the validated 123D/37D policy interface unchanged."""

    rewards: HumanoidRlLabG1UprightRewardsCfg = HumanoidRlLabG1UprightRewardsCfg()


@configclass
class HumanoidRlLabG1UprightHeadingRewardsCfg(
    HumanoidRlLabG1UprightRewardsCfg
):
    """Upright v1 rewards plus command-target heading hold."""

    track_heading_target_exp = RewTerm(
        func=track_heading_target_exp,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "std": 0.20,
            "yaw_rate_threshold": 0.05,
            "min_planar_speed": 0.10,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class HumanoidRlLabG1UprightHeadingEnvCfg(
    HumanoidRlLabG1UprightEnvCfg
):
    """Upright v2 candidate with one heading-target reward."""

    rewards: HumanoidRlLabG1UprightHeadingRewardsCfg = (
        HumanoidRlLabG1UprightHeadingRewardsCfg()
    )

@configclass
class HumanoidRlLabG1UprightHeadingLowEnvCfg(HumanoidRlLabG1UprightHeadingEnvCfg):
    """Upright Heading candidate with reward weight 0.10."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.track_heading_target_exp.weight = 0.1

@configclass
class HumanoidRlLabG1UprightHeadingDeadbandEnvCfg(HumanoidRlLabG1UprightHeadingLowEnvCfg):
    """Low-weight heading task with a 0.05 rad reward deadband."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.track_heading_target_exp.func = (
            track_heading_target_deadband_exp
        )
        params = dict(self.rewards.track_heading_target_exp.params)
        params["deadband"] = 0.05
        self.rewards.track_heading_target_exp.params = params
