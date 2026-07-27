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
from .mdp.upright_rewards import knee_flexion_above_threshold_l2


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