# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree G1 flat-terrain fine-tuning task for a less crouched gait."""

import copy
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import G1Rewards

from .humanoid_rl_lab_env_cfg import HumanoidRlLabEnvCfg
from .mdp.upright_rewards import (
    knee_flexion_above_threshold_l2,
    track_heading_target_exp,
    track_ang_vel_z_pseudo_huber,
    track_ang_vel_z_inverse_sqrt,
)
from .mdp.upright_rewards import track_heading_target_deadband_exp
from .mdp.upright_rewards import yaw_rate_near_heading_target_l2
from .mdp.upright_rewards import yaw_rate_change_near_heading_target_l2
from .mdp.upright_rewards import yaw_rate_deadzone_near_heading_target_l2
from .mdp.upright_rewards import track_heading_target_filtered_deadband_exp
from .mdp.upright_rewards import track_heading_target_progress_deadband


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
class HumanoidRlLabG1UprightYawDirectEnvCfg(HumanoidRlLabG1UprightEnvCfg):
    """Upright v1 with direct low-amplitude yaw-rate command training.

    This ablation changes only yaw-command generation: heading-target control
    is disabled and yaw rate is sampled directly from [-0.15, +0.15] rad/s,
    matching the complete output domain of the frozen external heading hold.
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        command = self.commands.base_velocity
        command.heading_command = False
        command.rel_heading_envs = 0.0
        command.ranges.ang_vel_z = (-0.15, 0.15)


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

@configclass
class HumanoidRlLabG1UprightHeadingDeadband010EnvCfg(HumanoidRlLabG1UprightHeadingDeadbandEnvCfg):
    # Same task as the 0.05-rad variant; only the deadband changes.

    def __post_init__(self):
        super().__post_init__()
        params = dict(
            self.rewards.track_heading_target_exp.params
        )
        params["deadband"] = 0.1
        self.rewards.track_heading_target_exp.params = params

@configclass
class HumanoidRlLabG1UprightHeadingDeadbandYawDampEnvCfg(HumanoidRlLabG1UprightHeadingDeadbandEnvCfg):
    # Add local yaw damping; keep heading reward unchanged.

    def __post_init__(self):
        super().__post_init__()
        source_term = self.rewards.track_heading_target_exp
        damping_term = copy.deepcopy(source_term)
        damping_term.func = yaw_rate_near_heading_target_l2
        damping_term.weight = -0.05
        damping_params = dict(source_term.params)
        damping_params["near_target"] = 0.1
        damping_term.params = damping_params
        self.rewards.yaw_rate_near_heading_target_l2 = (
            damping_term
        )

@configclass
class HumanoidRlLabG1UprightHeadingDeadbandYawDamp0025EnvCfg(HumanoidRlLabG1UprightHeadingDeadbandYawDampEnvCfg):
    # Same task as the -0.05 yaw-damping variant,
    # with only the damping weight reduced to -0.025.

    def __post_init__(self):
        super().__post_init__()
        self.rewards.yaw_rate_near_heading_target_l2.weight = (
            -0.025
        )

@configclass
class HumanoidRlLabG1UprightHeadingDeadbandYawSmoothEnvCfg(HumanoidRlLabG1UprightHeadingDeadbandEnvCfg):
    # Add temporal yaw smoothing; keep heading reward unchanged.

    def __post_init__(self):
        super().__post_init__()
        source_term = self.rewards.track_heading_target_exp
        smoothing_term = copy.deepcopy(source_term)
        smoothing_term.func = yaw_rate_change_near_heading_target_l2
        smoothing_term.weight = -0.05
        source_params = dict(source_term.params)
        smoothing_term.params = {
            "command_name": source_params["command_name"],
            "yaw_rate_threshold": source_params["yaw_rate_threshold"],
            "min_planar_speed": source_params["min_planar_speed"],
            "asset_cfg": source_params["asset_cfg"],
            "near_target": 0.1,
        }
        self.rewards.yaw_rate_change_near_heading_target_l2 = (
            smoothing_term
        )

@configclass
class HumanoidRlLabG1UprightHeadingDeadbandYawDeadzoneEnvCfg(HumanoidRlLabG1UprightHeadingDeadbandEnvCfg):
    # Add actual-yaw dead-zone damping; keep heading reward unchanged.

    def __post_init__(self):
        super().__post_init__()
        source_term = self.rewards.track_heading_target_exp
        damping_term = copy.deepcopy(source_term)
        damping_term.func = yaw_rate_deadzone_near_heading_target_l2
        damping_term.weight = -0.05
        source_params = dict(source_term.params)
        damping_term.params = {
            "command_name": source_params["command_name"],
            "yaw_rate_threshold": source_params["yaw_rate_threshold"],
            "min_planar_speed": source_params["min_planar_speed"],
            "asset_cfg": source_params["asset_cfg"],
            "near_target": 0.1,
            "free_yaw_rate": 0.05,
        }
        self.rewards.yaw_rate_deadzone_near_heading_target_l2 = (
            damping_term
        )

@configclass
class HumanoidRlLabG1UprightHeadingFilteredEnvCfg(HumanoidRlLabG1UprightHeadingDeadbandEnvCfg):
    # Filter persistent heading error; keep every other term unchanged.

    def __post_init__(self):
        super().__post_init__()
        filtered_term = copy.deepcopy(
            self.rewards.track_heading_target_exp
        )
        filtered_term.func = track_heading_target_filtered_deadband_exp
        filtered_term.params = dict(filtered_term.params)
        filtered_term.params["ema_alpha"] = 0.960789439152323
        self.rewards.track_heading_target_exp = filtered_term

@configclass
class HumanoidRlLabG1UprightHeadingProgressEnvCfg(HumanoidRlLabG1UprightHeadingDeadbandEnvCfg):
    # Use signed heading-error progress; keep every other term unchanged.

    def __post_init__(self):
        super().__post_init__()
        progress_term = copy.deepcopy(
            self.rewards.track_heading_target_exp
        )
        progress_term.func = track_heading_target_progress_deadband
        progress_term.params = dict(progress_term.params)
        progress_term.params["policy_dt"] = 0.02
        progress_term.params["target_change_tolerance"] = 0.0001
        self.rewards.track_heading_target_exp = progress_term

@configclass
class HumanoidRlLabG1UprightHeadingDeadbandW005EnvCfg(HumanoidRlLabG1UprightHeadingDeadbandEnvCfg):
    # Deadband-only heading objective at half the parent reward weight.

    def __post_init__(self):
        super().__post_init__()
        heading_term = copy.deepcopy(self.rewards.track_heading_target_exp)
        heading_term.weight = 0.05
        self.rewards.track_heading_target_exp = heading_term






@configclass
class HumanoidRlLabG1UprightYawDirectWideYawEnvCfg(HumanoidRlLabG1UprightYawDirectEnvCfg):
    """Yaw-Direct v2-F with a wider yaw-command training distribution."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Single scientific modification versus genuine-scratch v2-A:
        # widen only the sampled direct yaw-rate command range.
        self.commands.base_velocity.ranges.ang_vel_z = (-0.50, 0.50)


@configclass
class HumanoidRlLabG1UprightYawDirectWideYawFlatOrientationEnvCfg(
    HumanoidRlLabG1UprightYawDirectWideYawEnvCfg
):
    """Yaw-Direct v2-G: v2-F wide-yaw training with stronger flat-orientation regularization."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Single scientific modification versus v2-F:
        # strengthen only root flat-orientation regularization.
        self.rewards.flat_orientation_l2.weight = -2.0

@configclass
class HumanoidRlLabG1UprightYawDirectHeavyTailYawEnvCfg(HumanoidRlLabG1UprightYawDirectEnvCfg):
    """Yaw-Direct v2-E with an added bounded heavy-tail yaw reward."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Single scientific modification versus genuine-scratch v2-A:
        # keep the original Gaussian yaw reward unchanged and add a small,
        # positive, bounded heavy-tail yaw-tracking bonus.
        source_term = self.rewards.track_ang_vel_z_exp
        heavy_tail_term = copy.deepcopy(source_term)
        heavy_tail_term.func = track_ang_vel_z_inverse_sqrt
        heavy_tail_term.weight = 0.10
        heavy_tail_term.params = {
            "command_name": source_term.params["command_name"],
            "delta": 0.15,
        }
        self.rewards.track_ang_vel_z_heavy_tail = heavy_tail_term

@configclass
class HumanoidRlLabG1UprightYawDirectPseudoHuberYawEnvCfg(HumanoidRlLabG1UprightYawDirectEnvCfg):
    """Yaw-Direct v2-D with an added dense pseudo-Huber yaw-error penalty."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Single scientific modification versus genuine-scratch v2-A:
        # keep the original coarse Gaussian yaw reward unchanged and add a
        # dense pseudo-Huber tracking penalty with command-domain scale.
        source_term = self.rewards.track_ang_vel_z_exp
        pseudo_huber_term = copy.deepcopy(source_term)
        pseudo_huber_term.func = track_ang_vel_z_pseudo_huber
        pseudo_huber_term.weight = -0.10
        pseudo_huber_term.params = {
            "command_name": source_term.params["command_name"],
            "delta": 0.15,
        }
        self.rewards.track_ang_vel_z_pseudo_huber = pseudo_huber_term

@configclass
class HumanoidRlLabG1UprightYawDirectTwoScaleYawEnvCfg(HumanoidRlLabG1UprightYawDirectEnvCfg):
    """Yaw-Direct v2-C with complementary coarse and precision yaw-tracking kernels."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Single scientific modification versus genuine-scratch v2-A:
        # preserve broad capture range while adding equal-weight precision shaping.
        coarse_term = self.rewards.track_ang_vel_z_exp
        coarse_term.weight = 0.5
        coarse_term.params = dict(coarse_term.params)
        coarse_term.params["std"] = 0.50

        tight_term = copy.deepcopy(coarse_term)
        tight_term.weight = 0.5
        tight_term.params = dict(tight_term.params)
        tight_term.params["std"] = 0.15
        self.rewards.track_ang_vel_z_exp_tight = tight_term

@configclass
class HumanoidRlLabG1UprightYawDirectTightYawEnvCfg(HumanoidRlLabG1UprightYawDirectEnvCfg):
    """Yaw-Direct v2-B with yaw-tracking kernel matched to the +/-0.15 rad/s command domain."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Single scientific ablation versus genuine-scratch v2-A:
        # match yaw-tracking kernel width to the complete direct-yaw command range.
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.15


@configclass
class HumanoidRlLabG1UprightYawDirectTorsoRegEnvCfg(HumanoidRlLabG1UprightYawDirectEnvCfg):
    """Yaw-Direct v2 with stronger official torso-joint deviation regularization."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Single scientific ablation versus Yaw-Direct v1:
        # official G1 torso-joint L1 deviation penalty, conservative 2x weight.
        self.rewards.joint_deviation_torso.weight = -0.2
