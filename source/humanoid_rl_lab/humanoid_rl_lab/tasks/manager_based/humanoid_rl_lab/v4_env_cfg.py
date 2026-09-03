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

from isaaclab.envs.mdp import modify_reward_weight
from isaaclab.managers import CurriculumTermCfg as CurrTerm, RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import CurriculumCfg as LocomotionVelocityCurriculumCfg

from .mdp.v4_rewards import (
    low_frequency_command_residual_vy_wz_bounded,
    geometric_command_path_pose_bounded,
    geometric_command_path_cross_track_bounded,
    straight_line_lateral_velocity_ema_bounded,
    stand_still_joint_deviation_full_command_l1,
    zero_command_bounded_full_motion,
    zero_command_full_motion_l2,
    zero_command_motion_l2,
)

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

# ==================================================================================================
# EXPERIMENTAL v4-F STAND-STILL JOINT-DEVIATION ABLATION - GENUINE SCRATCH
# Scientific parent: v4-B Direct-Yaw.
# ONLY scientific intervention:
#   add stand_still_joint_deviation_full_command_l1
#   weight=-0.4
#   active only when ||[vx, vy, wz]|| < 0.06
#   acts on the 12 leg joints: hips, knees, ankle pitch/roll.
# rel_standing_envs remains the v4-B / Official value 0.02.
# No v4-C precision change, no v4-D standing-exposure change, no v4-E zero-motion reward.
# No imitation, reference motion, or gait clock.
# ==================================================================================================
@configclass
class HumanoidRlLabG1V4FStandStillJointDeviationEnvCfg(HumanoidRlLabG1V4BDirectYawEnvCfg):
    'v4-F: v4-B plus full-command-gated leg default-pose regularization.'

    def __post_init__(self) -> None:
        super().__post_init__()
        self.rewards.stand_still_full_command = RewTerm(
            func=stand_still_joint_deviation_full_command_l1,
            weight=-0.4,
            params={
                "command_name": "base_velocity",
                "command_threshold": 0.06,
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        ".*_hip_.*",
                        ".*_knee_joint",
                        ".*_ankle_pitch_joint",
                        ".*_ankle_roll_joint",
                    ],
                ),
            },
        )

# ==================================================================================================
# EXPERIMENTAL v4-G ZERO-COMMAND FULL-MOTION ABLATION - GENUINE SCRATCH
# Scientific parent: v4-B Direct-Yaw.
# ONLY scientific intervention:
#   add zero_command_full_motion_l2
#   gate: ||[vx, vy, wz]|| < 0.06
#   weight=-5.0, base_planar_scale=1.0, yaw_rate_scale=1.0, foot_xy_scale=0.25
# rel_standing_envs remains 0.02.
# No v4-C/v4-D/v4-E/v4-F scientific inheritance.
# ==================================================================================================
@configclass
class HumanoidRlLabG1V4GZeroCommandFullMotionEnvCfg(HumanoidRlLabG1V4BDirectYawEnvCfg):
    'v4-G: v4-B plus full physical-motion regularization at near-zero command.'

    def __post_init__(self) -> None:
        super().__post_init__()
        self.rewards.zero_command_full_motion = RewTerm(
            func=zero_command_full_motion_l2,
            weight=-5.0,
            params={
                "command_name": "base_velocity",
                "command_threshold": 0.06,
                "base_planar_scale": 1.0,
                "yaw_rate_scale": 1.0,
                "foot_xy_scale": 0.25,
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                    preserve_order=True,
                ),
            },
        )

# ==================================================================================================
# EXPERIMENTAL v4-H BOUNDED ZERO-COMMAND FULL-MOTION ABLATION - GENUINE SCRATCH
#
# Scientific parent: v4-B Direct-Yaw.
#
# ONLY scientific intervention versus v4-B:
#   add zero_command_bounded_full_motion
#
#   gate                  : ||[vx, vy, wz]|| < 0.06
#   weight                : -5.0
#   base_planar_scale     : 1.0
#   yaw bounded form      : 1 - exp(-(wz / 0.20)^2)
#   foot_xy_scale         : 0.25
#
# rel_standing_envs remains inherited from v4-B: 0.02.
# No v4-C/v4-D/v4-E/v4-F/v4-G scientific inheritance.
# ==================================================================================================
@configclass
class HumanoidRlLabG1V4HBoundedZeroCommandFullMotionEnvCfg(HumanoidRlLabG1V4BDirectYawEnvCfg):
    'v4-H: v4-B plus bounded full-motion regularization at near-zero command.'

    def __post_init__(self) -> None:
        super().__post_init__()
        self.rewards.zero_command_bounded_full_motion = RewTerm(
            func=zero_command_bounded_full_motion,
            weight=-5.0,
            params={
                "command_name": "base_velocity",
                "command_threshold": 0.06,
                "base_planar_scale": 1.0,
                "yaw_sigma": 0.20,
                "foot_xy_scale": 0.25,
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                    preserve_order=True,
                ),
            },
        )

# ==================================================================================================
# EXPERIMENTAL v4-I DELAYED BOUNDED ZERO-COMMAND FULL-MOTION - GENUINE SCRATCH
# Scientific parent: v4-B Direct-Yaw.
#
# ONLY scientific intervention versus v4-B:
#   add zero_command_bounded_full_motion, but delay its weight with Isaac Lab native curriculum.
#
# Reward semantics once active:
#   gate              : ||[vx, vy, wz]|| < 0.06
#   target weight     : -5.0
#   base_planar_scale : 1.0
#   yaw bounded form  : 1 - exp(-(wz / 0.20)^2)
#   foot_xy_scale     : 0.25
#
# Curriculum semantics:
#   initial weight    : 0.0
#   target weight     : -5.0
#   native scheduler  : isaaclab.envs.mdp.modify_reward_weight
#   switch condition  : common_step_counter > 4800, evaluated on reset
#
# With 24 control steps per PPO iteration, 4800 control steps correspond to 200 iterations.
# rel_standing_envs remains inherited from v4-B: 0.02.
# No v4-C/v4-D/v4-E/v4-F/v4-G/v4-H scientific inheritance.
# ==================================================================================================
@configclass
class HumanoidRlLabG1V4IDelayedBoundedZeroCommandFullMotionCurriculumCfg(LocomotionVelocityCurriculumCfg):
    "v4-I curriculum: delay bounded zero-command regularization until locomotion has emerged."

    zero_command_bounded_full_motion_weight = CurrTerm(
        func=modify_reward_weight,
        params={
            "term_name": "zero_command_bounded_full_motion",
            "weight": -5.0,
            "num_steps": 4800,
        },
    )


@configclass
class HumanoidRlLabG1V4IDelayedBoundedZeroCommandFullMotionEnvCfg(HumanoidRlLabG1V4BDirectYawEnvCfg):
    "v4-I: v4-B plus delayed bounded zero-command full-motion regularization."

    curriculum: HumanoidRlLabG1V4IDelayedBoundedZeroCommandFullMotionCurriculumCfg = HumanoidRlLabG1V4IDelayedBoundedZeroCommandFullMotionCurriculumCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        self.rewards.zero_command_bounded_full_motion = RewTerm(
            func=zero_command_bounded_full_motion,
            weight=0.0,
            params={
                "command_name": "base_velocity",
                "command_threshold": 0.06,
                "base_planar_scale": 1.0,
                "yaw_sigma": 0.2,
                "foot_xy_scale": 0.25,
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                    preserve_order=True,
                ),
            },
        )

# ==================================================================================================
# EXPERIMENTAL v4-J LOW-FREQUENCY STRAIGHT-LINE LATERAL-DRIFT PRECISION - GENUINE SCRATCH
# Scientific parent: v4-B Direct-Yaw.
#
# ONLY scientific intervention versus v4-B:
#   add straight_line_lateral_velocity_ema_bounded
#
# Reward semantics:
#   error                     : body-frame vy - commanded vy
#   EMA tau                   : 0.75 s
#   bounded sigma             : 0.03 m/s
#   weight                    : -0.25
#   active iff                : vx_cmd > 0.10 m/s
#                               |vy_cmd| < 0.10 m/s
#                               |wz_cmd| < 0.20 rad/s
#   EMA reset                 : episode reset and whenever the straight-line gate is inactive
#
# Everything else inherits unchanged from v4-B / Official G1 Flat:
# command ranges, direct-yaw semantics, rewards, observations, perturbation events, actions,
# scene, terminations, rel_standing_envs and PPO hyperparameters.
# No v4-C/v4-D/v4-E/v4-F/v4-G/v4-H/v4-I scientific inheritance.
# ==================================================================================================
@configclass
class HumanoidRlLabG1V4JStraightLinePrecisionEnvCfg(HumanoidRlLabG1V4BDirectYawEnvCfg):
    'v4-J: v4-B plus low-frequency lateral-drift precision during straight locomotion.'

    def __post_init__(self) -> None:
        super().__post_init__()

        self.rewards.straight_line_lateral_velocity_ema_bounded = RewTerm(
            func=straight_line_lateral_velocity_ema_bounded,
            weight=-0.25,
            params={
                "command_name": "base_velocity",
                "tau_s": 0.75,
                "sigma": 0.03,
                "min_forward_command": 0.10,
                "lateral_command_threshold": 0.10,
                "yaw_command_threshold": 0.20,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

# ==================================================================================================
# EXPERIMENTAL v4-K GEOMETRIC COMMAND-PATH PRECISION - GENUINE SCRATCH
# Scientific parent: v4-B Direct-Yaw.
#
# ONLY scientific intervention versus v4-B:
#   add geometric_command_path_cross_track_bounded
#
# Reward semantics:
#   quantity                    : geometric lateral cross-track error
#   reference                   : command-consistent planar path anchored at command-segment start
#   sigma                       : 0.20 m
#   weight                      : -0.15
#   active iff                 : vx_cmd > 0.10 m/s
#                                |vy_cmd| < 0.10 m/s
#                                |wz_cmd| < 0.20 rad/s
#   segment reset              : episode reset, gate exit/re-entry, command resample/change
#   command-change threshold   : 1e-6
#   yaw-rate analytic epsilon  : 1e-4 rad/s
#
# At the thesis straight command [0.40, 0, 0], the reference is exactly a straight line anchored
# to the robot pose at the start of the command segment.
#
# Everything else inherits unchanged from v4-B / Official G1 Flat:
# command ranges, direct-yaw semantics, Official rewards, observations, perturbation events, actions,
# scene, terminations, rel_standing_envs and PPO hyperparameters.
# No v4-C/v4-D/v4-E/v4-F/v4-G/v4-H/v4-I/v4-J scientific inheritance.
# ==================================================================================================
@configclass
class HumanoidRlLabG1V4KGeometricPathPrecisionEnvCfg(HumanoidRlLabG1V4BDirectYawEnvCfg):
    'v4-K: v4-B plus bounded geometric cross-track precision.'

    def __post_init__(self) -> None:
        super().__post_init__()

        self.rewards.geometric_command_path_cross_track_bounded = RewTerm(
            func=geometric_command_path_cross_track_bounded,
            weight=-0.15,
            params={
                "command_name": "base_velocity",
                "sigma_m": 0.20,
                "min_forward_command": 0.10,
                "lateral_command_threshold": 0.10,
                "yaw_command_threshold": 0.20,
                "command_change_threshold": 1.0e-6,
                "yaw_rate_epsilon": 1.0e-4,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

# ==================================================================================================
# EXPERIMENTAL v4-L PATH-POSE PRECISION - GENUINE SCRATCH
# Scientific parent: v4-B Direct-Yaw.
#
# ONLY scientific intervention versus v4-B:
#   add one composite geometric path-pose reward
#
# Composite reward:
#   cross-track sigma           : 0.20 m
#   heading sigma               : 5 deg = 0.08726646259971647 rad
#   weight                      : -0.15
#   active iff                  : vx_cmd > 0.10 m/s
#                                |vy_cmd| < 0.10 m/s
#                                |wz_cmd| < 0.20 rad/s
#   segment reset              : episode reset, gate exit/re-entry, command resample/change
#   command-change threshold   : 1e-6
#   yaw-rate analytic epsilon  : 1e-4 rad/s
#
# At [0.40, 0, 0], this jointly asks the policy to remain near the initial straight line AND keep
# body heading near the segment-start heading, closing the yaw/crab compensation observed in v4-K.
#
# Everything else inherits unchanged from v4-B / Official G1 Flat.
# No v4-C/v4-D/v4-E/v4-F/v4-G/v4-H/v4-I/v4-J/v4-K scientific inheritance.
# ==================================================================================================
@configclass
class HumanoidRlLabG1V4LPathPosePrecisionEnvCfg(HumanoidRlLabG1V4BDirectYawEnvCfg):
    'v4-L: v4-B plus bounded geometric path-pose precision.'

    def __post_init__(self) -> None:
        super().__post_init__()

        self.rewards.geometric_command_path_pose_bounded = RewTerm(
            func=geometric_command_path_pose_bounded,
            weight=-0.15,
            params={
                "command_name": "base_velocity",
                "sigma_cross_m": 0.20,
                "sigma_heading_rad": 0.08726646259971647,
                "min_forward_command": 0.10,
                "lateral_command_threshold": 0.10,
                "yaw_command_threshold": 0.20,
                "command_change_threshold": 1.0e-6,
                "yaw_rate_epsilon": 1.0e-4,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

# ==================================================================================================
# EXPERIMENTAL v4-M LOW-FREQUENCY LOCAL RESIDUAL - GENUINE SCRATCH
# Scientific parent: v4-B Direct-Yaw.
#
# ONLY scientific intervention versus v4-B:
#   add one bounded EMA reward on command-consistent body-frame vy/wz residuals
#
#   tau_s                       : 1.50 s
#   sigma_vy_mps                : 0.030 m/s
#   sigma_wz_radps              : 0.025 rad/s
#   weight                      : -0.10
#   active iff                  : vx_cmd > 0.10 m/s
#                                 |vy_cmd| < 0.10 m/s
#                                 |wz_cmd| < 0.20 rad/s
#   command-change threshold    : 1e-6
#
# residual_vy = body_vy - command_vy
# residual_wz = body_wz - command_wz
#
# Everything else inherits unchanged from v4-B / Official G1 Flat.
# No v4-C/v4-D/v4-E/v4-F/v4-G/v4-H/v4-I/v4-J/v4-K/v4-L scientific inheritance.
# ==================================================================================================
@configclass
class HumanoidRlLabG1V4MLowFreqLocalResidualEnvCfg(HumanoidRlLabG1V4BDirectYawEnvCfg):
    'v4-M: v4-B plus low-frequency command-residual vy/wz precision.'

    def __post_init__(self) -> None:
        super().__post_init__()

        self.rewards.low_frequency_command_residual_vy_wz_bounded = RewTerm(
            func=low_frequency_command_residual_vy_wz_bounded,
            weight=-0.10,
            params={
                "command_name": "base_velocity",
                "tau_s": 1.50,
                "sigma_vy_mps": 0.030,
                "sigma_wz_radps": 0.025,
                "min_forward_command": 0.10,
                "lateral_command_threshold": 0.10,
                "yaw_command_threshold": 0.20,
                "command_change_threshold": 1.0e-6,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
