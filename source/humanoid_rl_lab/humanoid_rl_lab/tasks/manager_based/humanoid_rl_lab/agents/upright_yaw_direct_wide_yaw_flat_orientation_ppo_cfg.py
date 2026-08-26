# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from .upright_yaw_direct_wide_yaw_ppo_cfg import UprightYawDirectWideYawPPORunnerCfg


@configclass
class UprightYawDirectWideYawFlatOrientationPPORunnerCfg(UprightYawDirectWideYawPPORunnerCfg):
    """PPO config for genuine-scratch v2-G posture-only refinement."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "humanoid_rl_lab_g1_upright_yaw_direct_wide_yaw_flat_orientation"

# ==================================================================================================
# EXPERIMENTAL v3-A GAIT REFINEMENT
# PPO hyperparameters inherit unchanged from frozen v2-G. Only experiment_name is separated so logs
# and checkpoints cannot be mixed with the frozen baseline.
# ==================================================================================================
@configclass
class UprightYawDirectWideYawFlatOrientationV3APPORunnerCfg(
    UprightYawDirectWideYawFlatOrientationPPORunnerCfg
):
    'PPO config for experimental v3-A gait refinement fine-tuning.'

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "humanoid_rl_lab_g1_v3a_gait_refinement"

# ==================================================================================================
# EXPERIMENTAL v3-B STAND-STILL REFINEMENT
# PPO hyperparameters remain identical to v3-A/v2-G; only experiment_name is separated.
# ==================================================================================================
@configclass
class UprightYawDirectWideYawFlatOrientationV3BPPORunnerCfg(
    UprightYawDirectWideYawFlatOrientationV3APPORunnerCfg
):
    """PPO config for experimental v3-B stand-still refinement."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "humanoid_rl_lab_g1_v3b_standstill_refinement"

# ==================================================================================================
# EXPERIMENTAL v3-C SWING-FOOT CLEARANCE REFINEMENT
# PPO hyperparameters inherit unchanged from frozen v2-G. Only experiment_name is separated.
# ==================================================================================================
@configclass
class UprightYawDirectWideYawFlatOrientationV3CPPORunnerCfg(
    UprightYawDirectWideYawFlatOrientationPPORunnerCfg
):
    'PPO config for experimental v3-C swing-clearance fine-tuning.'

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "humanoid_rl_lab_g1_v3c_swing_clearance_refinement"

