# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym registration for the project-owned Unitree G1 locomotion tasks."""

import gymnasium as gym

from . import agents


gym.register(
    id="Humanoid-RL-Lab-G1-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoid_rl_lab_env_cfg:HumanoidRlLabEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)


gym.register(
    id="Humanoid-RL-Lab-G1-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoid_rl_lab_env_cfg:HumanoidRlLabEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)


gym.register(
    id="Humanoid-RL-Lab-G1-Flat-Eval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoid_rl_lab_env_cfg:HumanoidRlLabEnvCfg_EVAL",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)


gym.register(
    id="Humanoid-RL-Lab-G1-Flat-Heading-Eval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoid_rl_lab_env_cfg:HumanoidRlLabEnvCfg_HEADING_EVAL",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Humanoid-RL-Lab-G1-Flat-Upright-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.upright_env_cfg:HumanoidRlLabG1UprightEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.upright_ppo_cfg:UprightPPORunnerCfg",
    },
)

gym.register(
    id="Humanoid-RL-Lab-G1-Flat-Upright-Yaw-Direct-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.upright_env_cfg:"
            "HumanoidRlLabG1UprightYawDirectEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.upright_yaw_direct_ppo_cfg:"
            "UprightYawDirectPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Humanoid-RL-Lab-G1-Flat-Upright-Speed-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.upright_speed_env_cfg:HumanoidRlLabG1UprightSpeedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.upright_speed_ppo_cfg:UprightSpeedPPORunnerCfg",
    },
)

gym.register(
    id="Humanoid-RL-Lab-G1-Flat-Robust-Eval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoid_rl_lab_env_cfg:HumanoidRlLabEnvCfg_ROBUST_EVAL",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)


gym.register(
    id="Humanoid-RL-Lab-G1-Flat-Upright-Heading-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.upright_env_cfg:"
            "HumanoidRlLabG1UprightHeadingEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Humanoid-RL-Lab-G1-Flat-Upright-Heading-Low-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.upright_env_cfg:"
            "HumanoidRlLabG1UprightHeadingLowEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Humanoid-RL-Lab-G1-Flat-Upright-Heading-Deadband-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.upright_env_cfg:"
            "HumanoidRlLabG1UprightHeadingDeadbandEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Humanoid-RL-Lab-G1-Flat-Upright-Heading-Deadband-W005-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.upright_env_cfg:"
            "HumanoidRlLabG1UprightHeadingDeadbandW005EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg"
        ),
    },
)
gym.register(
    id="Humanoid-RL-Lab-G1-Flat-Upright-Heading-Progress-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.upright_env_cfg:"
            "HumanoidRlLabG1UprightHeadingProgressEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg"
        ),
    },
)
gym.register(
    id="Humanoid-RL-Lab-G1-Flat-Upright-Heading-Filtered-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.upright_env_cfg:"
            "HumanoidRlLabG1UprightHeadingFilteredEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg"
        ),
    },
)
gym.register(
    id="Humanoid-RL-Lab-G1-Flat-Upright-Heading-Deadband-Yaw-Deadzone-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.upright_env_cfg:"
            "HumanoidRlLabG1UprightHeadingDeadbandYawDeadzoneEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg"
        ),
    },
)
gym.register(
    id="Humanoid-RL-Lab-G1-Flat-Upright-Heading-Deadband-Yaw-Smooth-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.upright_env_cfg:"
            "HumanoidRlLabG1UprightHeadingDeadbandYawSmoothEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg"
        ),
    },
)
gym.register(
    id="Humanoid-RL-Lab-G1-Flat-Upright-Heading-Deadband-Yaw-Damp-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.upright_env_cfg:"
            "HumanoidRlLabG1UprightHeadingDeadbandYawDampEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Humanoid-RL-Lab-G1-Flat-Upright-Heading-Deadband-Yaw-Damp-0025-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.upright_env_cfg:"
            "HumanoidRlLabG1UprightHeadingDeadbandYawDamp0025EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg"
        ),
    },
)
gym.register(
    id="Humanoid-RL-Lab-G1-Flat-Upright-Heading-Deadband-010-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.upright_env_cfg:"
            "HumanoidRlLabG1UprightHeadingDeadband010EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg"
        ),
    },
)

gym.register(
    id='Humanoid-RL-Lab-G1-Flat-Upright-Heading-Deadband-Yaw-Damp-Dense-v0',
    entry_point='isaaclab.envs:ManagerBasedRLEnv',
    disable_env_checker=True,
    kwargs={'env_cfg_entry_point': 'humanoid_rl_lab.tasks.manager_based.humanoid_rl_lab.upright_env_cfg:HumanoidRlLabG1UprightHeadingDeadbandYawDampEnvCfg',
         'rsl_rl_cfg_entry_point': 'humanoid_rl_lab.tasks.manager_based.humanoid_rl_lab.agents.rsl_rl_ppo_cfg:PPORunnerDenseSave5Cfg'},
)

gym.register(
    id="Humanoid-RL-Lab-G1-Flat-Upright-Yaw-Direct-Wide-Yaw-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.upright_env_cfg:"
            "HumanoidRlLabG1UprightYawDirectWideYawEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.upright_yaw_direct_wide_yaw_ppo_cfg:"
            "UprightYawDirectWideYawPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Humanoid-RL-Lab-G1-Flat-Upright-Yaw-Direct-Wide-Yaw-Flat-Orientation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.upright_env_cfg:"
            "HumanoidRlLabG1UprightYawDirectWideYawFlatOrientationEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.upright_yaw_direct_wide_yaw_flat_orientation_ppo_cfg:"
            "UprightYawDirectWideYawFlatOrientationPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Humanoid-RL-Lab-G1-Flat-Upright-Yaw-Direct-Heavy-Tail-Yaw-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.upright_env_cfg:"
            "HumanoidRlLabG1UprightYawDirectHeavyTailYawEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.upright_yaw_direct_heavy_tail_yaw_ppo_cfg:"
            "UprightYawDirectHeavyTailYawPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Humanoid-RL-Lab-G1-Flat-Upright-Yaw-Direct-Pseudo-Huber-Yaw-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.upright_env_cfg:"
            "HumanoidRlLabG1UprightYawDirectPseudoHuberYawEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.upright_yaw_direct_pseudo_huber_yaw_ppo_cfg:"
            "UprightYawDirectPseudoHuberYawPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Humanoid-RL-Lab-G1-Flat-Upright-Yaw-Direct-Two-Scale-Yaw-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.upright_env_cfg:"
            "HumanoidRlLabG1UprightYawDirectTwoScaleYawEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.upright_yaw_direct_two_scale_yaw_ppo_cfg:"
            "UprightYawDirectTwoScaleYawPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Humanoid-RL-Lab-G1-Flat-Upright-Yaw-Direct-Tight-Yaw-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.upright_env_cfg:"
            "HumanoidRlLabG1UprightYawDirectTightYawEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.upright_yaw_direct_tight_yaw_ppo_cfg:"
            "UprightYawDirectTightYawPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Humanoid-RL-Lab-G1-Flat-Upright-Yaw-Direct-Torso-Reg-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.upright_env_cfg:HumanoidRlLabG1UprightYawDirectTorsoRegEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.upright_yaw_direct_torso_reg_ppo_cfg:"
            "UprightYawDirectTorsoRegPPORunnerCfg"
        ),
    },
)

# ==================================================================================================
# EXPERIMENTAL v3-A GAIT REFINEMENT
# Separate task ID; frozen v2-G registration above remains untouched.
# ==================================================================================================
gym.register(
    id="Humanoid-RL-Lab-G1-Flat-Upright-Yaw-Direct-Wide-Yaw-Flat-Orientation-V3A-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.upright_env_cfg:"
            "HumanoidRlLabG1UprightYawDirectWideYawFlatOrientationV3AEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.upright_yaw_direct_wide_yaw_flat_orientation_ppo_cfg:"
            "UprightYawDirectWideYawFlatOrientationV3APPORunnerCfg"
        ),
    },
)

# ==================================================================================================
# EXPERIMENTAL v3-B STAND-STILL REFINEMENT
# Separate task ID; frozen v2-G and v3-A registrations above remain untouched.
# ==================================================================================================
gym.register(
    id="Humanoid-RL-Lab-G1-Flat-Upright-Yaw-Direct-Wide-Yaw-Flat-Orientation-V3B-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.upright_env_cfg:"
            "HumanoidRlLabG1UprightYawDirectWideYawFlatOrientationV3BEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.upright_yaw_direct_wide_yaw_flat_orientation_ppo_cfg:"
            "UprightYawDirectWideYawFlatOrientationV3BPPORunnerCfg"
        ),
    },
)

# Experimental v3-C: conservative swing-foot clearance refinement from frozen v2-G.
gym.register(
    id="Humanoid-RL-Lab-G1-Flat-Upright-Yaw-Direct-Wide-Yaw-Flat-Orientation-V3C-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.upright_env_cfg:"
            "HumanoidRlLabG1UprightYawDirectWideYawFlatOrientationV3CEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.upright_yaw_direct_wide_yaw_flat_orientation_ppo_cfg:"
            "UprightYawDirectWideYawFlatOrientationV3CPPORunnerCfg"
        ),
    },
)

# Experimental v3-D: zero-motion + mild knee-angle refinement from v3-C620.
gym.register(
    id="Humanoid-RL-Lab-G1-Flat-Upright-Yaw-Direct-Wide-Yaw-Flat-Orientation-V3D-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.upright_env_cfg:"
            "HumanoidRlLabG1UprightYawDirectWideYawFlatOrientationV3DEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.upright_yaw_direct_wide_yaw_flat_orientation_ppo_cfg:"
            "UprightYawDirectWideYawFlatOrientationV3DPPORunnerCfg"
        ),
    },
)



# ==================================================================================================
# v4-A GENUINE-SCRATCH OFFICIAL-BASELINE REPRODUCTION
# New scientific family. Frozen v2-G and all v3 registrations above remain untouched.
# Environment and PPO inherit the project-owned wrappers that are semantically identical to the
# official Isaac Lab G1 Flat baseline; only task/experiment provenance is isolated.
# ==================================================================================================
gym.register(
    id="Humanoid-RL-Lab-G1-Flat-V4A-Official-Baseline-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.v4_env_cfg:"
            "HumanoidRlLabG1V4AOfficialBaselineEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.v4a_ppo_cfg:"
            "V4AOfficialBaselinePPORunnerCfg"
        ),
    },
)

# v4-B: genuine-scratch direct-yaw causal ablation from the Official-baseline v4-A task.
gym.register(
    id="Humanoid-RL-Lab-G1-Flat-V4B-Direct-Yaw-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.v4_env_cfg:"
            "HumanoidRlLabG1V4BDirectYawEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.v4b_ppo_cfg:"
            "V4BDirectYawPPORunnerCfg"
        ),
    },
)

# Experimental v4-C: v4-B direct-yaw plus narrower XY velocity-tracking reward std.
gym.register(
    id="Humanoid-RL-Lab-G1-Flat-V4C-Precision-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": __name__ + ".v4_env_cfg:" + "HumanoidRlLabG1V4CPrecisionEnvCfg",
        "rsl_rl_cfg_entry_point": agents.__name__ + ".v4c_ppo_cfg:" + "V4CPrecisionPPORunnerCfg",
    },
)

# Experimental v4-D: v4-B direct-yaw with increased zero-command standing exposure only.
gym.register(
    id="Humanoid-RL-Lab-G1-Flat-V4D-Standing-Exposure-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": __name__ + ".v4_env_cfg:" + "HumanoidRlLabG1V4DStandingExposureEnvCfg",
        "rsl_rl_cfg_entry_point": agents.__name__ + ".v4d_ppo_cfg:" + "V4DStandingExposurePPORunnerCfg",
    },
)

# Experimental v4-E: v4-B direct-yaw plus a full-command-gated zero-command physical-motion cost.
gym.register(
    id="Humanoid-RL-Lab-G1-Flat-V4E-Zero-Command-Motion-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": __name__ + ".v4_env_cfg:" + "HumanoidRlLabG1V4EZeroCommandMotionEnvCfg",
        "rsl_rl_cfg_entry_point": agents.__name__ + ".v4e_ppo_cfg:" + "V4EZeroCommandMotionPPORunnerCfg",
    },
)

# Experimental v4-F: v4-B direct-yaw plus full-command-gated leg stand-still regularization.
gym.register(
    id="Humanoid-RL-Lab-G1-Flat-V4F-Stand-Still-Joint-Deviation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": __name__ + ".v4_env_cfg:" + "HumanoidRlLabG1V4FStandStillJointDeviationEnvCfg",
        "rsl_rl_cfg_entry_point": agents.__name__ + ".v4f_ppo_cfg:" + "V4FStandStillJointDeviationPPORunnerCfg",
    },
)

# Experimental v4-G: v4-B direct-yaw plus full zero-command physical-motion regularization.
gym.register(
    id="Humanoid-RL-Lab-G1-Flat-V4G-Zero-Command-Full-Motion-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": __name__ + ".v4_env_cfg:" + "HumanoidRlLabG1V4GZeroCommandFullMotionEnvCfg",
        "rsl_rl_cfg_entry_point": agents.__name__ + ".v4g_ppo_cfg:" + "V4GZeroCommandFullMotionPPORunnerCfg",
    },
)

# Experimental v4-H: v4-B direct-yaw plus bounded zero-command full-motion regularization.
gym.register(
    id="Humanoid-RL-Lab-G1-Flat-V4H-Bounded-Zero-Command-Full-Motion-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.v4_env_cfg:"
            "HumanoidRlLabG1V4HBoundedZeroCommandFullMotionEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.v4h_ppo_cfg:"
            "V4HBoundedZeroCommandFullMotionPPORunnerCfg"
        ),
    },
)

# Experimental v4-I: v4-B direct-yaw plus delayed bounded zero-command full-motion regularization.
gym.register(
    id="Humanoid-RL-Lab-G1-Flat-V4I-Delayed-Bounded-Zero-Command-Full-Motion-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.v4_env_cfg:"
            "HumanoidRlLabG1V4IDelayedBoundedZeroCommandFullMotionEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.v4i_ppo_cfg:"
            "V4IDelayedBoundedZeroCommandFullMotionPPORunnerCfg"
        ),
    },
)

# Experimental v4-J: v4-B direct-yaw plus low-frequency straight-line lateral-drift precision.
gym.register(
    id="Humanoid-RL-Lab-G1-Flat-V4J-Straight-Line-Precision-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.v4_env_cfg:"
            "HumanoidRlLabG1V4JStraightLinePrecisionEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.v4j_ppo_cfg:"
            "V4JStraightLinePrecisionPPORunnerCfg"
        ),
    },
)

