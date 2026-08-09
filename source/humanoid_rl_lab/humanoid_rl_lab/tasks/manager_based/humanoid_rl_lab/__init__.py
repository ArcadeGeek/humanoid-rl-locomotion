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
