# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL configuration for the project-owned Unitree G1 flat task."""

from isaaclab.utils.configclass import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import (
    G1FlatPPORunnerCfg,
)


@configclass
class PPORunnerCfg(G1FlatPPORunnerCfg):
    """Official G1 flat PPO settings with project-specific runtime metadata."""

    obs_groups = {
        "actor": ["policy"],
        "critic": ["policy"],
    }

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "humanoid_rl_lab_g1_flat"

@configclass
class PPORunnerDenseSave5Cfg(PPORunnerCfg):
    """Identical PPO runner with checkpoint saving every 5 iterations."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.save_interval = 5
