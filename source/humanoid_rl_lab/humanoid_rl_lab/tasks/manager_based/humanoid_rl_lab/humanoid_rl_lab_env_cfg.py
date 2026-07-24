# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Project-owned Unitree G1 flat-terrain locomotion configurations.

The first project milestone intentionally inherits the official Isaac Lab G1
configuration without changing its MDP. This gives the project a stable,
independently registered baseline before observations, actions, rewards, and
deployment interfaces are customized.
"""

from isaaclab.utils.configclass import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import (
    G1FlatEnvCfg,
    G1FlatEnvCfg_PLAY,
)


@configclass
class HumanoidRlLabEnvCfg(G1FlatEnvCfg):
    """Training configuration initially identical to the official G1 flat task."""

    pass


@configclass
class HumanoidRlLabEnvCfg_PLAY(G1FlatEnvCfg_PLAY):
    """Play configuration initially identical to the official G1 flat play task."""

    pass