# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 upright speed-recovery experiment."""

from isaaclab.utils.configclass import configclass

from .upright_env_cfg import HumanoidRlLabG1UprightEnvCfg


@configclass
class HumanoidRlLabG1UprightSpeedEnvCfg(HumanoidRlLabG1UprightEnvCfg):
    """Recover forward speed while preserving the upright posture reward."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.knee_flexion_above_threshold.weight = -0.5
        self.rewards.knee_flexion_above_threshold.params["threshold"] = 0.85