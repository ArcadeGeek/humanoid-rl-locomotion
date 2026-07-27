# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms used only by the project-owned G1 upright task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def knee_flexion_above_threshold_l2(
    env: ManagerBasedRLEnv,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize positive knee flexion only above ``threshold``.

    The official G1 asset uses positive knee angles for flexion. Angles at or
    below the threshold produce zero penalty. The squared excess is summed over
    the selected knee joints for each environment.
    """

    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos
    if hasattr(joint_pos, "torch"):
        joint_pos = joint_pos.torch

    selected_joint_pos = joint_pos[:, asset_cfg.joint_ids]
    excess_flexion = torch.clamp(selected_joint_pos - threshold, min=0.0)
    return torch.sum(torch.square(excess_flexion), dim=1)