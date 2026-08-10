# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""External heading-hold controller validated for frozen Upright680 locomotion.

Validated preset:
- checkpoint: checkpoints/local_g1_scratch_upright_v1/model_680.pt
- body vx: 0.4 m/s
- body vy: 0.0 m/s
- Kp: 0.5 s^-1
- deadband: 1.0 deg
- max |wz|: 0.15 rad/s

Validation campaigns:
- upright680_heading_hold_kp050_wz015_robust_40_49
- upright680_heading_hold_kp050_wz015_holdout_50_59

Seeds 40-59 are perturbation/evaluation seeds, not independent training seeds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HeadingHoldConfig:
    """Configuration for wrapped-error proportional heading hold."""

    body_vx_mps: float
    body_vy_mps: float
    kp_s_inv: float
    deadband_rad: float
    max_abs_wz_radps: float
    target_heading_rad: float = 0.0

    def __post_init__(self) -> None:
        if self.kp_s_inv < 0.0:
            raise ValueError("kp_s_inv must be non-negative.")
        if self.deadband_rad < 0.0:
            raise ValueError("deadband_rad must be non-negative.")
        if self.max_abs_wz_radps <= 0.0:
            raise ValueError("max_abs_wz_radps must be positive.")


VALIDATED_UPRIGHT680_HEADING_HOLD = HeadingHoldConfig(
    body_vx_mps=0.4,
    body_vy_mps=0.0,
    kp_s_inv=0.5,
    deadband_rad=math.radians(1.0),
    max_abs_wz_radps=0.15,
    target_heading_rad=0.0,
)


def compute_heading_hold_wz(
    heading_w: torch.Tensor,
    *,
    config: HeadingHoldConfig = VALIDATED_UPRIGHT680_HEADING_HOLD,
    target_heading_rad: float | None = None,
) -> torch.Tensor:
    """Return saturated body yaw-rate command from wrapped heading error."""

    target = (
        config.target_heading_rad
        if target_heading_rad is None
        else float(target_heading_rad)
    )
    target_tensor = torch.as_tensor(
        target,
        device=heading_w.device,
        dtype=heading_w.dtype,
    )
    wrapped_error = torch.atan2(
        torch.sin(target_tensor - heading_w),
        torch.cos(target_tensor - heading_w),
    )
    effective_error = torch.sign(wrapped_error) * torch.clamp(
        torch.abs(wrapped_error) - config.deadband_rad,
        min=0.0,
    )
    return torch.clamp(
        config.kp_s_inv * effective_error,
        min=-config.max_abs_wz_radps,
        max=config.max_abs_wz_radps,
    )
