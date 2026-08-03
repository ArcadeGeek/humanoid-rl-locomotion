# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Quantitative evaluator for the project-owned Unitree G1 locomotion policy.

The script loads an RSL-RL checkpoint using the same runner path as the normal
Isaac Lab playback script, executes one deterministic environment, prints live
telemetry, and saves both per-step samples and a JSON summary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata as metadata
import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.seed import configure_seed
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
import humanoid_rl_lab.tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, setup_preset_cli
from isaaclab_tasks.utils.hydra import hydra_task_config

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Quantitatively evaluate a Unitree G1 RSL-RL checkpoint.")
parser.add_argument("--task", type=str, required=True, help="Registered Isaac Lab task name.")
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="RSL-RL agent configuration registry entry point.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Must remain 1 for this evaluator.")
parser.add_argument("--seed", type=int, default=42, help="Environment seed.")
parser.add_argument("--warmup_s", type=float, default=3.0, help="Unmeasured stabilization time in seconds.")
parser.add_argument("--duration_s", type=float, default=20.0, help="Measured simulation time in seconds.")
parser.add_argument("--print_every_s", type=float, default=1.0, help="Live telemetry interval in seconds.")
parser.add_argument("--label", type=str, default=None, help="Human-readable label for the result directory.")
parser.add_argument("--output_dir", type=str, default=None, help="Optional exact result directory.")
parser.add_argument("--push_time_s", type=float, default=None, help="Measurement-relative time for one controlled impulse.")
parser.add_argument("--push_vx_mps", type=float, default=0.0, help="World-X root linear-velocity delta.")
parser.add_argument("--push_vy_mps", type=float, default=0.0, help="World-Y root linear-velocity delta.")
parser.add_argument("--push_wz_radps", type=float, default=0.0, help="World-Z root angular-velocity delta.")
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, remaining_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + remaining_args

installed_version = metadata.version("rsl-rl-lib")


def _torch(value) -> torch.Tensor:
    """Return a PyTorch view for either a ProxyArray or an existing tensor."""

    return value.torch if hasattr(value, "torch") else value


def _cpu_values(value) -> list[float]:
    """Convert one tensor row to ordinary Python floats."""

    tensor = _torch(value).detach().to("cpu")
    return [float(item) for item in tensor.reshape(-1).tolist()]


def _safe_label(value: str) -> str:
    """Create a filesystem-safe result label."""

    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return normalized.strip("._") or "evaluation"


def _sha256(path: str) -> str:
    """Calculate a checkpoint digest for result traceability."""

    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values))


def _std(values: np.ndarray) -> float:
    return float(np.std(values))


def _rmse(actual: np.ndarray, desired: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(actual - desired))))


def _quat_wxyz_to_euler_xyz(
    quaternion_wxyz: list[float],
) -> tuple[float, float, float]:
    """Convert a WXYZ quaternion to XYZ roll, pitch and yaw."""

    w, x, y, z = quaternion_wxyz
    sin_roll_cos_pitch = 2.0 * (w * x + y * z)
    cos_roll_cos_pitch = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sin_roll_cos_pitch, cos_roll_cos_pitch)

    sin_pitch = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sin_pitch)))

    sin_yaw_cos_pitch = 2.0 * (w * z + x * y)
    cos_yaw_cos_pitch = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(sin_yaw_cos_pitch, cos_yaw_cos_pitch)
    return roll, pitch, yaw


def _distribution_summary(values: np.ndarray) -> dict[str, float]:
    """Return reusable scalar distribution statistics."""

    absolute = np.abs(values)
    return {
        "mean": _mean(values),
        "std": _std(values),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean_abs": _mean(absolute),
        "rms": float(np.sqrt(np.mean(np.square(values)))),
        "p95_abs": float(np.quantile(absolute, 0.95)),
        "max_abs": float(np.max(absolute)),
    }


def _asymmetry_index(left: float, right: float) -> float:
    """Return a bounded magnitude asymmetry index when defined."""

    denominator = abs(left) + abs(right)
    if denominator <= 1.0e-12:
        return 0.0
    return 2.0 * abs(left - right) / denominator


def _build_locomotion_sustainability(
    *,
    arrays: dict[str, np.ndarray],
    dt: float,
    joint_names: list[str],
) -> dict[str, object]:
    """Build posture, actuation and torso-stability metrics."""

    joint_summaries: dict[str, object] = {}
    aggregate: dict[str, dict[str, float]] = {}

    for joint_name in joint_names:
        position_rad = arrays[f"{joint_name}_rad"]
        velocity_radps = arrays[f"{joint_name}_vel_radps"]
        acceleration_radps2 = arrays[f"{joint_name}_acc_radps2"]
        applied_torque_nm = arrays[f"{joint_name}_applied_torque_nm"]
        computed_torque_nm = arrays[f"{joint_name}_computed_torque_nm"]
        power_w = arrays[f"{joint_name}_mechanical_power_w"]

        position_deg = np.degrees(position_rad)
        velocity_degps = np.degrees(velocity_radps)
        acceleration_degps2 = np.degrees(acceleration_radps2)
        positive_power = np.maximum(power_w, 0.0)
        negative_power = np.minimum(power_w, 0.0)
        torque_difference = computed_torque_nm - applied_torque_nm

        position_summary = _distribution_summary(position_deg)
        velocity_summary = _distribution_summary(velocity_degps)
        acceleration_summary = _distribution_summary(
            acceleration_degps2
        )
        applied_torque_summary = _distribution_summary(
            applied_torque_nm
        )
        computed_torque_summary = _distribution_summary(
            computed_torque_nm
        )
        power_summary = _distribution_summary(power_w)

        joint_summaries[joint_name] = {
            "position_deg": position_summary,
            "velocity_degps": velocity_summary,
            "acceleration_degps2": acceleration_summary,
            "applied_torque_nm": applied_torque_summary,
            "computed_torque_nm": computed_torque_summary,
            "computed_minus_applied_torque_nm": {
                "mean_abs": _mean(np.abs(torque_difference)),
                "rms": float(
                    np.sqrt(np.mean(np.square(torque_difference)))
                ),
                "max_abs": float(np.max(np.abs(torque_difference))),
            },
            "mechanical_power_w": power_summary,
            "sampled_mechanical_work_j": {
                "positive": float(np.sum(positive_power) * dt),
                "negative": float(np.sum(negative_power) * dt),
                "negative_abs": float(
                    np.sum(np.abs(negative_power)) * dt
                ),
                "net": float(np.sum(power_w) * dt),
                "absolute": float(np.sum(np.abs(power_w)) * dt),
            },
        }

        aggregate[joint_name] = {
            "position_mean_deg": float(position_summary["mean"]),
            "position_range_deg": float(
                position_summary["max"] - position_summary["min"]
            ),
            "velocity_rms_degps": float(velocity_summary["rms"]),
            "acceleration_p95_abs_degps2": float(
                acceleration_summary["p95_abs"]
            ),
            "torque_rms_nm": float(applied_torque_summary["rms"]),
            "absolute_work_j": float(
                np.sum(np.abs(power_w)) * dt
            ),
        }

    pair_names = (
        ("hip_pitch", "left_hip_pitch_joint", "right_hip_pitch_joint"),
        ("knee", "left_knee_joint", "right_knee_joint"),
        ("ankle_pitch", "left_ankle_pitch_joint", "right_ankle_pitch_joint"),
    )
    symmetry: dict[str, object] = {}
    for pair_label, left_name, right_name in pair_names:
        left = aggregate[left_name]
        right = aggregate[right_name]
        symmetry[pair_label] = {
            "position_mean_left_minus_right_deg": (
                left["position_mean_deg"] - right["position_mean_deg"]
            ),
            "position_range_asymmetry_index": _asymmetry_index(
                left["position_range_deg"], right["position_range_deg"]
            ),
            "velocity_rms_asymmetry_index": _asymmetry_index(
                left["velocity_rms_degps"], right["velocity_rms_degps"]
            ),
            "acceleration_p95_abs_asymmetry_index": (
                _asymmetry_index(
                    left["acceleration_p95_abs_degps2"],
                    right["acceleration_p95_abs_degps2"],
                )
            ),
            "torque_rms_asymmetry_index": _asymmetry_index(
                left["torque_rms_nm"], right["torque_rms_nm"]
            ),
            "absolute_work_asymmetry_index": _asymmetry_index(
                left["absolute_work_j"], right["absolute_work_j"]
            ),
        }

    torso_roll_deg = np.degrees(arrays["torso_roll_rad"])
    torso_pitch_deg = np.degrees(arrays["torso_pitch_rad"])
    torso_ang_speed = arrays["torso_ang_speed_radps"]
    projected_gravity_x = arrays["projected_gravity_x"]
    projected_gravity_y = arrays["projected_gravity_y"]
    projected_gravity_z = arrays["projected_gravity_z"]
    root_tilt_rad = np.arctan2(
        np.sqrt(
            np.square(projected_gravity_x)
            + np.square(projected_gravity_y)
        ),
        np.maximum(-projected_gravity_z, 1.0e-12),
    )
    root_tilt_deg = np.degrees(root_tilt_rad)

    total_absolute_work_j = float(
        sum(
            aggregate[name]["absolute_work_j"]
            for name in joint_names
        )
    )

    return {
        "method": {
            "sample_period_s": dt,
            "sample_rate_hz": 1.0 / dt,
            "power_definition": (
                "applied_torque_nm * joint_velocity_radps"
            ),
            "sampled_work_integration": (
                "rectangle rule at evaluator policy sample rate"
            ),
            "sampled_work_limitation": (
                "not resolved at the physics-step frequency"
            ),
            "acceleration_asymmetry_statistic": (
                "p95 absolute joint acceleration"
            ),
        },
        "joint_count": len(joint_names),
        "joint_names": list(joint_names),
        "joints": joint_summaries,
        "left_right_symmetry": symmetry,
        "six_joint_sampled_total_absolute_work_j": total_absolute_work_j,
        "torso": {
            "roll_deg": _distribution_summary(torso_roll_deg),
            "pitch_deg": _distribution_summary(torso_pitch_deg),
            "angular_speed_radps": _distribution_summary(
                torso_ang_speed
            ),
        },
        "root_tilt_deg_from_projected_gravity": (
            _distribution_summary(root_tilt_deg)
        ),
    }


def _complete_episode_durations(
    mask: np.ndarray,
    *,
    target_value: bool,
    dt: float,
) -> np.ndarray:
    """Durations of runs that do not touch either measurement boundary."""

    values = np.asarray(mask, dtype=bool)
    durations: list[float] = []
    start = 0
    while start < len(values):
        value = bool(values[start])
        end = start + 1
        while end < len(values) and bool(values[end]) == value:
            end += 1
        if value == target_value and start > 0 and end < len(values):
            durations.append((end - start) * dt)
        start = end
    return np.asarray(durations, dtype=np.float64)


def _duration_summary(values: np.ndarray) -> dict[str, object]:
    if len(values) == 0:
        return {
            "available": False,
            "count": 0,
            "mean_s": None,
            "std_s": None,
            "median_s": None,
            "min_s": None,
            "max_s": None,
        }
    return {
        "available": True,
        "count": int(len(values)),
        "mean_s": _mean(values),
        "std_s": _std(values),
        "median_s": float(np.quantile(values, 0.50)),
        "min_s": float(np.min(values)),
        "max_s": float(np.max(values)),
    }


def _optional_asymmetry(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return _asymmetry_index(float(left), float(right))


def _one_foot_contact_summary(
    *,
    contact_mask: np.ndarray,
    normal_force_n: np.ndarray,
    net_force_n: np.ndarray,
    sensor_contact_time_s: np.ndarray,
    sensor_air_time_s: np.ndarray,
    dt: float,
) -> dict[str, object]:
    contact = np.asarray(contact_mask, dtype=bool)
    touchdowns = np.flatnonzero((~contact[:-1]) & contact[1:]) + 1
    liftoffs = np.flatnonzero(contact[:-1] & (~contact[1:])) + 1
    stance = _complete_episode_durations(contact, target_value=True, dt=dt)
    swing = _complete_episode_durations(contact, target_value=False, dt=dt)
    contact_normal = normal_force_n[contact]
    contact_net = net_force_n[contact]
    contact_only = (
        {
            "available": True,
            "samples": int(len(contact_normal)),
            "normal_force_n": _distribution_summary(contact_normal),
            "net_force_magnitude_n": _distribution_summary(contact_net),
        }
        if len(contact_normal) > 0
        else {
            "available": False,
            "samples": 0,
            "normal_force_n": None,
            "net_force_magnitude_n": None,
        }
    )
    return {
        "duty_factor": _mean(contact.astype(np.float64)),
        "contact_samples": int(np.count_nonzero(contact)),
        "air_samples": int(np.count_nonzero(~contact)),
        "touchdown_count": int(len(touchdowns)),
        "liftoff_count": int(len(liftoffs)),
        "touchdown_sample_indices": [int(x) for x in touchdowns],
        "liftoff_sample_indices": [int(x) for x in liftoffs],
        "stance_complete_episodes": _duration_summary(stance),
        "swing_complete_episodes": _duration_summary(swing),
        "normal_force_all_samples_n": _distribution_summary(normal_force_n),
        "net_force_magnitude_all_samples_n": _distribution_summary(net_force_n),
        "contact_only_force": contact_only,
        "sampled_normal_impulse_n_s": float(np.sum(normal_force_n) * dt),
        "sensor_current_contact_time_s": _distribution_summary(sensor_contact_time_s),
        "sensor_current_air_time_s": _distribution_summary(sensor_air_time_s),
    }


def _build_foot_contact_and_gait(
    *, arrays: dict[str, np.ndarray], dt: float
) -> dict[str, object]:
    left_contact = arrays["left_foot_contact"] > 0.5
    right_contact = arrays["right_foot_contact"] > 0.5
    left = _one_foot_contact_summary(
        contact_mask=left_contact,
        normal_force_n=arrays["left_foot_normal_force_n"],
        net_force_n=arrays["left_foot_net_force_magnitude_n"],
        sensor_contact_time_s=arrays["left_foot_sensor_current_contact_time_s"],
        sensor_air_time_s=arrays["left_foot_sensor_current_air_time_s"],
        dt=dt,
    )
    right = _one_foot_contact_summary(
        contact_mask=right_contact,
        normal_force_n=arrays["right_foot_normal_force_n"],
        net_force_n=arrays["right_foot_net_force_magnitude_n"],
        sensor_contact_time_s=arrays["right_foot_sensor_current_contact_time_s"],
        sensor_air_time_s=arrays["right_foot_sensor_current_air_time_s"],
        dt=dt,
    )

    events: list[tuple[int, str]] = []
    events += [(int(x), "left") for x in left["touchdown_sample_indices"]]
    events += [(int(x), "right") for x in right["touchdown_sample_indices"]]
    events.sort(key=lambda item: (item[0], item[1]))
    sequence: list[str] = []
    index = 0
    while index < len(events):
        sample_index = events[index][0]
        same_sample: list[str] = []
        while index < len(events) and events[index][0] == sample_index:
            same_sample.append(events[index][1])
            index += 1
        if len(same_sample) == 1:
            sequence.append(same_sample[0])
    alternating = sum(a != b for a, b in zip(sequence, sequence[1:]))
    repeated = sum(a == b for a, b in zip(sequence, sequence[1:]))
    pair_count = alternating + repeated
    duration_s = len(left_contact) * dt
    total_touchdowns = int(left["touchdown_count"] + right["touchdown_count"])
    left_stance = left["stance_complete_episodes"]["mean_s"]
    right_stance = right["stance_complete_episodes"]["mean_s"]
    left_swing = left["swing_complete_episodes"]["mean_s"]
    right_swing = right["swing_complete_episodes"]["mean_s"]

    return {
        "method": {
            "left_foot_body": "left_ankle_roll_link",
            "right_foot_body": "right_ankle_roll_link",
            "contact_signal": "absolute world-Z net contact force",
            "contact_force_threshold_n": 20.0,
            "sample_period_s": dt,
            "sample_rate_hz": 1.0 / dt,
            "duration_rule": "complete non-boundary-truncated episodes only",
            "duty_factor_rule": "all measured samples, including boundary episodes",
            "force_impulse_limitation": "sampled at policy rate, not physics rate",
        },
        "left": left,
        "right": right,
        "support": {
            "double_support_fraction": _mean((left_contact & right_contact).astype(np.float64)),
            "left_only_support_fraction": _mean((left_contact & ~right_contact).astype(np.float64)),
            "right_only_support_fraction": _mean((right_contact & ~left_contact).astype(np.float64)),
            "flight_fraction": _mean((~left_contact & ~right_contact).astype(np.float64)),
        },
        "cadence": {
            "total_touchdowns": total_touchdowns,
            "touchdown_event_rate_hz": total_touchdowns / duration_s,
            "touchdown_events_per_min": 60.0 * total_touchdowns / duration_s,
            "non_simultaneous_touchdown_sequence": sequence,
            "alternating_pair_count": alternating,
            "repeated_same_foot_pair_count": repeated,
            "alternation_fraction": alternating / pair_count if pair_count else None,
        },
        "left_right_symmetry": {
            "duty_factor_abs_difference": abs(float(left["duty_factor"]) - float(right["duty_factor"])),
            "duty_factor_asymmetry_index": _asymmetry_index(float(left["duty_factor"]), float(right["duty_factor"])),
            "complete_stance_mean_asymmetry_index": _optional_asymmetry(left_stance, right_stance),
            "complete_swing_mean_asymmetry_index": _optional_asymmetry(left_swing, right_swing),
            "sampled_normal_impulse_asymmetry_index": _asymmetry_index(
                float(left["sampled_normal_impulse_n_s"]),
                float(right["sampled_normal_impulse_n_s"]),
            ),
            "normal_force_rms_asymmetry_index": _asymmetry_index(
                float(left["normal_force_all_samples_n"]["rms"]),
                float(right["normal_force_all_samples_n"]["rms"]),
            ),
            "touchdown_count_left_minus_right": int(left["touchdown_count"] - right["touchdown_count"]),
        },
    }


def _signal_recovery_summary(
    *,
    post_times: np.ndarray,
    post_values: np.ndarray,
    reference_values: np.ndarray,
    push_time_s: float,
    fixed_floor: float,
    baseline_rms_multiplier: float,
    peak_search_window_s: float,
    recovery_hold_s: float,
    dt: float,
) -> dict[str, object]:
    """Summarize one post-push signal relative to its pre-push behavior."""

    reference_mean = _mean(reference_values)
    reference_std = _std(reference_values)
    reference_abs_deviation = np.abs(reference_values - reference_mean)
    reference_mean_abs_deviation = _mean(reference_abs_deviation)
    reference_p95 = float(np.quantile(reference_abs_deviation, 0.95))
    recovery_rms_tolerance = max(
        float(fixed_floor),
        float(baseline_rms_multiplier) * reference_std,
    )

    signed_deviation = post_values - reference_mean
    absolute_deviation = np.abs(signed_deviation)
    peak_limit_s = push_time_s + peak_search_window_s
    peak_candidates = np.flatnonzero(post_times <= peak_limit_s + 1.0e-12)
    if len(peak_candidates) == 0:
        peak_candidates = np.asarray([0], dtype=np.int64)
    peak_index = int(
        peak_candidates[int(np.argmax(absolute_deviation[peak_candidates]))]
    )

    recovery_hold_steps = max(1, int(math.ceil(recovery_hold_s / dt)))
    recovery_time_s: float | None = None
    recovery_window_start_s: float | None = None
    qualifying_rms: float | None = None
    rolling_rms_values: list[float] = []

    final_start = len(signed_deviation) - recovery_hold_steps
    if final_start >= peak_index:
        for start in range(peak_index, final_start + 1):
            window = signed_deviation[start : start + recovery_hold_steps]
            window_rms = float(np.sqrt(np.mean(np.square(window))))
            rolling_rms_values.append(window_rms)
            if recovery_time_s is None and window_rms <= recovery_rms_tolerance:
                end = start + recovery_hold_steps - 1
                recovery_window_start_s = max(
                    0.0, float(post_times[start] - push_time_s)
                )
                recovery_time_s = max(
                    0.0, float(post_times[end] - push_time_s)
                )
                qualifying_rms = window_rms

    excess_above_band = np.maximum(
        absolute_deviation - recovery_rms_tolerance,
        0.0,
    )

    return {
        "reference_mean": reference_mean,
        "reference_std": reference_std,
        "reference_mean_abs_deviation": reference_mean_abs_deviation,
        "reference_p95_abs_deviation": reference_p95,
        "baseline_rms_multiplier": baseline_rms_multiplier,
        "recovery_rms_tolerance": recovery_rms_tolerance,
        "peak_search_window_s": peak_search_window_s,
        "peak_value": float(post_values[peak_index]),
        "peak_abs_deviation_from_reference": float(
            absolute_deviation[peak_index]
        ),
        "peak_time_after_push_s": max(
            0.0, float(post_times[peak_index] - push_time_s)
        ),
        "integrated_abs_deviation": float(
            np.sum(absolute_deviation) * dt
        ),
        "integrated_excess_above_recovery_band": float(
            np.sum(excess_above_band) * dt
        ),
        "minimum_rolling_rms_after_peak": (
            min(rolling_rms_values) if rolling_rms_values else None
        ),
        "recovered": recovery_time_s is not None,
        "recovery_window_start_s": recovery_window_start_s,
        "recovery_time_s": recovery_time_s,
        "qualifying_window_rms": qualifying_rms,
        "recovery_hold_s": recovery_hold_steps * dt,
    }


def _build_post_push_recovery(
    *,
    arrays: dict[str, np.ndarray],
    dt: float,
    controlled_push: dict[str, object],
) -> dict[str, object]:
    """Build temporal recovery metrics for one controlled push."""

    reference_window_s = 1.0
    peak_search_window_s = 2.0
    recovery_hold_s = 0.5
    baseline_rms_multiplier = 1.25
    height_window_s = 2.0
    result: dict[str, object] = {
        "available": False,
        "reason": None,
        "method": {
            "pre_push_reference_window_s": reference_window_s,
            "peak_search_window_s": peak_search_window_s,
            "recovery_hold_s": recovery_hold_s,
            "threshold_definition": (
                "max(fixed_floor, baseline_rms_multiplier * pre_push_std)"
            ),
            "baseline_rms_multiplier": baseline_rms_multiplier,
            "lateral_velocity_fixed_floor_mps": 0.05,
            "yaw_rate_fixed_floor_radps": 0.10,
        },
    }

    if not bool(controlled_push["requested"]):
        result["reason"] = "no_controlled_push_requested"
        return result
    if not bool(controlled_push["applied"]):
        result["reason"] = "controlled_push_not_applied"
        return result

    marker_indices = np.flatnonzero(arrays["controlled_push_applied"] > 0.5)
    if len(marker_indices) != 1:
        result["reason"] = "expected_exactly_one_push_marker"
        result["push_marker_count"] = int(len(marker_indices))
        return result

    effective_time = controlled_push["effective_time_s"]
    if effective_time is None:
        result["reason"] = "missing_effective_push_time"
        return result

    push_time_s = float(effective_time)
    push_index = int(marker_indices[0])
    times = arrays["time_s"]
    reference_steps = max(2, int(round(reference_window_s / dt)))
    pre_start = max(0, push_index - reference_steps)
    pre_indices = np.arange(pre_start, push_index, dtype=np.int64)
    if len(pre_indices) < 2:
        result["reason"] = "insufficient_pre_push_samples"
        result["pre_push_samples"] = int(len(pre_indices))
        return result

    post_slice = slice(push_index, len(times))
    post_times = times[post_slice]
    if len(post_times) == 0:
        result["reason"] = "no_post_push_samples"
        return result

    lateral_recovery = _signal_recovery_summary(
        post_times=post_times,
        post_values=arrays["base_vy_mps"][post_slice],
        reference_values=arrays["base_vy_mps"][pre_indices],
        push_time_s=push_time_s,
        fixed_floor=0.05,
        baseline_rms_multiplier=baseline_rms_multiplier,
        peak_search_window_s=peak_search_window_s,
        recovery_hold_s=recovery_hold_s,
        dt=dt,
    )
    yaw_recovery = _signal_recovery_summary(
        post_times=post_times,
        post_values=arrays["base_wz_radps"][post_slice],
        reference_values=arrays["base_wz_radps"][pre_indices],
        push_time_s=push_time_s,
        fixed_floor=0.10,
        baseline_rms_multiplier=baseline_rms_multiplier,
        peak_search_window_s=peak_search_window_s,
        recovery_hold_s=recovery_hold_s,
        dt=dt,
    )

    world_y = arrays["world_y_m"]
    heading_unwrapped = np.unwrap(arrays["heading_rad"])
    reference_index = max(0, push_index - 1)
    reference_y = float(world_y[reference_index])
    reference_heading = float(heading_unwrapped[reference_index])
    post_y_relative = world_y[post_slice] - reference_y
    post_heading_relative = heading_unwrapped[post_slice] - reference_heading
    lateral_path = np.concatenate(([reference_y], world_y[post_slice]))

    height_mask = post_times <= push_time_s + height_window_s + 1.0e-12
    height_values = arrays["root_height_m"][post_slice][height_mask]
    height_times = post_times[height_mask]
    pre_height_mean = _mean(arrays["root_height_m"][pre_indices])
    min_height_index = int(np.argmin(height_values))
    minimum_height = float(height_values[min_height_index])

    result.update(
        {
            "available": True,
            "reason": None,
            "effective_push_time_s": push_time_s,
            "push_marker_sample_time_s": float(times[push_index]),
            "pre_push_samples": int(len(pre_indices)),
            "post_push_samples": int(len(post_times)),
            "post_push_observation_s": max(
                0.0, float(times[-1] - push_time_s)
            ),
            "lateral_velocity_body": {
                "units": "m/s",
                **lateral_recovery,
            },
            "yaw_rate_body": {
                "units": "rad/s",
                **yaw_recovery,
            },
            "trajectory_after_push": {
                "reference_world_y_m": reference_y,
                "final_lateral_displacement_m": float(post_y_relative[-1]),
                "max_abs_lateral_displacement_m": float(
                    np.max(np.abs(post_y_relative))
                ),
                "lateral_path_length_m": float(
                    np.sum(np.abs(np.diff(lateral_path)))
                ),
                "reference_heading_deg": float(
                    np.degrees(reference_heading)
                ),
                "final_heading_change_deg": float(
                    np.degrees(post_heading_relative[-1])
                ),
                "max_abs_heading_change_deg": float(
                    np.degrees(np.max(np.abs(post_heading_relative)))
                ),
            },
            "vertical_response_first_2s": {
                "requested_window_s": height_window_s,
                "observed_window_s": max(
                    0.0, float(height_times[-1] - push_time_s)
                ),
                "pre_push_mean_root_height_m": pre_height_mean,
                "minimum_root_height_m": minimum_height,
                "height_drop_from_pre_push_mean_m": max(
                    0.0, pre_height_mean - minimum_height
                ),
                "time_to_minimum_s": max(
                    0.0, float(height_times[min_height_index] - push_time_s)
                ),
            },
        }
    )
    return result

def _write_csv(path: Path, records: list[dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def _build_summary(
    *,
    records: list[dict[str, float]],
    dt: float,
    requested_steps: int,
    warmup_s: float,
    task: str,
    checkpoint: str,
    checkpoint_hash: str,
    seed: int,
    initial_state: dict[str, object],
    controlled_push: dict[str, object],
    joint_names: list[str],
    terminated_stage: str | None,
) -> dict:
    arrays = {name: np.asarray([row[name] for row in records], dtype=np.float64) for name in records[0]}
    post_push_recovery = _build_post_push_recovery(
        arrays=arrays,
        dt=dt,
        controlled_push=controlled_push,
    )
    locomotion_sustainability = _build_locomotion_sustainability(
        arrays=arrays,
        dt=dt,
        joint_names=joint_names,
    )
    foot_contact_and_gait = _build_foot_contact_and_gait(arrays=arrays, dt=dt)

    x = arrays["world_x_m"]
    y = arrays["world_y_m"]
    xy = np.column_stack((x, y))
    segment_lengths = np.linalg.norm(np.diff(xy, axis=0), axis=1) if len(xy) > 1 else np.zeros(0)
    path_length = float(np.sum(segment_lengths))
    net_displacement = float(np.linalg.norm(xy[-1] - xy[0]))
    straightness = net_displacement / path_length if path_length > 1.0e-9 else 0.0

    heading_unwrapped = np.unwrap(arrays["heading_rad"])
    posture_summary = {}
    for name in joint_names:
        key = f"{name}_rad"
        values = arrays[key]
        posture_summary[name] = {
            "mean_rad": _mean(values),
            "std_rad": _std(values),
            "mean_deg": float(np.degrees(np.mean(values))),
            "std_deg": float(np.degrees(np.std(values))),
        }

    completed = len(records) == requested_steps and terminated_stage is None

    return {
        "schema_version": 6,
        "status": "complete" if completed else "terminated_early",
        "task": task,
        "seed": seed,
        "initial_state": initial_state,
        "controlled_push": controlled_push,
        "post_push_recovery": post_push_recovery,
        "checkpoint": {
            "path": os.path.abspath(checkpoint),
            "sha256": checkpoint_hash,
            "size_bytes": os.path.getsize(checkpoint),
        },
        "timing": {
            "policy_dt_s": dt,
            "policy_hz": 1.0 / dt,
            "warmup_s": warmup_s,
            "requested_duration_s": requested_steps * dt,
            "measured_duration_s": len(records) * dt,
            "samples": len(records),
        },
        "termination": {
            "occurred": terminated_stage is not None,
            "stage": terminated_stage,
        },
        "command_mean": {
            "vx_mps": _mean(arrays["cmd_vx_mps"]),
            "vy_mps": _mean(arrays["cmd_vy_mps"]),
            "wz_radps": _mean(arrays["cmd_wz_radps"]),
        },
        "velocity_tracking": {
            "base_vx_mean_mps": _mean(arrays["base_vx_mps"]),
            "base_vx_std_mps": _std(arrays["base_vx_mps"]),
            "base_vx_rmse_mps": _rmse(arrays["base_vx_mps"], arrays["cmd_vx_mps"]),
            "base_vy_mean_mps": _mean(arrays["base_vy_mps"]),
            "base_vy_std_mps": _std(arrays["base_vy_mps"]),
            "base_vy_rmse_mps": _rmse(arrays["base_vy_mps"], arrays["cmd_vy_mps"]),
            "base_wz_mean_radps": _mean(arrays["base_wz_radps"]),
            "base_wz_std_radps": _std(arrays["base_wz_radps"]),
            "base_wz_rmse_radps": _rmse(arrays["base_wz_radps"], arrays["cmd_wz_radps"]),
        },
        "trajectory": {
            "forward_displacement_world_x_m": float(x[-1] - x[0]),
            "lateral_drift_world_y_m": float(y[-1] - y[0]),
            "max_abs_lateral_deviation_m": float(np.max(np.abs(y - y[0]))),
            "net_displacement_m": net_displacement,
            "path_length_m": path_length,
            "straightness_ratio": straightness,
        },
        "heading": {
            "initial_deg": float(np.degrees(heading_unwrapped[0])),
            "final_deg": float(np.degrees(heading_unwrapped[-1])),
            "mean_abs_deg": float(np.degrees(np.mean(np.abs(heading_unwrapped)))),
            "max_abs_deg": float(np.degrees(np.max(np.abs(heading_unwrapped)))),
        },
        "root_height": {
            "mean_m": _mean(arrays["root_height_m"]),
            "std_m": _std(arrays["root_height_m"]),
            "min_m": float(np.min(arrays["root_height_m"])),
            "max_m": float(np.max(arrays["root_height_m"])),
        },
        "posture_joints": posture_summary,
        "locomotion_sustainability": locomotion_sustainability,
        "foot_contact_and_gait": foot_contact_and_gait,
    }


def _print_summary(summary: dict, result_dir: Path) -> None:
    timing = summary["timing"]
    command = summary["command_mean"]
    tracking = summary["velocity_tracking"]
    trajectory = summary["trajectory"]
    heading = summary["heading"]
    height = summary["root_height"]
    posture = summary["posture_joints"]

    print("")
    print("=" * 72)
    print("G1 QUANTITATIVE EVALUATION")
    print("=" * 72)
    print(f"Status:                 {summary['status']}")
    print(f"Measured:               {timing['measured_duration_s']:.2f} s ({timing['samples']} samples)")
    print(
        "Mean command:           "
        f"vx={command['vx_mps']:.4f} m/s, vy={command['vy_mps']:.4f} m/s, "
        f"wz={command['wz_radps']:.4f} rad/s"
    )
    print(
        "Mean base velocity:     "
        f"vx={tracking['base_vx_mean_mps']:.4f} m/s, "
        f"vy={tracking['base_vy_mean_mps']:.4f} m/s, "
        f"wz={tracking['base_wz_mean_radps']:.4f} rad/s"
    )
    print(
        "Tracking RMSE:          "
        f"vx={tracking['base_vx_rmse_mps']:.4f}, "
        f"vy={tracking['base_vy_rmse_mps']:.4f}, "
        f"wz={tracking['base_wz_rmse_radps']:.4f}"
    )
    print(f"Forward displacement:   {trajectory['forward_displacement_world_x_m']:.4f} m")
    print(f"Lateral drift:          {trajectory['lateral_drift_world_y_m']:.4f} m")
    print(f"Max lateral deviation:  {trajectory['max_abs_lateral_deviation_m']:.4f} m")
    print(f"Path straightness:      {trajectory['straightness_ratio']:.6f}")
    print(
        "Heading:                "
        f"final={heading['final_deg']:.3f} deg, "
        f"mean_abs={heading['mean_abs_deg']:.3f} deg, "
        f"max_abs={heading['max_abs_deg']:.3f} deg"
    )
    print(
        "Root height:            "
        f"mean={height['mean_m']:.4f} m, std={height['std_m']:.4f} m, "
        f"min={height['min_m']:.4f} m"
    )
    print(
        "Mean knees:             "
        f"left={posture['left_knee_joint']['mean_deg']:.3f} deg, "
        f"right={posture['right_knee_joint']['mean_deg']:.3f} deg"
    )
    print(f"Result directory:       {result_dir}")
    print("=" * 72)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg) -> None:
    """Load a checkpoint and collect deterministic locomotion metrics."""

    if args_cli.checkpoint is None:
        raise ValueError("--checkpoint is required.")
    if args_cli.num_envs != 1:
        raise ValueError("This evaluator requires --num_envs=1.")
    if args_cli.warmup_s < 0.0:
        raise ValueError("--warmup_s must be non-negative.")
    if args_cli.duration_s <= 0.0:
        raise ValueError("--duration_s must be positive.")
    if args_cli.print_every_s <= 0.0:
        raise ValueError("--print_every_s must be positive.")

    push_delta = (float(args_cli.push_vx_mps), float(args_cli.push_vy_mps), float(args_cli.push_wz_radps))
    push_requested = args_cli.push_time_s is not None
    if not push_requested and any(abs(value) > 0.0 for value in push_delta):
        raise ValueError("--push_time_s is required when a push delta is non-zero.")
    if push_requested:
        if not 0.0 <= float(args_cli.push_time_s) < float(args_cli.duration_s):
            raise ValueError("--push_time_s must satisfy 0 <= push_time_s < duration_s.")
        if not any(abs(value) > 0.0 for value in push_delta):
            raise ValueError("At least one controlled push delta must be non-zero.")

    with launch_simulation(env_cfg, args_cli):
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

        env_cfg.scene.num_envs = 1
        env_cfg.seed = int(args_cli.seed)
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        env_cfg.episode_length_s = max(
            float(env_cfg.episode_length_s),
            float(args_cli.warmup_s + args_cli.duration_s + 10.0),
        )

        checkpoint = retrieve_file_path(args_cli.checkpoint)
        checkpoint_hash = _sha256(checkpoint)
        checkpoint_stem = Path(checkpoint).stem
        label = _safe_label(args_cli.label or checkpoint_stem)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        result_dir = (
            Path(args_cli.output_dir).resolve()
            if args_cli.output_dir
            else Path("logs") / "evaluations" / "g1" / f"{timestamp}_{label}"
        )
        result_dir.mkdir(parents=True, exist_ok=False)
        env_cfg.log_dir = str(result_dir)

        print(f"[INFO] Loading checkpoint: {checkpoint}")
        print(f"[INFO] Results: {result_dir.resolve()}")

        env = gym.make(args_cli.task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        try:
            if agent_cfg.class_name == "OnPolicyRunner":
                runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            elif agent_cfg.class_name == "DistillationRunner":
                runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            else:
                raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

            if args_cli.deterministic:
                configure_seed(env_cfg.seed, True)

            runner.load(checkpoint)
            policy = runner.get_inference_policy(device=env.unwrapped.device)

            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy_network = None
            elif version.parse(installed_version) >= version.parse("2.3.0"):
                policy_network = runner.alg.policy
            else:
                policy_network = runner.alg.actor_critic

            base_env = env.unwrapped
            robot = base_env.scene["robot"]
            posture_joint_names = [
                "left_hip_pitch_joint",
                "right_hip_pitch_joint",
                "left_knee_joint",
                "right_knee_joint",
                "left_ankle_pitch_joint",
                "right_ankle_pitch_joint",
            ]
            posture_joint_ids, resolved_joint_names = robot.find_joints(
                posture_joint_names,
                preserve_order=True,
            )
            if resolved_joint_names != posture_joint_names:
                raise RuntimeError(
                    "Unexpected posture joint resolution. "
                    f"Requested={posture_joint_names}, resolved={resolved_joint_names}"
                )

            torso_body_ids, resolved_torso_body_names = robot.find_bodies(
                ["torso_link"],
                preserve_order=True,
            )
            if resolved_torso_body_names != ["torso_link"]:
                raise RuntimeError(
                    "Unexpected torso body resolution. "
                    f"Resolved={resolved_torso_body_names}"
                )
            torso_body_id = int(torso_body_ids[0])

            foot_body_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
            _, resolved_foot_body_names = robot.find_bodies(
                foot_body_names, preserve_order=True
            )
            if resolved_foot_body_names != foot_body_names:
                raise RuntimeError(
                    f"Unexpected foot body resolution: {resolved_foot_body_names}"
                )
            contact_sensor = base_env.scene["contact_forces"]
            sensor_body_names_raw = getattr(contact_sensor, "body_names", None)
            if sensor_body_names_raw is None:
                raise RuntimeError(
                    "Contact sensor does not expose body_names; "
                    "cannot map foot contact signals safely."
                )
            sensor_body_names = list(sensor_body_names_raw)
            missing_sensor_foot_names = [
                name for name in foot_body_names
                if name not in sensor_body_names
            ]
            if missing_sensor_foot_names:
                raise RuntimeError(
                    "Foot bodies missing from contact sensor: "
                    f"{missing_sensor_foot_names}. "
                    f"Sensor bodies={sensor_body_names}"
                )
            foot_contact_sensor_body_ids = [
                sensor_body_names.index(name)
                for name in foot_body_names
            ]
            print(
                "[INFO] Contact sensor foot mapping: "
                + ", ".join(
                    f"{name}->{index}"
                    for name, index in zip(
                        foot_body_names,
                        foot_contact_sensor_body_ids,
                    )
                )
            )
            foot_contact_force_threshold_n = 20.0

            dt = float(base_env.step_dt)
            warmup_steps = int(round(args_cli.warmup_s / dt))
            requested_steps = max(1, int(round(args_cli.duration_s / dt)))
            print_every_steps = max(1, int(round(args_cli.print_every_s / dt)))
            total_steps = warmup_steps + requested_steps
            push_measurement_step = int(round(float(args_cli.push_time_s) / dt)) if push_requested else None
            if push_measurement_step is not None and push_measurement_step >= requested_steps:
                raise ValueError("Rounded push step falls outside the measurement window.")
            controlled_push: dict[str, object] = {
                "requested": push_requested,
                "applied": False,
                "requested_time_s": float(args_cli.push_time_s) if push_requested else None,
                "effective_time_s": push_measurement_step * dt if push_measurement_step is not None else None,
                "delta_root_velocity_world": {"x_mps": push_delta[0], "y_mps": push_delta[1], "yaw_radps": push_delta[2]},
                "before_root_velocity_world": None,
                "after_root_velocity_world": None,
            }

            print(
                f"[INFO] dt={dt:.6f} s, warmup={warmup_steps} steps, "
                f"measurement={requested_steps} steps"
            )

            obs = env.get_observations()

            initial_root_pos_w = _cpu_values(_torch(robot.data.root_pos_w)[0])
            initial_root_lin_vel_w = _cpu_values(_torch(robot.data.root_lin_vel_w)[0])
            initial_root_ang_vel_w = _cpu_values(_torch(robot.data.root_ang_vel_w)[0])
            initial_heading_w = float(_torch(robot.data.heading_w)[0].detach().cpu().item())
            initial_state: dict[str, object] = {
                "root_position_world_m": {
                    "x": initial_root_pos_w[0],
                    "y": initial_root_pos_w[1],
                    "z": initial_root_pos_w[2],
                },
                "root_linear_velocity_world_mps": {
                    "x": initial_root_lin_vel_w[0],
                    "y": initial_root_lin_vel_w[1],
                    "z": initial_root_lin_vel_w[2],
                },
                "root_angular_velocity_world_radps": {
                    "x": initial_root_ang_vel_w[0],
                    "y": initial_root_ang_vel_w[1],
                    "z": initial_root_ang_vel_w[2],
                },
                "heading_world_rad": initial_heading_w,
                "heading_world_deg": math.degrees(initial_heading_w),
            }

            records: list[dict[str, float]] = []
            terminated_stage: str | None = None

            for step in range(total_steps):
                push_applied_this_step = False
                measurement_step = step - warmup_steps
                if push_measurement_step is not None and measurement_step == push_measurement_step:
                    root_velocity_w = _torch(robot.data.root_vel_w)[0].detach().clone()
                    before_velocity_w = _cpu_values(root_velocity_w)
                    root_velocity_w[0] += push_delta[0]
                    root_velocity_w[1] += push_delta[1]
                    root_velocity_w[5] += push_delta[2]
                    push_env_ids = torch.tensor([0], device=root_velocity_w.device, dtype=torch.long)
                    robot.write_root_velocity_to_sim_index(root_velocity=root_velocity_w.unsqueeze(0), env_ids=push_env_ids)
                    obs = env.get_observations()
                    after_velocity_w = _cpu_values(_torch(robot.data.root_vel_w)[0])
                    controlled_push["applied"] = True
                    controlled_push["before_root_velocity_world"] = {
                        "x_mps": before_velocity_w[0], "y_mps": before_velocity_w[1], "z_mps": before_velocity_w[2],
                        "roll_radps": before_velocity_w[3], "pitch_radps": before_velocity_w[4], "yaw_radps": before_velocity_w[5],
                    }
                    controlled_push["after_root_velocity_world"] = {
                        "x_mps": after_velocity_w[0], "y_mps": after_velocity_w[1], "z_mps": after_velocity_w[2],
                        "roll_radps": after_velocity_w[3], "pitch_radps": after_velocity_w[4], "yaw_radps": after_velocity_w[5],
                    }
                    push_applied_this_step = True
                    print(f"[PUSH] t={controlled_push['effective_time_s']:.3f}s delta_world=({push_delta[0]:+.3f}, {push_delta[1]:+.3f}, {push_delta[2]:+.3f})")

                with torch.inference_mode():
                    actions = policy(obs)
                    obs, _, dones, _ = env.step(actions)
                    if version.parse(installed_version) >= version.parse("4.0.0"):
                        policy.reset(dones)
                    else:
                        policy_network.reset(dones)

                if bool(torch.any(dones).item()):
                    terminated_stage = "warmup" if step < warmup_steps else "measurement"
                    print(f"[ERROR] Environment terminated during {terminated_stage} at simulation step {step}.")
                    break

                if step < warmup_steps:
                    continue

                measurement_index = step - warmup_steps
                command = _cpu_values(base_env.command_manager.get_command("base_velocity")[0])
                root_pos = _cpu_values(_torch(robot.data.root_pos_w)[0])
                root_lin_vel_b = _cpu_values(_torch(robot.data.root_lin_vel_b)[0])
                root_ang_vel_b = _cpu_values(_torch(robot.data.root_ang_vel_b)[0])
                heading = float(_torch(robot.data.heading_w)[0].detach().cpu().item())
                posture = _cpu_values(_torch(robot.data.joint_pos)[0, posture_joint_ids])
                posture_vel = _cpu_values(
                    _torch(robot.data.joint_vel)[0, posture_joint_ids]
                )
                posture_acc = _cpu_values(
                    _torch(robot.data.joint_acc)[0, posture_joint_ids]
                )
                posture_applied_torque = _cpu_values(
                    _torch(robot.data.applied_torque)[0, posture_joint_ids]
                )
                posture_computed_torque = _cpu_values(
                    _torch(robot.data.computed_torque)[0, posture_joint_ids]
                )
                projected_gravity = _cpu_values(
                    _torch(robot.data.projected_gravity_b)[0]
                )
                torso_quat_w = _cpu_values(
                    _torch(robot.data.body_quat_w)[0, torso_body_id]
                )
                torso_ang_vel_w = _cpu_values(
                    _torch(robot.data.body_ang_vel_w)[0, torso_body_id]
                )
                torso_roll, torso_pitch, torso_yaw = (
                    _quat_wxyz_to_euler_xyz(torso_quat_w)
                )
                foot_net_forces_w = np.asarray(
                    _cpu_values(_torch(contact_sensor.data.net_forces_w)[0, foot_contact_sensor_body_ids]),
                    dtype=np.float64,
                )
                foot_sensor_contact_time = np.asarray(
                    _cpu_values(_torch(contact_sensor.data.current_contact_time)[0, foot_contact_sensor_body_ids]),
                    dtype=np.float64,
                )
                foot_sensor_air_time = np.asarray(
                    _cpu_values(_torch(contact_sensor.data.current_air_time)[0, foot_contact_sensor_body_ids]),
                    dtype=np.float64,
                )
                foot_net_forces_w = np.asarray(
                    foot_net_forces_w,
                    dtype=np.float64,
                ).reshape(2, 3)
                foot_normal_forces = np.abs(foot_net_forces_w[:, 2])
                foot_force_magnitudes = np.linalg.norm(foot_net_forces_w, axis=1)
                foot_contacts = foot_normal_forces >= foot_contact_force_threshold_n

                record: dict[str, float] = {
                    "time_s": float((measurement_index + 1) * dt),
                    "cmd_vx_mps": command[0],
                    "cmd_vy_mps": command[1],
                    "cmd_wz_radps": command[2],
                    "base_vx_mps": root_lin_vel_b[0],
                    "base_vy_mps": root_lin_vel_b[1],
                    "base_vz_mps": root_lin_vel_b[2],
                    "base_wx_radps": root_ang_vel_b[0],
                    "base_wy_radps": root_ang_vel_b[1],
                    "base_wz_radps": root_ang_vel_b[2],
                    "world_x_m": root_pos[0],
                    "world_y_m": root_pos[1],
                    "root_height_m": root_pos[2],
                    "heading_rad": heading,
                    "controlled_push_applied": 1.0 if push_applied_this_step else 0.0,
                }
                for name, value in zip(resolved_joint_names, posture, strict=True):
                    record[f"{name}_rad"] = value
                for index, joint_name in enumerate(resolved_joint_names):
                    record[f"{joint_name}_vel_radps"] = posture_vel[index]
                    record[f"{joint_name}_acc_radps2"] = posture_acc[index]
                    record[
                        f"{joint_name}_applied_torque_nm"
                    ] = posture_applied_torque[index]
                    record[
                        f"{joint_name}_computed_torque_nm"
                    ] = posture_computed_torque[index]
                    record[
                        f"{joint_name}_mechanical_power_w"
                    ] = (
                        posture_applied_torque[index]
                        * posture_vel[index]
                    )

                record.update(
                    {
                        "projected_gravity_x": projected_gravity[0],
                        "projected_gravity_y": projected_gravity[1],
                        "projected_gravity_z": projected_gravity[2],
                        "torso_roll_rad": torso_roll,
                        "torso_pitch_rad": torso_pitch,
                        "torso_yaw_rad": torso_yaw,
                        "torso_ang_vel_world_x_radps": (
                            torso_ang_vel_w[0]
                        ),
                        "torso_ang_vel_world_y_radps": (
                            torso_ang_vel_w[1]
                        ),
                        "torso_ang_vel_world_z_radps": (
                            torso_ang_vel_w[2]
                        ),
                        "torso_ang_speed_radps": float(
                            np.linalg.norm(torso_ang_vel_w)
                        ),
                        "left_foot_normal_force_n": float(foot_normal_forces[0]),
                        "right_foot_normal_force_n": float(foot_normal_forces[1]),
                        "left_foot_net_force_magnitude_n": float(foot_force_magnitudes[0]),
                        "right_foot_net_force_magnitude_n": float(foot_force_magnitudes[1]),
                        "left_foot_contact": 1.0 if bool(foot_contacts[0]) else 0.0,
                        "right_foot_contact": 1.0 if bool(foot_contacts[1]) else 0.0,
                        "left_foot_sensor_current_contact_time_s": float(foot_sensor_contact_time[0]),
                        "right_foot_sensor_current_contact_time_s": float(foot_sensor_contact_time[1]),
                        "left_foot_sensor_current_air_time_s": float(foot_sensor_air_time[0]),
                        "right_foot_sensor_current_air_time_s": float(foot_sensor_air_time[1]),
                    }
                )
                records.append(record)

                if measurement_index % print_every_steps == 0 or measurement_index == requested_steps - 1:
                    print(
                        "[EVAL] "
                        f"t={record['time_s']:6.2f}s "
                        f"cmd=({record['cmd_vx_mps']:+.3f}, {record['cmd_vy_mps']:+.3f}, "
                        f"{record['cmd_wz_radps']:+.3f}) "
                        f"vel=({record['base_vx_mps']:+.3f}, {record['base_vy_mps']:+.3f}, "
                        f"{record['base_wz_radps']:+.3f}) "
                        f"pos=({record['world_x_m']:+.3f}, {record['world_y_m']:+.3f}) "
                        f"heading={math.degrees(record['heading_rad']):+.2f}deg"
                    )

            if not records:
                raise RuntimeError("No measurement samples were collected.")

            summary = _build_summary(
                records=records,
                dt=dt,
                requested_steps=requested_steps,
                warmup_s=float(args_cli.warmup_s),
                task=args_cli.task,
                checkpoint=checkpoint,
                checkpoint_hash=checkpoint_hash,
                seed=int(env_cfg.seed),
                initial_state=initial_state,
                controlled_push=controlled_push,
                joint_names=resolved_joint_names,
                terminated_stage=terminated_stage,
            )

            csv_path = result_dir / "samples.csv"
            json_path = result_dir / "summary.json"
            _write_csv(csv_path, records)
            with json_path.open("w", encoding="utf-8") as stream:
                json.dump(summary, stream, indent=2, sort_keys=True)
                stream.write("\n")

            _print_summary(summary, result_dir.resolve())
        finally:
            env.close()


if __name__ == "__main__":
    main()
