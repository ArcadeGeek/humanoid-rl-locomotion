#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import random
import statistics
import traceback
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

parser = argparse.ArgumentParser(
    description=(
        "Standardized development-set straight-line benchmark for the historical v2/v3 policy family. "
        "All checkpoints are evaluated in the SAME canonical v2-G environment with the same command, "
        "seed protocol and zero-root-velocity startup conditioning. This benchmark is intended to "
        "generate thesis comparison data; seeds 100-109 are DEVELOPMENT seeds, not untouched holdout."
    )
)
parser.add_argument(
    "--case",
    required=True,
    choices=("V2-A", "V2-D", "V2-E", "V2-F", "V2-G", "V3-C"),
)
parser.add_argument("--seed-start", type=int, default=100)
parser.add_argument("--seed-end", type=int, default=109)
parser.add_argument("--duration-s", type=float, default=6.0)
parser.add_argument("--tail-s", type=float, default=3.0)
parser.add_argument(
    "--output-root",
    type=Path,
    default=Path(
        r"C:\humanoid_rl_lab\logs\thesis\figures_final"
        r"\01_v2_v3_evolution\data\straight_common_eval_v1"
    ),
)

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import humanoid_rl_lab.tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner


REPO = Path(r"C:\humanoid_rl_lab")

# Common inference environment for ALL historical policies. Reward differences among training
# tasks therefore cannot contaminate the evaluation conditions.
CANONICAL_EVAL_TASK = (
    "Humanoid-RL-Lab-G1-Flat-Upright-Yaw-Direct-Wide-Yaw-Flat-Orientation-v0"
)

CASES = {
    "V2-A": {
        "role": "direct-yaw genuine-scratch baseline",
        "native_task": "Humanoid-RL-Lab-G1-Flat-Upright-Yaw-Direct-v0",
        "checkpoint": REPO / "logs" / "rsl_rl" / "humanoid_rl_lab_g1_upright_yaw_direct"
            / "2026-08-15_03-33-18_thesis_yaw_direct_genuine_scratch_v2a_to700" / "model_600.pt",
        "sha256": "F1340C5E5CB44AE3157CC9C4D4493BBD1BC6316ED64A029FBD514C7FC0F7B413",
    },
    "V2-D": {
        "role": "pseudo-Huber negative result",
        "native_task": "Humanoid-RL-Lab-G1-Flat-Upright-Yaw-Direct-Pseudo-Huber-Yaw-v0",
        "checkpoint": REPO / "logs" / "rsl_rl" / "humanoid_rl_lab_g1_upright_yaw_direct_pseudo_huber_yaw"
            / "2026-08-15_17-13-57_thesis_yaw_direct_pseudo_huber_v2d_genuine_scratch_to700" / "model_600.pt",
        "sha256": "E298A2FAA3959421C5AE00CB3908F84EA34C36CB4D6126194921FFD06AC356FD",
    },
    "V2-E": {
        "role": "heavy-tail shaping, stable but insufficient",
        "native_task": "Humanoid-RL-Lab-G1-Flat-Upright-Yaw-Direct-Heavy-Tail-Yaw-v0",
        "checkpoint": REPO / "logs" / "rsl_rl" / "humanoid_rl_lab_g1_upright_yaw_direct_heavy_tail_yaw"
            / "2026-08-16_20-06-45_thesis_yaw_direct_heavy_tail_v2e_genuine_scratch_to700" / "model_600.pt",
        "sha256": "154AB9548FA829C59764AFA772D6897BD0F29104729131515A8A616F4879645F",
    },
    "V2-F": {
        "role": "wide-yaw training breakthrough",
        "native_task": "Humanoid-RL-Lab-G1-Flat-Upright-Yaw-Direct-Wide-Yaw-v0",
        "checkpoint": REPO / "logs" / "rsl_rl" / "humanoid_rl_lab_g1_upright_yaw_direct_wide_yaw"
            / "2026-08-16_22-19-58_thesis_yaw_direct_wide_yaw_v2f_genuine_scratch_to700" / "model_600.pt",
        "sha256": "F650608C94B50F1BD77D64F975B1F0B87033510B74A159CB3A56860DE4271EE7",
    },
    "V2-G": {
        "role": "flat-orientation refinement / frozen STAND controller",
        "native_task": CANONICAL_EVAL_TASK,
        "checkpoint": REPO / "checkpoints" / "local_g1_yaw_direct_wide_yaw_flat_orientation_v2g" / "model_600.pt",
        "sha256": "58468EED03F9EF35A9BC898063A3B14AC36C93A32F65DB42B404BDE93CD694D3",
    },
    "V3-C": {
        "role": "exploratory swing-clearance refinement / NOT promoted",
        "native_task": "Humanoid-RL-Lab-G1-Flat-Upright-Yaw-Direct-Wide-Yaw-Flat-Orientation-V3C-v0",
        "checkpoint": REPO / "logs" / "rsl_rl" / "humanoid_rl_lab_g1_v3c_swing_clearance_refinement"
            / "2026-08-26_12-24-10_v3c_from_v2g_model600_30it" / "model_620.pt",
        "sha256": "C9D93ADE8F32D8F52D1583F4596EAB8906377802265B975E66719C5B41F08458",
    },
}

COMMAND = (0.40, 0.0, 0.0)
EXPECTED_DT = 0.02
EXPECTED_OBS_DIM = 123
EXPECTED_ACTION_DIM = 37
CONTACT_FORCE_THRESHOLD_N = 20.0

FORMAT_ID = "G1_THESIS_V2V3_STRAIGHT_COMMON_BENCHMARK_V1"
PASS_MARKER = "G1_THESIS_FIG_CORE1_V2V3_STRAIGHT_COMMON_BENCHMARK_V1_PASS"
FAIL_MARKER = "G1_THESIS_FIG_CORE1_V2V3_STRAIGHT_COMMON_BENCHMARK_V1_FAIL"

CLASSIFICATION = (
    "standardized development-set post-training evaluation for thesis historical comparison; "
    "same canonical inference environment across v2/v3; NOT untouched holdout"
)


def fail(msg: str) -> None:
    raise RuntimeError(msg)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def tv(value: Any) -> torch.Tensor:
    if hasattr(value, "torch"):
        value = value.torch
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    return value


def extract_policy_obs(obs: Any) -> torch.Tensor:
    if torch.is_tensor(obs) or hasattr(obs, "torch"):
        candidate = obs
    else:
        candidate = None
        try:
            candidate = obs["policy"]
        except Exception:
            get_fn = getattr(obs, "get", None)
            if callable(get_fn):
                try:
                    candidate = get_fn("policy")
                except Exception:
                    candidate = None
    if candidate is None:
        fail(f"cannot resolve policy observation from {type(obs).__name__}")
    tensor = tv(candidate)
    if tensor.ndim != 2 or tuple(tensor.shape) != (1, EXPECTED_OBS_DIM):
        fail(f"unexpected policy observation shape: {tuple(tensor.shape)}")
    return tensor


def reset_global_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_command(base_env: Any) -> None:
    cmd = base_env.command_manager.get_command("base_velocity")
    with torch.no_grad():
        cmd.zero_()
        cmd[:, 0] = COMMAND[0]
        cmd[:, 1] = COMMAND[1]
        cmd[:, 2] = COMMAND[2]


def tilt_deg(robot: Any) -> float:
    g = tv(robot.data.projected_gravity_b)[0]
    c = torch.clamp(-g[2], -1.0, 1.0)
    return float(torch.rad2deg(torch.acos(c)).detach().cpu().item())


def action_rms(action: torch.Tensor) -> float:
    a = tv(action).detach().float()
    return float(torch.sqrt(torch.mean(a * a)).cpu().item())


def make_common_env_cfg(seed: int):
    cfg = parse_env_cfg(
        CANONICAL_EVAL_TASK,
        device=args_cli.device,
        num_envs=1,
    )
    cfg.scene.num_envs = 1
    cfg.seed = int(seed)
    if args_cli.device is not None:
        cfg.sim.device = args_cli.device

    # Prevent timeout from truncating the requested scientific window.
    cfg.episode_length_s = max(
        float(cfg.episode_length_s),
        float(args_cli.duration_s + 10.0),
    )

    if hasattr(cfg.observations, "policy") and hasattr(cfg.observations.policy, "enable_corruption"):
        cfg.observations.policy.enable_corruption = False

    if hasattr(cfg, "events"):
        if hasattr(cfg.events, "push_robot"):
            cfg.events.push_robot = None
        if hasattr(cfg.events, "base_external_force_torque"):
            cfg.events.base_external_force_torque = None

    # Preserve randomized reset POSE so seeds provide development-set variability,
    # but remove reset root velocity as frozen engineering startup conditioning.
    reset_base = getattr(cfg.events, "reset_base", None)
    if reset_base is not None:
        reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }

    cmd_cfg = cfg.commands.base_velocity
    cmd_cfg.resampling_time_range = (1.0e9, 1.0e9)
    cmd_cfg.rel_standing_envs = 0.0
    cmd_cfg.ranges.lin_vel_x = (0.0, 0.0)
    cmd_cfg.ranges.lin_vel_y = (0.0, 0.0)
    cmd_cfg.ranges.ang_vel_z = (0.0, 0.0)
    if hasattr(cmd_cfg, "heading_command"):
        cmd_cfg.heading_command = False
    if hasattr(cmd_cfg, "rel_heading_envs"):
        cmd_cfg.rel_heading_envs = 0.0
    if hasattr(cmd_cfg, "debug_vis"):
        cmd_cfg.debug_vis = False

    return cfg


def make_native_agent_cfg(native_task: str, seed: int):
    cfg = load_cfg_from_registry(native_task, "rsl_rl_cfg_entry_point")
    if args_cli.device is not None and hasattr(cfg, "device"):
        cfg.device = args_cli.device
    if hasattr(cfg, "seed"):
        cfg.seed = int(seed)
    cfg = handle_deprecated_rsl_rl_cfg(cfg, metadata.version("rsl-rl-lib"))
    if cfg.class_name != "OnPolicyRunner":
        fail(f"unexpected runner class: {cfg.class_name}")
    return cfg


def fingerprint_reset(robot: Any) -> list[float]:
    tensors = (
        tv(robot.data.root_pos_w)[0],
        tv(robot.data.root_quat_w)[0],
        tv(robot.data.root_lin_vel_w)[0],
        tv(robot.data.root_ang_vel_w)[0],
        tv(robot.data.joint_pos)[0],
    )
    out: list[float] = []
    for tensor in tensors:
        out.extend(float(x) for x in tensor.detach().cpu().tolist())
    return out


def summarize_rows(rows: list[dict[str, Any]], completed: bool) -> dict[str, Any]:
    if not rows:
        return {
            "samples_recorded": 0,
            "completed": False,
            "duration_recorded_s": 0.0,
        }

    requested_tail_n = int(round(args_cli.tail_s / EXPECTED_DT))
    tail_n = min(len(rows), requested_tail_n)
    tail = rows[-tail_n:]

    def mean(key: str, data: list[dict[str, Any]]) -> float:
        return statistics.fmean(float(r[key]) for r in data)

    def p95(key: str, data: list[dict[str, Any]]) -> float:
        vals = sorted(float(r[key]) for r in data)
        if len(vals) == 1:
            return vals[0]
        pos = 0.95 * (len(vals) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return vals[lo]
        frac = pos - lo
        return vals[lo] * (1.0 - frac) + vals[hi] * frac

    def maximum(key: str, data: list[dict[str, Any]]) -> float:
        return max(float(r[key]) for r in data)

    ds_valid = all(float(r["double_support"]) >= 0.0 for r in rows)
    tail_ds_valid = all(float(r["double_support"]) >= 0.0 for r in tail)

    return {
        "samples_recorded": len(rows),
        "completed": bool(completed),
        "duration_recorded_s": len(rows) * EXPECTED_DT,
        "mean_vx_b_mps": mean("vx_b_mps", rows),
        "mean_abs_vx_tracking_error_mps": mean("abs_vx_tracking_error_mps", rows),
        "mean_vy_b_mps": mean("vy_b_mps", rows),
        "mean_abs_wz_radps": mean("abs_wz_radps", rows),
        "mean_tilt_deg": mean("tilt_deg", rows),
        "p95_tilt_deg": p95("tilt_deg", rows),
        "max_tilt_deg": maximum("tilt_deg", rows),
        "mean_root_z_m": mean("root_z_m", rows),
        "std_root_z_m": statistics.pstdev(float(r["root_z_m"]) for r in rows) if len(rows) > 1 else 0.0,
        "mean_action_rms": mean("action_rms", rows),
        "double_support_pct": 100.0 * mean("double_support", rows) if ds_valid else float("nan"),
        "tail_window_requested_s": float(args_cli.tail_s),
        "tail_window_actual_s": tail_n * EXPECTED_DT,
        "tail_mean_vx_b_mps": mean("vx_b_mps", tail),
        "tail_mean_abs_vx_tracking_error_mps": mean("abs_vx_tracking_error_mps", tail),
        "tail_mean_vy_b_mps": mean("vy_b_mps", tail),
        "tail_mean_abs_wz_radps": mean("abs_wz_radps", tail),
        "tail_mean_tilt_deg": mean("tilt_deg", tail),
        "tail_p95_tilt_deg": p95("tilt_deg", tail),
        "tail_max_tilt_deg": maximum("tilt_deg", tail),
        "tail_mean_root_z_m": mean("root_z_m", tail),
        "tail_std_root_z_m": statistics.pstdev(float(r["root_z_m"]) for r in tail) if len(tail) > 1 else 0.0,
        "tail_double_support_pct": (
            100.0 * mean("double_support", tail) if tail_ds_valid else float("nan")
        ),
    }


def run_seed(
    case_name: str,
    case: dict[str, Any],
    seed: int,
    case_dir: Path,
) -> dict[str, Any]:
    reset_global_rng(seed)
    env_cfg = make_common_env_cfg(seed)
    agent_cfg = make_native_agent_cfg(case["native_task"], seed)

    env = None
    rows: list[dict[str, Any]] = []

    try:
        raw_env = gym.make(CANONICAL_EVAL_TASK, cfg=env_cfg)
        env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
        base_env = env.unwrapped
        robot = base_env.scene["robot"]

        dt = float(base_env.step_dt)
        if not math.isclose(dt, EXPECTED_DT, rel_tol=0.0, abs_tol=1.0e-12):
            fail(f"{case_name} seed {seed}: unexpected dt={dt}")
        if int(base_env.action_manager.total_action_dim) != EXPECTED_ACTION_DIM:
            fail(f"{case_name} seed {seed}: unexpected action dim")

        runner = OnPolicyRunner(
            env,
            agent_cfg.to_dict(),
            log_dir=None,
            device=agent_cfg.device,
        )
        runner.load(str(case["checkpoint"]))
        policy = runner.get_inference_policy(device=base_env.device)

        obs = env.get_observations()
        extract_policy_obs(obs)

        root_lin0 = tv(robot.data.root_lin_vel_w)[0].detach().cpu()
        root_ang0 = tv(robot.data.root_ang_vel_w)[0].detach().cpu()
        if float(torch.max(torch.abs(root_lin0)).item()) > 1.0e-6:
            fail(f"{case_name} seed {seed}: root linear reset velocity not zero")
        if float(torch.max(torch.abs(root_ang0)).item()) > 1.0e-6:
            fail(f"{case_name} seed {seed}: root angular reset velocity not zero")

        reset_fp = fingerprint_reset(robot)
        reset_fp_text = ",".join(f"{x:.12g}" for x in reset_fp)
        reset_fp_sha = hashlib.sha256(reset_fp_text.encode("utf-8")).hexdigest().upper()

        contact_sensor = None
        left_idx = None
        right_idx = None
        try:
            contact_sensor = base_env.scene.sensors["contact_forces"]
            names = list(contact_sensor.body_names)
            if "left_ankle_roll_link" in names and "right_ankle_roll_link" in names:
                left_idx = names.index("left_ankle_roll_link")
                right_idx = names.index("right_ankle_roll_link")
            else:
                contact_sensor = None
        except Exception:
            contact_sensor = None

        steps = int(round(args_cli.duration_s / dt))
        completed = True
        termination_step = None

        for step in range(steps):
            set_command(base_env)
            obs = env.get_observations()

            with torch.inference_mode():
                action = policy(obs)

            action_t = tv(action)
            if tuple(action_t.shape) != (1, EXPECTED_ACTION_DIM):
                fail(f"{case_name} seed {seed}: unexpected action shape {tuple(action_t.shape)}")
            if not bool(torch.isfinite(action_t).all().item()):
                fail(f"{case_name} seed {seed}: non-finite action at step {step}")

            with torch.inference_mode():
                obs, _, dones, _ = env.step(action)

            # Isaac Lab may reset the environment in the same step as done.
            # Do NOT append post-reset state to the scientific trajectory.
            if bool(torch.any(tv(dones)).item()):
                completed = False
                termination_step = step
                break

            lin_b = tv(robot.data.root_lin_vel_b)[0]
            ang_b = tv(robot.data.root_ang_vel_b)[0]
            pos = tv(robot.data.root_pos_w)[0]

            double_support = -1.0
            if contact_sensor is not None and left_idx is not None and right_idx is not None:
                try:
                    net = tv(contact_sensor.data.net_forces_w)[0]
                    lf = abs(float(net[left_idx, 2].detach().cpu().item()))
                    rf = abs(float(net[right_idx, 2].detach().cpu().item()))
                    double_support = 1.0 if (
                        lf >= CONTACT_FORCE_THRESHOLD_N and rf >= CONTACT_FORCE_THRESHOLD_N
                    ) else 0.0
                except Exception:
                    contact_sensor = None
                    double_support = -1.0

            vx_b = float(lin_b[0].detach().cpu().item())
            vy_b = float(lin_b[1].detach().cpu().item())
            wz = float(ang_b[2].detach().cpu().item())

            rows.append(
                {
                    "case": case_name,
                    "seed": seed,
                    "step": step,
                    "time_s": (step + 1) * dt,
                    "command_vx_mps": COMMAND[0],
                    "command_vy_mps": COMMAND[1],
                    "command_wz_radps": COMMAND[2],
                    "vx_b_mps": vx_b,
                    "vy_b_mps": vy_b,
                    "vxy_mps": math.hypot(vx_b, vy_b),
                    "abs_vx_tracking_error_mps": abs(vx_b - COMMAND[0]),
                    "abs_wz_radps": abs(wz),
                    "root_z_m": float(pos[2].detach().cpu().item()),
                    "tilt_deg": tilt_deg(robot),
                    "action_rms": action_rms(action_t),
                    "double_support": double_support,
                }
            )

        seed_dir = case_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        trajectory_path = seed_dir / "trajectory.csv"
        if rows:
            with trajectory_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        else:
            trajectory_path.write_text("", encoding="utf-8")

        summary = summarize_rows(rows, completed)
        summary.update(
            {
                "format": FORMAT_ID,
                "classification": CLASSIFICATION,
                "case": case_name,
                "role": case["role"],
                "seed": seed,
                "canonical_eval_task": CANONICAL_EVAL_TASK,
                "native_training_task": case["native_task"],
                "checkpoint": str(case["checkpoint"]),
                "checkpoint_sha256": case["sha256"],
                "command": list(COMMAND),
                "requested_duration_s": float(args_cli.duration_s),
                "termination_step": termination_step,
                "reset_fingerprint_sha256": reset_fp_sha,
                "trajectory_csv": str(trajectory_path),
                "trajectory_csv_sha256": sha256_file(trajectory_path),
            }
        )

        summary_path = seed_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    finally:
        if env is not None:
            env.close()
        del env
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def finite_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    vals = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            vals.append(float(value))
    return vals


def aggregate_metric(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    vals = finite_values(rows, key)
    if not vals:
        return {
            f"{key}_n": 0,
            f"{key}_mean": None,
            f"{key}_sd": None,
            f"{key}_median": None,
            f"{key}_min": None,
            f"{key}_max": None,
        }
    return {
        f"{key}_n": len(vals),
        f"{key}_mean": statistics.fmean(vals),
        f"{key}_sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        f"{key}_median": statistics.median(vals),
        f"{key}_min": min(vals),
        f"{key}_max": max(vals),
    }


def main() -> None:
    if args_cli.seed_end < args_cli.seed_start:
        fail("--seed-end must be >= --seed-start")
    if args_cli.duration_s < 3.0:
        fail("--duration-s must be >= 3.0")
    if args_cli.tail_s <= 0.0 or args_cli.tail_s > args_cli.duration_s:
        fail("--tail-s must satisfy 0 < tail-s <= duration-s")

    case_name = args_cli.case
    case = CASES[case_name]
    checkpoint: Path = case["checkpoint"]

    if not checkpoint.is_file():
        fail(f"checkpoint missing: {checkpoint}")
    actual_sha = sha256_file(checkpoint)
    if actual_sha != case["sha256"]:
        fail(f"{case_name}: checkpoint SHA mismatch: expected {case['sha256']}, got {actual_sha}")

    case_slug = case_name.replace("-", "")
    case_dir = Path(args_cli.output_root).resolve() / case_slug
    case_dir.mkdir(parents=True, exist_ok=True)

    seeds = list(range(args_cli.seed_start, args_cli.seed_end + 1))

    print("=" * 132)
    print("G1 THESIS CORE 1 — v2/v3 STANDARDIZED STRAIGHT-LINE BENCHMARK")
    print("=" * 132)
    print(f"Case / role                  : {case_name} / {case['role']}")
    print(f"Native training task         : {case['native_task']}")
    print(f"COMMON evaluation task       : {CANONICAL_EVAL_TASK}")
    print(f"Checkpoint SHA256            : {actual_sha}")
    print(f"Command                      : vx={COMMAND[0]:.2f}, vy=0, wz=0")
    print(f"Duration / tail              : {args_cli.duration_s:.2f} s / {args_cli.tail_s:.2f} s")
    print(f"Seeds                        : {seeds[0]}–{seeds[-1]} (N={len(seeds)})")
    print("Reset pose                   : canonical randomized pose, seed-controlled")
    print("Reset root velocities        : ZERO all 6 components")
    print("Observation corruption       : OFF")
    print("External pushes/forces       : OFF")
    print("Physical terminations        : ON")
    print(f"Classification               : {CLASSIFICATION}")
    print(f"Output                       : {case_dir}")
    print("=" * 132)

    rows: list[dict[str, Any]] = []

    for i, seed in enumerate(seeds, start=1):
        print(f"[RUN {i:02d}/{len(seeds):02d}] {case_name} seed={seed}", flush=True)
        summary = run_seed(case_name, case, seed, case_dir)
        rows.append(summary)
        print(
            f"    completed={summary['completed']} "
            f"samples={summary['samples_recorded']} "
            f"tail_vx={summary.get('tail_mean_vx_b_mps', float('nan')):.6f} "
            f"tail_abs_err={summary.get('tail_mean_abs_vx_tracking_error_mps', float('nan')):.6f} "
            f"tail_tilt={summary.get('tail_mean_tilt_deg', float('nan')):.6f}",
            flush=True,
        )

    # Case-level CSV: one row per seed. This is the canonical table used later by the plotter.
    seed_summary_csv = case_dir / f"{case_slug}_seed_summary.csv"
    all_keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in all_keys:
                all_keys.append(key)
    with seed_summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)

    complete_rows = [r for r in rows if bool(r.get("completed"))]
    aggregate: dict[str, Any] = {
        "format": FORMAT_ID,
        "status": "PASS",
        "classification": CLASSIFICATION,
        "case": case_name,
        "role": case["role"],
        "canonical_eval_task": CANONICAL_EVAL_TASK,
        "native_training_task": case["native_task"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": actual_sha,
        "command": list(COMMAND),
        "seed_start": seeds[0],
        "seed_end": seeds[-1],
        "num_seeds": len(seeds),
        "completed_trials": len(complete_rows),
        "terminated_trials": len(rows) - len(complete_rows),
        "completion_rate_pct": 100.0 * len(complete_rows) / len(rows),
        "requested_duration_s": float(args_cli.duration_s),
        "tail_s": float(args_cli.tail_s),
        "seed_summary_csv": str(seed_summary_csv),
        "seed_summary_csv_sha256": sha256_file(seed_summary_csv),
    }

    # Aggregate sustained metrics over COMPLETED trials only.
    # Completion rate is reported separately so failures can never be silently hidden.
    for key in (
        "tail_mean_vx_b_mps",
        "tail_mean_abs_vx_tracking_error_mps",
        "tail_mean_vy_b_mps",
        "tail_mean_abs_wz_radps",
        "tail_mean_tilt_deg",
        "tail_p95_tilt_deg",
        "tail_max_tilt_deg",
        "tail_mean_root_z_m",
        "tail_std_root_z_m",
        "tail_double_support_pct",
    ):
        aggregate.update(aggregate_metric(complete_rows, key))

    aggregate_path = case_dir / f"{case_slug}_aggregate_summary.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    runner_sha = sha256_file(Path(__file__).resolve())

    print()
    print("=" * 132)
    print("CORE 1 STRAIGHT BENCHMARK SUMMARY")
    print("=" * 132)
    print(f"Status                              : PASS")
    print(f"Case                                : {case_name}")
    print(f"Trials completed                    : {len(complete_rows)} / {len(rows)}")
    print(f"Completion rate                     : {aggregate['completion_rate_pct']:.1f}%")
    print(
        "Tail mean vx                       : "
        f"{aggregate.get('tail_mean_vx_b_mps_mean')} "
        f"± {aggregate.get('tail_mean_vx_b_mps_sd')} m/s"
    )
    print(
        "Tail mean |vx-command|             : "
        f"{aggregate.get('tail_mean_abs_vx_tracking_error_mps_mean')} "
        f"± {aggregate.get('tail_mean_abs_vx_tracking_error_mps_sd')} m/s"
    )
    print(
        "Tail mean tilt                     : "
        f"{aggregate.get('tail_mean_tilt_deg_mean')} "
        f"± {aggregate.get('tail_mean_tilt_deg_sd')} deg"
    )
    print(f"Per-seed CSV                        : {seed_summary_csv}")
    print(f"Aggregate JSON                      : {aggregate_path}")
    print(f"Runner SHA256                       : {runner_sha}")
    print("Repository source modified          : NO")
    print("Checkpoint modified                 : NO")
    print(PASS_MARKER)


try:
    main()
except BaseException as exc:
    print(FAIL_MARKER, flush=True)
    print(f"EXCEPTION_TYPE={type(exc).__name__}", flush=True)
    print(f"EXCEPTION_MESSAGE={exc}", flush=True)
    traceback.print_exc()
    raise
finally:
    gc.collect()
    simulation_app.close()
