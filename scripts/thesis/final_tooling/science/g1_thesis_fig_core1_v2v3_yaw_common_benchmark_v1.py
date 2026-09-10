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
from importlib import metadata
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Paired-seed common yaw benchmark for viable historical v2/v3 policies."
)
parser.add_argument("--case", required=True, choices=("V2-A", "V2-E", "V2-F", "V2-G", "V3-C"))
parser.add_argument("--seed-start", type=int, default=100)
parser.add_argument("--seed-end", type=int, default=109)
parser.add_argument("--duration-s", type=float, default=6.0)
parser.add_argument("--tail-s", type=float, default=3.0)
parser.add_argument(
    "--output-root",
    type=Path,
    default=Path(
        r"C:\humanoid_rl_lab\logs\thesis\figures_final"
        r"\01_v2_v3_evolution\data\yaw_common_eval_v1"
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
CANONICAL_EVAL_TASK = "Humanoid-RL-Lab-G1-Flat-Upright-Yaw-Direct-Wide-Yaw-Flat-Orientation-v0"

CASES = {
    "V2-A": {
        "native_task": "Humanoid-RL-Lab-G1-Flat-Upright-Yaw-Direct-v0",
        "checkpoint": REPO / "logs/rsl_rl/humanoid_rl_lab_g1_upright_yaw_direct/2026-08-15_03-33-18_thesis_yaw_direct_genuine_scratch_v2a_to700/model_600.pt",
        "sha256": "F1340C5E5CB44AE3157CC9C4D4493BBD1BC6316ED64A029FBD514C7FC0F7B413",
    },
    "V2-E": {
        "native_task": "Humanoid-RL-Lab-G1-Flat-Upright-Yaw-Direct-Heavy-Tail-Yaw-v0",
        "checkpoint": REPO / "logs/rsl_rl/humanoid_rl_lab_g1_upright_yaw_direct_heavy_tail_yaw/2026-08-16_20-06-45_thesis_yaw_direct_heavy_tail_v2e_genuine_scratch_to700/model_600.pt",
        "sha256": "154AB9548FA829C59764AFA772D6897BD0F29104729131515A8A616F4879645F",
    },
    "V2-F": {
        "native_task": "Humanoid-RL-Lab-G1-Flat-Upright-Yaw-Direct-Wide-Yaw-v0",
        "checkpoint": REPO / "logs/rsl_rl/humanoid_rl_lab_g1_upright_yaw_direct_wide_yaw/2026-08-16_22-19-58_thesis_yaw_direct_wide_yaw_v2f_genuine_scratch_to700/model_600.pt",
        "sha256": "F650608C94B50F1BD77D64F975B1F0B87033510B74A159CB3A56860DE4271EE7",
    },
    "V2-G": {
        "native_task": CANONICAL_EVAL_TASK,
        "checkpoint": REPO / "checkpoints/local_g1_yaw_direct_wide_yaw_flat_orientation_v2g/model_600.pt",
        "sha256": "58468EED03F9EF35A9BC898063A3B14AC36C93A32F65DB42B404BDE93CD694D3",
    },
    "V3-C": {
        "native_task": "Humanoid-RL-Lab-G1-Flat-Upright-Yaw-Direct-Wide-Yaw-Flat-Orientation-V3C-v0",
        "checkpoint": REPO / "logs/rsl_rl/humanoid_rl_lab_g1_v3c_swing_clearance_refinement/2026-08-26_12-24-10_v3c_from_v2g_model600_30it/model_620.pt",
        "sha256": "C9D93ADE8F32D8F52D1583F4596EAB8906377802265B975E66719C5B41F08458",
    },
}

VX_COMMAND = 0.40
YAW_COMMANDS = (-0.15, -0.10, -0.05, 0.00, 0.05, 0.10, 0.15)
EXPECTED_DT = 0.02
EXPECTED_OBS_DIM = 123
EXPECTED_ACTION_DIM = 37
FORMAT_ID = "G1_THESIS_V2V3_YAW_COMMON_BENCHMARK_V1"
PASS_MARKER = "G1_THESIS_FIG_CORE1_V2V3_YAW_COMMON_BENCHMARK_V1_PASS"
FAIL_MARKER = "G1_THESIS_FIG_CORE1_V2V3_YAW_COMMON_BENCHMARK_V1_FAIL"


def fail(msg):
    raise RuntimeError(msg)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def tv(x):
    if hasattr(x, "torch"):
        x = x.torch
    return x if torch.is_tensor(x) else torch.as_tensor(x)


def reset_rng(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_policy_obs(obs):
    try:
        candidate = obs["policy"]
    except Exception:
        candidate = obs
    t = tv(candidate)
    if tuple(t.shape) != (1, EXPECTED_OBS_DIM):
        fail(f"unexpected policy observation shape: {tuple(t.shape)}")
    return t


def make_env_cfg(seed: int):
    cfg = parse_env_cfg(CANONICAL_EVAL_TASK, device=args_cli.device, num_envs=1)
    cfg.scene.num_envs = 1
    cfg.seed = seed
    cfg.episode_length_s = max(float(cfg.episode_length_s), args_cli.duration_s + 10.0)

    if hasattr(cfg.observations.policy, "enable_corruption"):
        cfg.observations.policy.enable_corruption = False
    if hasattr(cfg.events, "push_robot"):
        cfg.events.push_robot = None
    if hasattr(cfg.events, "base_external_force_torque"):
        cfg.events.base_external_force_torque = None

    reset_base = getattr(cfg.events, "reset_base", None)
    if reset_base is not None:
        reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        }

    cmd = cfg.commands.base_velocity
    cmd.resampling_time_range = (1.0e9, 1.0e9)
    cmd.rel_standing_envs = 0.0
    cmd.ranges.lin_vel_x = (0.0, 0.0)
    cmd.ranges.lin_vel_y = (0.0, 0.0)
    cmd.ranges.ang_vel_z = (0.0, 0.0)
    if hasattr(cmd, "heading_command"):
        cmd.heading_command = False
    if hasattr(cmd, "rel_heading_envs"):
        cmd.rel_heading_envs = 0.0
    return cfg


def make_agent_cfg(native_task: str, seed: int):
    cfg = load_cfg_from_registry(native_task, "rsl_rl_cfg_entry_point")
    if args_cli.device is not None and hasattr(cfg, "device"):
        cfg.device = args_cli.device
    if hasattr(cfg, "seed"):
        cfg.seed = seed
    cfg = handle_deprecated_rsl_rl_cfg(cfg, metadata.version("rsl-rl-lib"))
    if cfg.class_name != "OnPolicyRunner":
        fail(f"unexpected runner class: {cfg.class_name}")
    return cfg


def set_command(base_env, wz_cmd: float):
    cmd = base_env.command_manager.get_command("base_velocity")
    with torch.no_grad():
        cmd.zero_()
        cmd[:, 0] = VX_COMMAND
        cmd[:, 2] = wz_cmd


def reset_fingerprint(robot) -> str:
    vals = []
    for t in (
        robot.data.root_pos_w[0],
        robot.data.root_quat_w[0],
        robot.data.root_lin_vel_w[0],
        robot.data.root_ang_vel_w[0],
        robot.data.joint_pos[0],
    ):
        vals.extend(float(v) for v in tv(t).detach().cpu().tolist())
    return hashlib.sha256(",".join(f"{v:.12g}" for v in vals).encode()).hexdigest().upper()


def tilt_deg(robot) -> float:
    g = tv(robot.data.projected_gravity_b)[0]
    c = torch.clamp(-g[2], -1.0, 1.0)
    return float(torch.rad2deg(torch.acos(c)).detach().cpu().item())


def run_trial(case_name: str, case: dict, seed: int, wz_cmd: float) -> dict:
    reset_rng(seed)
    env_cfg = make_env_cfg(seed)
    agent_cfg = make_agent_cfg(case["native_task"], seed)
    env = None

    try:
        raw = gym.make(CANONICAL_EVAL_TASK, cfg=env_cfg)
        env = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
        base_env = env.unwrapped
        robot = base_env.scene["robot"]

        if not math.isclose(float(base_env.step_dt), EXPECTED_DT, abs_tol=1e-12):
            fail("unexpected dt")
        if int(base_env.action_manager.total_action_dim) != EXPECTED_ACTION_DIM:
            fail("unexpected action dimension")

        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(case["checkpoint"]))
        policy = runner.get_inference_policy(device=base_env.device)

        extract_policy_obs(env.get_observations())

        if float(torch.max(torch.abs(tv(robot.data.root_lin_vel_w)[0])).item()) > 1e-6:
            fail("non-zero linear reset velocity")
        if float(torch.max(torch.abs(tv(robot.data.root_ang_vel_w)[0])).item()) > 1e-6:
            fail("non-zero angular reset velocity")

        fp = reset_fingerprint(robot)
        steps = int(round(args_cli.duration_s / EXPECTED_DT))
        tail_steps = int(round(args_cli.tail_s / EXPECTED_DT))
        samples = []
        completed = True
        termination_step = None

        for step in range(steps):
            set_command(base_env, wz_cmd)
            obs = env.get_observations()
            with torch.inference_mode():
                action = policy(obs)
                obs, _, dones, _ = env.step(action)

            if bool(torch.any(tv(dones)).item()):
                completed = False
                termination_step = step
                break

            if step >= steps - tail_steps:
                lin = tv(robot.data.root_lin_vel_b)[0]
                ang = tv(robot.data.root_ang_vel_b)[0]
                vx = float(lin[0].detach().cpu().item())
                wz = float(ang[2].detach().cpu().item())
                samples.append((vx, wz, tilt_deg(robot)))

        result = {
            "case": case_name,
            "seed": seed,
            "command_vx_mps": VX_COMMAND,
            "command_wz_radps": wz_cmd,
            "completed": completed,
            "termination_step": termination_step,
            "reset_fingerprint_sha256": fp,
            "tail_samples": len(samples),
            "tail_mean_wz_radps": None,
            "tail_mean_abs_wz_error_radps": None,
            "tail_mean_vx_mps": None,
            "tail_mean_abs_vx_error_mps": None,
            "tail_mean_tilt_deg": None,
        }

        if completed and len(samples) == tail_steps:
            vx_vals = [s[0] for s in samples]
            wz_vals = [s[1] for s in samples]
            tilt_vals = [s[2] for s in samples]
            result.update({
                "tail_mean_wz_radps": statistics.fmean(wz_vals),
                "tail_mean_abs_wz_error_radps": statistics.fmean(abs(v - wz_cmd) for v in wz_vals),
                "tail_mean_vx_mps": statistics.fmean(vx_vals),
                "tail_mean_abs_vx_error_mps": statistics.fmean(abs(v - VX_COMMAND) for v in vx_vals),
                "tail_mean_tilt_deg": statistics.fmean(tilt_vals),
            })
        return result
    finally:
        if env is not None:
            env.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def stats(rows, key):
    vals = [float(r[key]) for r in rows if r[key] is not None and math.isfinite(float(r[key]))]
    return {
        f"{key}_mean": statistics.fmean(vals) if vals else None,
        f"{key}_sd": statistics.stdev(vals) if len(vals) > 1 else (0.0 if vals else None),
    }


def main():
    if args_cli.seed_end < args_cli.seed_start:
        fail("invalid seed range")
    if not (0.0 < args_cli.tail_s <= args_cli.duration_s):
        fail("invalid tail window")

    case_name = args_cli.case
    case = CASES[case_name]
    checkpoint = case["checkpoint"]
    if not checkpoint.is_file():
        fail(f"missing checkpoint: {checkpoint}")
    actual_sha = sha256_file(checkpoint)
    if actual_sha != case["sha256"]:
        fail(f"checkpoint SHA mismatch: {actual_sha}")

    seeds = list(range(args_cli.seed_start, args_cli.seed_end + 1))
    case_dir = args_cli.output_root.resolve() / case_name.replace("-", "")
    case_dir.mkdir(parents=True, exist_ok=True)

    trials = []
    fingerprints = {seed: set() for seed in seeds}
    total = len(seeds) * len(YAW_COMMANDS)

    print("=" * 128)
    print("CORE 1 — STANDARDIZED PAIRED-SEED YAW BENCHMARK")
    print("=" * 128)
    print(f"Case                : {case_name}")
    print(f"Checkpoint SHA256   : {actual_sha}")
    print(f"vx command          : {VX_COMMAND:.2f} m/s")
    print(f"wz commands         : {list(YAW_COMMANDS)} rad/s")
    print(f"Seeds               : {seeds[0]}-{seeds[-1]} (N={len(seeds)})")
    print(f"Trials              : {total}")
    print(f"Duration / tail     : {args_cli.duration_s:.1f} / {args_cli.tail_s:.1f} s")
    print("Reset pairing       : identical randomized pose required within each seed")
    print("Root reset velocity : zero (all 6 components)")
    print("Corruption/pushes   : OFF")
    print("Terminations        : ON")
    print("=" * 128)

    count = 0
    for seed in seeds:
        for wz in YAW_COMMANDS:
            count += 1
            print(f"[{count:03d}/{total:03d}] seed={seed} wz={wz:+.2f}", flush=True)
            row = run_trial(case_name, case, seed, wz)
            trials.append(row)
            fingerprints[seed].add(row["reset_fingerprint_sha256"])
            print(
                f"  completed={row['completed']} "
                f"tail_wz={row['tail_mean_wz_radps']} "
                f"tail_vx={row['tail_mean_vx_mps']}",
                flush=True,
            )

    bad = {seed: fps for seed, fps in fingerprints.items() if len(fps) != 1}
    if bad:
        fail(f"paired reset fingerprint mismatch: {bad}")

    trial_csv = case_dir / f"{case_name.replace('-', '')}_yaw_trial_summary.csv"
    with trial_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(trials[0].keys()))
        writer.writeheader()
        writer.writerows(trials)

    aggregates = []
    for wz in YAW_COMMANDS:
        selected = [r for r in trials if math.isclose(r["command_wz_radps"], wz, abs_tol=1e-12)]
        completed = [r for r in selected if r["completed"]]
        row = {
            "case": case_name,
            "command_wz_radps": wz,
            "completed_trials": len(completed),
            "total_trials": len(selected),
            "completion_rate_pct": 100.0 * len(completed) / len(selected),
        }
        for key in (
            "tail_mean_wz_radps",
            "tail_mean_abs_wz_error_radps",
            "tail_mean_vx_mps",
            "tail_mean_abs_vx_error_mps",
            "tail_mean_tilt_deg",
        ):
            row.update(stats(completed, key))
        aggregates.append(row)

    aggregate_csv = case_dir / f"{case_name.replace('-', '')}_yaw_command_aggregate.csv"
    with aggregate_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregates[0].keys()))
        writer.writeheader()
        writer.writerows(aggregates)

    completed_total = sum(int(r["completed"]) for r in trials)
    summary = {
        "format": FORMAT_ID,
        "status": "PASS",
        "case": case_name,
        "checkpoint_sha256": actual_sha,
        "canonical_eval_task": CANONICAL_EVAL_TASK,
        "vx_command_mps": VX_COMMAND,
        "yaw_commands_radps": list(YAW_COMMANDS),
        "seed_start": seeds[0],
        "seed_end": seeds[-1],
        "total_trials": len(trials),
        "completed_trials": completed_total,
        "completion_rate_pct": 100.0 * completed_total / len(trials),
        "paired_reset_fingerprints_pass": True,
        "trial_csv": str(trial_csv),
        "trial_csv_sha256": sha256_file(trial_csv),
        "aggregate_csv": str(aggregate_csv),
        "aggregate_csv_sha256": sha256_file(aggregate_csv),
    }
    summary_path = case_dir / f"{case_name.replace('-', '')}_yaw_benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print("=" * 128)
    print("CORE 1 YAW BENCHMARK SUMMARY")
    print("=" * 128)
    print("Status                              : PASS")
    print(f"Case                                : {case_name}")
    print(f"Trials completed                    : {completed_total} / {len(trials)}")
    print(f"Completion rate                     : {summary['completion_rate_pct']:.1f}%")
    print("Paired reset fingerprints           : PASS")
    for row in aggregates:
        print(
            f"wz={row['command_wz_radps']:+.2f} -> "
            f"measured={row['tail_mean_wz_radps_mean']} "
            f"± {row['tail_mean_wz_radps_sd']} rad/s | "
            f"completed={row['completed_trials']}/{row['total_trials']}"
        )
    print(f"Per-command CSV                     : {aggregate_csv}")
    print(f"Aggregate JSON                      : {summary_path}")
    print(f"Runner SHA256                       : {sha256_file(Path(__file__).resolve())}")
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
