from __future__ import annotations

import argparse
import csv
import gc
import hashlib
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
        "Paired low-speed competence probe at vx=0.10 m/s for frozen v2-G and frozen v4-N. "
        "No router, no handoff, no threshold acceptance decision. Same seed and zero-root-velocity "
        "startup conditioning are used for both policy trials."
    )
)
parser.add_argument("--seed", type=int, default=104)
parser.add_argument("--duration_s", type=float, default=6.0)

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import humanoid_rl_lab.tasks  # noqa: F401
from humanoid_rl_lab.tasks.manager_based.humanoid_rl_lab.agents.upright_yaw_direct_wide_yaw_flat_orientation_ppo_cfg import (
    UprightYawDirectWideYawFlatOrientationPPORunnerCfg,
)
from humanoid_rl_lab.tasks.manager_based.humanoid_rl_lab.agents.v4n_ppo_cfg import (
    V4NSymmetryAugmentationPPORunnerCfg,
)
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import parse_env_cfg
from rsl_rl.runners import OnPolicyRunner


TASK = "Humanoid-RL-Lab-G1-Flat-Upright-Yaw-Direct-Wide-Yaw-Flat-Orientation-v0"

V2G_CHECKPOINT = Path(
    r"C:\humanoid_rl_lab\checkpoints"
    r"\local_g1_yaw_direct_wide_yaw_flat_orientation_v2g\model_600.pt"
)
V2G_SHA256 = "58468EED03F9EF35A9BC898063A3B14AC36C93A32F65DB42B404BDE93CD694D3"

V4N_CHECKPOINT = Path(
    r"C:\humanoid_rl_lab\logs\rsl_rl\humanoid_rl_lab_g1_v4n_symmetry_augmentation_scratch"
    r"\2026-09-04_13-58-13_v4n_genuine_scratch_seed42_official1500\model_1499.pt"
)
V4N_SHA256 = "367D318AB4940A3AAE8F0FA568384212762258067DFC4CF9F624FA472BEB9342"

COMMAND_VX = 0.10
EXPECTED_DT = 0.02
EXPECTED_OBS_DIM = 123
EXPECTED_ACTION_DIM = 37
CONTACT_FORCE_THRESHOLD_N = 20.0

PASS_MARKER = "V2G_V4N_VX010_PAIRED_COMPETENCE_PROBE_PASS"
FAIL_MARKER = "V2G_V4N_VX010_PAIRED_COMPETENCE_PROBE_FAIL"


def fail(msg: str) -> None:
    raise RuntimeError(msg)


def sha256(path: Path) -> str:
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
                candidate = get_fn("policy")
    if candidate is None:
        fail(f"cannot resolve policy observation from {type(obs).__name__}")
    tensor = tv(candidate)
    if tensor.ndim != 2 or tensor.shape != (1, EXPECTED_OBS_DIM):
        fail(f"unexpected policy observation shape: {tuple(tensor.shape)}")
    return tensor


def tilt_deg(robot: Any) -> float:
    g = tv(robot.data.projected_gravity_b)[0]
    c = torch.clamp(-g[2], -1.0, 1.0)
    return float(torch.rad2deg(torch.acos(c)).detach().cpu().item())


def action_rms(action: torch.Tensor) -> float:
    a = action.detach().float()
    return float(torch.sqrt(torch.mean(a * a)).cpu().item())


def set_command(base_env: Any, vx: float) -> None:
    cmd = base_env.command_manager.get_command("base_velocity")
    with torch.no_grad():
        cmd[:, 0] = float(vx)
        cmd[:, 1] = 0.0
        cmd[:, 2] = 0.0


def reset_global_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env_cfg(seed: int):
    cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=1)
    cfg.scene.num_envs = 1
    cfg.seed = int(seed)
    if args_cli.device is not None:
        cfg.sim.device = args_cli.device

    cfg.episode_length_s = max(float(cfg.episode_length_s), float(args_cli.duration_s + 10.0))

    if hasattr(cfg.observations.policy, "enable_corruption"):
        cfg.observations.policy.enable_corruption = False

    if hasattr(cfg, "events"):
        if hasattr(cfg.events, "base_external_force_torque"):
            cfg.events.base_external_force_torque = None
        if hasattr(cfg.events, "push_robot"):
            cfg.events.push_robot = None

    cfg.commands.base_velocity.heading_command = False
    if hasattr(cfg.commands.base_velocity, "rel_heading_envs"):
        cfg.commands.base_velocity.rel_heading_envs = 0.0

    # Same engineering-only startup conditioning used in handoff/router development.
    cfg.events.reset_base.params["velocity_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    return cfg


def make_agent_cfg(policy_name: str):
    if policy_name == "v2-G":
        cfg = UprightYawDirectWideYawFlatOrientationPPORunnerCfg()
    elif policy_name == "v4-N":
        cfg = V4NSymmetryAugmentationPPORunnerCfg()
    else:
        fail(f"unknown policy_name={policy_name}")

    if args_cli.device is not None:
        cfg.device = args_cli.device
    return handle_deprecated_rsl_rl_cfg(cfg, metadata.version("rsl-rl-lib"))


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        fail("empty trajectory")

    # Whole trial plus a terminal 3 s window to separate startup transient from sustained behavior.
    tail_n = min(len(rows), int(round(3.0 / EXPECTED_DT)))
    tail = rows[-tail_n:]

    def mean(key: str, data: list[dict[str, Any]]) -> float:
        return statistics.fmean(float(r[key]) for r in data)

    def maxv(key: str, data: list[dict[str, Any]]) -> float:
        return max(float(r[key]) for r in data)

    return {
        "samples": float(len(rows)),
        "mean_vx_b_mps": mean("vx_b_mps", rows),
        "mean_abs_tracking_error_mps": mean("abs_tracking_error_mps", rows),
        "mean_vxy_mps": mean("vxy_mps", rows),
        "max_vxy_mps": maxv("vxy_mps", rows),
        "mean_tilt_deg": mean("tilt_deg", rows),
        "max_tilt_deg": maxv("tilt_deg", rows),
        "mean_abs_wz_radps": mean("abs_wz_radps", rows),
        "max_abs_wz_radps": maxv("abs_wz_radps", rows),
        "double_support_pct": (
            100.0 * mean("double_support", rows)
            if all(float(r["double_support"]) >= 0.0 for r in rows)
            else float("nan")
        ),
        "tail3_mean_vx_b_mps": mean("vx_b_mps", tail),
        "tail3_mean_abs_tracking_error_mps": mean("abs_tracking_error_mps", tail),
        "tail3_mean_vxy_mps": mean("vxy_mps", tail),
        "tail3_mean_tilt_deg": mean("tilt_deg", tail),
        "tail3_mean_abs_wz_radps": mean("abs_wz_radps", tail),
        "tail3_double_support_pct": (
            100.0 * mean("double_support", tail)
            if all(float(r["double_support"]) >= 0.0 for r in tail)
            else float("nan")
        ),
    }


def run_trial(
    policy_name: str,
    checkpoint: Path,
    expected_sha: str,
    seed: int,
    result_dir: Path,
) -> tuple[dict[str, float], bool, list[float]]:
    actual_sha = sha256(checkpoint)
    if actual_sha != expected_sha:
        fail(f"{policy_name} checkpoint SHA mismatch: {actual_sha}")

    reset_global_rng(seed)
    env_cfg = make_env_cfg(seed)
    agent_cfg = make_agent_cfg(policy_name)

    env = None
    rows: list[dict[str, Any]] = []
    reset_fingerprint: list[float] = []

    try:
        raw_env = gym.make(TASK, cfg=env_cfg)
        env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
        base_env = env.unwrapped
        robot = base_env.scene["robot"]

        dt = float(base_env.step_dt)
        if abs(dt - EXPECTED_DT) > 1e-12:
            fail(f"{policy_name}: unexpected dt={dt}")
        if int(base_env.action_manager.total_action_dim) != EXPECTED_ACTION_DIM:
            fail(f"{policy_name}: unexpected action dim")

        runner = OnPolicyRunner(
            env,
            agent_cfg.to_dict(),
            log_dir=None,
            device=agent_cfg.device,
        )
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=base_env.device)

        obs = env.get_observations()
        extract_policy_obs(obs)

        # Physical reset fingerprint before first control step.
        root_pos = tv(robot.data.root_pos_w)[0].detach().cpu()
        root_quat = tv(robot.data.root_quat_w)[0].detach().cpu()
        root_lin = tv(robot.data.root_lin_vel_w)[0].detach().cpu()
        root_ang = tv(robot.data.root_ang_vel_w)[0].detach().cpu()
        joint_pos = tv(robot.data.joint_pos)[0].detach().cpu()
        reset_fingerprint = [
            *[float(x) for x in root_pos.tolist()],
            *[float(x) for x in root_quat.tolist()],
            *[float(x) for x in root_lin.tolist()],
            *[float(x) for x in root_ang.tolist()],
            *[float(x) for x in joint_pos.tolist()],
        ]

        if max(abs(x) for x in root_lin.tolist()) > 1e-6:
            fail(f"{policy_name}: root linear reset velocity not zero")
        if max(abs(x) for x in root_ang.tolist()) > 1e-6:
            fail(f"{policy_name}: root angular reset velocity not zero")

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
        terminated = False

        for step in range(steps):
            set_command(base_env, COMMAND_VX)
            obs = env.get_observations()

            with torch.inference_mode():
                action = policy(obs)
            if not torch.isfinite(action).all():
                fail(f"{policy_name}: non-finite action step={step}")

            with torch.inference_mode():
                obs, _, dones, _ = env.step(action)

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
            done = int(bool(torch.any(dones).item()))

            rows.append(
                {
                    "policy": policy_name,
                    "seed": seed,
                    "step": step,
                    "time_s": (step + 1) * dt,
                    "command_vx_mps": COMMAND_VX,
                    "vx_b_mps": vx_b,
                    "vy_b_mps": vy_b,
                    "vxy_mps": math.hypot(vx_b, vy_b),
                    "abs_tracking_error_mps": abs(vx_b - COMMAND_VX),
                    "root_z_m": float(pos[2].detach().cpu().item()),
                    "tilt_deg": tilt_deg(robot),
                    "abs_wz_radps": abs(wz),
                    "action_rms": action_rms(action),
                    "double_support": double_support,
                    "done": done,
                }
            )

            if done:
                terminated = True
                break

        csv_path = result_dir / f"{policy_name.replace('-', '').lower()}_trajectory.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fobj:
            writer = csv.DictWriter(fobj, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        return summarize(rows), terminated, reset_fingerprint

    finally:
        if env is not None:
            env.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def print_summary(label: str, summary: dict[str, float], terminated: bool) -> None:
    print("-" * 124)
    print(label)
    print("-" * 124)
    print(f"termination                    : {'YES' if terminated else 'NO'}")
    for key, value in summary.items():
        if key == "samples":
            print(f"{key:<31}: {int(value)}")
        else:
            print(f"{key:<31}: {value:.9f}")


def main() -> None:
    if args_cli.duration_s < 3.0:
        fail("--duration_s must be >= 3.0")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_dir = (
        Path(r"C:\humanoid_rl_lab\logs\thesis\router")
        / f"{timestamp}_vx010_paired_competence_seed{args_cli.seed}"
    )
    result_dir.mkdir(parents=True, exist_ok=False)

    print("=" * 124)
    print("v2-G / v4-N PAIRED LOW-SPEED COMPETENCE PROBE")
    print("=" * 124)
    print(f"seed                         : {args_cli.seed}")
    print(f"command                      : vx={COMMAND_VX:.3f} m/s, vy=0, wz=0")
    print(f"duration per policy          : {args_cli.duration_s:.2f} s")
    print("router                       : ABSENT")
    print("handoff                      : ABSENT")
    print("threshold acceptance         : ABSENT")
    print("startup root velocities      : ZERO [engineering-only conditioning]")
    print("test role                    : characterize competence at the provisional EXIT boundary")
    print(f"result_dir                   : {result_dir}")
    print("=" * 124)

    v2_summary, v2_term, v2_fp = run_trial(
        "v2-G",
        V2G_CHECKPOINT,
        V2G_SHA256,
        int(args_cli.seed),
        result_dir,
    )
    v4_summary, v4_term, v4_fp = run_trial(
        "v4-N",
        V4N_CHECKPOINT,
        V4N_SHA256,
        int(args_cli.seed),
        result_dir,
    )

    if len(v2_fp) != len(v4_fp):
        fail("reset fingerprint length mismatch")
    reset_max_delta = max(abs(a - b) for a, b in zip(v2_fp, v4_fp))
    if reset_max_delta > 1e-6:
        fail(
            f"paired reset fingerprint mismatch: max delta={reset_max_delta:.9e}; "
            "do not interpret policy comparison"
        )

    print_summary("FROZEN v2-G @ vx=0.10", v2_summary, v2_term)
    print_summary("FROZEN v4-N @ vx=0.10", v4_summary, v4_term)

    print("-" * 124)
    print("PAIRED DIAGNOSTIC")
    print("-" * 124)
    print(f"reset fingerprint max delta     : {reset_max_delta:.9e}")
    print(
        f"tail3 tracking error v2-G/v4-N  : "
        f"{v2_summary['tail3_mean_abs_tracking_error_mps']:.9f} / "
        f"{v4_summary['tail3_mean_abs_tracking_error_mps']:.9f} m/s"
    )
    print(
        f"tail3 tilt v2-G/v4-N            : "
        f"{v2_summary['tail3_mean_tilt_deg']:.9f} / "
        f"{v4_summary['tail3_mean_tilt_deg']:.9f} deg"
    )
    print(
        f"tail3 DS v2-G/v4-N              : "
        f"{v2_summary['tail3_double_support_pct']:.9f} / "
        f"{v4_summary['tail3_double_support_pct']:.9f} %"
    )
    print("")
    print(
        "INTERPRETATION RULE            : this probe measures whether vx=0.10 is an appropriate lower-overlap / EXIT-boundary region. "
        "Do not select ENTER/EXIT from one seed or from a single metric."
    )

    summary_path = result_dir / "summary.txt"
    lines = [
        "v2-G / v4-N PAIRED LOW-SPEED COMPETENCE PROBE",
        f"seed={args_cli.seed}",
        f"command_vx={COMMAND_VX}",
        f"duration_s={args_cli.duration_s}",
        "router=ABSENT",
        "handoff=ABSENT",
        "threshold_acceptance=ABSENT",
        "startup_root_velocity_test=ZERO_ALL_6_COMPONENTS",
        f"reset_fingerprint_max_delta={reset_max_delta}",
        f"v2g_termination={'YES' if v2_term else 'NO'}",
        f"v4n_termination={'YES' if v4_term else 'NO'}",
    ]
    for prefix, summary in (("v2g", v2_summary), ("v4n", v4_summary)):
        for key, value in summary.items():
            lines.append(f"{prefix}_{key}={value}")
    lines.extend(
        [
            "scientific_role=OVERLAP_CHARACTERIZATION_ONLY",
            "threshold_status=PROVISIONAL_NOT_CALIBRATED",
            PASS_MARKER,
        ]
    )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"summary                       : {summary_path}")
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
