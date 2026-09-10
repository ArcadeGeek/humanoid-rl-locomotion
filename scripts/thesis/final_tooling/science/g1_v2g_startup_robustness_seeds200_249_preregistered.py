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
        "Preregistered v2-G startup-robustness failure-rate experiment. "
        "Runs frozen v2-G standalone over a fixed new seed block under the ORIGINAL "
        "randomized root-velocity reset distribution. No v4-N, router, handoff, "
        "training, blending, or post-hoc seed selection."
    )
)
parser.add_argument("--seed_start", type=int, default=200)
parser.add_argument("--seed_count", type=int, default=50)
parser.add_argument("--startup_window_s", type=float, default=3.0)

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
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import parse_env_cfg
from rsl_rl.runners import OnPolicyRunner


TASK = "Humanoid-RL-Lab-G1-Flat-Upright-Yaw-Direct-Wide-Yaw-Flat-Orientation-v0"

CHECKPOINT = Path(
    r"C:\humanoid_rl_lab\checkpoints"
    r"\local_g1_yaw_direct_wide_yaw_flat_orientation_v2g\model_600.pt"
)
CHECKPOINT_SHA256 = "58468EED03F9EF35A9BC898063A3B14AC36C93A32F65DB42B404BDE93CD694D3"

EXPECTED_DT = 0.02
EXPECTED_OBS_DIM = 123
EXPECTED_ACTION_DIM = 37

# Preregistered experiment design.
EXPECTED_SEED_START = 200
EXPECTED_SEED_COUNT = 50
EXPECTED_STARTUP_WINDOW_S = 3.0
EXPECTED_STEPS = 150

ORIGINAL_ROOT_VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.5, 0.5),
    "roll": (-0.5, 0.5),
    "pitch": (-0.5, 0.5),
    "yaw": (-0.5, 0.5),
}

PASS_MARKER = "V2G_STARTUP_ROBUSTNESS_SEEDS200_249_EXPERIMENT_PASS"
FAIL_MARKER = "V2G_STARTUP_ROBUSTNESS_SEEDS200_249_EXPERIMENT_FAIL"


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
    candidate = None
    if torch.is_tensor(obs) or hasattr(obs, "torch"):
        candidate = obs
    else:
        try:
            candidate = obs["policy"]
        except Exception:
            get_fn = getattr(obs, "get", None)
            if callable(get_fn):
                candidate = get_fn("policy")

    if candidate is None:
        fail(f"cannot resolve policy observation from {type(obs).__name__}")

    tensor = tv(candidate)
    if tuple(tensor.shape) != (1, EXPECTED_OBS_DIM):
        fail(f"unexpected policy observation shape: {tuple(tensor.shape)}")
    return tensor


def set_zero_command(base_env: Any) -> None:
    cmd = base_env.command_manager.get_command("base_velocity")
    with torch.no_grad():
        cmd[:, 0] = 0.0
        cmd[:, 1] = 0.0
        cmd[:, 2] = 0.0


def reset_global_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tilt_deg(robot: Any) -> float:
    g = tv(robot.data.projected_gravity_b)[0]
    c = torch.clamp(-g[2], -1.0, 1.0)
    return float(torch.rad2deg(torch.acos(c)).detach().cpu().item())


def wilson_interval_95(k: int, n: int) -> tuple[float, float]:
    if n <= 0:
        fail("Wilson interval requires n > 0")
    z = 1.959963984540054
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    half = (
        z
        * math.sqrt(
            phat * (1.0 - phat) / n
            + z * z / (4.0 * n * n)
        )
        / denom
    )
    return max(0.0, center - half), min(1.0, center + half)


def make_env_cfg(seed: int):
    cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=1)
    cfg.scene.num_envs = 1
    cfg.seed = int(seed)

    if args_cli.device is not None:
        cfg.sim.device = args_cli.device

    # Prevent time-out contamination inside the preregistered 3 s startup window.
    cfg.episode_length_s = max(
        float(cfg.episode_length_s),
        float(args_cli.startup_window_s + 10.0),
    )

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

    actual_range = dict(cfg.events.reset_base.params["velocity_range"])
    if actual_range != ORIGINAL_ROOT_VELOCITY_RANGE:
        fail(
            f"original root velocity reset range changed for seed {seed}: "
            f"{actual_range} != {ORIGINAL_ROOT_VELOCITY_RANGE}"
        )

    # IMPORTANT: do NOT modify reset_base velocity_range.
    return cfg


def make_agent_cfg():
    cfg = UprightYawDirectWideYawFlatOrientationPPORunnerCfg()
    if args_cli.device is not None:
        cfg.device = args_cli.device
    return handle_deprecated_rsl_rl_cfg(cfg, metadata.version("rsl-rl-lib"))


def run_seed(seed: int) -> dict[str, Any]:
    reset_global_rng(seed)
    env_cfg = make_env_cfg(seed)
    agent_cfg = make_agent_cfg()

    env = None
    try:
        raw_env = gym.make(TASK, cfg=env_cfg)
        env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
        base_env = env.unwrapped
        robot = base_env.scene["robot"]

        dt = float(base_env.step_dt)
        if not math.isclose(dt, EXPECTED_DT, rel_tol=0.0, abs_tol=1e-12):
            fail(f"seed {seed}: step_dt={dt}, expected {EXPECTED_DT}")

        if int(base_env.action_manager.total_action_dim) != EXPECTED_ACTION_DIM:
            fail(
                f"seed {seed}: action_dim={base_env.action_manager.total_action_dim}, "
                f"expected {EXPECTED_ACTION_DIM}"
            )

        runner = OnPolicyRunner(
            env,
            agent_cfg.to_dict(),
            log_dir=None,
            device=agent_cfg.device,
        )
        runner.load(str(CHECKPOINT))
        policy = runner.get_inference_policy(device=base_env.device)

        obs = env.get_observations()
        extract_policy_obs(obs)

        set_zero_command(base_env)

        root_lin_w = tv(robot.data.root_lin_vel_w)[0].detach().cpu()
        root_ang_w = tv(robot.data.root_ang_vel_w)[0].detach().cpu()

        init = {
            "init_vx_w": float(root_lin_w[0].item()),
            "init_vy_w": float(root_lin_w[1].item()),
            "init_vz_w": float(root_lin_w[2].item()),
            "init_wx_w": float(root_ang_w[0].item()),
            "init_wy_w": float(root_ang_w[1].item()),
            "init_wz_w": float(root_ang_w[2].item()),
        }

        for key, value in init.items():
            if not (-0.500001 <= value <= 0.500001):
                fail(f"seed {seed}: {key} outside original reset support: {value}")

        terminated = False
        termination_step = -1
        termination_time_s = float("nan")

        # Store the last known state BEFORE a done-triggering step, because the
        # manager-based env auto-resets terminated envs before returning.
        prev_state = {
            "preterm_z_m": float(tv(robot.data.root_pos_w)[0, 2].detach().cpu().item()),
            "preterm_vxy_mps": math.hypot(
                float(tv(robot.data.root_lin_vel_b)[0, 0].detach().cpu().item()),
                float(tv(robot.data.root_lin_vel_b)[0, 1].detach().cpu().item()),
            ),
            "preterm_tilt_deg": tilt_deg(robot),
            "preterm_abs_wz_radps": abs(
                float(tv(robot.data.root_ang_vel_b)[0, 2].detach().cpu().item())
            ),
        }

        steps = int(round(args_cli.startup_window_s / dt))
        if steps != EXPECTED_STEPS:
            fail(f"seed {seed}: expected {EXPECTED_STEPS} steps, got {steps}")

        for step in range(steps):
            set_zero_command(base_env)

            obs = env.get_observations()
            with torch.inference_mode():
                action = policy(obs)

            if not torch.isfinite(action).all():
                fail(f"seed {seed}: non-finite action at step {step}")

            with torch.inference_mode():
                obs, _, dones, _ = env.step(action)

            done = bool(torch.any(dones).item())
            if done:
                terminated = True
                termination_step = step
                termination_time_s = (step + 1) * dt
                break

            prev_state = {
                "preterm_z_m": float(
                    tv(robot.data.root_pos_w)[0, 2].detach().cpu().item()
                ),
                "preterm_vxy_mps": math.hypot(
                    float(tv(robot.data.root_lin_vel_b)[0, 0].detach().cpu().item()),
                    float(tv(robot.data.root_lin_vel_b)[0, 1].detach().cpu().item()),
                ),
                "preterm_tilt_deg": tilt_deg(robot),
                "preterm_abs_wz_radps": abs(
                    float(tv(robot.data.root_ang_vel_b)[0, 2].detach().cpu().item())
                ),
            }

        return {
            "seed": seed,
            **init,
            "failure_within_3s": int(terminated),
            "termination_step": termination_step,
            "termination_time_s": termination_time_s,
            **prev_state,
        }

    finally:
        if env is not None:
            env.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    if args_cli.seed_start != EXPECTED_SEED_START:
        fail(
            f"preregistered seed_start is {EXPECTED_SEED_START}; "
            f"got {args_cli.seed_start}. Do not alter after design freeze."
        )
    if args_cli.seed_count != EXPECTED_SEED_COUNT:
        fail(
            f"preregistered seed_count is {EXPECTED_SEED_COUNT}; "
            f"got {args_cli.seed_count}. Do not stop early or extend after outcomes."
        )
    if not math.isclose(
        args_cli.startup_window_s,
        EXPECTED_STARTUP_WINDOW_S,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        fail(
            f"preregistered startup window is {EXPECTED_STARTUP_WINDOW_S:.1f}s; "
            f"got {args_cli.startup_window_s}"
        )

    if not CHECKPOINT.is_file():
        fail(f"checkpoint missing: {CHECKPOINT}")
    actual_sha = sha256(CHECKPOINT)
    if actual_sha != CHECKPOINT_SHA256:
        fail(f"checkpoint SHA mismatch: {actual_sha}")

    seeds = tuple(
        range(
            args_cli.seed_start,
            args_cli.seed_start + args_cli.seed_count,
        )
    )
    if seeds != tuple(range(200, 250)):
        fail(f"preregistered seed block mismatch: {seeds[0]}..{seeds[-1]}")

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_dir = (
        Path(r"C:\humanoid_rl_lab\logs\thesis\startup_robustness")
        / f"{stamp}_v2g_original_reset_seeds200_249"
    )
    result_dir.mkdir(parents=True, exist_ok=False)

    print("=" * 136)
    print("v2-G STARTUP ROBUSTNESS — PREREGISTERED FAILURE-RATE EXPERIMENT")
    print("=" * 136)
    print("policy                   : frozen v2-G ONLY")
    print(f"checkpoint SHA256        : {actual_sha}")
    print("v4-N / router / handoff  : ABSENT")
    print("command                  : [0,0,0]")
    print("reset velocity range     : ORIGINAL ±0.5 on all 6 root velocity components")
    print("push / external force    : ABSENT")
    print("T0 observation lifecycle : UNCHANGED training/runtime stale-cache contract")
    print(f"seed block               : {seeds[0]}-{seeds[-1]} ({len(seeds)} seeds) [FROZEN]")
    print(f"startup window           : {args_cli.startup_window_s:.2f} s = {EXPECTED_STEPS} steps [FROZEN]")
    print("primary endpoint         : any env termination within 3.0 s")
    print("early stopping           : FORBIDDEN")
    print("post-hoc seed exclusion  : FORBIDDEN")
    print("holdout/router results   : NOT USED TO CHANGE THIS DESIGN")
    print(f"result_dir               : {result_dir}")
    print("=" * 136)

    rows: list[dict[str, Any]] = []

    for index, seed in enumerate(seeds, start=1):
        row = run_seed(seed)
        rows.append(row)

        status = "FAIL" if row["failure_within_3s"] else "PASS"
        ttxt = (
            f"{row['termination_time_s']:.3f}s"
            if row["failure_within_3s"]
            else "n/a"
        )
        print(
            f"[{index:02d}/{len(seeds)}] seed={seed} startup={status} "
            f"term_t={ttxt} "
            f"init_lin=({row['init_vx_w']:+.4f},{row['init_vy_w']:+.4f},{row['init_vz_w']:+.4f}) "
            f"init_ang=({row['init_wx_w']:+.4f},{row['init_wy_w']:+.4f},{row['init_wz_w']:+.4f})",
            flush=True,
        )

    csv_path = result_dir / "startup_robustness_seeds200_249.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    failures = [r for r in rows if int(r["failure_within_3s"]) == 1]
    successes = [r for r in rows if int(r["failure_within_3s"]) == 0]

    k = len(failures)
    n = len(rows)
    rate = k / n
    lo, hi = wilson_interval_95(k, n)

    report_lines = [
        "=" * 136,
        "v2-G STARTUP ROBUSTNESS — PREREGISTERED RESULT",
        "=" * 136,
        f"seed block               : {seeds[0]}-{seeds[-1]}",
        f"n                        : {n}",
        f"startup window           : {args_cli.startup_window_s:.2f} s",
        "reset velocity range     : ORIGINAL ±0.5 all 6 root velocity components",
        "command                  : [0,0,0]",
        "policy                   : frozen v2-G only",
        f"checkpoint SHA256        : {actual_sha}",
        "",
        "[PRIMARY ENDPOINT]",
        f"failures within 3.0 s    : {k}/{n}",
        f"failure rate             : {100.0 * rate:.3f}%",
        f"Wilson 95% CI            : [{100.0 * lo:.3f}%, {100.0 * hi:.3f}%]",
        f"successes                : {len(successes)}/{n}",
        "",
        "[FAILURE SEEDS]",
        (
            ",".join(str(r["seed"]) for r in failures)
            if failures
            else "NONE"
        ),
        "",
        "[INTERPRETATION GUARDRAIL]",
        "This experiment quantifies startup failure frequency only.",
        "It does NOT modify the already-closed router/handoff architecture.",
        "It does NOT by itself identify a universal causal threshold in root-velocity space.",
        "Any later startup intervention must be motivated by this preregistered failure-rate result.",
        "",
        f"trajectory table         : {csv_path}",
        PASS_MARKER,
        "=" * 136,
    ]

    report_path = result_dir / "startup_robustness_report.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print()
    print("\n".join(report_lines), flush=True)
    print(f"\nreport                   : {report_path}", flush=True)


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
