from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import math
import traceback
from datetime import datetime
from importlib import metadata
from pathlib import Path

parser = argparse.ArgumentParser(
    description=(
        "Causal isolation test for frozen v2-G seed-227 startup failure. "
        "Single experimental delta: set all six reset root-velocity ranges to zero."
    )
)
parser.add_argument("--seed", type=int, default=227)
parser.add_argument("--duration_s", type=float, default=3.0)

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
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
EXPECTED_SHA256 = "58468EED03F9EF35A9BC898063A3B14AC36C93A32F65DB42B404BDE93CD694D3"
DT = 0.02

RESCUE_MARKER = "V2G_SEED227_ZERO_ROOT_VELOCITY_RESCUE_PASS"
PERSISTS_MARKER = "V2G_SEED227_ZERO_ROOT_VELOCITY_FAILURE_PERSISTS"
PASS_MARKER = "V2G_SEED227_ZERO_ROOT_VELOCITY_CAUSAL_TEST_PASS"
FAIL_MARKER = "V2G_SEED227_ZERO_ROOT_VELOCITY_CAUSAL_TEST_RUNTIME_FAIL"


def fail(msg: str) -> None:
    raise RuntimeError(msg)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def tt(value):
    if hasattr(value, "torch"):
        value = value.torch
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    return value


def tolist(value):
    return [float(v) for v in tt(value).detach().cpu().flatten().tolist()]


def tilt_deg(robot) -> float:
    g = tt(robot.data.projected_gravity_b)[0]
    c = torch.clamp(-g[2], -1.0, 1.0)
    return float(torch.rad2deg(torch.acos(c)).detach().cpu().item())


def action_rms(action: torch.Tensor) -> float:
    a = action.detach().float()
    return float(torch.sqrt(torch.mean(a * a)).cpu().item())


def set_zero_command(base_env) -> None:
    buf = base_env.command_manager.get_command("base_velocity")
    with torch.no_grad():
        buf[:, 0] = 0.0
        buf[:, 1] = 0.0
        buf[:, 2] = 0.0


def reset_policy(policy, runner, dones) -> None:
    if hasattr(policy, "reset") and callable(policy.reset):
        policy.reset(dones)
        return
    alg = getattr(runner, "alg", None)
    for name in ("policy", "actor_critic"):
        net = getattr(alg, name, None)
        if net is not None and hasattr(net, "reset") and callable(net.reset):
            net.reset(dones)
            return


def main() -> None:
    if args_cli.duration_s <= 0.0:
        fail("--duration_s must be positive")
    if not CHECKPOINT.is_file():
        fail(f"checkpoint missing: {CHECKPOINT}")

    ckpt_sha = sha256(CHECKPOINT)
    if ckpt_sha != EXPECTED_SHA256:
        fail(f"checkpoint SHA mismatch: {ckpt_sha}")

    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=1)
    env_cfg.scene.num_envs = 1
    env_cfg.seed = int(args_cli.seed)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    env_cfg.episode_length_s = max(
        float(env_cfg.episode_length_s),
        float(args_cli.duration_s + 10.0),
    )

    if hasattr(env_cfg.observations.policy, "enable_corruption"):
        env_cfg.observations.policy.enable_corruption = False
    if hasattr(env_cfg, "events"):
        if hasattr(env_cfg.events, "base_external_force_torque"):
            env_cfg.events.base_external_force_torque = None
        if hasattr(env_cfg.events, "push_robot"):
            env_cfg.events.push_robot = None

    env_cfg.commands.base_velocity.heading_command = False
    if hasattr(env_cfg.commands.base_velocity, "rel_heading_envs"):
        env_cfg.commands.base_velocity.rel_heading_envs = 0.0

    # ------------------------------------------------------------------
    # SINGLE CAUSAL DELTA
    # Preregistered failing seed227 baseline used [-0.5, +0.5] for all six root velocities.
    # Here ONLY these six reset ranges are changed to exactly zero.
    # Pose reset (x/y/yaw), policy, checkpoint, command, physics and seed
    # remain unchanged.
    # ------------------------------------------------------------------
    original_velocity_range = dict(
        env_cfg.events.reset_base.params["velocity_range"]
    )
    zero_velocity_range = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    env_cfg.events.reset_base.params["velocity_range"] = zero_velocity_range

    agent_cfg = UprightYawDirectWideYawFlatOrientationPPORunnerCfg()
    if args_cli.device is not None:
        agent_cfg.device = args_cli.device
    agent_cfg = handle_deprecated_rsl_rl_cfg(
        agent_cfg, metadata.version("rsl-rl-lib")
    )

    raw_env = None
    env = None
    rows = []

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_dir = (
        Path(r"C:\humanoid_rl_lab\logs\thesis\startup_robustness")
        / f"{stamp}_v2g_seed227_zero_root_velocity"
    )
    result_dir.mkdir(parents=True, exist_ok=False)

    try:
        raw_env = gym.make(TASK, cfg=env_cfg)
        env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
        base_env = env.unwrapped
        robot = base_env.scene["robot"]

        set_zero_command(base_env)

        reset_lin_w = tolist(robot.data.root_lin_vel_w[0])
        reset_ang_w = tolist(robot.data.root_ang_vel_w[0])

        # With default root velocity = zero, this should be essentially zero.
        reset_vel_max = max(abs(x) for x in reset_lin_w + reset_ang_w)
        if reset_vel_max > 1e-5:
            fail(
                "zero-root-velocity intervention was not realized at wrapper reset: "
                f"max_abs={reset_vel_max}, lin={reset_lin_w}, ang={reset_ang_w}"
            )

        runner = OnPolicyRunner(
            env,
            agent_cfg.to_dict(),
            log_dir=None,
            device=agent_cfg.device,
        )
        runner.load(str(CHECKPOINT))
        policy = runner.get_inference_policy(device=base_env.device)

        total_steps = int(round(args_cli.duration_s / DT))

        print("=" * 116, flush=True)
        print("v2-G SEED-227 ZERO ROOT-VELOCITY CAUSAL ISOLATION", flush=True)
        print("=" * 116, flush=True)
        print(f"seed                     : {args_cli.seed}", flush=True)
        print(f"checkpoint SHA256        : {ckpt_sha}", flush=True)
        print(f"duration                 : {args_cli.duration_s:.2f} s", flush=True)
        print("command                  : [0,0,0]", flush=True)
        print("policy                   : frozen v2-G ONLY", flush=True)
        print("v4-N / router / handoff  : ABSENT", flush=True)
        print("T0 observation           : UNCHANGED training/runtime stale-cache contract", flush=True)
        print("prior baseline result    : seed227 terminated at 1.50 s in preregistered seeds200-249 experiment", flush=True)
        print(f"baseline velocity_range  : {original_velocity_range}", flush=True)
        print(f"test velocity_range      : {zero_velocity_range}", flush=True)
        print(
            "single causal delta       : ALL SIX reset root velocities [-0.5,+0.5] -> 0",
            flush=True,
        )
        print(
            "wrapper-reset root lin_w  : "
            + " ".join(f"{x:+.9f}" for x in reset_lin_w),
            flush=True,
        )
        print(
            "wrapper-reset root ang_w  : "
            + " ".join(f"{x:+.9f}" for x in reset_ang_w),
            flush=True,
        )
        print(f"result_dir                : {result_dir}", flush=True)
        print("=" * 116, flush=True)

        terminated = False

        for step in range(total_steps):
            set_zero_command(base_env)
            obs = env.get_observations()

            with torch.inference_mode():
                actions = policy(obs)

            if not torch.isfinite(actions).all():
                fail(f"non-finite policy action at step {step}")

            with torch.inference_mode():
                obs, _, dones, _ = env.step(actions)

            reset_policy(policy, runner, dones)

            pos = tt(robot.data.root_pos_w)[0]
            lin_b = tt(robot.data.root_lin_vel_b)[0]
            ang_b = tt(robot.data.root_ang_vel_b)[0]

            row = {
                "step": step,
                "time_s": (step + 1) * DT,
                "root_x_m": float(pos[0].detach().cpu().item()),
                "root_y_m": float(pos[1].detach().cpu().item()),
                "root_z_m": float(pos[2].detach().cpu().item()),
                "root_vxy_mps": math.hypot(
                    float(lin_b[0].detach().cpu().item()),
                    float(lin_b[1].detach().cpu().item()),
                ),
                "root_wz_radps": float(ang_b[2].detach().cpu().item()),
                "root_tilt_deg": tilt_deg(robot),
                "action_rms": action_rms(actions),
                "done": int(bool(torch.any(dones).item())),
            }
            rows.append(row)

            if step % 10 == 0 or row["done"]:
                print(
                    f"[STATE] step={step:03d} "
                    f"t={row['time_s']:.3f}s "
                    f"z={row['root_z_m']:.4f} "
                    f"vxy={row['root_vxy_mps']:.4f} "
                    f"tilt={row['root_tilt_deg']:.3f} "
                    f"|wz|={abs(row['root_wz_radps']):.4f} "
                    f"action_rms={row['action_rms']:.4f} "
                    f"done={row['done']}",
                    flush=True,
                )

            if row["done"]:
                terminated = True
                print(
                    f"[TERMINATION] step={step} t={row['time_s']:.3f}s",
                    flush=True,
                )
                break

        csv_path = result_dir / "trajectory.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        print(f"trajectory CSV           : {csv_path}", flush=True)

        if terminated:
            print(
                "CAUSAL_RESULT            : FAILURE PERSISTS -> the six-component root-velocity block is not necessary for seed227 failure",
                flush=True,
            )
            print(PERSISTS_MARKER, flush=True)
        else:
            last = rows[-1]
            print("termination              : NO", flush=True)
            print(
                f"final state              : z={last['root_z_m']:.6f} "
                f"vxy={last['root_vxy_mps']:.6f} "
                f"tilt={last['root_tilt_deg']:.6f} "
                f"|wz|={abs(last['root_wz_radps']):.6f}",
                flush=True,
            )
            print(
                "CAUSAL_RESULT            : RESCUE -> the original root-velocity reset block is causally necessary for the seed227 baseline failure",
                flush=True,
            )
            print(RESCUE_MARKER, flush=True)

        print(PASS_MARKER, flush=True)

    finally:
        if env is not None:
            env.close()
        elif raw_env is not None:
            raw_env.close()
        gc.collect()


try:
    main()
except BaseException as exc:
    print(FAIL_MARKER, flush=True)
    print(f"EXCEPTION_TYPE={type(exc).__name__}", flush=True)
    print(f"EXCEPTION_MESSAGE={exc}", flush=True)
    traceback.print_exc()
    raise
finally:
    simulation_app.close()
