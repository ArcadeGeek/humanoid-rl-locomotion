from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import math
import statistics
import traceback
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

parser = argparse.ArgumentParser(
    description=(
        "Controlled braking-dwell handoff test for frozen Unitree G1 controllers: "
        "v2-G STAND -> v4-N LOCOMOTION -> v4-N zero-command brake -> v2-G STAND."
    )
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--stand_pre_s", type=float, default=3.0)
parser.add_argument("--locomotion_s", type=float, default=8.0)
parser.add_argument("--stand_post_s", type=float, default=8.0)
parser.add_argument("--brake_s", type=float, default=0.36)
parser.add_argument("--locomotion_vx", type=float, default=0.40)
parser.add_argument("--output_dir", type=str, default=None)

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
from humanoid_rl_lab.tasks.manager_based.humanoid_rl_lab.agents.v4n_ppo_cfg import (
    V4NSymmetryAugmentationPPORunnerCfg,
)
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import parse_env_cfg
from rsl_rl.runners import OnPolicyRunner


STAND_TASK = "Humanoid-RL-Lab-G1-Flat-Upright-Yaw-Direct-Wide-Yaw-Flat-Orientation-v0"

STAND_CHECKPOINT = Path(
    r"C:\humanoid_rl_lab\checkpoints\local_g1_yaw_direct_wide_yaw_flat_orientation_v2g\model_600.pt"
)
STAND_CHECKPOINT_SHA256 = "58468EED03F9EF35A9BC898063A3B14AC36C93A32F65DB42B404BDE93CD694D3"

LOCO_CHECKPOINT = Path(
    r"C:\humanoid_rl_lab\logs\rsl_rl\humanoid_rl_lab_g1_v4n_symmetry_augmentation_scratch"
    r"\2026-09-04_13-58-13_v4n_genuine_scratch_seed42_official1500\model_1499.pt"
)
LOCO_CHECKPOINT_SHA256 = "367D318AB4940A3AAE8F0FA568384212762258067DFC4CF9F624FA472BEB9342"

EXPECTED_OBS_DIM = 123
EXPECTED_ACTION_DIM = 37
EXPECTED_STEP_DT = 0.02
CONTACT_FORCE_THRESHOLD_N = 20.0

MARKER = "V2G_V4N_BRAKE_DWELL_036_ZERO_ROOT_VEL_START_SMOKE_PASS"


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def torch_value(value: Any) -> torch.Tensor:
    if hasattr(value, "torch"):
        value = value.torch
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    return value


def extract_policy_observation_tensor(obs: Any) -> torch.Tensor:
    """Resolve the 123-D policy tensor without assuming a plain dict container.

    RSL-RL 5 / Isaac Lab may return a TensorDict-like batched container whose
    own batch shape is (1,), while obs["policy"] is the actual (1, 123) tensor.
    The policy itself must continue receiving the original observation container.
    """
    candidate = None

    if torch.is_tensor(obs) or hasattr(obs, "torch"):
        candidate = obs
    else:
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
        fail(
            "unable to resolve 'policy' observation tensor from "
            f"container type={type(obs).__name__}"
        )

    tensor = torch_value(candidate)
    if tensor.ndim != 2 or tensor.shape[0] != 1:
        fail(
            "unexpected resolved policy observation shape="
            f"{tuple(tensor.shape)} from container type={type(obs).__name__}"
        )
    return tensor


def cpu_list(value: Any) -> list[float]:
    return [float(x) for x in torch_value(value).detach().cpu().flatten().tolist()]


def scalar(value: Any) -> float:
    t = torch_value(value).detach().cpu().flatten()
    if t.numel() != 1:
        fail(f"expected scalar tensor, got shape={tuple(t.shape)}")
    return float(t.item())


def write_command(base_env: Any, vx: float, vy: float, wz: float) -> None:
    command_buffer = base_env.command_manager.get_command("base_velocity")
    with torch.no_grad():
        command_buffer[:, 0] = float(vx)
        command_buffer[:, 1] = float(vy)
        command_buffer[:, 2] = float(wz)


def projected_gravity_tilt_deg(robot: Any) -> float:
    g_b = torch_value(robot.data.projected_gravity_b)[0]
    # Upright G1 has projected gravity approximately [0, 0, -1].
    c = torch.clamp(-g_b[2], -1.0, 1.0)
    return float(torch.rad2deg(torch.acos(c)).detach().cpu().item())


def action_rms(action: torch.Tensor) -> float:
    a = action.detach().float()
    return float(torch.sqrt(torch.mean(a * a)).cpu().item())


def action_delta_metrics(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    delta = (b.detach().float() - a.detach().float()).flatten()
    rms = float(torch.sqrt(torch.mean(delta * delta)).cpu().item())
    max_abs = float(torch.max(torch.abs(delta)).cpu().item())
    return rms, max_abs


def validate_checkpoint(path: Path, expected_sha: str, label: str) -> str:
    if not path.is_file():
        fail(f"{label} checkpoint missing: {path}")
    actual = sha256(path)
    if actual != expected_sha:
        fail(
            f"{label} checkpoint SHA256 mismatch.\n"
            f"Expected: {expected_sha}\n"
            f"Actual  : {actual}"
        )
    print(f"{label} checkpoint       : {path}")
    print(f"{label} checkpoint SHA256: {actual}")
    return actual


def reset_policy_after_done(policy: Any, runner: Any, dones: torch.Tensor) -> None:
    if hasattr(policy, "reset") and callable(policy.reset):
        policy.reset(dones)
        return

    alg = getattr(runner, "alg", None)
    for name in ("policy", "actor_critic"):
        network = getattr(alg, name, None)
        if network is not None and hasattr(network, "reset") and callable(network.reset):
            network.reset(dones)
            return


def phase_summary(rows: list[dict[str, Any]], phase: str) -> dict[str, float]:
    subset = [r for r in rows if r["phase"] == phase]
    if not subset:
        fail(f"no rows recorded for phase {phase}")

    x0, y0 = subset[0]["root_x_m"], subset[0]["root_y_m"]
    x1, y1 = subset[-1]["root_x_m"], subset[-1]["root_y_m"]
    drift = math.hypot(x1 - x0, y1 - y0)

    out = {
        "samples": float(len(subset)),
        "duration_s": len(subset) * EXPECTED_STEP_DT,
        "xy_drift_m": drift,
        "mean_vxy_mps": statistics.fmean(r["root_vxy_mps"] for r in subset),
        "max_vxy_mps": max(r["root_vxy_mps"] for r in subset),
        "mean_tilt_deg": statistics.fmean(r["root_tilt_deg"] for r in subset),
        "max_tilt_deg": max(r["root_tilt_deg"] for r in subset),
        "mean_abs_wz_radps": statistics.fmean(abs(r["root_wz_radps"]) for r in subset),
    }

    if all(r["double_support"] >= 0 for r in subset):
        out["double_support_pct"] = (
            100.0 * statistics.fmean(r["double_support"] for r in subset)
        )
    return out


def tail_summary(rows: list[dict[str, Any]], phase: str, tail_s: float = 2.0) -> dict[str, float]:
    subset = [r for r in rows if r["phase"] == phase]
    count = max(1, int(round(tail_s / EXPECTED_STEP_DT)))
    tail = subset[-count:]

    out = {
        "samples": float(len(tail)),
        "mean_vxy_mps": statistics.fmean(r["root_vxy_mps"] for r in tail),
        "max_vxy_mps": max(r["root_vxy_mps"] for r in tail),
        "mean_tilt_deg": statistics.fmean(r["root_tilt_deg"] for r in tail),
        "max_tilt_deg": max(r["root_tilt_deg"] for r in tail),
        "mean_abs_wz_radps": statistics.fmean(abs(r["root_wz_radps"]) for r in tail),
        "mean_action_rms": statistics.fmean(r["action_rms"] for r in tail),
    }
    if all(r["double_support"] >= 0 for r in tail):
        out["double_support_pct"] = 100.0 * statistics.fmean(
            r["double_support"] for r in tail
        )
    return out


def print_summary(title: str, data: dict[str, float]) -> None:
    print()
    print("-" * 104)
    print(title)
    print("-" * 104)
    for key, value in data.items():
        if key == "samples":
            print(f"{key:<32}: {int(value)}")
        else:
            print(f"{key:<32}: {value:.9f}")


def main() -> None:
    if args_cli.stand_pre_s <= 0.0:
        fail("--stand_pre_s must be positive")
    if args_cli.locomotion_s <= 0.0:
        fail("--locomotion_s must be positive")
    if args_cli.stand_post_s <= 0.0:
        fail("--stand_post_s must be positive")
    if args_cli.brake_s <= 0.0:
        fail("--brake_s must be positive")
    if args_cli.locomotion_vx <= 0.0:
        fail("--locomotion_vx must be positive")

    print("=" * 104)
    print("v2-G STAND -> v4-N LOCOMOTION -> v4-N BRAKE 0.36 s (0 cmd) -> v2-G STAND")
    print("=" * 104)
    print("Scientific role : frozen 0.36 s handoff with engineering-only ZERO root-velocity startup conditioning; NO blending, NO training")
    print("Environment     : frozen v2-G task used as the shared physical runtime")
    print("Commands        : [0,0,0] -> [vx,0,0] -> [0,0,0] (v4-N brake) -> [0,0,0] (v2-G)")
    print("Seed            :", args_cli.seed)
    print()

    stand_sha = validate_checkpoint(
        STAND_CHECKPOINT, STAND_CHECKPOINT_SHA256, "v2-G STAND"
    )
    loco_sha = validate_checkpoint(
        LOCO_CHECKPOINT, LOCO_CHECKPOINT_SHA256, "v4-N LOCOMOTION"
    )

    installed_version = metadata.version("rsl-rl-lib")

    env_cfg = parse_env_cfg(
        STAND_TASK,
        device=args_cli.device,
        num_envs=1,
    )
    env_cfg.scene.num_envs = 1
    env_cfg.seed = int(args_cli.seed)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    total_requested_s = (
        float(args_cli.stand_pre_s)
        + float(args_cli.locomotion_s)
        + float(args_cli.brake_s)
        + float(args_cli.stand_post_s)
    )
    env_cfg.episode_length_s = max(
        float(env_cfg.episode_length_s),
        total_requested_s + 10.0,
    )

    # Engineering-only startup conditioning for handoff isolation:
    # remove randomized root-velocity startup stress so every run can reach
    # the handoff phases. Policies, commands, dwell, pose/joints, physics,
    # and all handoff logic remain unchanged.
    baseline_root_velocity_range = dict(
        env_cfg.events.reset_base.params["velocity_range"]
    )
    zero_root_velocity_range = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    env_cfg.events.reset_base.params["velocity_range"] = zero_root_velocity_range

    # Match the already validated PLAY-equivalent evaluation conditions.
    if hasattr(env_cfg.observations.policy, "enable_corruption"):
        env_cfg.observations.policy.enable_corruption = False
    if hasattr(env_cfg, "events"):
        if hasattr(env_cfg.events, "base_external_force_torque"):
            env_cfg.events.base_external_force_torque = None
        if hasattr(env_cfg.events, "push_robot"):
            env_cfg.events.push_robot = None

    command_cfg = env_cfg.commands.base_velocity
    command_cfg.heading_command = False
    if hasattr(command_cfg, "rel_heading_envs"):
        command_cfg.rel_heading_envs = 0.0

    stand_agent_cfg = UprightYawDirectWideYawFlatOrientationPPORunnerCfg()
    loco_agent_cfg = V4NSymmetryAugmentationPPORunnerCfg()

    if args_cli.device is not None:
        stand_agent_cfg.device = args_cli.device
        loco_agent_cfg.device = args_cli.device

    stand_agent_cfg = handle_deprecated_rsl_rl_cfg(stand_agent_cfg, installed_version)
    loco_agent_cfg = handle_deprecated_rsl_rl_cfg(loco_agent_cfg, installed_version)

    if stand_agent_cfg.class_name != "OnPolicyRunner":
        fail(f"unexpected v2-G runner class: {stand_agent_cfg.class_name}")
    if loco_agent_cfg.class_name != "OnPolicyRunner":
        fail(f"unexpected v4-N runner class: {loco_agent_cfg.class_name}")

    if stand_agent_cfg.clip_actions != loco_agent_cfg.clip_actions:
        fail(
            "clip_actions differs between v2-G and v4-N: "
            f"{stand_agent_cfg.clip_actions!r} vs {loco_agent_cfg.clip_actions!r}"
        )

    env = None
    try:
        raw_env = gym.make(STAND_TASK, cfg=env_cfg)
        env = RslRlVecEnvWrapper(raw_env, clip_actions=stand_agent_cfg.clip_actions)

        base_env = env.unwrapped
        robot = base_env.scene["robot"]

        if not math.isclose(float(base_env.step_dt), EXPECTED_STEP_DT, rel_tol=0.0, abs_tol=1e-12):
            fail(f"unexpected step_dt={base_env.step_dt}; expected {EXPECTED_STEP_DT}")

        if int(getattr(base_env.action_manager, "total_action_dim", -1)) != EXPECTED_ACTION_DIM:
            fail(
                f"unexpected action dimension={getattr(base_env.action_manager, 'total_action_dim', None)}; "
                f"expected {EXPECTED_ACTION_DIM}"
            )

        print()
        print("[INFO] Constructing frozen v2-G runner...", flush=True)
        stand_runner = OnPolicyRunner(
            env,
            stand_agent_cfg.to_dict(),
            log_dir=None,
            device=stand_agent_cfg.device,
        )
        stand_runner.load(str(STAND_CHECKPOINT))
        stand_policy = stand_runner.get_inference_policy(device=base_env.device)

        print("[INFO] Constructing frozen v4-N runner...", flush=True)
        loco_runner = OnPolicyRunner(
            env,
            loco_agent_cfg.to_dict(),
            log_dir=None,
            device=loco_agent_cfg.device,
        )
        loco_runner.load(str(LOCO_CHECKPOINT))
        loco_policy = loco_runner.get_inference_policy(device=base_env.device)

        obs = env.get_observations()
        policy_obs = extract_policy_observation_tensor(obs)
        if int(policy_obs.shape[1]) != EXPECTED_OBS_DIM:
            fail(
                f"unexpected observation dimension={policy_obs.shape[1]}; "
                f"expected {EXPECTED_OBS_DIM}"
            )

        dt = float(base_env.step_dt)
        pre_steps = int(round(args_cli.stand_pre_s / dt))
        loco_steps = int(round(args_cli.locomotion_s / dt))
        brake_steps = int(round(args_cli.brake_s / dt))
        post_steps = int(round(args_cli.stand_post_s / dt))
        total_steps = pre_steps + loco_steps + brake_steps + post_steps
        brake_start_step = pre_steps + loco_steps
        stand_return_step = brake_start_step + brake_steps

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if args_cli.output_dir:
            result_dir = Path(args_cli.output_dir).resolve()
            result_dir.mkdir(parents=True, exist_ok=False)
        else:
            result_dir = (
                Path(r"C:\humanoid_rl_lab\logs\thesis\handoff")
                / f"{timestamp}_v2g_v4n_brake_dwell_036_zero_root_vel_start_seed{args_cli.seed}"
            )
            result_dir.mkdir(parents=True, exist_ok=False)

        csv_path = result_dir / "trajectory.csv"
        meta_path = result_dir / "summary.txt"

        contact_sensor = None
        left_contact_idx = None
        right_contact_idx = None
        try:
            contact_sensor = base_env.scene.sensors["contact_forces"]
            contact_names = list(contact_sensor.body_names)
            left_name = "left_ankle_roll_link"
            right_name = "right_ankle_roll_link"
            if left_name in contact_names and right_name in contact_names:
                left_contact_idx = contact_names.index(left_name)
                right_contact_idx = contact_names.index(right_name)
            else:
                contact_sensor = None
        except Exception:
            contact_sensor = None

        rows: list[dict[str, Any]] = []
        terminated = False
        terminated_step = -1
        terminated_phase = ""

        switch_metrics: dict[str, dict[str, float]] = {}
        previous_selected_action: torch.Tensor | None = None

        print()
        print("=" * 104)
        print("RUNTIME PREFLIGHT", flush=True)
        print("=" * 104)
        print(f"step_dt                  : {dt:.12f}")
        print(f"obs_container_type        : {type(obs).__name__}", flush=True)
        print(f"policy_obs_shape          : {tuple(policy_obs.shape)}", flush=True)
        print(f"policy_obs_dim            : {policy_obs.shape[1]}", flush=True)
        print(f"action_dim                : {base_env.action_manager.total_action_dim}")
        print(f"clip_actions              : {stand_agent_cfg.clip_actions}")
        print(f"startup root vel baseline : {baseline_root_velocity_range}")
        print(f"startup root vel test     : {zero_root_velocity_range}")
        print("startup role              : engineering-only conditioning; isolates handoff from v2-G randomized-start robustness")
        print(f"pre-stand                 : {pre_steps} steps = {pre_steps * dt:.2f} s")
        print(f"locomotion                : {loco_steps} steps = {loco_steps * dt:.2f} s")
        print(f"v4-N zero-command brake   : {brake_steps} steps = {brake_steps * dt:.2f} s")
        print(f"post-stand                : {post_steps} steps = {post_steps * dt:.2f} s")
        print(f"locomotion command        : [{args_cli.locomotion_vx:.3f}, 0.000, 0.000]")
        print(f"contact instrumentation   : {'AVAILABLE' if contact_sensor is not None else 'UNAVAILABLE'}")
        print(f"result_dir                : {result_dir}")
        print("=" * 104)

        for step in range(total_steps):
            if step < pre_steps:
                phase = "STAND_PRE"
                active_name = "v2-G"
                active_policy = stand_policy
                active_runner = stand_runner
                command = (0.0, 0.0, 0.0)
            elif step < brake_start_step:
                phase = "LOCOMOTION"
                active_name = "v4-N"
                active_policy = loco_policy
                active_runner = loco_runner
                command = (float(args_cli.locomotion_vx), 0.0, 0.0)
            elif step < stand_return_step:
                phase = "BRAKE_V4N"
                active_name = "v4-N"
                active_policy = loco_policy
                active_runner = loco_runner
                command = (0.0, 0.0, 0.0)
            else:
                phase = "STAND_POST"
                active_name = "v2-G"
                active_policy = stand_policy
                active_runner = stand_runner
                command = (0.0, 0.0, 0.0)

            # At the locomotion-command drop, measure the pure command-change effect
            # on v4-N at the same physical state, before stepping physics.
            old_command_obs = obs if step == brake_start_step else None
            old_command_action = None
            if old_command_obs is not None:
                with torch.inference_mode():
                    old_command_action = loco_policy(old_command_obs)

            write_command(base_env, *command)
            # Canonical evaluator behavior: refresh observations after command-buffer overwrite.
            obs = env.get_observations()

            with torch.inference_mode():
                # At each switch, evaluate both controllers on the exact same refreshed observation
                # before applying either action. This is diagnostic only and does not touch physics.
                precomputed_active_action = None

                if step == pre_steps:
                    old_action_same_obs = stand_policy(obs)
                    new_action_same_obs = loco_policy(obs)
                    precomputed_active_action = new_action_same_obs
                    rms_delta, max_delta = action_delta_metrics(
                        old_action_same_obs, new_action_same_obs
                    )
                    switch_metrics["STAND_TO_LOCO"] = {
                        "counterfactual_action_rms_delta": rms_delta,
                        "counterfactual_action_max_abs_delta": max_delta,
                    }
                    print(
                        f"[SWITCH] STAND->LOCO t={step * dt:.3f}s "
                        f"same-obs action delta RMS={rms_delta:.6f} max={max_delta:.6f}",
                        flush=True,
                    )

                if step == brake_start_step:
                    if old_command_action is None:
                        fail("missing old-command v4-N action at brake transition")
                    new_command_action = loco_policy(obs)
                    precomputed_active_action = new_command_action
                    rms_delta, max_delta = action_delta_metrics(
                        old_command_action, new_command_action
                    )
                    switch_metrics["V4N_CMD040_TO_ZERO"] = {
                        "counterfactual_action_rms_delta": rms_delta,
                        "counterfactual_action_max_abs_delta": max_delta,
                    }
                    print(
                        f"[COMMAND] v4-N 0.40->0.00 t={step * dt:.3f}s "
                        f"same-state command-response delta RMS={rms_delta:.6f} max={max_delta:.6f}",
                        flush=True,
                    )

                if step == stand_return_step:
                    old_action_same_obs = loco_policy(obs)
                    new_action_same_obs = stand_policy(obs)
                    precomputed_active_action = new_action_same_obs
                    rms_delta, max_delta = action_delta_metrics(
                        old_action_same_obs, new_action_same_obs
                    )
                    switch_metrics["BRAKE_TO_STAND"] = {
                        "counterfactual_action_rms_delta": rms_delta,
                        "counterfactual_action_max_abs_delta": max_delta,
                    }
                    print(
                        f"[SWITCH] v4-N BRAKE->v2-G STAND t={step * dt:.3f}s "
                        f"same-obs action delta RMS={rms_delta:.6f} max={max_delta:.6f}",
                        flush=True,
                    )

                actions = (
                    precomputed_active_action
                    if precomputed_active_action is not None
                    else active_policy(obs)
                )

            if not torch.isfinite(actions).all():
                fail(f"non-finite action produced by {active_name} at step {step}")

            actual_jump_rms = 0.0
            actual_jump_max = 0.0
            if previous_selected_action is not None:
                actual_jump_rms, actual_jump_max = action_delta_metrics(
                    previous_selected_action, actions
                )
            previous_selected_action = actions.detach().clone()

            with torch.inference_mode():
                obs, _, dones, _ = env.step(actions)

            reset_policy_after_done(active_policy, active_runner, dones)

            root_pos = torch_value(robot.data.root_pos_w)[0]
            root_lin_b = torch_value(robot.data.root_lin_vel_b)[0]
            root_ang_b = torch_value(robot.data.root_ang_vel_b)[0]

            double_support = -1.0
            left_force_z = float("nan")
            right_force_z = float("nan")
            if (
                contact_sensor is not None
                and left_contact_idx is not None
                and right_contact_idx is not None
            ):
                try:
                    net_forces = torch_value(contact_sensor.data.net_forces_w)[0]
                    left_force_z = abs(float(net_forces[left_contact_idx, 2].detach().cpu().item()))
                    right_force_z = abs(float(net_forces[right_contact_idx, 2].detach().cpu().item()))
                    left_contact = left_force_z >= CONTACT_FORCE_THRESHOLD_N
                    right_contact = right_force_z >= CONTACT_FORCE_THRESHOLD_N
                    double_support = 1.0 if (left_contact and right_contact) else 0.0
                except Exception:
                    contact_sensor = None
                    double_support = -1.0

            row = {
                "step": step,
                "time_s": (step + 1) * dt,
                "phase": phase,
                "active_policy": active_name,
                "command_vx_mps": command[0],
                "command_vy_mps": command[1],
                "command_wz_radps": command[2],
                "root_x_m": float(root_pos[0].detach().cpu().item()),
                "root_y_m": float(root_pos[1].detach().cpu().item()),
                "root_z_m": float(root_pos[2].detach().cpu().item()),
                "root_vx_mps": float(root_lin_b[0].detach().cpu().item()),
                "root_vy_mps": float(root_lin_b[1].detach().cpu().item()),
                "root_vxy_mps": math.hypot(
                    float(root_lin_b[0].detach().cpu().item()),
                    float(root_lin_b[1].detach().cpu().item()),
                ),
                "root_wz_radps": float(root_ang_b[2].detach().cpu().item()),
                "root_tilt_deg": projected_gravity_tilt_deg(robot),
                "action_rms": action_rms(actions),
                "actual_action_jump_rms": actual_jump_rms,
                "actual_action_jump_max_abs": actual_jump_max,
                "left_contact_force_z_n": left_force_z,
                "right_contact_force_z_n": right_force_z,
                "double_support": double_support,
                "done": int(bool(torch.any(dones).item())),
            }
            rows.append(row)

            if bool(torch.any(dones).item()):
                terminated = True
                terminated_step = step
                terminated_phase = phase
                print(
                    f"[ERROR] Environment terminated at step={step}, "
                    f"t={(step + 1) * dt:.3f}s, phase={phase}"
                )
                break

        if not rows:
            fail("no trajectory rows recorded")

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        if terminated:
            fail(
                "dynamic handoff smoke terminated before completion: "
                f"step={terminated_step}, phase={terminated_phase}"
            )

        if len(rows) != total_steps:
            fail(f"incomplete trajectory: {len(rows)} rows vs expected {total_steps}")

        pre = phase_summary(rows, "STAND_PRE")
        loco = phase_summary(rows, "LOCOMOTION")
        brake = phase_summary(rows, "BRAKE_V4N")
        post = phase_summary(rows, "STAND_POST")
        post_tail = tail_summary(rows, "STAND_POST", tail_s=2.0)

        brake_rows = [r for r in rows if r["phase"] == "BRAKE_V4N"]
        post_rows = [r for r in rows if r["phase"] == "STAND_POST"]
        if not brake_rows or not post_rows:
            fail("missing BRAKE_V4N or STAND_POST rows")

        brake_start_state = brake_rows[0]
        brake_end_state = brake_rows[-1]

        def first_n_phase_summary(phase_rows, seconds):
            n = max(1, int(round(seconds / dt)))
            seg = phase_rows[:n]
            return {
                "samples": float(len(seg)),
                "duration_s": len(seg) * dt,
                "mean_vxy_mps": statistics.fmean(r["root_vxy_mps"] for r in seg),
                "max_vxy_mps": max(r["root_vxy_mps"] for r in seg),
                "mean_tilt_deg": statistics.fmean(r["root_tilt_deg"] for r in seg),
                "max_tilt_deg": max(r["root_tilt_deg"] for r in seg),
                "mean_abs_wz_radps": statistics.fmean(abs(r["root_wz_radps"]) for r in seg),
                "max_abs_wz_radps": max(abs(r["root_wz_radps"]) for r in seg),
                "double_support_pct": 100.0 * statistics.fmean(r["double_support"] for r in seg)
                if all(r["double_support"] >= 0 for r in seg) else float("nan"),
            }

        post_first_05 = first_n_phase_summary(post_rows, 0.5)
        post_first_10 = first_n_phase_summary(post_rows, 1.0)
        post_first_20 = first_n_phase_summary(post_rows, 2.0)

        print_summary("PHASE SUMMARY: STAND_PRE", pre)
        print_summary("PHASE SUMMARY: LOCOMOTION", loco)
        print_summary("PHASE SUMMARY: BRAKE_V4N (zero command)", brake)
        print(
            f"BRAKE start state: vxy={brake_start_state['root_vxy_mps']:.9f} "
            f"tilt={brake_start_state['root_tilt_deg']:.9f} "
            f"|wz|={abs(brake_start_state['root_wz_radps']):.9f} "
            f"double_support={brake_start_state['double_support']:.0f}"
        )
        print(
            f"BRAKE end state  : vxy={brake_end_state['root_vxy_mps']:.9f} "
            f"tilt={brake_end_state['root_tilt_deg']:.9f} "
            f"|wz|={abs(brake_end_state['root_wz_radps']):.9f} "
            f"double_support={brake_end_state['double_support']:.0f}"
        )
        print_summary("FIRST 0.5 s AFTER BRAKE->STAND", post_first_05)
        print_summary("FIRST 1.0 s AFTER BRAKE->STAND", post_first_10)
        print_summary("FIRST 2.0 s AFTER BRAKE->STAND", post_first_20)
        print_summary("PHASE SUMMARY: STAND_POST", post)
        print_summary("FINAL 2 s OF STAND_POST", post_tail)

        print()
        print("-" * 104)
        print("SWITCH ACTION-DISCONTINUITY DIAGNOSTICS")
        print("-" * 104)
        for switch_name in ("STAND_TO_LOCO", "V4N_CMD040_TO_ZERO", "BRAKE_TO_STAND"):
            data = switch_metrics.get(switch_name)
            if data is None:
                fail(f"missing switch diagnostics for {switch_name}")
            print(
                f"{switch_name:<20} same-obs RMS={data['counterfactual_action_rms_delta']:.9f} "
                f"max_abs={data['counterfactual_action_max_abs_delta']:.9f}"
            )

        script_path = Path(__file__).resolve()
        script_hash = sha256(script_path)

        summary_lines = [
            "v2-G STAND -> v4-N LOCOMOTION -> v4-N zero-command brake -> v2-G STAND",
            f"timestamp={timestamp}",
            f"seed={args_cli.seed}",
            f"script={script_path}",
            f"script_sha256={script_hash}",
            f"stand_checkpoint={STAND_CHECKPOINT}",
            f"stand_checkpoint_sha256={stand_sha}",
            f"loco_checkpoint={LOCO_CHECKPOINT}",
            f"loco_checkpoint_sha256={loco_sha}",
            f"step_dt={dt}",
            f"stand_pre_s={pre_steps * dt}",
            f"locomotion_s={loco_steps * dt}",
            f"brake_s={brake_steps * dt}",
            f"stand_post_s={post_steps * dt}",
            f"locomotion_vx={args_cli.locomotion_vx}",
            f"startup_root_velocity_baseline={baseline_root_velocity_range}",
            f"startup_root_velocity_test={zero_root_velocity_range}",
            "startup_conditioning_role=engineering-only; isolates handoff from randomized v2-G startup robustness",
            f"stand_to_loco_same_obs_action_rms_delta={switch_metrics['STAND_TO_LOCO']['counterfactual_action_rms_delta']}",
            f"stand_to_loco_same_obs_action_max_abs_delta={switch_metrics['STAND_TO_LOCO']['counterfactual_action_max_abs_delta']}",
            f"v4n_cmd040_to_zero_action_rms_delta={switch_metrics['V4N_CMD040_TO_ZERO']['counterfactual_action_rms_delta']}",
            f"v4n_cmd040_to_zero_action_max_abs_delta={switch_metrics['V4N_CMD040_TO_ZERO']['counterfactual_action_max_abs_delta']}",
            f"brake_to_stand_same_obs_action_rms_delta={switch_metrics['BRAKE_TO_STAND']['counterfactual_action_rms_delta']}",
            f"brake_to_stand_same_obs_action_max_abs_delta={switch_metrics['BRAKE_TO_STAND']['counterfactual_action_max_abs_delta']}",
        ]
        for prefix, data in (
            ("stand_pre", pre),
            ("locomotion", loco),
            ("brake_v4n", brake),
            ("stand_post_first_05", post_first_05),
            ("stand_post_first_10", post_first_10),
            ("stand_post_first_20", post_first_20),
            ("stand_post", post),
            ("stand_post_final2s", post_tail),
        ):
            for key, value in data.items():
                summary_lines.append(f"{prefix}_{key}={value}")

        summary_lines.extend(
            [
                "termination=NO",
                "interpretation=SMOKE COMPLETED; quantitative compatibility must be judged from the printed/logged transients.",
                MARKER,
            ]
        )
        meta_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

        print()
        print("=" * 104)
        print("BRAKING-DWELL HANDOFF SMOKE COMPLETED")
        print("=" * 104)
        print(f"trajectory CSV : {csv_path}")
        print(f"summary        : {meta_path}")
        print(f"script SHA256  : {script_hash}")
        print("termination    : NO")
        print("NOTE           : PASS means the controlled smoke completed without runtime termination.")
        print("                 It does NOT yet declare the handoff scientifically acceptable.")
        print("NEXT           : compare against immediate LOCO->STAND baseline before any router/blending change.")
        print(MARKER)
        print("=" * 104)

    finally:
        if env is not None:
            env.close()
        gc.collect()


try:
    main()
except BaseException as exc:
    print("V2G_V4N_BRAKE_DWELL_036_SMOKE_FAIL", flush=True)
    print(f"EXCEPTION_TYPE={type(exc).__name__}", flush=True)
    print(f"EXCEPTION_MESSAGE={exc}", flush=True)
    traceback.print_exc()
    raise
finally:
    simulation_app.close()
