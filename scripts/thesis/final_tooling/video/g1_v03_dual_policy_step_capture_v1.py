#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import traceback
from dataclasses import dataclass
from enum import Enum, auto
from importlib import metadata
from pathlib import Path
from typing import Any

parser = argparse.ArgumentParser(
    description=(
        "Deterministic simulation-step capture for final V03 engineering curve-negative video. "
        "Runs the frozen v2-G/v4-N controller for exactly 900 control steps "
        "(3 s ZERO + 12 s [0.40,0,0] + 3 s ZERO) at dt=0.02 s. "
        "This is a video master trajectory generator, not ROS deployment evidence."
    )
)
parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path(r"C:\humanoid_rl_lab\logs\thesis\video_final\V03_curve_neg\capture"),
)
parser.add_argument(
    "--trajectory-path",
    type=Path,
    default=None,
    help="Optional explicit output .pt path. Defaults inside --output-dir.",
)
parser.add_argument("--seed", type=int, default=42)

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# This capture is intentionally headless by default. Rendering happens later in offline replay.
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


TASK_ID = "Humanoid-RL-Lab-G1-Flat-Upright-Yaw-Direct-Wide-Yaw-Flat-Orientation-v0"

V2G_CHECKPOINT = Path(
    r"C:\humanoid_rl_lab\checkpoints\local_g1_yaw_direct_wide_yaw_flat_orientation_v2g\model_600.pt"
)
V2G_SHA256 = "58468EED03F9EF35A9BC898063A3B14AC36C93A32F65DB42B404BDE93CD694D3"

V4N_CHECKPOINT = Path(
    r"C:\humanoid_rl_lab\logs\rsl_rl\humanoid_rl_lab_g1_v4n_symmetry_augmentation_scratch"
    r"\2026-09-04_13-58-13_v4n_genuine_scratch_seed42_official1500\model_1499.pt"
)
V4N_SHA256 = "367D318AB4940A3AAE8F0FA568384212762258067DFC4CF9F624FA472BEB9342"

EXPECTED_DT = 0.02
EXPECTED_OBS_DIM = 123
EXPECTED_ACTION_DIM = 37

EXIT_LOCO_SPEED = 0.10
ENTER_LOCO_SPEED = 0.20
BRAKE_S = 0.36
BRAKE_STEPS = 18

ZERO_BEFORE_STEPS = 150
CURVE_NEG_STEPS = 600
ZERO_AFTER_STEPS = 150
TOTAL_STEPS = ZERO_BEFORE_STEPS + CURVE_NEG_STEPS + ZERO_AFTER_STEPS
EXPECTED_DURATION_S = TOTAL_STEPS * EXPECTED_DT

ZERO_CMD = (0.0, 0.0, 0.0)
CURVE_NEG_CMD = (0.40, 0.0, -0.25)

PASS_MARKER = "G1_V03_DUAL_POLICY_STEP_CAPTURE_PASS"
FAIL_MARKER = "G1_V03_DUAL_POLICY_STEP_CAPTURE_FAIL"
FORMAT_ID = "G1_DUAL_POLICY_VISUAL_TRAJECTORY_V3"


class RouterState(Enum):
    STAND = auto()
    LOCO = auto()
    BRAKE = auto()


@dataclass(frozen=True)
class RouterOutput:
    state: RouterState
    active_policy: str
    policy_command: tuple[float, float, float]
    brake_index: int


class Router:
    def __init__(self) -> None:
        if not (0.0 <= EXIT_LOCO_SPEED < ENTER_LOCO_SPEED):
            raise RuntimeError("router threshold invariant failed")
        if BRAKE_STEPS != int(round(BRAKE_S / EXPECTED_DT)):
            raise RuntimeError("BRAKE 0.36 s must equal 18 control steps")
        self.state = RouterState.STAND
        self._brake_followup_count = 0

    def update(self, requested_cmd: tuple[float, float, float]) -> RouterOutput:
        vx, vy, wz = (float(v) for v in requested_cmd)
        speed = abs(vx)  # Frozen calibration is based on |vx| only.

        if self.state is RouterState.STAND:
            if speed >= ENTER_LOCO_SPEED:
                self.state = RouterState.LOCO

        elif self.state is RouterState.LOCO:
            if speed <= EXIT_LOCO_SPEED:
                self.state = RouterState.BRAKE
                self._brake_followup_count = 0

        elif self.state is RouterState.BRAKE:
            if speed >= ENTER_LOCO_SPEED:
                self.state = RouterState.LOCO
                self._brake_followup_count = 0
            else:
                self._brake_followup_count += 1
                if self._brake_followup_count >= BRAKE_STEPS:
                    self.state = RouterState.STAND
                    self._brake_followup_count = 0
        else:
            raise RuntimeError(f"unknown router state: {self.state}")

        if self.state is RouterState.STAND:
            return RouterOutput(self.state, "v2-G", ZERO_CMD, 0)
        if self.state is RouterState.LOCO:
            return RouterOutput(self.state, "v4-N", (vx, vy, wz), 0)
        return RouterOutput(
            self.state,
            "v4-N",
            ZERO_CMD,
            self._brake_followup_count + 1,
        )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def fail(message: str) -> None:
    raise RuntimeError(message)


def as_tensor(value: Any) -> torch.Tensor:
    if hasattr(value, "torch"):
        value = value.torch
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    return value


def extract_policy_observation_tensor(obs: Any) -> torch.Tensor:
    candidate = None
    if torch.is_tensor(obs):
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
        fail(f"cannot resolve policy observation from {type(obs).__name__}")
    tensor = as_tensor(candidate)
    if tensor.ndim != 2 or tensor.shape[0] != 1:
        fail(f"unexpected policy observation shape: {tuple(tensor.shape)}")
    return tensor


def requested_command_for_step(step: int) -> tuple[tuple[float, float, float], str]:
    if 0 <= step < ZERO_BEFORE_STEPS:
        return ZERO_CMD, "ZERO_BEFORE"
    if ZERO_BEFORE_STEPS <= step < ZERO_BEFORE_STEPS + CURVE_NEG_STEPS:
        return CURVE_NEG_CMD, "CURVE_NEG"
    if ZERO_BEFORE_STEPS + CURVE_NEG_STEPS <= step < TOTAL_STEPS:
        return ZERO_CMD, "ZERO_AFTER"
    raise IndexError(step)


def write_command(base_env: Any, cmd: tuple[float, float, float]) -> None:
    buf = base_env.command_manager.get_command("base_velocity")
    with torch.no_grad():
        buf[:, 0] = float(cmd[0])
        buf[:, 1] = float(cmd[1])
        buf[:, 2] = float(cmd[2])


def cpu_row(tensor: Any) -> torch.Tensor:
    return as_tensor(tensor)[0].detach().to(device="cpu").clone()


def main() -> None:
    out_dir = Path(args_cli.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = (
        Path(args_cli.trajectory_path).resolve()
        if args_cli.trajectory_path is not None
        else out_dir / "V03_master_trajectory_900steps_50hz.pt"
    )
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    report_json_path = out_dir / "V03_step_capture_summary.json"
    report_txt_path = out_dir / "V03_step_capture_summary.txt"

    if not V2G_CHECKPOINT.is_file():
        fail(f"v2-G checkpoint missing: {V2G_CHECKPOINT}")
    if not V4N_CHECKPOINT.is_file():
        fail(f"v4-N checkpoint missing: {V4N_CHECKPOINT}")
    v2g_actual_sha = sha256_file(V2G_CHECKPOINT)
    v4n_actual_sha = sha256_file(V4N_CHECKPOINT)
    if v2g_actual_sha != V2G_SHA256:
        fail(f"v2-G SHA mismatch: {v2g_actual_sha}")
    if v4n_actual_sha != V4N_SHA256:
        fail(f"v4-N SHA mismatch: {v4n_actual_sha}")

    installed_rsl = metadata.version("rsl-rl-lib")

    env_cfg = parse_env_cfg(TASK_ID, device=args_cli.device, num_envs=1)
    env_cfg.scene.num_envs = 1
    env_cfg.seed = int(args_cli.seed)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # Preserve the already-frozen deployment startup/physics contract.
    env_cfg.observations.policy.enable_corruption = False
    if hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None
    if hasattr(env_cfg.events, "base_external_force_torque"):
        env_cfg.events.base_external_force_torque = None
    if hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None

    reset_base = getattr(env_cfg.events, "reset_base", None)
    if reset_base is not None:
        pose_range = reset_base.params.get("pose_range", {})
        velocity_range = reset_base.params.get("velocity_range", {})
        for key in list(pose_range.keys()):
            pose_range[key] = (0.0, 0.0)
        for key in list(velocity_range.keys()):
            velocity_range[key] = (0.0, 0.0)

    cmd_cfg = env_cfg.commands.base_velocity
    cmd_cfg.resampling_time_range = (1.0e9, 1.0e9)
    cmd_cfg.rel_standing_envs = 0.0
    cmd_cfg.ranges.lin_vel_x = (0.0, 0.0)
    cmd_cfg.ranges.lin_vel_y = (0.0, 0.0)
    cmd_cfg.ranges.ang_vel_z = (0.0, 0.0)
    if hasattr(cmd_cfg, "heading_command"):
        cmd_cfg.heading_command = False
    if hasattr(cmd_cfg, "debug_vis"):
        cmd_cfg.debug_vis = False

    v2g_cfg = UprightYawDirectWideYawFlatOrientationPPORunnerCfg()
    v4n_cfg = V4NSymmetryAugmentationPPORunnerCfg()
    if args_cli.device is not None:
        v2g_cfg.device = args_cli.device
        v4n_cfg.device = args_cli.device
    if hasattr(v2g_cfg, "seed"):
        v2g_cfg.seed = int(args_cli.seed)
    if hasattr(v4n_cfg, "seed"):
        v4n_cfg.seed = int(args_cli.seed)

    v2g_cfg = handle_deprecated_rsl_rl_cfg(v2g_cfg, installed_rsl)
    v4n_cfg = handle_deprecated_rsl_rl_cfg(v4n_cfg, installed_rsl)
    if v2g_cfg.class_name != "OnPolicyRunner" or v4n_cfg.class_name != "OnPolicyRunner":
        fail("both policies must use OnPolicyRunner")
    if v2g_cfg.clip_actions != v4n_cfg.clip_actions:
        fail("v2-G/v4-N clip_actions mismatch")

    env = None
    try:
        raw_env = gym.make(TASK_ID, cfg=env_cfg)
        env = RslRlVecEnvWrapper(raw_env, clip_actions=v2g_cfg.clip_actions)
        base_env = env.unwrapped
        robot = base_env.scene["robot"]

        dt = float(base_env.step_dt)
        if not math.isclose(dt, EXPECTED_DT, rel_tol=0.0, abs_tol=1e-12):
            fail(f"unexpected dt={dt}")
        if int(base_env.action_manager.total_action_dim) != EXPECTED_ACTION_DIM:
            fail(f"unexpected action dim={base_env.action_manager.total_action_dim}")

        # Same command-manager startup behavior as the final deployment runner.
        env_ids = torch.tensor([0], dtype=torch.int32, device=base_env.device)
        base_env.command_manager.reset(env_ids=env_ids)
        base_env.command_manager.get_command("base_velocity").zero_()

        v2g_runner = OnPolicyRunner(
            env,
            v2g_cfg.to_dict(),
            log_dir=None,
            device=v2g_cfg.device,
        )
        v2g_runner.load(str(V2G_CHECKPOINT))
        v2g_policy = v2g_runner.get_inference_policy(device=base_env.device)

        v4n_runner = OnPolicyRunner(
            env,
            v4n_cfg.to_dict(),
            log_dir=None,
            device=v4n_cfg.device,
        )
        v4n_runner.load(str(V4N_CHECKPOINT))
        v4n_policy = v4n_runner.get_inference_policy(device=base_env.device)

        obs = env.get_observations()
        policy_obs = extract_policy_observation_tensor(obs)
        if int(policy_obs.shape[1]) != EXPECTED_OBS_DIM:
            fail(f"unexpected observation dimension={policy_obs.shape[1]}")

        root_lin = cpu_row(robot.data.root_lin_vel_w)
        root_ang = cpu_row(robot.data.root_ang_vel_w)
        if float(torch.max(torch.abs(root_lin)).item()) > 1.0e-6:
            fail(f"startup root linear velocity is nonzero: {root_lin.tolist()}")
        if float(torch.max(torch.abs(root_ang)).item()) > 1.0e-6:
            fail(f"startup root angular velocity is nonzero: {root_ang.tolist()}")

        # Deployment-only optimization from the validated final runner. It changes no policy/physics state.
        reward_manager = base_env.reward_manager
        reward_buf = getattr(reward_manager, "_reward_buf", None)
        if reward_buf is None or not torch.is_tensor(reward_buf):
            fail("reward_manager._reward_buf unavailable")

        def reward_bypass(*_args, **_kwargs):
            reward_buf.zero_()
            return reward_buf

        reward_manager.compute = reward_bypass

        router = Router()
        previous_state = router.state
        transitions: list[tuple[str, str, int, float]] = []
        completed_brake_lengths: list[int] = []
        active_brake_len = 0

        root_pos_rows: list[torch.Tensor] = []
        root_quat_rows: list[torch.Tensor] = []
        root_lin_vel_rows: list[torch.Tensor] = []
        root_ang_vel_rows: list[torch.Tensor] = []
        joint_pos_rows: list[torch.Tensor] = []
        joint_vel_rows: list[torch.Tensor] = []
        action_rows: list[torch.Tensor] = []
        requested_cmd_rows: list[list[float]] = []
        policy_cmd_rows: list[list[float]] = []
        router_state_rows: list[int] = []
        active_policy_rows: list[int] = []
        brake_index_rows: list[int] = []
        phase_rows: list[str] = []

        print("=" * 120)
        print("G1 V03 DUAL-POLICY DETERMINISTIC STEP CAPTURE")
        print("=" * 120)
        print(f"task                    : {TASK_ID}")
        print(f"seed                    : {args_cli.seed}")
        print(f"dt                      : {dt:.5f} s / 50 Hz")
        print(f"schedule                : {ZERO_BEFORE_STEPS} ZERO + {CURVE_NEG_STEPS} CURVE_NEG + {ZERO_AFTER_STEPS} ZERO")
        print(f"total                   : {TOTAL_STEPS} steps = {EXPECTED_DURATION_S:.2f} s")
        print("pacing                  : NONE — simulation-step indexed, wall-clock irrelevant")
        print("ROS                     : ABSENT — deployment already validated separately")
        print(f"output                  : {trajectory_path}")
        print("=" * 120)

        for step in range(TOTAL_STEPS):
            requested_cmd, phase = requested_command_for_step(step)
            old_state = router.state
            route = router.update(requested_cmd)

            if route.state is RouterState.BRAKE:
                active_brake_len += 1
            elif old_state is RouterState.BRAKE and route.state is RouterState.STAND:
                completed_brake_lengths.append(active_brake_len)
                active_brake_len = 0

            if route.state is not old_state:
                transitions.append((old_state.name, route.state.name, step, step * dt))
                print(
                    f"[ROUTER] step={step:04d} t={step * dt:7.3f}s "
                    f"{old_state.name}->{route.state.name} request={requested_cmd} "
                    f"policy={route.active_policy} brake_index={route.brake_index}",
                    flush=True,
                )

            if route.state is RouterState.STAND:
                if route.active_policy != "v2-G" or route.policy_command != ZERO_CMD:
                    fail(f"STAND invariant failed at step={step}")
            elif route.state is RouterState.LOCO:
                if route.active_policy != "v4-N" or route.policy_command != requested_cmd:
                    fail(f"LOCO invariant failed at step={step}")
            elif route.state is RouterState.BRAKE:
                if route.active_policy != "v4-N" or route.policy_command != ZERO_CMD:
                    fail(f"BRAKE invariant failed at step={step}")

            write_command(base_env, route.policy_command)
            obs = env.get_observations()
            active_policy = v2g_policy if route.active_policy == "v2-G" else v4n_policy

            with torch.inference_mode():
                actions = active_policy(obs)
            if not bool(torch.isfinite(actions).all().item()):
                fail(f"non-finite actions at step={step}")

            with torch.inference_mode():
                obs, _, dones, _ = env.step(actions)

            if bool(torch.any(dones).item()):
                fail(f"environment termination/reset at step={step}")

            # Post-physics state for exactly this control step. Wall-clock cost is intentionally irrelevant.
            root_pos_rows.append(cpu_row(robot.data.root_pos_w))
            root_quat_rows.append(cpu_row(robot.data.root_quat_w))
            root_lin_vel_rows.append(cpu_row(robot.data.root_lin_vel_w))
            root_ang_vel_rows.append(cpu_row(robot.data.root_ang_vel_w))
            joint_pos_rows.append(cpu_row(robot.data.joint_pos))
            joint_vel_rows.append(cpu_row(robot.data.joint_vel))
            action_rows.append(cpu_row(actions))
            requested_cmd_rows.append([float(v) for v in requested_cmd])
            policy_cmd_rows.append([float(v) for v in route.policy_command])
            router_state_rows.append({RouterState.STAND: 0, RouterState.LOCO: 1, RouterState.BRAKE: 2}[route.state])
            active_policy_rows.append(0 if route.active_policy == "v2-G" else 1)
            brake_index_rows.append(int(route.brake_index))
            phase_rows.append(phase)
            previous_state = route.state

        observed_transitions = [(a, b) for a, b, _, _ in transitions]
        expected_transitions = [("STAND", "LOCO"), ("LOCO", "BRAKE"), ("BRAKE", "STAND")]
        if observed_transitions != expected_transitions:
            fail(f"unexpected transitions: {observed_transitions}")
        if completed_brake_lengths != [BRAKE_STEPS]:
            fail(f"unexpected completed BRAKE lengths: {completed_brake_lengths}")
        if router.state is not RouterState.STAND:
            fail(f"final router state is {router.state.name}, expected STAND")

        phase_counts = {name: phase_rows.count(name) for name in ("ZERO_BEFORE", "CURVE_NEG", "ZERO_AFTER")}
        if phase_counts != {
            "ZERO_BEFORE": ZERO_BEFORE_STEPS,
            "CURVE_NEG": CURVE_NEG_STEPS,
            "ZERO_AFTER": ZERO_AFTER_STEPS,
        }:
            fail(f"phase counts mismatch: {phase_counts}")

        state_counts = {
            "STAND": router_state_rows.count(0),
            "LOCO": router_state_rows.count(1),
            "BRAKE": router_state_rows.count(2),
        }
        if state_counts != {"STAND": 282, "LOCO": 600, "BRAKE": 18}:
            fail(f"router-state counts mismatch: {state_counts}")

        trajectory = {
            "format": FORMAT_ID,
            "classification": "deterministic engineering curve-negative video master trajectory; NOT ROS deployment evidence",
            "sample_semantics": "post-physics state after control step n; command/router metadata correspond to the action applied during that step",
            "task_id": TASK_ID,
            "seed": int(args_cli.seed),
            "dt_s": EXPECTED_DT,
            "fps_master": 50,
            "num_frames": TOTAL_STEPS,
            "duration_s": EXPECTED_DURATION_S,
            "schedule": {
                "zero_before_steps": ZERO_BEFORE_STEPS,
                "curve_neg_steps": CURVE_NEG_STEPS,
                "zero_after_steps": ZERO_AFTER_STEPS,
                "zero_before_s": ZERO_BEFORE_STEPS * EXPECTED_DT,
                "curve_neg_s": CURVE_NEG_STEPS * EXPECTED_DT,
                "zero_after_s": ZERO_AFTER_STEPS * EXPECTED_DT,
            },
            "controller": {
                "v2g_checkpoint": str(V2G_CHECKPOINT),
                "v2g_sha256": v2g_actual_sha,
                "v4n_checkpoint": str(V4N_CHECKPOINT),
                "v4n_sha256": v4n_actual_sha,
                "exit_loco_speed_mps": EXIT_LOCO_SPEED,
                "enter_loco_speed_mps": ENTER_LOCO_SPEED,
                "brake_s": BRAKE_S,
                "brake_steps": BRAKE_STEPS,
                "blending": False,
            },
            "joint_names": list(robot.joint_names),
            "root_pos_w": torch.stack(root_pos_rows, dim=0),
            "root_quat_w_wxyz": torch.stack(root_quat_rows, dim=0),
            "root_lin_vel_w": torch.stack(root_lin_vel_rows, dim=0),
            "root_ang_vel_w": torch.stack(root_ang_vel_rows, dim=0),
            "joint_pos": torch.stack(joint_pos_rows, dim=0),
            "joint_vel": torch.stack(joint_vel_rows, dim=0),
            "actions": torch.stack(action_rows, dim=0),
            "requested_cmd": torch.tensor(requested_cmd_rows, dtype=torch.float32),
            "policy_cmd": torch.tensor(policy_cmd_rows, dtype=torch.float32),
            "router_state_code": torch.tensor(router_state_rows, dtype=torch.int8),
            "active_policy_code": torch.tensor(active_policy_rows, dtype=torch.int8),
            "brake_index": torch.tensor(brake_index_rows, dtype=torch.int16),
            "phase": phase_rows,
            "sim_time_s": torch.arange(1, TOTAL_STEPS + 1, dtype=torch.float64) * EXPECTED_DT,
            "transitions": transitions,
            "completed_brake_lengths": completed_brake_lengths,
        }

        shape_pass = (
            trajectory["root_pos_w"].shape == (TOTAL_STEPS, 3)
            and trajectory["root_quat_w_wxyz"].shape == (TOTAL_STEPS, 4)
            and trajectory["joint_pos"].shape == (TOTAL_STEPS, EXPECTED_ACTION_DIM)
            and trajectory["actions"].shape == (TOTAL_STEPS, EXPECTED_ACTION_DIM)
            and trajectory["requested_cmd"].shape == (TOTAL_STEPS, 3)
        )
        if not shape_pass:
            fail("trajectory shape guardrail failed")

        torch.save(trajectory, trajectory_path)
        trajectory_sha = sha256_file(trajectory_path)
        runner_sha = sha256_file(Path(__file__).resolve())

        summary = {
            "status": "PASS",
            "format": FORMAT_ID,
            "classification": trajectory["classification"],
            "task_id": TASK_ID,
            "seed": int(args_cli.seed),
            "frames": TOTAL_STEPS,
            "fps": 50,
            "duration_s": EXPECTED_DURATION_S,
            "phase_counts": phase_counts,
            "router_state_counts": state_counts,
            "router_transitions": transitions,
            "completed_brake_lengths": completed_brake_lengths,
            "final_router_state": router.state.name,
            "environment_resets": 0,
            "trajectory_shape_pass": bool(shape_pass),
            "trajectory_path": str(trajectory_path),
            "trajectory_sha256": trajectory_sha,
            "runner_sha256": runner_sha,
            "v2g_sha256": v2g_actual_sha,
            "v4n_sha256": v4n_actual_sha,
            "wall_clock_timing": "NOT_APPLICABLE / intentionally unpaced",
            "ros": "ABSENT / deployment evidence validated separately",
        }
        report_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        txt_lines = [
            "G1 V03 DUAL-POLICY DETERMINISTIC STEP CAPTURE",
            "=" * 120,
            "Status                              : PASS",
            f"Classification                      : {summary['classification']}",
            f"Frames / FPS / duration             : {TOTAL_STEPS} / 50 / {EXPECTED_DURATION_S:.2f} s",
            f"Phase counts                        : {phase_counts}",
            f"Router state counts                 : {state_counts}",
            f"Router transitions                  : {transitions}",
            f"Completed BRAKE lengths             : {completed_brake_lengths}",
            f"Final router state                  : {router.state.name}",
            "Environment resets                  : 0",
            f"Trajectory shape PASS               : {shape_pass}",
            f"Trajectory path                     : {trajectory_path}",
            f"Trajectory SHA256                   : {trajectory_sha}",
            f"Runner SHA256                       : {runner_sha}",
            f"v2-G / v4-N SHA256                  : {v2g_actual_sha} / {v4n_actual_sha}",
            "Wall-clock timing                   : NOT_APPLICABLE / intentionally unpaced",
            "ROS                                 : ABSENT / deployment evidence validated separately",
            "Repository source modified          : NO",
            "Checkpoint modified                 : NO",
            PASS_MARKER,
        ]
        report_txt_path.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")

        print()
        print("=" * 120)
        print("V03 STEP-CAPTURE SUMMARY")
        print("=" * 120)
        for line in txt_lines[2:]:
            print(line)

    finally:
        if env is not None:
            env.close()


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
