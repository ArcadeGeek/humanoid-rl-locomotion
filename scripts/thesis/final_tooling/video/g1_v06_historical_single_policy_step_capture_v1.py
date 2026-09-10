#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import traceback
from importlib import metadata
from pathlib import Path
from typing import Any

parser = argparse.ArgumentParser(
    description=(
        "Deterministic single-policy step capture for V06 historical presentation clips. "
        "Each selected historical checkpoint is run with the same short presentation schedule: "
        "1.0 s ZERO + 3.0 s [0.40,0,0], 200 control steps at 50 Hz. "
        "This is presentation-only historical material, NOT new scientific evidence."
    )
)
parser.add_argument(
    "--case",
    required=True,
    choices=("V2-A", "V2-D", "V2-E", "V2-F", "V2-G", "V3-C", "V4-A", "V4-B", "V4-N"),
)
parser.add_argument(
    "--output-root",
    type=Path,
    default=Path(r"C:\humanoid_rl_lab\logs\thesis\video_final\V06_historical_progression\capture"),
)
parser.add_argument("--seed", type=int, default=42)

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import humanoid_rl_lab.tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner


REPO = Path(r"C:\humanoid_rl_lab")

CASES = {
    "V2-A": {
        "role": "direct-yaw genuine-scratch baseline",
        "task": "Humanoid-RL-Lab-G1-Flat-Upright-Yaw-Direct-v0",
        "checkpoint": REPO / "logs" / "rsl_rl" / "humanoid_rl_lab_g1_upright_yaw_direct"
            / "2026-08-15_03-33-18_thesis_yaw_direct_genuine_scratch_v2a_to700" / "model_600.pt",
        "sha256": "F1340C5E5CB44AE3157CC9C4D4493BBD1BC6316ED64A029FBD514C7FC0F7B413",
    },
    "V2-D": {
        "role": "pseudo-Huber negative result",
        "task": "Humanoid-RL-Lab-G1-Flat-Upright-Yaw-Direct-Pseudo-Huber-Yaw-v0",
        "checkpoint": REPO / "logs" / "rsl_rl" / "humanoid_rl_lab_g1_upright_yaw_direct_pseudo_huber_yaw"
            / "2026-08-15_17-13-57_thesis_yaw_direct_pseudo_huber_v2d_genuine_scratch_to700" / "model_600.pt",
        "sha256": "E298A2FAA3959421C5AE00CB3908F84EA34C36CB4D6126194921FFD06AC356FD",
    },
    "V2-E": {
        "role": "heavy-tail shaping, stable but insufficient",
        "task": "Humanoid-RL-Lab-G1-Flat-Upright-Yaw-Direct-Heavy-Tail-Yaw-v0",
        "checkpoint": REPO / "logs" / "rsl_rl" / "humanoid_rl_lab_g1_upright_yaw_direct_heavy_tail_yaw"
            / "2026-08-16_20-06-45_thesis_yaw_direct_heavy_tail_v2e_genuine_scratch_to700" / "model_600.pt",
        "sha256": "154AB9548FA829C59764AFA772D6897BD0F29104729131515A8A616F4879645F",
    },
    "V2-F": {
        "role": "wide-yaw training breakthrough",
        "task": "Humanoid-RL-Lab-G1-Flat-Upright-Yaw-Direct-Wide-Yaw-v0",
        "checkpoint": REPO / "logs" / "rsl_rl" / "humanoid_rl_lab_g1_upright_yaw_direct_wide_yaw"
            / "2026-08-16_22-19-58_thesis_yaw_direct_wide_yaw_v2f_genuine_scratch_to700" / "model_600.pt",
        "sha256": "F650608C94B50F1BD77D64F975B1F0B87033510B74A159CB3A56860DE4271EE7",
    },
    "V2-G": {
        "role": "flat-orientation refinement / frozen STAND controller",
        "task": "Humanoid-RL-Lab-G1-Flat-Upright-Yaw-Direct-Wide-Yaw-Flat-Orientation-v0",
        "checkpoint": REPO / "checkpoints" / "local_g1_yaw_direct_wide_yaw_flat_orientation_v2g" / "model_600.pt",
        "sha256": "58468EED03F9EF35A9BC898063A3B14AC36C93A32F65DB42B404BDE93CD694D3",
    },
    "V3-C": {
        "role": "exploratory swing-clearance refinement / NOT promoted",
        "task": "Humanoid-RL-Lab-G1-Flat-Upright-Yaw-Direct-Wide-Yaw-Flat-Orientation-V3C-v0",
        "checkpoint": REPO / "logs" / "rsl_rl" / "humanoid_rl_lab_g1_v3c_swing_clearance_refinement"
            / "2026-08-26_12-24-10_v3c_from_v2g_model600_30it" / "model_620.pt",
        "sha256": "C9D93ADE8F32D8F52D1583F4596EAB8906377802265B975E66719C5B41F08458",
    },
    "V4-A": {
        "role": "official G1 flat genuine-scratch reproduction",
        "task": "Humanoid-RL-Lab-G1-Flat-V4A-Official-Baseline-v0",
        "checkpoint": REPO / "logs" / "rsl_rl" / "humanoid_rl_lab_g1_v4a_official_baseline_scratch"
            / "2026-09-01_14-49-13_v4a_genuine_scratch_seed42_official1500" / "model_1499.pt",
        "sha256": "9729904F7766217D3466F4B1935BC769E4B4AAD8872982E70C1CE3AA35EDFB60",
    },
    "V4-B": {
        "role": "direct-yaw causal scratch baseline",
        "task": "Humanoid-RL-Lab-G1-Flat-V4B-Direct-Yaw-v0",
        "checkpoint": REPO / "logs" / "rsl_rl" / "humanoid_rl_lab_g1_v4b_direct_yaw_scratch"
            / "2026-09-01_20-08-18_v4b_genuine_scratch_seed42_official1500" / "model_1499.pt",
        "sha256": "B6B9E94DC222F21AF698BA158B9E91C18071FE90ABC255B9C4311874E106EC7F",
    },
    "V4-N": {
        "role": "symmetry augmentation / final LOCOMOTION controller",
        "task": "Humanoid-RL-Lab-G1-Flat-V4N-Symmetry-Augmentation-v0",
        "checkpoint": REPO / "logs" / "rsl_rl" / "humanoid_rl_lab_g1_v4n_symmetry_augmentation_scratch"
            / "2026-09-04_13-58-13_v4n_genuine_scratch_seed42_official1500" / "model_1499.pt",
        "sha256": "367D318AB4940A3AAE8F0FA568384212762258067DFC4CF9F624FA472BEB9342",
    },
}

EXPECTED_DT = 0.02
EXPECTED_OBS_DIM = 123
EXPECTED_ACTION_DIM = 37
FPS = 50

ZERO_STEPS = 50
STRAIGHT_STEPS = 150
TOTAL_STEPS = ZERO_STEPS + STRAIGHT_STEPS
DURATION_S = TOTAL_STEPS * EXPECTED_DT

ZERO_CMD = (0.0, 0.0, 0.0)
STRAIGHT_CMD = (0.40, 0.0, 0.0)

FORMAT_ID = "G1_HISTORICAL_SINGLE_POLICY_VISUAL_TRAJECTORY_V1"
PASS_MARKER = "G1_V06_HISTORICAL_SINGLE_POLICY_STEP_CAPTURE_V1_PASS"
FAIL_MARKER = "G1_V06_HISTORICAL_SINGLE_POLICY_STEP_CAPTURE_V1_FAIL"


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


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
    if tensor.ndim != 2 or int(tensor.shape[0]) != 1:
        fail(f"unexpected policy observation shape: {tuple(tensor.shape)}")
    return tensor


def cpu_row(tensor: Any) -> torch.Tensor:
    return as_tensor(tensor)[0].detach().to(device="cpu").clone()


def write_command(base_env: Any, cmd: tuple[float, float, float]) -> None:
    buf = base_env.command_manager.get_command("base_velocity")
    with torch.no_grad():
        buf.zero_()
        buf[:, 0] = float(cmd[0])
        buf[:, 1] = float(cmd[1])
        buf[:, 2] = float(cmd[2])


def command_for_step(step: int) -> tuple[tuple[float, float, float], str]:
    if 0 <= step < ZERO_STEPS:
        return ZERO_CMD, "ZERO"
    if ZERO_STEPS <= step < TOTAL_STEPS:
        return STRAIGHT_CMD, "STRAIGHT"
    raise IndexError(step)


def normalize_environment_for_presentation(env_cfg: Any) -> None:
    env_cfg.scene.num_envs = 1
    env_cfg.seed = int(args_cli.seed)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    if hasattr(env_cfg.observations, "policy"):
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
    if hasattr(cmd_cfg, "rel_heading_envs"):
        cmd_cfg.rel_heading_envs = 0.0
    if hasattr(cmd_cfg, "debug_vis"):
        cmd_cfg.debug_vis = False


def main() -> None:
    case = CASES[args_cli.case]
    checkpoint: Path = case["checkpoint"]
    task_id: str = case["task"]

    if not checkpoint.is_file():
        fail(f"checkpoint missing: {checkpoint}")
    actual_sha = sha256_file(checkpoint)
    if actual_sha != case["sha256"]:
        fail(f"{args_cli.case}: SHA mismatch: expected {case['sha256']}, got {actual_sha}")

    case_slug = args_cli.case.replace("-", "")
    out_dir = Path(args_cli.output_root).resolve() / case_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = out_dir / f"{case_slug}_master_trajectory_{TOTAL_STEPS}steps_50hz.pt"
    summary_json = out_dir / f"{case_slug}_step_capture_summary.json"
    summary_txt = out_dir / f"{case_slug}_step_capture_summary.txt"

    installed_rsl = metadata.version("rsl-rl-lib")
    env_cfg = parse_env_cfg(task_id, device=args_cli.device, num_envs=1)
    normalize_environment_for_presentation(env_cfg)

    agent_cfg = load_cfg_from_registry(task_id, "rsl_rl_cfg_entry_point")
    if args_cli.device is not None and hasattr(agent_cfg, "device"):
        agent_cfg.device = args_cli.device
    if hasattr(agent_cfg, "seed"):
        agent_cfg.seed = int(args_cli.seed)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_rsl)

    if agent_cfg.class_name != "OnPolicyRunner":
        fail(f"unexpected runner class: {agent_cfg.class_name}")

    env = None
    try:
        raw_env = gym.make(task_id, cfg=env_cfg)
        env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
        base_env = env.unwrapped
        robot = base_env.scene["robot"]

        dt = float(base_env.step_dt)
        action_dim = int(base_env.action_manager.total_action_dim)
        if not math.isclose(dt, EXPECTED_DT, rel_tol=0.0, abs_tol=1.0e-12):
            fail(f"unexpected dt={dt}")
        if action_dim != EXPECTED_ACTION_DIM:
            fail(f"unexpected action_dim={action_dim}")

        env_ids = torch.tensor([0], dtype=torch.int32, device=base_env.device)
        base_env.command_manager.reset(env_ids=env_ids)
        write_command(base_env, ZERO_CMD)

        runner = OnPolicyRunner(
            env,
            agent_cfg.to_dict(),
            log_dir=None,
            device=agent_cfg.device,
        )
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=base_env.device)

        obs = env.get_observations()
        policy_obs = extract_policy_observation_tensor(obs)
        if int(policy_obs.shape[1]) != EXPECTED_OBS_DIM:
            fail(f"unexpected obs_dim={policy_obs.shape[1]}")

        root_lin0 = cpu_row(robot.data.root_lin_vel_w)
        root_ang0 = cpu_row(robot.data.root_ang_vel_w)
        if float(torch.max(torch.abs(root_lin0)).item()) > 1.0e-6:
            fail(f"startup root linear velocity nonzero: {root_lin0.tolist()}")
        if float(torch.max(torch.abs(root_ang0)).item()) > 1.0e-6:
            fail(f"startup root angular velocity nonzero: {root_ang0.tolist()}")

        # Same safe performance optimization already validated in final video capture.
        reward_manager = base_env.reward_manager
        reward_buf = getattr(reward_manager, "_reward_buf", None)
        if reward_buf is not None and torch.is_tensor(reward_buf):
            def reward_bypass(*_args, **_kwargs):
                reward_buf.zero_()
                return reward_buf
            reward_manager.compute = reward_bypass

        root_pos_rows = []
        root_quat_rows = []
        root_lin_vel_rows = []
        root_ang_vel_rows = []
        joint_pos_rows = []
        joint_vel_rows = []
        action_rows = []
        command_rows = []
        phase_rows = []

        print("=" * 124)
        print("G1 V06 HISTORICAL SINGLE-POLICY DETERMINISTIC STEP CAPTURE")
        print("=" * 124)
        print(f"case                    : {args_cli.case}")
        print(f"role                    : {case['role']}")
        print(f"task                    : {task_id}")
        print(f"checkpoint              : {checkpoint}")
        print(f"checkpoint SHA256       : {actual_sha}")
        print(f"seed                    : {args_cli.seed}")
        print(f"dt                      : {dt:.5f} s / 50 Hz")
        print(f"schedule                : {ZERO_STEPS} ZERO + {STRAIGHT_STEPS} STRAIGHT")
        print(f"command                 : {STRAIGHT_CMD}")
        print(f"total                   : {TOTAL_STEPS} steps = {DURATION_S:.2f} s")
        print("classification          : presentation-only historical trajectory; NOT new scientific evidence")
        print("pacing                  : NONE — simulation-step indexed")
        print(f"output                  : {trajectory_path}")
        print("=" * 124)

        initial_pos = None
        max_abs_action = 0.0

        for step in range(TOTAL_STEPS):
            cmd, phase = command_for_step(step)
            write_command(base_env, cmd)
            obs = env.get_observations()

            with torch.inference_mode():
                actions = policy(obs)

            actions_t = as_tensor(actions)
            if tuple(actions_t.shape) != (1, EXPECTED_ACTION_DIM):
                fail(f"unexpected action shape at step={step}: {tuple(actions_t.shape)}")
            if not bool(torch.isfinite(actions_t).all().item()):
                fail(f"non-finite actions at step={step}")

            max_abs_action = max(
                max_abs_action,
                float(torch.max(torch.abs(actions_t)).item()),
            )

            with torch.inference_mode():
                obs, _, dones, _ = env.step(actions)

            if bool(torch.any(as_tensor(dones)).item()):
                fail(
                    f"environment termination/reset at step={step}; "
                    "do not silently include resets in a historical presentation clip"
                )

            root_pos = cpu_row(robot.data.root_pos_w)
            if initial_pos is None:
                initial_pos = root_pos.clone()

            root_pos_rows.append(root_pos)
            root_quat_rows.append(cpu_row(robot.data.root_quat_w))
            root_lin_vel_rows.append(cpu_row(robot.data.root_lin_vel_w))
            root_ang_vel_rows.append(cpu_row(robot.data.root_ang_vel_w))
            joint_pos_rows.append(cpu_row(robot.data.joint_pos))
            joint_vel_rows.append(cpu_row(robot.data.joint_vel))
            action_rows.append(cpu_row(actions_t))
            command_rows.append([float(v) for v in cmd])
            phase_rows.append(phase)

            if step in {0, ZERO_STEPS - 1, ZERO_STEPS, TOTAL_STEPS - 1}:
                print(
                    f"[CAPTURE] step={step:03d}/{TOTAL_STEPS-1:03d} "
                    f"t={(step+1)*dt:5.2f}s phase={phase:8s} "
                    f"root=({root_pos[0]:+.3f},{root_pos[1]:+.3f},{root_pos[2]:+.3f}) "
                    f"max|a|={max_abs_action:.4f}",
                    flush=True,
                )

        if initial_pos is None:
            fail("no samples captured")

        final_pos = root_pos_rows[-1]
        planar_displacement = math.hypot(
            float(final_pos[0] - initial_pos[0]),
            float(final_pos[1] - initial_pos[1]),
        )

        trajectory = {
            "format": FORMAT_ID,
            "classification": (
                "deterministic presentation-only historical single-policy video master trajectory; "
                "NOT new scientific evidence and NOT ROS deployment evidence"
            ),
            "sample_semantics": (
                "post-physics state after control step n; command metadata corresponds "
                "to the action applied during that step"
            ),
            "case": args_cli.case,
            "role": case["role"],
            "task_id": task_id,
            "seed": int(args_cli.seed),
            "dt_s": EXPECTED_DT,
            "fps_master": FPS,
            "num_frames": TOTAL_STEPS,
            "duration_s": DURATION_S,
            "schedule": {
                "zero_steps": ZERO_STEPS,
                "straight_steps": STRAIGHT_STEPS,
                "zero_s": ZERO_STEPS * EXPECTED_DT,
                "straight_s": STRAIGHT_STEPS * EXPECTED_DT,
            },
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": actual_sha,
            "command_straight": list(STRAIGHT_CMD),
            "joint_names": list(robot.joint_names),
            "root_pos_w": torch.stack(root_pos_rows, dim=0),
            "root_quat_w_wxyz": torch.stack(root_quat_rows, dim=0),
            "root_lin_vel_w": torch.stack(root_lin_vel_rows, dim=0),
            "root_ang_vel_w": torch.stack(root_ang_vel_rows, dim=0),
            "joint_pos": torch.stack(joint_pos_rows, dim=0),
            "joint_vel": torch.stack(joint_vel_rows, dim=0),
            "actions": torch.stack(action_rows, dim=0),
            "policy_cmd": torch.tensor(command_rows, dtype=torch.float32),
            "phase": phase_rows,
            "sim_time_s": torch.arange(1, TOTAL_STEPS + 1, dtype=torch.float64) * EXPECTED_DT,
        }

        shape_pass = (
            trajectory["root_pos_w"].shape == (TOTAL_STEPS, 3)
            and trajectory["root_quat_w_wxyz"].shape == (TOTAL_STEPS, 4)
            and trajectory["joint_pos"].shape == (TOTAL_STEPS, EXPECTED_ACTION_DIM)
            and trajectory["joint_vel"].shape == (TOTAL_STEPS, EXPECTED_ACTION_DIM)
            and trajectory["actions"].shape == (TOTAL_STEPS, EXPECTED_ACTION_DIM)
            and trajectory["policy_cmd"].shape == (TOTAL_STEPS, 3)
            and len(trajectory["phase"]) == TOTAL_STEPS
        )
        if not shape_pass:
            fail("trajectory shape guardrail failed")

        torch.save(trajectory, trajectory_path)
        trajectory_sha = sha256_file(trajectory_path)
        runner_sha = sha256_file(Path(__file__).resolve())

        phase_counts = {
            "ZERO": phase_rows.count("ZERO"),
            "STRAIGHT": phase_rows.count("STRAIGHT"),
        }
        if phase_counts != {"ZERO": ZERO_STEPS, "STRAIGHT": STRAIGHT_STEPS}:
            fail(f"phase counts mismatch: {phase_counts}")

        summary = {
            "status": "PASS",
            "classification": trajectory["classification"],
            "case": args_cli.case,
            "role": case["role"],
            "task_id": task_id,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": actual_sha,
            "seed": int(args_cli.seed),
            "frames": TOTAL_STEPS,
            "fps": FPS,
            "duration_s": DURATION_S,
            "phase_counts": phase_counts,
            "environment_resets": 0,
            "obs_dim": EXPECTED_OBS_DIM,
            "action_dim": EXPECTED_ACTION_DIM,
            "step_dt": dt,
            "max_abs_action": max_abs_action,
            "planar_displacement_m": planar_displacement,
            "trajectory_shape_pass": bool(shape_pass),
            "trajectory_path": str(trajectory_path),
            "trajectory_sha256": trajectory_sha,
            "runner_sha256": runner_sha,
            "repository_source_modified": False,
            "checkpoint_modified": False,
        }

        summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        lines = [
            "G1 V06 HISTORICAL SINGLE-POLICY STEP-CAPTURE SUMMARY",
            "=" * 124,
            "Status                              : PASS",
            f"Classification                      : {summary['classification']}",
            f"Case / role                         : {args_cli.case} / {case['role']}",
            f"Task                                : {task_id}",
            f"Checkpoint SHA256                   : {actual_sha}",
            f"Frames / FPS / duration             : {TOTAL_STEPS} / {FPS} / {DURATION_S:.2f} s",
            f"Phase counts                        : {phase_counts}",
            "Environment resets                  : 0",
            f"Obs / action / dt                   : {EXPECTED_OBS_DIM} / {EXPECTED_ACTION_DIM} / {dt:.5f}",
            f"Max |action|                        : {max_abs_action:.6f}",
            f"Planar displacement                 : {planar_displacement:.6f} m",
            f"Trajectory shape PASS               : {shape_pass}",
            f"Trajectory path                     : {trajectory_path}",
            f"Trajectory SHA256                   : {trajectory_sha}",
            f"Runner SHA256                       : {runner_sha}",
            "Repository source modified          : NO",
            "Checkpoint modified                 : NO",
            PASS_MARKER,
        ]
        summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

        print()
        print("=" * 124)
        print("V06 HISTORICAL SINGLE-POLICY STEP-CAPTURE SUMMARY")
        print("=" * 124)
        for line in lines[2:]:
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
