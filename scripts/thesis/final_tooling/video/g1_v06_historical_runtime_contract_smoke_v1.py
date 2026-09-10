#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import traceback
from importlib import metadata
from pathlib import Path
from typing import Any

parser = argparse.ArgumentParser(
    description=(
        "Headless runtime compatibility smoke for the nine frozen V06 historical checkpoints. "
        "Loads each checkpoint with its exact registered task/PPO contract, runs only a few "
        "deterministic PhysX control steps, and verifies the 123D observation / 37D action interface. "
        "This is NOT qualitative evaluation, NOT checkpoint promotion, and NOT new scientific evidence."
    )
)
parser.add_argument("--steps-per-case", type=int, default=3)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--output-json",
    type=Path,
    default=Path(
        r"C:\humanoid_rl_lab\logs\thesis\video_final\V06_historical_progression"
        r"\replay\V06_runtime_contract_smoke_v1.json"
    ),
)

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

CASES = [
    {
        "label": "V2-A",
        "role": "direct-yaw genuine-scratch baseline",
        "task": "Humanoid-RL-Lab-G1-Flat-Upright-Yaw-Direct-v0",
        "checkpoint": REPO / "logs" / "rsl_rl" / "humanoid_rl_lab_g1_upright_yaw_direct"
            / "2026-08-15_03-33-18_thesis_yaw_direct_genuine_scratch_v2a_to700" / "model_600.pt",
        "sha256": "F1340C5E5CB44AE3157CC9C4D4493BBD1BC6316ED64A029FBD514C7FC0F7B413",
    },
    {
        "label": "V2-D",
        "role": "pseudo-Huber negative result",
        "task": "Humanoid-RL-Lab-G1-Flat-Upright-Yaw-Direct-Pseudo-Huber-Yaw-v0",
        "checkpoint": REPO / "logs" / "rsl_rl" / "humanoid_rl_lab_g1_upright_yaw_direct_pseudo_huber_yaw"
            / "2026-08-15_17-13-57_thesis_yaw_direct_pseudo_huber_v2d_genuine_scratch_to700" / "model_600.pt",
        "sha256": "E298A2FAA3959421C5AE00CB3908F84EA34C36CB4D6126194921FFD06AC356FD",
    },
    {
        "label": "V2-E",
        "role": "heavy-tail shaping, stable but insufficient",
        "task": "Humanoid-RL-Lab-G1-Flat-Upright-Yaw-Direct-Heavy-Tail-Yaw-v0",
        "checkpoint": REPO / "logs" / "rsl_rl" / "humanoid_rl_lab_g1_upright_yaw_direct_heavy_tail_yaw"
            / "2026-08-16_20-06-45_thesis_yaw_direct_heavy_tail_v2e_genuine_scratch_to700" / "model_600.pt",
        "sha256": "154AB9548FA829C59764AFA772D6897BD0F29104729131515A8A616F4879645F",
    },
    {
        "label": "V2-F",
        "role": "wide-yaw training breakthrough",
        "task": "Humanoid-RL-Lab-G1-Flat-Upright-Yaw-Direct-Wide-Yaw-v0",
        "checkpoint": REPO / "logs" / "rsl_rl" / "humanoid_rl_lab_g1_upright_yaw_direct_wide_yaw"
            / "2026-08-16_22-19-58_thesis_yaw_direct_wide_yaw_v2f_genuine_scratch_to700" / "model_600.pt",
        "sha256": "F650608C94B50F1BD77D64F975B1F0B87033510B74A159CB3A56860DE4271EE7",
    },
    {
        "label": "V2-G",
        "role": "flat-orientation refinement / frozen STAND controller",
        "task": "Humanoid-RL-Lab-G1-Flat-Upright-Yaw-Direct-Wide-Yaw-Flat-Orientation-v0",
        "checkpoint": REPO / "checkpoints" / "local_g1_yaw_direct_wide_yaw_flat_orientation_v2g" / "model_600.pt",
        "sha256": "58468EED03F9EF35A9BC898063A3B14AC36C93A32F65DB42B404BDE93CD694D3",
    },
    {
        "label": "V3-C",
        "role": "exploratory swing-clearance refinement / NOT promoted",
        "task": "Humanoid-RL-Lab-G1-Flat-Upright-Yaw-Direct-Wide-Yaw-Flat-Orientation-V3C-v0",
        "checkpoint": REPO / "logs" / "rsl_rl" / "humanoid_rl_lab_g1_v3c_swing_clearance_refinement"
            / "2026-08-26_12-24-10_v3c_from_v2g_model600_30it" / "model_620.pt",
        "sha256": "C9D93ADE8F32D8F52D1583F4596EAB8906377802265B975E66719C5B41F08458",
    },
    {
        "label": "V4-A",
        "role": "official G1 flat genuine-scratch reproduction",
        "task": "Humanoid-RL-Lab-G1-Flat-V4A-Official-Baseline-v0",
        "checkpoint": REPO / "logs" / "rsl_rl" / "humanoid_rl_lab_g1_v4a_official_baseline_scratch"
            / "2026-09-01_14-49-13_v4a_genuine_scratch_seed42_official1500" / "model_1499.pt",
        "sha256": "9729904F7766217D3466F4B1935BC769E4B4AAD8872982E70C1CE3AA35EDFB60",
    },
    {
        "label": "V4-B",
        "role": "direct-yaw causal scratch baseline",
        "task": "Humanoid-RL-Lab-G1-Flat-V4B-Direct-Yaw-v0",
        "checkpoint": REPO / "logs" / "rsl_rl" / "humanoid_rl_lab_g1_v4b_direct_yaw_scratch"
            / "2026-09-01_20-08-18_v4b_genuine_scratch_seed42_official1500" / "model_1499.pt",
        "sha256": "B6B9E94DC222F21AF698BA158B9E91C18071FE90ABC255B9C4311874E106EC7F",
    },
    {
        "label": "V4-N",
        "role": "symmetry-augmentation / final LOCOMOTION controller",
        "task": "Humanoid-RL-Lab-G1-Flat-V4N-Symmetry-Augmentation-v0",
        "checkpoint": REPO / "logs" / "rsl_rl" / "humanoid_rl_lab_g1_v4n_symmetry_augmentation_scratch"
            / "2026-09-04_13-58-13_v4n_genuine_scratch_seed42_official1500" / "model_1499.pt",
        "sha256": "367D318AB4940A3AAE8F0FA568384212762258067DFC4CF9F624FA472BEB9342",
    },
]

EXPECTED_OBS_DIM = 123
EXPECTED_ACTION_DIM = 37
EXPECTED_DT = 0.02
SMOKE_COMMAND = (0.40, 0.0, 0.0)

PASS_MARKER = "G1_V06_HISTORICAL_RUNTIME_CONTRACT_SMOKE_V1_PASS"
FAIL_MARKER = "G1_V06_HISTORICAL_RUNTIME_CONTRACT_SMOKE_V1_FAIL"


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


def neutralize_runtime_randomization(env_cfg: Any, seed: int) -> None:
    env_cfg.scene.num_envs = 1
    env_cfg.seed = int(seed)
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
    if hasattr(cmd_cfg, "debug_vis"):
        cmd_cfg.debug_vis = False


def write_command(base_env: Any, command: tuple[float, float, float]) -> None:
    tensor = base_env.command_manager.get_command("base_velocity")
    tensor.zero_()
    tensor[:, 0] = float(command[0])
    tensor[:, 1] = float(command[1])
    tensor[:, 2] = float(command[2])


def run_case(case: dict[str, Any], installed_rsl: str) -> dict[str, Any]:
    checkpoint: Path = case["checkpoint"]
    if not checkpoint.is_file():
        fail(f"{case['label']}: checkpoint missing: {checkpoint}")

    actual_sha = sha256_file(checkpoint)
    if actual_sha != case["sha256"]:
        fail(
            f"{case['label']}: SHA mismatch: expected {case['sha256']}, got {actual_sha}"
        )

    env_cfg = parse_env_cfg(case["task"], device=args_cli.device, num_envs=1)
    agent_cfg = load_cfg_from_registry(case["task"], "rsl_rl_cfg_entry_point")
    neutralize_runtime_randomization(env_cfg, args_cli.seed)

    if args_cli.device is not None and hasattr(agent_cfg, "device"):
        agent_cfg.device = args_cli.device
    if hasattr(agent_cfg, "seed"):
        agent_cfg.seed = int(args_cli.seed)

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_rsl)
    if agent_cfg.class_name != "OnPolicyRunner":
        fail(f"{case['label']}: unexpected runner class {agent_cfg.class_name}")

    env = None
    runner = None
    result = {
        "label": case["label"],
        "role": case["role"],
        "task": case["task"],
        "checkpoint": str(checkpoint),
        "sha256": actual_sha,
        "sha_pass": True,
        "runner_class": agent_cfg.class_name,
        "steps_requested": int(args_cli.steps_per_case),
    }

    try:
        raw_env = gym.make(case["task"], cfg=env_cfg)
        env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
        base_env = env.unwrapped

        action_dim = int(base_env.action_manager.total_action_dim)
        dt = float(base_env.step_dt)
        if action_dim != EXPECTED_ACTION_DIM:
            fail(f"{case['label']}: action_dim={action_dim}, expected {EXPECTED_ACTION_DIM}")
        if not math.isclose(dt, EXPECTED_DT, rel_tol=0.0, abs_tol=1.0e-12):
            fail(f"{case['label']}: step_dt={dt}, expected {EXPECTED_DT}")

        env_ids = torch.tensor([0], dtype=torch.int32, device=base_env.device)
        base_env.command_manager.reset(env_ids=env_ids)
        write_command(base_env, SMOKE_COMMAND)

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
        obs_dim = int(policy_obs.shape[1])
        if obs_dim != EXPECTED_OBS_DIM:
            fail(f"{case['label']}: obs_dim={obs_dim}, expected {EXPECTED_OBS_DIM}")

        max_abs_action = 0.0
        dones_seen = 0
        completed_steps = 0

        for _ in range(int(args_cli.steps_per_case)):
            write_command(base_env, SMOKE_COMMAND)
            obs = env.get_observations()
            with torch.inference_mode():
                actions = policy(obs)

            actions_t = as_tensor(actions)
            if actions_t.ndim != 2 or tuple(actions_t.shape) != (1, EXPECTED_ACTION_DIM):
                fail(f"{case['label']}: unexpected action shape {tuple(actions_t.shape)}")
            if not bool(torch.isfinite(actions_t).all().item()):
                fail(f"{case['label']}: non-finite actions")

            max_abs_action = max(
                max_abs_action,
                float(torch.max(torch.abs(actions_t)).item()),
            )

            with torch.inference_mode():
                obs, _, dones, _ = env.step(actions)
            dones_seen += int(torch.count_nonzero(as_tensor(dones)).item())
            completed_steps += 1

        result.update(
            {
                "status": "PASS",
                "obs_dim": obs_dim,
                "action_dim": action_dim,
                "step_dt": dt,
                "action_shape": [1, EXPECTED_ACTION_DIM],
                "actions_finite": True,
                "max_abs_action": max_abs_action,
                "steps_completed": completed_steps,
                "dones_seen": dones_seen,
                "note": (
                    "dones_seen is diagnostic only; historical behavior quality is deliberately "
                    "NOT a PASS/FAIL criterion in this contract smoke."
                ),
            }
        )
        return result

    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        del runner
        del env
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    if int(args_cli.steps_per_case) < 1 or int(args_cli.steps_per_case) > 10:
        fail("--steps-per-case must be in [1, 10]")

    installed_rsl = metadata.version("rsl-rl-lib")

    print("=" * 124)
    print("G1 V06 HISTORICAL RUNTIME CONTRACT SMOKE")
    print("=" * 124)
    print("Classification       : compatibility smoke only; NOT qualitative evaluation / NOT new scientific evidence")
    print(f"Cases                : {len(CASES)}")
    print(f"Steps per case       : {int(args_cli.steps_per_case)}")
    print(f"Seed                 : {int(args_cli.seed)}")
    print(f"Smoke command        : {SMOKE_COMMAND}")
    print(f"Expected obs/action  : {EXPECTED_OBS_DIM} / {EXPECTED_ACTION_DIM}")
    print(f"Expected dt          : {EXPECTED_DT:.5f} s")
    print("=" * 124)

    results = []
    for index, case in enumerate(CASES, 1):
        print(
            f"[SMOKE {index:02d}/{len(CASES):02d}] {case['label']} | "
            f"{case['task']}",
            flush=True,
        )
        result = run_case(case, installed_rsl)
        results.append(result)
        print(
            f"[PASS] {case['label']} obs={result['obs_dim']} action={result['action_dim']} "
            f"dt={result['step_dt']:.5f} steps={result['steps_completed']} "
            f"dones={result['dones_seen']} max|a|={result['max_abs_action']:.6f}",
            flush=True,
        )

    payload = {
        "classification": (
            "V06 historical checkpoint runtime compatibility smoke; "
            "NOT qualitative evaluation, NOT checkpoint promotion, NOT new scientific evidence"
        ),
        "seed": int(args_cli.seed),
        "smoke_command": list(SMOKE_COMMAND),
        "expected_obs_dim": EXPECTED_OBS_DIM,
        "expected_action_dim": EXPECTED_ACTION_DIM,
        "expected_step_dt": EXPECTED_DT,
        "case_count": len(CASES),
        "all_pass": all(r.get("status") == "PASS" for r in results),
        "results": results,
    }
    args_cli.output_json.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print()
    print("=" * 124)
    print("V06 HISTORICAL RUNTIME CONTRACT SMOKE SUMMARY")
    print("=" * 124)
    print(f"Status              : {'PASS' if payload['all_pass'] else 'FAIL'}")
    print(f"Cases PASS          : {sum(r.get('status') == 'PASS' for r in results)} / {len(CASES)}")
    print(f"Output JSON         : {args_cli.output_json}")
    print("Behavior assessment : NOT PERFORMED / intentionally outside this smoke")
    if not payload["all_pass"]:
        fail("one or more V06 historical runtime contract cases failed")
    print(PASS_MARKER)


try:
    main()
except Exception as exc:
    print(FAIL_MARKER)
    print(f"EXCEPTION_TYPE={type(exc).__name__}")
    print(f"EXCEPTION_MESSAGE={exc}")
    traceback.print_exc()
    raise
finally:
    simulation_app.close()
