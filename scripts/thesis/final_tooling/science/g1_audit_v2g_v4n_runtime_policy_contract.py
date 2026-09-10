from __future__ import annotations

import argparse
import gc
import math
from typing import Any

parser = argparse.ArgumentParser(
    description="Runtime-resolved policy-contract audit: frozen v2-G STAND vs frozen v4-N LOCOMOTION."
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
from isaaclab_tasks.utils import parse_env_cfg


STAND_TASK = "Humanoid-RL-Lab-G1-Flat-Upright-Yaw-Direct-Wide-Yaw-Flat-Orientation-v0"
LOCO_TASK = "Humanoid-RL-Lab-G1-Flat-V4N-Symmetry-Augmentation-v0"

EXPECTED_OBS_TERMS = (
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "joint_pos",
    "joint_vel",
    "actions",
)
EXPECTED_OBS_DIM = 123
EXPECTED_ACTION_DIM = 37
EXPECTED_STEP_DT = 0.02


def fail(message: str) -> None:
    raise RuntimeError(message)


def to_tensor(value: Any) -> torch.Tensor:
    if hasattr(value, "torch"):
        value = value.torch
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    return value.detach().cpu()


def get_policy_tensor(obs: Any) -> torch.Tensor:
    if isinstance(obs, dict):
        if "policy" not in obs:
            fail(f"observation dict has no 'policy' key: {list(obs.keys())}")
        obs = obs["policy"]
    obs = to_tensor(obs)
    if obs.ndim != 2 or obs.shape[0] != 1:
        fail(f"unexpected policy observation shape: {tuple(obs.shape)}")
    return obs


def extract_obs_terms(manager: Any) -> tuple[str, ...]:
    # Public/protected layouts seen across Isaac Lab releases.
    active = getattr(manager, "active_terms", None)
    if callable(active):
        try:
            active = active()
        except TypeError:
            pass

    if isinstance(active, dict) and "policy" in active:
        return tuple(str(x) for x in active["policy"])

    for attr in ("_group_obs_term_names", "group_obs_term_names"):
        value = getattr(manager, attr, None)
        if isinstance(value, dict) and "policy" in value:
            return tuple(str(x) for x in value["policy"])

    fail(
        "unable to resolve runtime policy observation term order; "
        f"manager type={type(manager).__name__}"
    )


def extract_action_term_names(manager: Any) -> tuple[str, ...]:
    for attr in ("active_terms", "_term_names", "term_names"):
        value = getattr(manager, attr, None)
        if callable(value):
            try:
                value = value()
            except TypeError:
                continue
        if isinstance(value, (list, tuple)):
            return tuple(str(x) for x in value)

    # ActionManager.__str__ is not sufficient for a machine-checkable contract.
    fail(
        "unable to resolve runtime action term names; "
        f"manager type={type(manager).__name__}"
    )


def get_joint_pos_action_term(manager: Any) -> Any:
    get_term = getattr(manager, "get_term", None)
    if callable(get_term):
        try:
            return get_term("joint_pos")
        except Exception:
            pass

    term_names = getattr(manager, "_term_names", None)
    terms = getattr(manager, "_terms", None)

    if isinstance(term_names, (list, tuple)) and isinstance(terms, (list, tuple)):
        if "joint_pos" in term_names:
            return terms[list(term_names).index("joint_pos")]

    if isinstance(terms, dict) and "joint_pos" in terms:
        return terms["joint_pos"]

    fail("unable to resolve runtime JointPositionAction term object")


def extract_string_sequence(obj: Any, preferred: tuple[str, ...]) -> tuple[str, ...] | None:
    for attr in preferred:
        value = getattr(obj, attr, None)
        if isinstance(value, (list, tuple)) and value and all(isinstance(x, str) for x in value):
            return tuple(value)

    for attr in dir(obj):
        if "joint" not in attr.lower() or "name" not in attr.lower():
            continue
        try:
            value = getattr(obj, attr)
        except Exception:
            continue
        if isinstance(value, (list, tuple)) and value and all(isinstance(x, str) for x in value):
            return tuple(value)

    return None


def extract_int_sequence(obj: Any, preferred: tuple[str, ...]) -> tuple[int, ...] | None:
    for attr in preferred:
        value = getattr(obj, attr, None)
        if value is None:
            continue
        if torch.is_tensor(value):
            value = value.detach().cpu().flatten().tolist()
        if isinstance(value, range):
            value = list(value)
        if isinstance(value, (list, tuple)) and value and all(isinstance(x, int) for x in value):
            return tuple(int(x) for x in value)

    for attr in dir(obj):
        if "joint" not in attr.lower() or "id" not in attr.lower():
            continue
        try:
            value = getattr(obj, attr)
        except Exception:
            continue
        if torch.is_tensor(value):
            value = value.detach().cpu().flatten().tolist()
        if isinstance(value, range):
            value = list(value)
        if isinstance(value, (list, tuple)) and value and all(isinstance(x, int) for x in value):
            return tuple(int(x) for x in value)

    return None


def force_play_equivalent_zero_command(env_cfg: Any) -> None:
    env_cfg.seed = args_cli.seed
    env_cfg.scene.num_envs = 1

    if hasattr(env_cfg.observations.policy, "enable_corruption"):
        env_cfg.observations.policy.enable_corruption = False

    if hasattr(env_cfg, "events"):
        if hasattr(env_cfg.events, "base_external_force_torque"):
            env_cfg.events.base_external_force_torque = None
        if hasattr(env_cfg.events, "push_robot"):
            env_cfg.events.push_robot = None

    cmd = env_cfg.commands.base_velocity
    cmd.heading_command = False
    if hasattr(cmd, "rel_heading_envs"):
        cmd.rel_heading_envs = 0.0
    cmd.ranges.lin_vel_x = (0.0, 0.0)
    cmd.ranges.lin_vel_y = (0.0, 0.0)
    cmd.ranges.ang_vel_z = (0.0, 0.0)


def fingerprint(task_id: str, label: str) -> dict[str, Any]:
    env_cfg = parse_env_cfg(
        task_id,
        device=args_cli.device,
        num_envs=1,
    )
    force_play_equivalent_zero_command(env_cfg)

    env = None
    try:
        env = gym.make(task_id, cfg=env_cfg)
        uenv = env.unwrapped

        robot = uenv.scene["robot"]
        robot_joint_names = tuple(robot.joint_names)

        default_joint_pos = to_tensor(robot.data.default_joint_pos)
        if default_joint_pos.ndim != 2 or default_joint_pos.shape[0] < 1:
            fail(
                f"{label}: unexpected default_joint_pos shape "
                f"{tuple(default_joint_pos.shape)}"
            )
        default_joint_pos = default_joint_pos[0].flatten()

        obs_terms = extract_obs_terms(uenv.observation_manager)
        action_terms = extract_action_term_names(uenv.action_manager)

        action_term = get_joint_pos_action_term(uenv.action_manager)
        action_joint_names = extract_string_sequence(
            action_term,
            ("joint_names", "_joint_names"),
        )
        action_joint_ids = extract_int_sequence(
            action_term,
            ("joint_ids", "_joint_ids"),
        )

        if action_joint_names is None:
            fail(f"{label}: could not resolve JointPositionAction joint names")
        if action_joint_ids is None:
            fail(f"{label}: could not resolve JointPositionAction joint ids")

        reset_out = env.reset()
        if not isinstance(reset_out, tuple) or len(reset_out) < 1:
            fail(f"{label}: unexpected env.reset() return type")
        policy_obs = get_policy_tensor(reset_out[0])

        total_action_dim = getattr(uenv.action_manager, "total_action_dim", None)
        if total_action_dim is None:
            total_action_dim = len(action_joint_names)
        total_action_dim = int(total_action_dim)

        step_dt = float(uenv.step_dt)

        result = {
            "label": label,
            "task": task_id,
            "robot_joint_names": robot_joint_names,
            "default_joint_pos": default_joint_pos,
            "obs_terms": obs_terms,
            "obs_dim": int(policy_obs.shape[1]),
            "action_terms": action_terms,
            "action_dim": total_action_dim,
            "action_joint_names": action_joint_names,
            "action_joint_ids": action_joint_ids,
            "step_dt": step_dt,
        }

        print()
        print("=" * 118)
        print(f"{label} RUNTIME FINGERPRINT")
        print("=" * 118)
        print(f"task                  : {task_id}")
        print(f"step_dt               : {step_dt:.12f}")
        print(f"policy_obs_dim         : {result['obs_dim']}")
        print(f"policy_obs_terms       : {list(obs_terms)}")
        print(f"action_dim             : {total_action_dim}")
        print(f"action_terms           : {list(action_terms)}")
        print(f"robot_joint_count      : {len(robot_joint_names)}")
        print(f"action_joint_count     : {len(action_joint_names)}")
        print(f"action_joint_ids       : {list(action_joint_ids)}")
        print("action_joint_names:")
        for idx, name in enumerate(action_joint_names):
            print(f"  {idx:02d}: {name}")
        print("default_joint_pos_by_robot_order:")
        for idx, (name, value) in enumerate(zip(robot_joint_names, default_joint_pos.tolist())):
            print(f"  {idx:02d}: {name:<34} {value:+.9f}")

        return result
    finally:
        if env is not None:
            env.close()
        gc.collect()


def compare(a: dict[str, Any], b: dict[str, Any]) -> None:
    checks: list[tuple[str, bool]] = []

    checks.append(("step_dt", math.isclose(a["step_dt"], b["step_dt"], rel_tol=0.0, abs_tol=1e-12)))
    checks.append(("step_dt_expected_50Hz", math.isclose(a["step_dt"], EXPECTED_STEP_DT, rel_tol=0.0, abs_tol=1e-12)))
    checks.append(("obs_dim", a["obs_dim"] == b["obs_dim"] == EXPECTED_OBS_DIM))
    checks.append(("obs_term_order", a["obs_terms"] == b["obs_terms"] == EXPECTED_OBS_TERMS))
    checks.append(("action_dim", a["action_dim"] == b["action_dim"] == EXPECTED_ACTION_DIM))
    checks.append(("action_term_order", a["action_terms"] == b["action_terms"] == ("joint_pos",)))
    checks.append(("robot_joint_order", a["robot_joint_names"] == b["robot_joint_names"]))
    checks.append(("action_joint_order", a["action_joint_names"] == b["action_joint_names"]))
    checks.append(("action_joint_ids", a["action_joint_ids"] == b["action_joint_ids"]))
    checks.append(
        (
            "action_maps_exact_robot_order",
            a["action_joint_names"] == a["robot_joint_names"]
            and b["action_joint_names"] == b["robot_joint_names"],
        )
    )

    default_equal = (
        a["default_joint_pos"].shape == b["default_joint_pos"].shape
        and torch.equal(a["default_joint_pos"], b["default_joint_pos"])
    )
    checks.append(("default_joint_pose_exact", default_equal))

    if default_equal:
        max_default_delta = 0.0
    else:
        if a["default_joint_pos"].shape != b["default_joint_pos"].shape:
            max_default_delta = float("inf")
        else:
            max_default_delta = float(
                torch.max(torch.abs(a["default_joint_pos"] - b["default_joint_pos"])).item()
            )

    print()
    print("=" * 118)
    print("v2-G STAND <-> v4-N LOCOMOTION RUNTIME POLICY-CONTRACT COMPARISON")
    print("=" * 118)
    for name, ok in checks:
        print(f"{name:<42}: {'PASS' if ok else 'FAIL'}")
    print(f"{'max_default_joint_pose_delta_rad':<42}: {max_default_delta:.12g}")

    failed = [name for name, ok in checks if not ok]
    if failed:
        print(f"FAILED_CHECKS={failed}")
        fail("runtime policy-contract compatibility audit failed")

    print()
    print("RUNTIME_COMPATIBILITY=PASS")
    print("Observation order       : EXACT MATCH")
    print("Action joint order      : EXACT MATCH")
    print("Action joint ids        : EXACT MATCH")
    print("Default joint pose      : EXACT MATCH")
    print("Policy/action dimensions: 123 / 37")
    print("NEXT_REQUIRED_CHECK     : load both frozen checkpoints and measure live policy handoff behavior")
    print("V2G_V4N_RUNTIME_POLICY_CONTRACT_AUDIT_PASS")
    print("=" * 118)


try:
    print("=" * 118)
    print("v2-G STAND <-> v4-N LOCOMOTION RUNTIME-RESOLVED POLICY-CONTRACT AUDIT")
    print("=" * 118)
    print(f"seed       : {args_cli.seed}")
    print("conditions : 1 env, zero command, observation corruption OFF, external forces OFF, pushes OFF")
    print()

    stand = fingerprint(STAND_TASK, "v2-G STAND")
    loco = fingerprint(LOCO_TASK, "v4-N LOCOMOTION")
    compare(stand, loco)
finally:
    simulation_app.close()
