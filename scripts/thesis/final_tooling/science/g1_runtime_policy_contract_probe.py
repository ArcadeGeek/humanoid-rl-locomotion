from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Runtime-resolved policy I/O contract probe for the G1 dual-controller audit."
)
parser.add_argument("--task", required=True, type=str)
parser.add_argument("--label", required=True, type=str)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry

import isaaclab_tasks  # noqa: F401
import humanoid_rl_lab.tasks  # noqa: F401


EXPECTED_OBS_TERM_NAMES = (
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "joint_pos",
    "joint_vel",
    "actions",
)
EXPECTED_OBS_TERM_DIMS = (3, 3, 3, 3, 37, 37, 37)
EXPECTED_POLICY_OBS_DIM = 123
EXPECTED_ACTION_DIM = 37


def to_plain(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, slice):
        return {
            "slice": [
                value.start,
                value.stop,
                value.step,
            ]
        }
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_plain(v) for k, v in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def get_action_term(manager: Any, name: str) -> Any:
    getter = getattr(manager, "get_term", None)
    if callable(getter):
        try:
            return getter(name)
        except Exception:
            pass

    terms = getattr(manager, "_terms", None)
    if isinstance(terms, dict) and name in terms:
        return terms[name]

    raise RuntimeError(f"cannot resolve action term {name!r}")


def resolve_action_joint_names(term: Any, robot_joint_names: list[str]) -> list[str]:
    for attr in ("joint_names", "_joint_names"):
        value = getattr(term, attr, None)
        if value is not None:
            names = list(value)
            if names and all(isinstance(x, str) for x in names):
                return names

    ids = None
    for attr in ("joint_ids", "_joint_ids"):
        value = getattr(term, attr, None)
        if value is not None:
            ids = value
            break

    if ids is None:
        raise RuntimeError("cannot resolve action joint ids/names")

    if isinstance(ids, slice):
        return list(robot_joint_names[ids])

    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().reshape(-1).tolist()

    return [robot_joint_names[int(i)] for i in ids]


def resolve_obs_term_names(manager: Any) -> list[str]:
    active = getattr(manager, "active_terms", None)
    if isinstance(active, dict):
        value = active.get("policy")
        if value is not None:
            return list(value)

    private = getattr(manager, "_group_obs_term_names", None)
    if isinstance(private, dict) and "policy" in private:
        return list(private["policy"])

    raise RuntimeError("cannot resolve runtime policy observation term names")


def resolve_obs_term_dims(manager: Any) -> list[int] | None:
    candidates = (
        "_group_obs_term_dim",
        "_group_obs_term_dims",
        "group_obs_term_dim",
        "group_obs_term_dims",
    )

    for attr in candidates:
        data = getattr(manager, attr, None)
        if not isinstance(data, dict) or "policy" not in data:
            continue

        raw = data["policy"]
        dims: list[int] = []
        try:
            for item in raw:
                if isinstance(item, int):
                    dims.append(int(item))
                elif isinstance(item, torch.Size):
                    product = 1
                    for x in item:
                        product *= int(x)
                    dims.append(product)
                elif isinstance(item, (list, tuple)):
                    product = 1
                    for x in item:
                        product *= int(x)
                    dims.append(product)
                else:
                    product = 1
                    for x in tuple(item):
                        product *= int(x)
                    dims.append(product)
            return dims
        except Exception:
            continue

    return None


def resolve_tensor_attr(term: Any, names: tuple[str, ...]) -> torch.Tensor | None:
    for name in names:
        value = getattr(term, name, None)
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
    return None


def rounded_list(tensor: torch.Tensor, digits: int = 10) -> list[float]:
    return [round(float(v), digits) for v in tensor.reshape(-1).tolist()]


env = None
wrapped = None

try:
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    env_cfg.scene.num_envs = 1
    env_cfg.seed = 42

    if hasattr(env_cfg.observations.policy, "enable_corruption"):
        env_cfg.observations.policy.enable_corruption = False

    if hasattr(env_cfg, "events"):
        if hasattr(env_cfg.events, "base_external_force_torque"):
            env_cfg.events.base_external_force_torque = None
        if hasattr(env_cfg.events, "push_robot"):
            env_cfg.events.push_robot = None

    env = gym.make(args_cli.task, cfg=env_cfg)
    uenv = env.unwrapped

    robot = uenv.scene["robot"]
    robot_joint_names = list(robot.joint_names)

    action_manager = uenv.action_manager
    action_term = get_action_term(action_manager, "joint_pos")
    action_joint_names = resolve_action_joint_names(
        action_term,
        robot_joint_names,
    )

    wrapped = RslRlVecEnvWrapper(
        env,
        clip_actions=agent_cfg.clip_actions,
    )
    obs = wrapped.get_observations()

    if not isinstance(obs, torch.Tensor):
        raise RuntimeError(
            f"RSL-RL policy observation is not a tensor: {type(obs)!r}"
        )

    policy_obs_dim = int(obs.shape[-1])
    action_dim = int(uenv.action_manager.total_action_dim)

    obs_term_names = resolve_obs_term_names(uenv.observation_manager)
    obs_term_dims = resolve_obs_term_dims(uenv.observation_manager)

    default_joint_pos = robot.data.default_joint_pos[0].detach().cpu()

    scale_tensor = resolve_tensor_attr(
        action_term,
        ("scale", "_scale"),
    )
    offset_tensor = resolve_tensor_attr(
        action_term,
        ("offset", "_offset"),
    )

    if len(robot_joint_names) != EXPECTED_ACTION_DIM:
        raise RuntimeError(
            f"robot joint count mismatch: {len(robot_joint_names)} != {EXPECTED_ACTION_DIM}"
        )

    if action_dim != EXPECTED_ACTION_DIM:
        raise RuntimeError(
            f"resolved action dim mismatch: {action_dim} != {EXPECTED_ACTION_DIM}"
        )

    if policy_obs_dim != EXPECTED_POLICY_OBS_DIM:
        raise RuntimeError(
            f"policy observation dim mismatch: {policy_obs_dim} != {EXPECTED_POLICY_OBS_DIM}"
        )

    if tuple(obs_term_names) != EXPECTED_OBS_TERM_NAMES:
        raise RuntimeError(
            "runtime observation term order mismatch\n"
            f"expected={EXPECTED_OBS_TERM_NAMES}\n"
            f"actual  ={tuple(obs_term_names)}"
        )

    if obs_term_dims is not None and tuple(obs_term_dims) != EXPECTED_OBS_TERM_DIMS:
        raise RuntimeError(
            "runtime observation term dimensions mismatch\n"
            f"expected={EXPECTED_OBS_TERM_DIMS}\n"
            f"actual  ={tuple(obs_term_dims)}"
        )

    if action_joint_names != robot_joint_names:
        raise RuntimeError(
            "resolved action joint order differs from robot joint order"
        )

    default_offset_error = None
    if offset_tensor is not None:
        offset_flat = offset_tensor.reshape(-1)
        default_flat = default_joint_pos.reshape(-1)
        if offset_flat.numel() == 1:
            # A scalar offset is not the expanded runtime default-offset tensor.
            default_offset_error = None
        elif offset_flat.numel() == default_flat.numel():
            default_offset_error = float(
                torch.max(torch.abs(offset_flat - default_flat)).item()
            )
            if default_offset_error > 1.0e-7:
                raise RuntimeError(
                    "runtime action offset differs from robot default joint pose: "
                    f"max_abs_error={default_offset_error:.12g}"
                )
        else:
            raise RuntimeError(
                "unexpected runtime action offset size: "
                f"{offset_flat.numel()} vs {default_flat.numel()}"
            )

    if scale_tensor is not None:
        scale_flat = scale_tensor.reshape(-1)
        if scale_flat.numel() not in (1, EXPECTED_ACTION_DIM):
            raise RuntimeError(
                f"unexpected runtime action scale size: {scale_flat.numel()}"
            )
        max_scale_error = float(
            torch.max(torch.abs(scale_flat - 0.5)).item()
        )
        if max_scale_error > 1.0e-7:
            raise RuntimeError(
                "runtime action scale differs from 0.5: "
                f"max_abs_error={max_scale_error:.12g}"
            )
    else:
        max_scale_error = None

    contract = {
        "policy_obs_dim": policy_obs_dim,
        "obs_term_names": obs_term_names,
        "obs_term_dims": (
            obs_term_dims
            if obs_term_dims is not None
            else list(EXPECTED_OBS_TERM_DIMS)
        ),
        "action_dim": action_dim,
        "robot_joint_names": robot_joint_names,
        "action_joint_names": action_joint_names,
        "default_joint_pos": rounded_list(default_joint_pos),
        "action_scale": (
            rounded_list(scale_tensor)
            if scale_tensor is not None
            else None
        ),
        "action_offset": (
            rounded_list(offset_tensor)
            if offset_tensor is not None
            else None
        ),
    }

    canonical = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
    )
    fingerprint = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest().upper()

    print("=" * 120)
    print("G1 RUNTIME POLICY-CONTRACT PROBE")
    print("=" * 120)
    print(f"label                         : {args_cli.label}")
    print(f"task                          : {args_cli.task}")
    print(f"policy observation shape      : {tuple(obs.shape)}")
    print(f"policy observation dim        : {policy_obs_dim}")
    print(f"runtime observation terms     : {obs_term_names}")
    print(
        "runtime observation dims      : "
        f"{obs_term_dims if obs_term_dims is not None else 'API_NOT_EXPOSED; total verified'}"
    )
    print(f"robot joint count             : {len(robot_joint_names)}")
    print(f"resolved action dim           : {action_dim}")
    print(
        "action order == robot order   : "
        f"{action_joint_names == robot_joint_names}"
    )
    print(f"first joint                   : {robot_joint_names[0]}")
    print(f"last joint                    : {robot_joint_names[-1]}")
    print(
        "runtime action scale error    : "
        f"{max_scale_error if max_scale_error is not None else 'API_NOT_EXPOSED'}"
    )
    print(
        "default-offset max error      : "
        f"{default_offset_error if default_offset_error is not None else 'API_NOT_EXPOSED_OR_SCALAR'}"
    )
    print(f"resolved contract SHA256      : {fingerprint}")
    print("=" * 120)
    print("G1_RUNTIME_POLICY_CONTRACT_PROBE_PASS")
    print("=" * 120)

finally:
    if wrapped is not None:
        try:
            wrapped.close()
        except Exception:
            pass
    elif env is not None:
        try:
            env.close()
        except Exception:
            pass
    simulation_app.close()
