from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Runtime smoke for v4-G zero-command full-motion task.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import humanoid_rl_lab.tasks  # noqa: F401


TASK = "Humanoid-RL-Lab-G1-Flat-V4G-Zero-Command-Full-Motion-v0"
NUM_ENVS = 4
STEPS = 16
EXPECTED_FEET = ["left_ankle_roll_link", "right_ankle_roll_link"]


def fail(msg: str) -> None:
    raise RuntimeError(msg)


def iter_tensors(obj):
    if torch.is_tensor(obj):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from iter_tensors(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from iter_tensors(value)


env = None

try:
    env_cfg = parse_env_cfg(
        TASK,
        device=args_cli.device,
        num_envs=NUM_ENVS,
    )

    if env_cfg.commands.base_velocity.rel_standing_envs != 0.02:
        fail(f"rel_standing_envs mismatch: {env_cfg.commands.base_velocity.rel_standing_envs!r}")
    if env_cfg.commands.base_velocity.heading_command is not False:
        fail(f"heading_command mismatch: {env_cfg.commands.base_velocity.heading_command!r}")
    if env_cfg.rewards.track_lin_vel_xy_exp.params["std"] != 0.5:
        fail(f"track_lin_vel_xy_exp std mismatch: {env_cfg.rewards.track_lin_vel_xy_exp.params['std']!r}")

    active_rewards = [
        name for name, term in vars(env_cfg.rewards).items()
        if not name.startswith("_") and term is not None
    ]
    if len(active_rewards) != 17:
        fail(f"expected 17 active rewards, found {len(active_rewards)}: {active_rewards}")
    if "zero_command_full_motion" not in active_rewards:
        fail("zero_command_full_motion reward missing")

    term = env_cfg.rewards.zero_command_full_motion
    if term.weight != -5.0:
        fail(f"zero_command_full_motion weight mismatch: {term.weight!r}")

    params = term.params
    expected = {
        "command_name": "base_velocity",
        "command_threshold": 0.06,
        "base_planar_scale": 1.0,
        "yaw_rate_scale": 1.0,
        "foot_xy_scale": 0.25,
    }
    for key, value in expected.items():
        if params[key] != value:
            fail(f"{key} mismatch: {params[key]!r}")

    asset_cfg = params["asset_cfg"]
    if asset_cfg.name != "robot":
        fail(f"asset_cfg.name mismatch: {asset_cfg.name!r}")
    if list(asset_cfg.body_names) != EXPECTED_FEET:
        fail(f"asset_cfg.body_names mismatch: {asset_cfg.body_names!r}")

    env = gym.make(TASK, cfg=env_cfg)
    obs, _ = env.reset()
    unwrapped = env.unwrapped

    robot = unwrapped.scene["robot"]
    robot_body_names = list(robot.data.body_names)

    missing_feet = [name for name in EXPECTED_FEET if name not in robot_body_names]
    if missing_feet:
        fail(f"expected foot bodies missing from instantiated robot: {missing_feet}")

    resolved_foot_indices = [robot_body_names.index(name) for name in EXPECTED_FEET]
    if len(set(resolved_foot_indices)) != 2:
        fail(f"foot body indices are not unique: {resolved_foot_indices}")

    obs_tensors = list(iter_tensors(obs))
    if not obs_tensors:
        fail("no tensor observations found after reset")
    for tensor in obs_tensors:
        if not torch.isfinite(tensor).all():
            fail("non-finite observation immediately after reset")

    action_dim = int(unwrapped.action_manager.total_action_dim)
    if action_dim != 37:
        fail(f"expected 37 actions, found {action_dim}")

    action = torch.zeros(
        (NUM_ENVS, action_dim),
        dtype=torch.float32,
        device=unwrapped.device,
    )

    terminated_any = False
    truncated_any = False

    for step_idx in range(STEPS):
        obs, reward, terminated, truncated, _ = env.step(action)

        for tensor in iter_tensors(obs):
            if not torch.isfinite(tensor).all():
                fail(f"non-finite observation at step {step_idx + 1}")

        if torch.is_tensor(reward) and not torch.isfinite(reward).all():
            fail(f"non-finite reward at step {step_idx + 1}")

        terminated_any = terminated_any or bool(torch.as_tensor(terminated).any().item())
        truncated_any = truncated_any or bool(torch.as_tensor(truncated).any().item())

    print("=" * 112)
    print("V4-G RUNTIME SMOKE")
    print("=" * 112)
    print(f"task                    : {TASK}")
    print(f"num envs                : {NUM_ENVS}")
    print(f"steps                   : {STEPS}")
    print(f"action dimension        : {action_dim}")
    print(f"rel_standing_envs       : {env_cfg.commands.base_velocity.rel_standing_envs}")
    print(f"heading_command         : {env_cfg.commands.base_velocity.heading_command}")
    print(f"track_lin_vel_xy std    : {env_cfg.rewards.track_lin_vel_xy_exp.params['std']}")
    print(f"active rewards          : {len(active_rewards)}")
    print(f"full-motion weight      : {term.weight}")
    print(f"full-motion threshold   : {params['command_threshold']}")
    print(f"base planar scale       : {params['base_planar_scale']}")
    print(f"yaw-rate scale          : {params['yaw_rate_scale']}")
    print(f"foot XY scale           : {params['foot_xy_scale']}")
    print(f"resolved foot bodies    : {EXPECTED_FEET}")
    print(f"resolved foot indices   : {resolved_foot_indices}")
    print(f"terminated any          : {terminated_any}")
    print(f"truncated any           : {truncated_any}")
    print("finite observations     : PASS")
    print("finite rewards          : PASS")
    print("checkpoint loaded       : NO")
    print("PPO update              : NO")
    print()
    print("V4G_RUNTIME_SMOKE_PASS")
    print("=" * 112)

finally:
    if env is not None:
        env.close()
    simulation_app.close()
