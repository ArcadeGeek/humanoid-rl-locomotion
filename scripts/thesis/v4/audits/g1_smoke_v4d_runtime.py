from __future__ import annotations

import argparse
import math
import sys

from isaaclab.app import AppLauncher


# Launch Isaac Sim before importing simulation-dependent modules.
parser = argparse.ArgumentParser(description="Runtime smoke for v4-D standing-exposure task.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

# Import project tasks only after the simulation app exists so Gym registrations are active.
import humanoid_rl_lab.tasks  # noqa: F401


TASK = "Humanoid-RL-Lab-G1-Flat-V4D-Standing-Exposure-v0"
NUM_ENVS = 4
STEPS = 16


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

    # Runtime-resolved scientific guard.
    if env_cfg.commands.base_velocity.rel_standing_envs != 0.05:
        fail(
            "resolved rel_standing_envs mismatch: "
            f"{env_cfg.commands.base_velocity.rel_standing_envs!r}"
        )
    if env_cfg.commands.base_velocity.heading_command is not False:
        fail(
            "resolved heading_command mismatch: "
            f"{env_cfg.commands.base_velocity.heading_command!r}"
        )
    if env_cfg.rewards.track_lin_vel_xy_exp.params["std"] != 0.5:
        fail(
            "resolved track_lin_vel_xy_exp std mismatch: "
            f"{env_cfg.rewards.track_lin_vel_xy_exp.params['std']!r}"
        )

    env = gym.make(TASK, cfg=env_cfg)

    obs, _ = env.reset()

    obs_tensors = list(iter_tensors(obs))
    if not obs_tensors:
        fail("no tensor observations found after reset")
    for tensor in obs_tensors:
        if not torch.isfinite(tensor).all():
            fail("non-finite observation detected immediately after reset")

    unwrapped = env.unwrapped
    if not hasattr(unwrapped, "action_manager"):
        fail("runtime environment has no action_manager")

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

        for label, tensor in (
            ("reward", reward),
            ("terminated", terminated),
            ("truncated", truncated),
        ):
            if torch.is_tensor(tensor) and tensor.dtype.is_floating_point:
                if not torch.isfinite(tensor).all():
                    fail(f"non-finite {label} at step {step_idx + 1}")

        terminated_any = terminated_any or bool(torch.as_tensor(terminated).any().item())
        truncated_any = truncated_any or bool(torch.as_tensor(truncated).any().item())

    print("=" * 104)
    print("V4-D RUNTIME SMOKE")
    print("=" * 104)
    print(f"task                    : {TASK}")
    print(f"num envs                : {NUM_ENVS}")
    print(f"steps                   : {STEPS}")
    print(f"action dimension        : {action_dim}")
    print(f"rel_standing_envs       : {env_cfg.commands.base_velocity.rel_standing_envs}")
    print(f"heading_command         : {env_cfg.commands.base_velocity.heading_command}")
    print(f"track_lin_vel_xy std    : {env_cfg.rewards.track_lin_vel_xy_exp.params['std']}")
    print(f"terminated any          : {terminated_any}")
    print(f"truncated any           : {truncated_any}")
    print("finite observations     : PASS")
    print("checkpoint loaded       : NO")
    print("PPO update              : NO")
    print()
    print("V4D_RUNTIME_SMOKE_PASS")
    print("=" * 104)

finally:
    if env is not None:
        env.close()
    simulation_app.close()
