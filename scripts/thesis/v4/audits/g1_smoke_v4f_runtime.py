from __future__ import annotations

import argparse
import re

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Runtime smoke for v4-F stand-still joint-deviation task.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import humanoid_rl_lab.tasks  # noqa: F401


TASK = "Humanoid-RL-Lab-G1-Flat-V4F-Stand-Still-Joint-Deviation-v0"
NUM_ENVS = 4
STEPS = 16

EXPECTED_PATTERNS = [
    ".*_hip_.*",
    ".*_knee_joint",
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
]


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

    # Resolved-config invariants.
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
    if "stand_still_full_command" not in active_rewards:
        fail("stand_still_full_command reward missing")

    term = env_cfg.rewards.stand_still_full_command
    if term.weight != -0.4:
        fail(f"stand_still reward weight mismatch: {term.weight!r}")

    params = term.params
    if params["command_name"] != "base_velocity":
        fail(f"command_name mismatch: {params['command_name']!r}")
    if params["command_threshold"] != 0.06:
        fail(f"command_threshold mismatch: {params['command_threshold']!r}")

    asset_cfg = params["asset_cfg"]
    if asset_cfg.name != "robot":
        fail(f"asset_cfg.name mismatch: {asset_cfg.name!r}")
    if list(asset_cfg.joint_names) != EXPECTED_PATTERNS:
        fail(f"joint regexes mismatch: {asset_cfg.joint_names!r}")

    env = gym.make(TASK, cfg=env_cfg)
    obs, _ = env.reset()

    unwrapped = env.unwrapped

    # Resolve the configured regexes against the actual instantiated robot joint names,
    # using Python fullmatch semantics and de-duplicating while preserving robot order.
    robot = unwrapped.scene["robot"]
    joint_names = list(robot.data.joint_names)

    resolved_names = []
    for joint_name in joint_names:
        if any(re.fullmatch(pattern, joint_name) is not None for pattern in EXPECTED_PATTERNS):
            resolved_names.append(joint_name)

    if len(resolved_names) != 12:
        fail(
            "expected exactly 12 resolved leg joints, "
            f"found {len(resolved_names)}: {resolved_names}"
        )
    if len(set(resolved_names)) != 12:
        fail(f"resolved leg joints are not unique: {resolved_names}")

    # Sanity by category: 6 hips, 2 knees, 2 ankle-pitch, 2 ankle-roll.
    category_counts = {
        "hip": sum(re.fullmatch(".*_hip_.*", n) is not None for n in resolved_names),
        "knee": sum(re.fullmatch(".*_knee_joint", n) is not None for n in resolved_names),
        "ankle_pitch": sum(re.fullmatch(".*_ankle_pitch_joint", n) is not None for n in resolved_names),
        "ankle_roll": sum(re.fullmatch(".*_ankle_roll_joint", n) is not None for n in resolved_names),
    }
    if category_counts != {"hip": 6, "knee": 2, "ankle_pitch": 2, "ankle_roll": 2}:
        fail(f"unexpected resolved joint category counts: {category_counts}")

    obs_tensors = list(iter_tensors(obs))
    if not obs_tensors:
        fail("no tensor observations found after reset")
    for tensor in obs_tensors:
        if not torch.isfinite(tensor).all():
            fail("non-finite observation immediately after reset")

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

        if torch.is_tensor(reward) and not torch.isfinite(reward).all():
            fail(f"non-finite reward at step {step_idx + 1}")

        terminated_any = terminated_any or bool(torch.as_tensor(terminated).any().item())
        truncated_any = truncated_any or bool(torch.as_tensor(truncated).any().item())

    print("=" * 112)
    print("V4-F RUNTIME SMOKE")
    print("=" * 112)
    print(f"task                    : {TASK}")
    print(f"num envs                : {NUM_ENVS}")
    print(f"steps                   : {STEPS}")
    print(f"action dimension        : {action_dim}")
    print(f"rel_standing_envs       : {env_cfg.commands.base_velocity.rel_standing_envs}")
    print(f"heading_command         : {env_cfg.commands.base_velocity.heading_command}")
    print(f"track_lin_vel_xy std    : {env_cfg.rewards.track_lin_vel_xy_exp.params['std']}")
    print(f"active rewards          : {len(active_rewards)}")
    print(f"stand-still weight      : {term.weight}")
    print(f"stand-still threshold   : {params['command_threshold']}")
    print(f"resolved leg joints     : {len(resolved_names)}")
    print(f"resolved names          : {resolved_names}")
    print(f"joint category counts   : {category_counts}")
    print(f"terminated any          : {terminated_any}")
    print(f"truncated any           : {truncated_any}")
    print("finite observations     : PASS")
    print("finite rewards          : PASS")
    print("checkpoint loaded       : NO")
    print("PPO update              : NO")
    print()
    print("V4F_RUNTIME_SMOKE_PASS")
    print("=" * 112)

finally:
    if env is not None:
        env.close()
    simulation_app.close()
