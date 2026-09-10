from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Minimal runtime smoke test for v4-C.")
parser.add_argument(
    "--task",
    type=str,
    default="Humanoid-RL-Lab-G1-Flat-V4C-Precision-v0",
)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--smoke_steps", type=int, default=16)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import humanoid_rl_lab.tasks.manager_based.humanoid_rl_lab  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def fail(msg: str) -> None:
    raise RuntimeError(f"ABORT: {msg}")


def policy_tensor(obs):
    if isinstance(obs, tuple):
        obs = obs[0]
    if isinstance(obs, dict):
        if "policy" not in obs:
            fail(f"observation dict has no 'policy' key; keys={list(obs.keys())}")
        obs = obs["policy"]
    if not torch.is_tensor(obs):
        fail(f"policy observation is not a tensor: {type(obs).__name__}")
    return obs


env = None
try:
    if args_cli.num_envs != 4:
        fail(f"canonical smoke requires --num_envs 4, got {args_cli.num_envs}")
    if args_cli.smoke_steps != 16:
        fail(f"canonical smoke requires --smoke_steps 16, got {args_cli.smoke_steps}")
    if args_cli.seed != 42:
        fail(f"canonical smoke requires --seed 42, got {args_cli.seed}")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    env_cfg.seed = args_cli.seed

    # Runtime-specific guard for the intended scientific delta.
    std = float(env_cfg.rewards.track_lin_vel_xy_exp.params["std"])
    if abs(std - 0.25) > 1.0e-12:
        fail(f"resolved v4-C tracking std != 0.25: {std}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    uenv = env.unwrapped

    if int(uenv.num_envs) != 4:
        fail(f"runtime num_envs != 4: {uenv.num_envs}")
    if abs(float(uenv.step_dt) - 0.02) > 1.0e-12:
        fail(f"runtime step_dt != 0.02: {uenv.step_dt}")

    action_dim = int(uenv.action_manager.total_action_dim)
    if action_dim != 37:
        fail(f"runtime action dim != 37: {action_dim}")

    reset_out = env.reset(seed=args_cli.seed)
    obs0 = policy_tensor(reset_out)

    if tuple(obs0.shape) != (4, 123):
        fail(f"initial policy obs shape != (4, 123): {tuple(obs0.shape)}")
    if not torch.isfinite(obs0).all():
        fail("initial policy observation contains non-finite values")

    actions = torch.zeros(
        (uenv.num_envs, action_dim),
        device=uenv.device,
        dtype=obs0.dtype,
    )

    reset_count = 0
    reward_finite = True
    obs = obs0

    for _ in range(args_cli.smoke_steps):
        step_out = env.step(actions)

        if not isinstance(step_out, tuple):
            fail(f"env.step returned unexpected type: {type(step_out).__name__}")

        if len(step_out) == 5:
            obs_raw, reward, terminated, truncated, _info = step_out
            done = torch.logical_or(terminated, truncated)
        elif len(step_out) == 4:
            obs_raw, reward, done, _info = step_out
        else:
            fail(f"env.step returned tuple length {len(step_out)}, expected 4 or 5")

        obs = policy_tensor(obs_raw)

        if tuple(obs.shape) != (4, 123):
            fail(f"policy obs shape changed during rollout: {tuple(obs.shape)}")
        if not torch.isfinite(obs).all():
            fail("policy observation became non-finite")

        if torch.is_tensor(reward) and not torch.isfinite(reward).all():
            reward_finite = False
            fail("reward became non-finite")

        if torch.is_tensor(done):
            reset_count += int(torch.count_nonzero(done).item())

    if reset_count != 0:
        fail(f"unexpected reset/termination count during 16-step smoke: {reset_count}")

    print("=" * 108)
    print("V4-C RUNTIME SMOKE")
    print("=" * 108)
    print(f"task                                  : {args_cli.task}")
    print(f"num_envs                              : {uenv.num_envs}")
    print(f"seed                                  : {args_cli.seed}")
    print(f"device                                : {uenv.device}")
    print(f"step_dt                               : {float(uenv.step_dt):.5f} s")
    print(f"policy observation shape              : {tuple(obs.shape)}")
    print(f"action dim                            : {action_dim}")
    print(f"resolved track_lin_vel_xy_exp std     : {std}")
    print(f"steps                                 : {args_cli.smoke_steps}")
    print(f"reset/termination count               : {reset_count}")
    print(f"initial/final policy obs finite       : PASS")
    print(f"reward finite                         : {'PASS' if reward_finite else 'FAIL'}")
    print("checkpoint loaded                     : NO")
    print("PPO update                            : NO")
    print()
    print("V4C_RUNTIME_SMOKE_PASS")
    print("=" * 108)

finally:
    if env is not None:
        env.close()
    simulation_app.close()
