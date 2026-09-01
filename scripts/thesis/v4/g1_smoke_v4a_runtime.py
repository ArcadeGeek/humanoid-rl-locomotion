from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Runtime smoke test for humanoid_rl_lab v4-A.")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--smoke_steps", type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import humanoid_rl_lab.tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

TASK_ID = "Humanoid-RL-Lab-G1-Flat-V4A-Official-Baseline-v0"
EXPECTED_OBS_DIM = 123
EXPECTED_ACTION_DIM = 37
EXPECTED_STEP_DT = 0.02

def policy_tensor(obs):
    """Extract the policy tensor from either TensorDict/dict or a direct tensor."""
    if torch.is_tensor(obs):
        return obs
    if hasattr(obs, "keys"):
        keys = list(obs.keys())
        if "policy" not in keys:
            raise RuntimeError(f"Observation container has no 'policy' key. keys={keys}")
        tensor = obs["policy"]
        if not torch.is_tensor(tensor):
            raise RuntimeError(f"obs['policy'] is not a torch.Tensor: {type(tensor)}")
        return tensor
    raise RuntimeError(f"Unsupported observation container type: {type(obs)}")

env = None
try:
    env_cfg = load_cfg_from_registry(TASK_ID, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(TASK_ID, "rsl_rl_cfg_entry_point")

    env_cfg.scene.num_envs = int(args_cli.num_envs)
    env_cfg.seed = int(agent_cfg.seed)
    if getattr(args_cli, "device", None) is not None:
        env_cfg.sim.device = args_cli.device

    raw_env = gym.make(TASK_ID, cfg=env_cfg)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    uenv = env.unwrapped

    step_dt = float(uenv.step_dt)
    action_dim = int(uenv.action_manager.total_action_dim)

    obs = env.get_observations()
    policy_obs = policy_tensor(obs)
    obs_shape = tuple(policy_obs.shape)
    obs_dim = int(obs_shape[-1])

    print("=" * 104)
    print("V4-A RUNTIME SMOKE TEST")
    print(f"Task ID                             : {TASK_ID}")
    print(f"num_envs                            : {args_cli.num_envs}")
    print(f"device                              : {uenv.device}")
    print(f"step_dt                             : {step_dt}")
    print(f"observation container               : {type(obs).__name__}")
    print(f"policy observation shape            : {obs_shape}")
    print(f"policy observation dim              : {obs_dim}")
    print(f"action dim                          : {action_dim}")
    print(f"initial policy obs finite           : {bool(torch.isfinite(policy_obs).all().item())}")
    print("=" * 104)

    checks = {
        "step_dt_50hz": abs(step_dt - EXPECTED_STEP_DT) <= 1.0e-12,
        "obs_dim_123": obs_dim == EXPECTED_OBS_DIM,
        "action_dim_37": action_dim == EXPECTED_ACTION_DIM,
        "initial_obs_finite": bool(torch.isfinite(policy_obs).all().item()),
    }

    total_done = 0
    reward_min = float("inf")
    reward_max = float("-inf")

    for i in range(int(args_cli.smoke_steps)):
        actions = torch.zeros(
            (int(args_cli.num_envs), action_dim),
            dtype=policy_obs.dtype,
            device=uenv.device,
        )
        obs, rewards, dones, extras = env.step(actions)
        policy_obs = policy_tensor(obs)

        if not torch.isfinite(policy_obs).all():
            raise RuntimeError(f"Non-finite policy observation at smoke step {i + 1}")
        if not torch.isfinite(rewards).all():
            raise RuntimeError(f"Non-finite reward at smoke step {i + 1}")

        reward_min = min(reward_min, float(torch.min(rewards).item()))
        reward_max = max(reward_max, float(torch.max(rewards).item()))
        total_done += int(torch.count_nonzero(dones).item())

    checks["post_step_obs_finite"] = bool(torch.isfinite(policy_obs).all().item())

    print("RUNTIME")
    print(f"  smoke steps                       : {args_cli.smoke_steps}")
    print(f"  total env done/reset indications  : {total_done}")
    print(f"  reward range                      : [{reward_min:.6f}, {reward_max:.6f}]")
    print(f"  final policy obs finite           : {bool(torch.isfinite(policy_obs).all().item())}")

    print("CHECKS")
    failed = []
    for name, ok in checks.items():
        print(f"  {name:<34}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            failed.append(name)

    if failed:
        print("=" * 104)
        print("V4A_RUNTIME_SMOKE_FAIL")
        print("FAILED:", ", ".join(failed))
        raise SystemExit(2)

    print("=" * 104)
    print("V4A_RUNTIME_SMOKE_PASS")
    print("=" * 104)

finally:
    if env is not None:
        env.close()
    simulation_app.close()
