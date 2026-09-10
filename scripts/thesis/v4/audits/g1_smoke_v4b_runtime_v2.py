import argparse
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Robust runtime smoke test for v4-B direct-yaw task.")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--smoke_steps", type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

TASK_ID = "Humanoid-RL-Lab-G1-Flat-V4B-Direct-Yaw-v0"
env = None

def policy_tensor(obs):
    """Extract the policy tensor from Isaac Lab/Gym observation containers."""
    if isinstance(obs, tuple) and len(obs) >= 1:
        obs = obs[0]
    if hasattr(obs, "keys") and "policy" in obs.keys():
        return obs["policy"]
    if isinstance(obs, dict) and "policy" in obs:
        return obs["policy"]
    raise RuntimeError(
        f"Observation container {type(obs).__name__} does not expose a 'policy' group."
    )

try:
    import gymnasium as gym
    import torch
    import humanoid_rl_lab.tasks  # noqa: F401

    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(
        TASK_ID,
        device="cuda:0",
        num_envs=int(args_cli.num_envs),
    )
    env_cfg.seed = 42

    print("V4B_SMOKE_PREMAKE_PASS", flush=True)
    env = gym.make(TASK_ID, cfg=env_cfg)
    base_env = env.unwrapped
    print("V4B_SMOKE_ENV_CREATED_PASS", flush=True)

    # Use the public Gymnasium lifecycle. Do not call base_env.get_observations()
    # before reset; that was the fragile point in the previous smoke script.
    reset_result = env.reset()
    policy_obs = policy_tensor(reset_result)
    print("V4B_SMOKE_RESET_PASS", flush=True)

    step_dt = float(base_env.step_dt)
    action_dim = int(base_env.action_manager.total_action_dim)
    num_envs = int(policy_obs.shape[0])
    obs_dim = int(policy_obs.shape[-1])
    initial_finite = bool(torch.isfinite(policy_obs).all().item())

    actions = torch.zeros(
        (num_envs, action_dim),
        device=policy_obs.device,
        dtype=torch.float32,
    )

    total_done = 0
    reward_min = float("inf")
    reward_max = float("-inf")

    for step_index in range(int(args_cli.smoke_steps)):
        result = env.step(actions)
        if not isinstance(result, tuple):
            raise RuntimeError(
                f"Unexpected env.step return type at step {step_index}: {type(result).__name__}"
            )

        if len(result) == 5:
            step_obs, rewards, terminated, truncated, _info = result
            dones = torch.logical_or(
                torch.as_tensor(terminated), torch.as_tensor(truncated)
            )
        elif len(result) == 4:
            step_obs, rewards, dones, _info = result
            dones = torch.as_tensor(dones)
        else:
            raise RuntimeError(
                f"Unexpected env.step tuple length at step {step_index}: {len(result)}"
            )

        policy_obs = policy_tensor(step_obs)
        if tuple(policy_obs.shape) != (num_envs, obs_dim):
            raise RuntimeError(
                f"Policy observation shape changed at step {step_index}: "
                f"{tuple(policy_obs.shape)}"
            )
        if not bool(torch.isfinite(policy_obs).all().item()):
            raise RuntimeError(f"Non-finite policy observation at step {step_index}.")

        rewards_t = torch.as_tensor(rewards)
        if not bool(torch.isfinite(rewards_t).all().item()):
            raise RuntimeError(f"Non-finite reward at step {step_index}.")
        reward_min = min(reward_min, float(rewards_t.min().item()))
        reward_max = max(reward_max, float(rewards_t.max().item()))
        total_done += int(torch.count_nonzero(dones).item())

    final_finite = bool(torch.isfinite(policy_obs).all().item())

    checks = {
        "seed_42": int(base_env.cfg.seed) == 42,
        "step_dt_50hz": abs(step_dt - 0.02) <= 1.0e-9,
        "obs_dim_123": obs_dim == 123,
        "action_dim_37": action_dim == 37,
        "initial_obs_finite": initial_finite,
        "post_step_obs_finite": final_finite,
    }

    print("=" * 104)
    print("V4-B RUNTIME SMOKE TEST")
    print("=" * 104)
    print(f"Task ID                             : {TASK_ID}")
    print(f"num_envs                            : {num_envs}")
    print(f"seed                                : {base_env.cfg.seed}")
    print(f"device                              : {policy_obs.device}")
    print(f"step_dt                             : {step_dt}")
    print(f"policy observation shape            : {tuple(policy_obs.shape)}")
    print(f"policy observation dim              : {obs_dim}")
    print(f"action dim                          : {action_dim}")
    print(f"smoke steps                         : {int(args_cli.smoke_steps)}")
    print(f"total env done/reset indications    : {total_done}")
    print(f"reward range                        : [{reward_min:.6f}, {reward_max:.6f}]")
    print()
    print("CHECKS")
    print("-" * 104)
    for name, ok in checks.items():
        print(f"{name:<36} {'PASS' if ok else 'FAIL'}")

    if not all(checks.values()):
        failed = [name for name, ok in checks.items() if not ok]
        raise RuntimeError(f"Runtime smoke checks failed: {failed}")

    print()
    print("V4B_RUNTIME_SMOKE_PASS")
    print("No checkpoint loaded. No PPO update. No training performed.")
    print("=" * 104)
    sys.stdout.flush()

except BaseException as exc:
    print("=" * 104, file=sys.stderr, flush=True)
    print("V4B_RUNTIME_SMOKE_ABORT", file=sys.stderr, flush=True)
    print(f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    traceback.print_exc()
    sys.stderr.flush()
    raise

finally:
    if env is not None:
        try:
            env.close()
        except Exception as close_exc:
            print(f"[WARN] env.close(): {close_exc}", file=sys.stderr, flush=True)
    try:
        simulation_app.close()
    except Exception as close_exc:
        print(f"[WARN] simulation_app.close(): {close_exc}", file=sys.stderr, flush=True)
