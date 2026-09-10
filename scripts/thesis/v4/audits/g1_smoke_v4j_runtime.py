from __future__ import annotations

import argparse


TASK_ID = "Humanoid-RL-Lab-G1-Flat-V4J-Straight-Line-Precision-v0"
REWARD_NAME = "straight_line_lateral_velocity_ema_bounded"
NUM_ENVS = 4
SEED = 42


def fail(message: str) -> None:
    raise RuntimeError(message)


parser = argparse.ArgumentParser(description="Runtime smoke test for v4-J.")
from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

raw_env = None

try:
    import gymnasium as gym
    import torch

    import humanoid_rl_lab.tasks  # noqa: F401
    from isaaclab.managers import ManagerTermBase
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(
        TASK_ID,
        device=args_cli.device,
        num_envs=NUM_ENVS,
    )
    env_cfg.seed = SEED

    raw_env = gym.make(TASK_ID, cfg=env_cfg)
    env = raw_env.unwrapped

    if env.num_envs != NUM_ENVS:
        fail(f"unexpected num_envs: {env.num_envs}")
    if abs(float(env.step_dt) - 0.02) > 1.0e-12:
        fail(f"unexpected step_dt: {env.step_dt}")

    if REWARD_NAME not in env.reward_manager.active_terms:
        fail(f"reward term not active: {REWARD_NAME}")

    reward_cfg = env.reward_manager.get_term_cfg(REWARD_NAME)
    reward_term = reward_cfg.func

    if not isinstance(reward_term, ManagerTermBase):
        fail(f"reward term is not ManagerTermBase: {type(reward_term)}")
    if reward_term.__class__.__name__ != REWARD_NAME:
        fail(
            "unexpected reward class: "
            f"{reward_term.__class__.__name__}"
        )
    if not hasattr(reward_term, "_ema_error"):
        fail("reward term does not expose _ema_error runtime state")
    if reward_term._ema_error.shape != (NUM_ENVS,):
        fail(f"unexpected EMA state shape: {tuple(reward_term._ema_error.shape)}")
    if reward_term._ema_error.device != torch.device(env.device):
        fail(
            f"EMA state device mismatch: {reward_term._ema_error.device} vs {env.device}"
        )

    expected_alpha = 1.0 - torch.exp(
        torch.tensor(
            -float(env.step_dt) / 0.75,
            device=env.device,
            dtype=torch.float32,
        )
    )
    if not torch.allclose(
        reward_term._alpha,
        expected_alpha,
        rtol=0.0,
        atol=1.0e-7,
    ):
        fail(
            f"unexpected EMA alpha: {float(reward_term._alpha)} "
            f"expected {float(expected_alpha)}"
        )

    # Verify RewardManager propagates selective resets to the class-based term.
    reward_term._ema_error.fill_(0.123)
    reset_ids = torch.tensor([0, 2], device=env.device, dtype=torch.long)
    env.reward_manager.reset(env_ids=reset_ids)

    ema_after_selective_reset = reward_term._ema_error.detach().clone()
    if not torch.allclose(
        ema_after_selective_reset[reset_ids],
        torch.zeros(2, device=env.device),
        rtol=0.0,
        atol=0.0,
    ):
        fail(
            "RewardManager selective reset did not zero requested EMA states: "
            f"{ema_after_selective_reset.tolist()}"
        )

    keep_ids = torch.tensor([1, 3], device=env.device, dtype=torch.long)
    expected_kept = torch.full((2,), 0.123, device=env.device)
    if not torch.allclose(
        ema_after_selective_reset[keep_ids],
        expected_kept,
        rtol=0.0,
        atol=1.0e-7,
    ):
        fail(
            "RewardManager selective reset unexpectedly altered non-requested EMA states: "
            f"{ema_after_selective_reset.tolist()}"
        )

    reward_term.reset()

    # Reset the physical environment before runtime reward evaluation.
    raw_env.reset(seed=SEED)

    command = env.command_manager.get_command("base_velocity")
    if command.shape[0] != NUM_ENVS or command.shape[1] < 3:
        fail(f"unexpected command-buffer shape: {tuple(command.shape)}")

    # Active-gate direct call: finite bounded values and retained EMA state.
    command.zero_()
    command[:, 0] = 0.40
    reward_term._ema_error.fill_(0.020)

    active_cost = reward_term(env, **reward_cfg.params)

    if active_cost.shape != (NUM_ENVS,):
        fail(f"unexpected active cost shape: {tuple(active_cost.shape)}")
    if not torch.isfinite(active_cost).all():
        fail(f"non-finite active cost: {active_cost}")
    if torch.any(active_cost < 0.0) or torch.any(active_cost > 1.0 + 1.0e-6):
        fail(f"active cost outside [0,1]: {active_cost}")
    if not torch.any(torch.abs(reward_term._ema_error) > 0.0):
        fail("active gate unexpectedly zeroed all EMA state")

    active_cost_mean = float(active_cost.mean().item())

    # Inactive-gate direct call: exact zero output and exact EMA reset.
    command.zero_()
    reward_term._ema_error.fill_(0.020)

    inactive_cost = reward_term(env, **reward_cfg.params)

    if not torch.allclose(
        inactive_cost,
        torch.zeros_like(inactive_cost),
        rtol=0.0,
        atol=0.0,
    ):
        fail(f"inactive gate returned non-zero cost: {inactive_cost}")
    if not torch.allclose(
        reward_term._ema_error,
        torch.zeros_like(reward_term._ema_error),
        rtol=0.0,
        atol=0.0,
    ):
        fail(
            "inactive gate did not clear EMA state: "
            f"{reward_term._ema_error.tolist()}"
        )

    # Real environment steps: ensure RewardManager can execute the class term in the normal loop.
    command.zero_()
    command[:, 0] = 0.40
    reward_term.reset()

    action_dim = int(env.action_manager.total_action_dim)
    actions = torch.zeros(
        (NUM_ENVS, action_dim),
        device=env.device,
        dtype=torch.float32,
    )

    for _ in range(5):
        raw_env.step(actions)

    if not torch.isfinite(reward_term._ema_error).all():
        fail(
            "EMA state became non-finite during normal env.step loop: "
            f"{reward_term._ema_error}"
        )

    runtime_ema_abs_max = float(torch.max(torch.abs(reward_term._ema_error)).item())

    print("=" * 108)
    print("v4-J RUNTIME SMOKE")
    print("=" * 108)
    print(f"Task ID                         : {TASK_ID}")
    print(f"num_envs                        : {env.num_envs}")
    print(f"device                          : {env.device}")
    print(f"step_dt                         : {float(env.step_dt):.5f} s")
    print(f"action dim                      : {action_dim}")
    print(f"reward active                   : YES")
    print(f"reward class                    : {reward_term.__class__.__name__}")
    print(f"EMA state shape                 : {tuple(reward_term._ema_error.shape)}")
    print(f"EMA alpha                       : {float(reward_term._alpha):.8f}")
    print("RewardManager selective reset   : PASS")
    print(f"active-gate bounded cost mean   : {active_cost_mean:.8f}")
    print("inactive-gate cost              : 0")
    print("inactive-gate EMA reset         : PASS")
    print("normal env.step executions      : 5")
    print(f"post-step EMA max abs           : {runtime_ema_abs_max:.8f}")
    print()
    print("V4J_RUNTIME_SMOKE_PASS")
    print("=" * 108)

finally:
    if raw_env is not None:
        raw_env.close()
    simulation_app.close()
