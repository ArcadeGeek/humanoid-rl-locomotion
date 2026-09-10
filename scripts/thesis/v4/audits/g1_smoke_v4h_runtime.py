from __future__ import annotations

import argparse
import math
import sys
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Minimal runtime smoke test for v4-H bounded zero-command full-motion."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import humanoid_rl_lab.tasks  # noqa: F401


TASK = "Humanoid-RL-Lab-G1-Flat-V4H-Bounded-Zero-Command-Full-Motion-v0"
NUM_ENVS = 4
STEPS = 16
SEED = 42
EXPECTED_ACTION_DIM = 37
EXPECTED_REWARD_COUNT = 17
EXPECTED_FOOT_IDS = [24, 25]
REWARD_NAME = "zero_command_bounded_full_motion"


def fail(msg: str) -> None:
    raise RuntimeError(msg)


def assert_finite_tree(value, path: str = "obs") -> None:
    if torch.is_tensor(value):
        if not torch.isfinite(value).all():
            fail(f"non-finite tensor at {path}")
        return

    if isinstance(value, dict):
        for key, child in value.items():
            assert_finite_tree(child, f"{path}.{key}")
        return

    if hasattr(value, "items"):
        for key, child in value.items():
            assert_finite_tree(child, f"{path}.{key}")
        return

    # Non-tensor metadata values are ignored intentionally.


def get_reward_term_cfg(reward_manager, name: str):
    getter = getattr(reward_manager, "get_term_cfg", None)
    if callable(getter):
        return getter(name)

    term_names = getattr(reward_manager, "_term_names", None)
    term_cfgs = getattr(reward_manager, "_term_cfgs", None)
    if term_names is None or term_cfgs is None:
        fail("unable to inspect reward-manager term configuration")

    matches = [cfg for term_name, cfg in zip(term_names, term_cfgs, strict=True) if term_name == name]
    if len(matches) != 1:
        fail(f"expected one reward cfg for {name!r}, found {len(matches)}")
    return matches[0]


env = None

try:
    env_cfg = parse_env_cfg(
        TASK,
        device=args_cli.device,
        num_envs=NUM_ENVS,
    )
    env_cfg.seed = SEED

    env = gym.make(TASK, cfg=env_cfg)
    unwrapped = env.unwrapped

    action_dim = int(unwrapped.action_manager.total_action_dim)
    if action_dim != EXPECTED_ACTION_DIM:
        fail(f"action dimension mismatch: expected {EXPECTED_ACTION_DIM}, got {action_dim}")

    reward_manager = unwrapped.reward_manager
    active_terms = list(reward_manager.active_terms)
    if len(active_terms) != EXPECTED_REWARD_COUNT:
        fail(
            f"active reward count mismatch: expected {EXPECTED_REWARD_COUNT}, "
            f"got {len(active_terms)}"
        )
    if REWARD_NAME not in active_terms:
        fail(f"missing active reward term: {REWARD_NAME}")

    reward_cfg = get_reward_term_cfg(reward_manager, REWARD_NAME)

    if float(reward_cfg.weight) != -5.0:
        fail(f"reward weight mismatch: {reward_cfg.weight!r}")

    params = reward_cfg.params
    expected_params = {
        "command_name": "base_velocity",
        "command_threshold": 0.06,
        "base_planar_scale": 1.0,
        "yaw_sigma": 0.20,
        "foot_xy_scale": 0.25,
    }
    for key, expected in expected_params.items():
        actual = params[key]
        if isinstance(expected, float):
            if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1.0e-12):
                fail(f"{key} mismatch: expected {expected}, got {actual!r}")
        elif actual != expected:
            fail(f"{key} mismatch: expected {expected!r}, got {actual!r}")

    asset_cfg = params["asset_cfg"]
    body_ids = list(asset_cfg.body_ids)
    if body_ids != EXPECTED_FOOT_IDS:
        fail(f"resolved foot ids mismatch: expected {EXPECTED_FOOT_IDS}, got {body_ids}")

    reset_result = env.reset(seed=SEED)
    if not isinstance(reset_result, tuple) or len(reset_result) != 2:
        fail("env.reset(seed=SEED) did not return (observations, info)")

    obs, _ = reset_result
    assert_finite_tree(obs, "reset_obs")

    device = unwrapped.device
    actions = torch.zeros((NUM_ENVS, action_dim), device=device, dtype=torch.float32)

    total_terminated = 0
    total_truncated = 0
    bounded_min = float("inf")
    bounded_max = float("-inf")
    term_logged_min = float("inf")
    term_logged_max = float("-inf")

    term_index = active_terms.index(REWARD_NAME)

    for step_idx in range(STEPS):
        obs, rewards, terminated, truncated, _ = env.step(actions)

        assert_finite_tree(obs, f"step_{step_idx}.obs")

        for name, tensor in (
            ("rewards", rewards),
            ("terminated", terminated),
            ("truncated", truncated),
        ):
            if not torch.is_tensor(tensor):
                fail(f"{name} is not a torch.Tensor: {type(tensor)!r}")

        if not torch.isfinite(rewards).all():
            fail(f"non-finite total reward at step {step_idx}")

        total_terminated += int(torch.count_nonzero(terminated).item())
        total_truncated += int(torch.count_nonzero(truncated).item())

        robot = unwrapped.scene["robot"]
        root_ang_vel_b = robot.data.root_ang_vel_b
        if hasattr(root_ang_vel_b, "torch"):
            root_ang_vel_b = root_ang_vel_b.torch

        wz = root_ang_vel_b[:, 2]
        sigma = float(params["yaw_sigma"])
        bounded_yaw = 1.0 - torch.exp(-torch.square(wz / sigma))

        if not torch.isfinite(bounded_yaw).all():
            fail(f"non-finite bounded yaw at step {step_idx}")

        this_min = float(bounded_yaw.min().item())
        this_max = float(bounded_yaw.max().item())
        bounded_min = min(bounded_min, this_min)
        bounded_max = max(bounded_max, this_max)

        if this_min < -1.0e-7 or this_max > 1.0 + 1.0e-7:
            fail(
                f"bounded yaw escaped [0,1] at step {step_idx}: "
                f"min={this_min}, max={this_max}"
            )

        step_reward = reward_manager._step_reward[:, term_index]
        if not torch.isfinite(step_reward).all():
            fail(f"non-finite {REWARD_NAME} logged term at step {step_idx}")

        logged_min = float(step_reward.min().item())
        logged_max = float(step_reward.max().item())
        term_logged_min = min(term_logged_min, logged_min)
        term_logged_max = max(term_logged_max, logged_max)

        # Positive motion cost times negative weight must never become positive.
        if logged_max > 1.0e-6:
            fail(
                f"{REWARD_NAME} logged contribution became positive at step {step_idx}: "
                f"max={logged_max}"
            )

    print("=" * 124)
    print("V4-H RUNTIME SMOKE")
    print("=" * 124)
    print(f"task                     : {TASK}")
    print(f"num envs                 : {NUM_ENVS}")
    print(f"steps                    : {STEPS}")
    print(f"seed                     : {SEED}")
    print(f"action dim               : {action_dim}")
    print(f"active rewards           : {len(active_terms)}")
    print(f"new reward               : {REWARD_NAME}")
    print(f"weight                   : {reward_cfg.weight}")
    print(f"command threshold        : {params['command_threshold']}")
    print(f"base planar scale        : {params['base_planar_scale']}")
    print(f"yaw sigma                : {params['yaw_sigma']}")
    print(f"foot XY scale            : {params['foot_xy_scale']}")
    print(f"resolved foot ids        : {body_ids}")
    print(f"bounded yaw observed min : {bounded_min:.9f}")
    print(f"bounded yaw observed max : {bounded_max:.9f}")
    print(f"logged term observed min : {term_logged_min:.9f}")
    print(f"logged term observed max : {term_logged_max:.9f}")
    print(f"terminations observed    : {total_terminated}")
    print(f"truncations observed     : {total_truncated}")
    print("finite observations      : PASS")
    print("finite total rewards     : PASS")
    print("bounded yaw in [0,1]     : PASS")
    print("new reward sign          : PASS")
    print("checkpoint loaded        : NO")
    print("PPO update performed     : NO")
    print()
    print("V4H_RUNTIME_SMOKE_PASS")
    print("=" * 124)

except BaseException as exc:
    print("", flush=True)
    print("=" * 120, flush=True)
    print("V4H_RUNTIME_SMOKE_EXCEPTION", flush=True)
    print(f"type    : {type(exc).__name__}", flush=True)
    print(f"message : {exc}", flush=True)
    print("traceback:", flush=True)
    traceback.print_exc(file=sys.stdout)
    print("=" * 120, flush=True)
    raise
finally:
    if env is not None:
        env.close()
    simulation_app.close()
