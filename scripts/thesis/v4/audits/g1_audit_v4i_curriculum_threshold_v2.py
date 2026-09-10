from __future__ import annotations

import argparse
import math
import sys
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Runtime audit for the v4-I delayed reward-weight curriculum threshold."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import humanoid_rl_lab.tasks  # noqa: F401


TASK = "Humanoid-RL-Lab-G1-Flat-V4I-Delayed-Bounded-Zero-Command-Full-Motion-v0"
REWARD_TERM = "zero_command_bounded_full_motion"
CURRICULUM_TERM = "zero_command_bounded_full_motion_weight"
NUM_ENVS = 4
SEED = 42
INITIAL_WEIGHT = 0.0
TARGET_WEIGHT = -5.0
THRESHOLD = 4800


def fail(message: str) -> None:
    raise RuntimeError(message)


def assert_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1.0e-12):
        fail(f"{label}: expected {expected}, got {actual}")


def get_named_term_cfg(manager, term_name: str, manager_label: str):
    getter = getattr(manager, "get_term_cfg", None)
    if callable(getter):
        return getter(term_name)

    term_names = getattr(manager, "_term_names", None)
    term_cfgs = getattr(manager, "_term_cfgs", None)

    if term_names is None or term_cfgs is None:
        fail(
            f"{manager_label}: unable to inspect term cfg; "
            "no get_term_cfg() and no _term_names/_term_cfgs"
        )

    names = list(term_names)
    if term_name not in names:
        fail(
            f"{manager_label}: term {term_name!r} not found; "
            f"available={names}"
        )

    idx = names.index(term_name)
    if idx >= len(term_cfgs):
        fail(
            f"{manager_label}: inconsistent internal term lists: "
            f"index={idx}, cfg_count={len(term_cfgs)}"
        )
    return term_cfgs[idx]


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

    reward_manager = unwrapped.reward_manager
    curriculum_manager = unwrapped.curriculum_manager

    reward_active_terms = list(reward_manager.active_terms)
    curriculum_active_terms = list(curriculum_manager.active_terms)

    if REWARD_TERM not in reward_active_terms:
        fail(
            f"reward term not registered: {REWARD_TERM}; "
            f"active={reward_active_terms}"
        )

    if CURRICULUM_TERM not in curriculum_active_terms:
        fail(
            f"curriculum term not registered: {CURRICULUM_TERM}; "
            f"active={curriculum_active_terms}"
        )

    reward_cfg = get_named_term_cfg(
        reward_manager,
        REWARD_TERM,
        "RewardManager",
    )
    assert_close(reward_cfg.weight, INITIAL_WEIGHT, "initial reward weight")

    curriculum_cfg = get_named_term_cfg(
        curriculum_manager,
        CURRICULUM_TERM,
        "CurriculumManager",
    )
    params = dict(curriculum_cfg.params)

    if params.get("term_name") != REWARD_TERM:
        fail(
            f"curriculum term_name mismatch: expected {REWARD_TERM!r}, "
            f"got {params.get('term_name')!r}"
        )
    assert_close(params.get("weight"), TARGET_WEIGHT, "curriculum target weight")

    if int(params.get("num_steps")) != THRESHOLD:
        fail(
            f"curriculum num_steps mismatch: expected {THRESHOLD}, "
            f"got {params.get('num_steps')!r}"
        )

    env_ids = torch.arange(
        NUM_ENVS,
        device=unwrapped.device,
        dtype=torch.long,
    )

    observations = []

    for counter, expected_weight in (
        (4799, INITIAL_WEIGHT),
        (4800, INITIAL_WEIGHT),
        (4801, TARGET_WEIGHT),
    ):
        unwrapped.common_step_counter = counter
        curriculum_manager.compute(env_ids=env_ids)

        current_cfg = get_named_term_cfg(
            reward_manager,
            REWARD_TERM,
            "RewardManager",
        )
        actual_weight = float(current_cfg.weight)

        assert_close(
            actual_weight,
            expected_weight,
            f"reward weight at common_step_counter={counter}",
        )
        observations.append((counter, actual_weight))

    unwrapped.common_step_counter = 5000
    curriculum_manager.compute(env_ids=env_ids)

    persistent_cfg = get_named_term_cfg(
        reward_manager,
        REWARD_TERM,
        "RewardManager",
    )
    persistent_weight = float(persistent_cfg.weight)
    assert_close(
        persistent_weight,
        TARGET_WEIGHT,
        "post-threshold persistent weight",
    )

    print("=" * 124)
    print("V4-I CURRICULUM THRESHOLD AUDIT V2")
    print("=" * 124)
    print(f"task                        : {TASK}")
    print(f"reward term                 : {REWARD_TERM}")
    print(f"curriculum term             : {CURRICULUM_TERM}")
    print(f"num envs                    : {NUM_ENVS}")
    print(f"seed                        : {SEED}")
    print(f"configured initial weight   : {INITIAL_WEIGHT}")
    print(f"configured target weight    : {TARGET_WEIGHT}")
    print(f"configured num_steps        : {THRESHOLD}")
    print("native switch condition     : common_step_counter > num_steps")
    print()
    for counter, weight in observations:
        print(f"counter {counter:4d} -> reward weight {weight: .1f}")
    print(f"counter 5000 -> reward weight {persistent_weight: .1f}")
    print()
    print("4799 keeps initial weight   : PASS")
    print("4800 keeps initial weight   : PASS")
    print("4801 switches to target     : PASS")
    print("post-threshold persistence  : PASS")
    print("checkpoint loaded           : NO")
    print("PPO update performed        : NO")
    print()
    print("V4I_CURRICULUM_THRESHOLD_AUDIT_V2_PASS")
    print("=" * 124)

except BaseException as exc:
    print("", flush=True)
    print("=" * 124, flush=True)
    print("V4I_CURRICULUM_THRESHOLD_AUDIT_V2_EXCEPTION", flush=True)
    print(f"type    : {type(exc).__name__}", flush=True)
    print(f"message : {exc}", flush=True)
    print("traceback:", flush=True)
    traceback.print_exc(file=sys.stdout)
    print("=" * 124, flush=True)
    raise
finally:
    if env is not None:
        env.close()
    simulation_app.close()
