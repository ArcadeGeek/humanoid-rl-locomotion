from __future__ import annotations

import copy
import sys
import traceback

import gymnasium as gym

import humanoid_rl_lab.tasks  # noqa: F401

from isaaclab_tasks.utils import load_cfg_from_registry


TASK_B = "Humanoid-RL-Lab-G1-Flat-V4B-Direct-Yaw-v0"
TASK_N = "Humanoid-RL-Lab-G1-Flat-V4N-Symmetry-Augmentation-v0"

EXPECTED_ENV_CLASS = "HumanoidRlLabG1V4BDirectYawEnvCfg"
EXPECTED_PPO_CLASS = "V4NSymmetryAugmentationPPORunnerCfg"
EXPECTED_EXPERIMENT = "humanoid_rl_lab_g1_v4n_symmetry_augmentation_scratch"
EXPECTED_SYMMETRY_FUNC = "compute_g1_left_right_symmetric_states"


def p(*args):
    print(*args, flush=True)


def as_dict(cfg):
    if hasattr(cfg, "to_dict") and callable(cfg.to_dict):
        return cfg.to_dict()
    raise RuntimeError(
        f"config object has no to_dict(): {type(cfg).__module__}.{type(cfg).__qualname__}"
    )


def canonicalize(value):
    if isinstance(value, dict):
        return {
            str(k): canonicalize(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [canonicalize(v) for v in value]
    if callable(value):
        return (
            getattr(value, "__module__", type(value).__module__)
            + "."
            + getattr(
                value,
                "__qualname__",
                getattr(value, "__name__", type(value).__qualname__),
            )
        )
    return value


def drop_path(data: dict, path: tuple[str, ...]) -> None:
    cursor = data
    for key in path[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            return
        cursor = cursor[key]
    cursor.pop(path[-1], None)


failed = False

try:
    spec_b = gym.spec(TASK_B)
    spec_n = gym.spec(TASK_N)

    env_ep_b = spec_b.kwargs["env_cfg_entry_point"]
    env_ep_n = spec_n.kwargs["env_cfg_entry_point"]

    p("=" * 132)
    p("v4-N RESOLVED CONFIG AUDIT")
    p("=" * 132)
    p(f"TASK_B                  : {TASK_B}")
    p(f"TASK_N                  : {TASK_N}")
    p(f"v4-B env entry point    : {env_ep_b}")
    p(f"v4-N env entry point    : {env_ep_n}")

    if env_ep_b != env_ep_n:
        raise RuntimeError(
            "v4-N env entry point differs from v4-B; expected exact reuse"
        )

    env_b = load_cfg_from_registry(TASK_B, "env_cfg_entry_point")
    env_n = load_cfg_from_registry(TASK_N, "env_cfg_entry_point")

    env_class_b = type(env_b).__name__
    env_class_n = type(env_n).__name__

    p(f"v4-B env class          : {env_class_b}")
    p(f"v4-N env class          : {env_class_n}")

    if env_class_b != EXPECTED_ENV_CLASS or env_class_n != EXPECTED_ENV_CLASS:
        raise RuntimeError(
            f"unexpected env classes: B={env_class_b}, N={env_class_n}"
        )

    env_dict_b = canonicalize(as_dict(env_b))
    env_dict_n = canonicalize(as_dict(env_n))

    if env_dict_b != env_dict_n:
        raise RuntimeError(
            "resolved v4-N env cfg is not exactly equal to resolved v4-B env cfg"
        )

    p("resolved env equality  : PASS")

    ppo_b = load_cfg_from_registry(TASK_B, "rsl_rl_cfg_entry_point")
    ppo_n = load_cfg_from_registry(TASK_N, "rsl_rl_cfg_entry_point")

    p(f"v4-B PPO class          : {type(ppo_b).__name__}")
    p(f"v4-N PPO class          : {type(ppo_n).__name__}")

    if type(ppo_n).__name__ != EXPECTED_PPO_CLASS:
        raise RuntimeError(
            f"unexpected v4-N PPO class: {type(ppo_n).__name__}"
        )

    if ppo_n.experiment_name != EXPECTED_EXPERIMENT:
        raise RuntimeError(
            f"unexpected experiment_name: {ppo_n.experiment_name}"
        )

    symmetry = ppo_n.algorithm.symmetry_cfg

    if symmetry is None:
        raise RuntimeError("v4-N symmetry_cfg resolved to None")

    func = symmetry.data_augmentation_func
    func_name = getattr(func, "__name__", None)
    func_module = getattr(func, "__module__", None)

    p()
    p("-" * 132)
    p("SYMMETRY CFG")
    p("-" * 132)
    p(f"use_data_augmentation    : {symmetry.use_data_augmentation}")
    p(f"use_mirror_loss          : {symmetry.use_mirror_loss}")
    p(f"mirror_loss_coeff         : {symmetry.mirror_loss_coeff}")
    p(f"data_augmentation_func    : {func_module}.{func_name}")

    if symmetry.use_data_augmentation is not True:
        raise RuntimeError("use_data_augmentation must be True")
    if symmetry.use_mirror_loss is not False:
        raise RuntimeError("use_mirror_loss must be False")
    if float(symmetry.mirror_loss_coeff) != 0.0:
        raise RuntimeError(
            f"mirror_loss_coeff must be 0.0, got {symmetry.mirror_loss_coeff}"
        )
    if func_name != EXPECTED_SYMMETRY_FUNC:
        raise RuntimeError(
            f"unexpected symmetry function: {func_module}.{func_name}"
        )

    # Sole-PPO-delta audit:
    # v4-B has no enabled symmetry config. Remove the two intentional v4-N changes:
    # experiment_name and algorithm.symmetry_cfg, then require exact equality.
    ppo_dict_b = canonicalize(as_dict(ppo_b))
    ppo_dict_n = canonicalize(as_dict(ppo_n))

    b_cmp = copy.deepcopy(ppo_dict_b)
    n_cmp = copy.deepcopy(ppo_dict_n)

    drop_path(b_cmp, ("experiment_name",))
    drop_path(n_cmp, ("experiment_name",))
    drop_path(b_cmp, ("algorithm", "symmetry_cfg"))
    drop_path(n_cmp, ("algorithm", "symmetry_cfg"))

    if b_cmp != n_cmp:
        raise RuntimeError(
            "PPO configs differ beyond experiment_name and algorithm.symmetry_cfg"
        )

    p()
    p("-" * 132)
    p("CAUSAL DELTA CLOSURE")
    p("-" * 132)
    p("env cfg delta           : NONE — exact v4-B reuse")
    p("PPO delta               : algorithm.symmetry_cfg only")
    p("metadata delta          : experiment_name only")
    p("mirror loss             : OFF")
    p("data augmentation       : ON")
    p("genuine scratch config  : no resume/checkpoint fields introduced by v4-N")
    p()
    p("V4N_RESOLVED_CONFIG_AUDIT_PASS")
    p("=" * 132)

except Exception:
    failed = True
    p()
    p("=" * 132)
    p("V4N_RESOLVED_CONFIG_AUDIT_FAIL")
    p("=" * 132)
    traceback.print_exc(file=sys.stdout)
    sys.stdout.flush()

if failed:
    raise SystemExit(1)
