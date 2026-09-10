from __future__ import annotations

import copy
import math
from pathlib import Path

import gymnasium as gym

import humanoid_rl_lab.tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry


TASK_B = "Humanoid-RL-Lab-G1-Flat-V4B-Direct-Yaw-v0"
TASK_K = "Humanoid-RL-Lab-G1-Flat-V4K-Geometric-Path-Precision-v0"

EXPECTED_ENV_CLASS = "HumanoidRlLabG1V4KGeometricPathPrecisionEnvCfg"
EXPECTED_ENV_PARENT = "HumanoidRlLabG1V4BDirectYawEnvCfg"
EXPECTED_PPO_CLASS = "V4KGeometricPathPrecisionPPORunnerCfg"
EXPECTED_PPO_PARENT = "V4BDirectYawPPORunnerCfg"
EXPECTED_REWARD_CLASS = "geometric_command_path_cross_track_bounded"
EXPECTED_EXPERIMENT = "humanoid_rl_lab_g1_v4k_geometric_path_precision_scratch"


def canonical(value):
    """Convert resolved configs into deterministic comparable Python values."""
    if value is None or isinstance(value, (bool, int, str)):
        return value

    if isinstance(value, float):
        if math.isnan(value):
            return ("float", "nan")
        if math.isinf(value):
            return ("float", "inf" if value > 0 else "-inf")
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): canonical(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }

    if isinstance(value, (list, tuple)):
        return [canonical(item) for item in value]

    if isinstance(value, set):
        return sorted(canonical(item) for item in value)

    if callable(value):
        return (
            "callable",
            getattr(value, "__module__", type(value).__module__),
            getattr(value, "__qualname__", getattr(value, "__name__", type(value).__qualname__)),
        )

    if hasattr(value, "tolist"):
        try:
            return canonical(value.tolist())
        except Exception:
            pass

    if hasattr(value, "to_dict") and callable(value.to_dict):
        return canonical(value.to_dict())

    if hasattr(value, "__dict__"):
        public = {
            key: val
            for key, val in vars(value).items()
            if not key.startswith("_")
        }
        if public:
            return canonical(public)

    return repr(value)


def cfg_dict(cfg):
    if not hasattr(cfg, "to_dict") or not callable(cfg.to_dict):
        raise RuntimeError(
            f"resolved config {type(cfg).__name__} does not expose to_dict()"
        )
    result = cfg.to_dict()
    if not isinstance(result, dict):
        raise RuntimeError(
            f"{type(cfg).__name__}.to_dict() did not return dict"
        )
    return canonical(result)


def assert_close(actual, expected, label, atol=1.0e-12):
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=atol):
        raise RuntimeError(
            f"{label}: expected {expected!r}, got {actual!r}"
        )


# Registry presence first.
for task_id in (TASK_B, TASK_K):
    try:
        gym.spec(task_id)
    except Exception as exc:
        raise RuntimeError(
            f"task is not registered: {task_id}"
        ) from exc


env_b = load_cfg_from_registry(TASK_B, "env_cfg_entry_point")
env_k = load_cfg_from_registry(TASK_K, "env_cfg_entry_point")
ppo_b = load_cfg_from_registry(TASK_B, "rsl_rl_cfg_entry_point")
ppo_k = load_cfg_from_registry(TASK_K, "rsl_rl_cfg_entry_point")


# --------------------------------------------------------------------------------------------------
# Direct scientific inheritance.
# --------------------------------------------------------------------------------------------------
if type(env_k).__name__ != EXPECTED_ENV_CLASS:
    raise RuntimeError(
        f"unexpected v4-K env class: {type(env_k).__name__}"
    )

if type(env_k).__mro__[1].__name__ != EXPECTED_ENV_PARENT:
    raise RuntimeError(
        "v4-K env is not a direct child of v4-B: "
        f"{type(env_k).__mro__[1].__name__}"
    )

if type(ppo_k).__name__ != EXPECTED_PPO_CLASS:
    raise RuntimeError(
        f"unexpected v4-K PPO class: {type(ppo_k).__name__}"
    )

if type(ppo_k).__mro__[1].__name__ != EXPECTED_PPO_PARENT:
    raise RuntimeError(
        "v4-K PPO is not a direct child of v4-B PPO: "
        f"{type(ppo_k).__mro__[1].__name__}"
    )


# --------------------------------------------------------------------------------------------------
# New reward: exact resolved semantics.
# --------------------------------------------------------------------------------------------------
if not hasattr(env_k.rewards, EXPECTED_REWARD_CLASS):
    raise RuntimeError(
        f"v4-K resolved rewards missing {EXPECTED_REWARD_CLASS}"
    )

reward_k = getattr(env_k.rewards, EXPECTED_REWARD_CLASS)

assert_close(reward_k.weight, -0.15, "v4-K reward weight")

params = reward_k.params
expected_params = {
    "command_name": "base_velocity",
    "sigma_m": 0.20,
    "min_forward_command": 0.10,
    "lateral_command_threshold": 0.10,
    "yaw_command_threshold": 0.20,
    "command_change_threshold": 1.0e-6,
    "yaw_rate_epsilon": 1.0e-4,
}

for key, expected in expected_params.items():
    if key not in params:
        raise RuntimeError(f"v4-K reward param missing: {key}")
    actual = params[key]
    if isinstance(expected, float):
        assert_close(actual, expected, f"v4-K reward param {key}")
    elif actual != expected:
        raise RuntimeError(
            f"v4-K reward param {key}: expected {expected!r}, got {actual!r}"
        )

if "asset_cfg" not in params:
    raise RuntimeError("v4-K reward param asset_cfg missing")

asset_cfg = params["asset_cfg"]
if getattr(asset_cfg, "name", None) != "robot":
    raise RuntimeError(
        f"v4-K asset_cfg expected robot, got {getattr(asset_cfg, 'name', None)!r}"
    )

func = reward_k.func
func_name = getattr(func, "__name__", type(func).__name__)
if func_name != EXPECTED_REWARD_CLASS:
    raise RuntimeError(
        f"unexpected resolved reward func: {func_name}"
    )


# --------------------------------------------------------------------------------------------------
# Whole resolved ENV comparison: remove only the added v4-K reward, then require equality with v4-B.
# --------------------------------------------------------------------------------------------------
dict_b = cfg_dict(env_b)
dict_k = cfg_dict(env_k)

if "rewards" not in dict_b or "rewards" not in dict_k:
    raise RuntimeError("resolved env dict is missing rewards")

if EXPECTED_REWARD_CLASS not in dict_k["rewards"]:
    raise RuntimeError(
        "new v4-K reward is absent from resolved env to_dict()"
    )

dict_k_without_new_reward = copy.deepcopy(dict_k)
dict_k_without_new_reward["rewards"].pop(EXPECTED_REWARD_CLASS)

if dict_k_without_new_reward != dict_b:
    # Produce a compact top-level diagnostic rather than dumping huge configs.
    changed_top_level = []
    all_keys = sorted(set(dict_b) | set(dict_k_without_new_reward))
    for key in all_keys:
        if dict_b.get(key) != dict_k_without_new_reward.get(key):
            changed_top_level.append(key)

    raise RuntimeError(
        "v4-K resolved env differs from v4-B beyond the intended reward. "
        f"Changed top-level sections: {changed_top_level}"
    )


# --------------------------------------------------------------------------------------------------
# Whole resolved PPO comparison: only experiment_name may differ.
# --------------------------------------------------------------------------------------------------
ppo_b_dict = cfg_dict(ppo_b)
ppo_k_dict = cfg_dict(ppo_k)

if ppo_k_dict.get("experiment_name") != EXPECTED_EXPERIMENT:
    raise RuntimeError(
        "unexpected v4-K experiment_name: "
        f"{ppo_k_dict.get('experiment_name')!r}"
    )

ppo_b_cmp = copy.deepcopy(ppo_b_dict)
ppo_k_cmp = copy.deepcopy(ppo_k_dict)
ppo_b_cmp.pop("experiment_name", None)
ppo_k_cmp.pop("experiment_name", None)

if ppo_b_cmp != ppo_k_cmp:
    changed = []
    all_keys = sorted(set(ppo_b_cmp) | set(ppo_k_cmp))
    for key in all_keys:
        if ppo_b_cmp.get(key) != ppo_k_cmp.get(key):
            changed.append(key)
    raise RuntimeError(
        "v4-K PPO differs from v4-B beyond experiment_name. "
        f"Changed sections: {changed}"
    )


# --------------------------------------------------------------------------------------------------
# Critical v4-B semantics explicitly surfaced for human audit.
# --------------------------------------------------------------------------------------------------
commands = env_k.commands.base_velocity

if bool(commands.heading_command) is not False:
    raise RuntimeError(
        f"v4-K heading_command expected False, got {commands.heading_command}"
    )

assert_close(commands.ranges.lin_vel_x[0], 0.0, "vx lower")
assert_close(commands.ranges.lin_vel_x[1], 1.0, "vx upper")
assert_close(commands.ranges.lin_vel_y[0], -0.5, "vy lower")
assert_close(commands.ranges.lin_vel_y[1], +0.5, "vy upper")
assert_close(commands.ranges.ang_vel_z[0], -1.0, "wz lower")
assert_close(commands.ranges.ang_vel_z[1], +1.0, "wz upper")
assert_close(commands.rel_standing_envs, 0.02, "rel_standing_envs")

print("=" * 116)
print("v4-K RESOLVED CONFIG AUDIT")
print("=" * 116)
print(f"Task                            : {TASK_K}")
print(f"Env class                       : {type(env_k).__name__}")
print(f"Direct env parent               : {type(env_k).__mro__[1].__name__}")
print(f"PPO class                       : {type(ppo_k).__name__}")
print(f"Direct PPO parent               : {type(ppo_k).__mro__[1].__name__}")
print()
print("Resolved sole env delta         : PASS")
print("Resolved sole PPO delta         : PASS")
print()
print(f"Reward func                     : {func_name}")
print(f"Reward weight                   : {reward_k.weight}")
print(f"sigma_m                         : {params['sigma_m']}")
print(f"min_forward_command             : {params['min_forward_command']}")
print(f"lateral_command_threshold       : {params['lateral_command_threshold']}")
print(f"yaw_command_threshold           : {params['yaw_command_threshold']}")
print(f"command_change_threshold        : {params['command_change_threshold']}")
print(f"yaw_rate_epsilon                : {params['yaw_rate_epsilon']}")
print(f"asset                           : {asset_cfg.name}")
print()
print(f"heading_command                 : {commands.heading_command}")
print(f"vx range                        : {commands.ranges.lin_vel_x}")
print(f"vy range                        : {commands.ranges.lin_vel_y}")
print(f"wz range                        : {commands.ranges.ang_vel_z}")
print(f"rel_standing_envs               : {commands.rel_standing_envs}")
print(f"experiment_name                 : {ppo_k.experiment_name}")
print()
print("V4K_RESOLVED_CONFIG_AUDIT_PASS")
print("=" * 116)
