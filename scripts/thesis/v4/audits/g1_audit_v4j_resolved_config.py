from __future__ import annotations

import copy
import math

from isaaclab.utils import class_to_dict

from humanoid_rl_lab.tasks.manager_based.humanoid_rl_lab.v4_env_cfg import (
    HumanoidRlLabG1V4BDirectYawEnvCfg,
    HumanoidRlLabG1V4JStraightLinePrecisionEnvCfg,
)
from humanoid_rl_lab.tasks.manager_based.humanoid_rl_lab.agents.v4b_ppo_cfg import (
    V4BDirectYawPPORunnerCfg,
)
from humanoid_rl_lab.tasks.manager_based.humanoid_rl_lab.agents.v4j_ppo_cfg import (
    V4JStraightLinePrecisionPPORunnerCfg,
)
from humanoid_rl_lab.tasks.manager_based.humanoid_rl_lab.mdp.v4_rewards import (
    straight_line_lateral_velocity_ema_bounded,
)


EXPECTED_TASK_ID = "Humanoid-RL-Lab-G1-Flat-V4J-Straight-Line-Precision-v0"
EXPECTED_EXPERIMENT = "humanoid_rl_lab_g1_v4j_straight_line_precision_scratch"
EXPECTED_REWARD_NAME = "straight_line_lateral_velocity_ema_bounded"


def fail(message: str) -> None:
    raise RuntimeError(message)


def normalize(value):
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(v) for v in value]
    if callable(value):
        module = getattr(value, "__module__", "")
        name = getattr(value, "__qualname__", getattr(value, "__name__", repr(value)))
        return f"{module}:{name}"
    return value


def strip_j_delta(env_dict: dict) -> dict:
    cleaned = copy.deepcopy(env_dict)
    rewards = cleaned.get("rewards")
    if not isinstance(rewards, dict):
        fail("resolved env config does not contain a rewards dictionary")
    if EXPECTED_REWARD_NAME not in rewards:
        fail(f"v4-J resolved rewards missing {EXPECTED_REWARD_NAME}")
    del rewards[EXPECTED_REWARD_NAME]
    return cleaned


b_env = HumanoidRlLabG1V4BDirectYawEnvCfg()
j_env = HumanoidRlLabG1V4JStraightLinePrecisionEnvCfg()

b_ppo = V4BDirectYawPPORunnerCfg()
j_ppo = V4JStraightLinePrecisionPPORunnerCfg()

b_env_dict = normalize(class_to_dict(b_env))
j_env_dict = normalize(class_to_dict(j_env))
b_ppo_dict = normalize(class_to_dict(b_ppo))
j_ppo_dict = normalize(class_to_dict(j_ppo))

# -------------------------------------------------------------------------------------------------
# Environment inheritance / sole scientific delta.
# -------------------------------------------------------------------------------------------------
if not isinstance(j_env, HumanoidRlLabG1V4BDirectYawEnvCfg):
    fail("v4-J env is not a direct behavioral descendant of v4-B")

j_without_delta = strip_j_delta(j_env_dict)
if j_without_delta != b_env_dict:
    # Give compact top-level diagnostics without dumping huge configs.
    b_keys = set(b_env_dict.keys())
    j_keys = set(j_without_delta.keys())
    differing = sorted(
        key for key in (b_keys & j_keys)
        if b_env_dict[key] != j_without_delta[key]
    )
    fail(
        "v4-J has unexpected resolved env differences versus v4-B after removing "
        f"the intended reward. differing_top_level_keys={differing}, "
        f"only_in_b={sorted(b_keys-j_keys)}, only_in_j={sorted(j_keys-b_keys)}"
    )

reward = j_env.rewards.straight_line_lateral_velocity_ema_bounded
if reward is None:
    fail("v4-J reward term resolved to None")

if reward.func is not straight_line_lateral_velocity_ema_bounded:
    fail(f"unexpected reward func: {reward.func}")

if not math.isclose(float(reward.weight), -0.25, rel_tol=0.0, abs_tol=1e-12):
    fail(f"unexpected reward weight: {reward.weight}")

expected_params = {
    "command_name": "base_velocity",
    "tau_s": 0.75,
    "sigma": 0.03,
    "min_forward_command": 0.10,
    "lateral_command_threshold": 0.10,
    "yaw_command_threshold": 0.20,
}

for key, expected in expected_params.items():
    if key not in reward.params:
        fail(f"reward param missing: {key}")
    actual = reward.params[key]
    if isinstance(expected, float):
        if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12):
            fail(f"reward param {key}: expected {expected}, got {actual}")
    elif actual != expected:
        fail(f"reward param {key}: expected {expected!r}, got {actual!r}")

asset_cfg = reward.params.get("asset_cfg")
if asset_cfg is None:
    fail("reward asset_cfg missing")
if getattr(asset_cfg, "name", None) != "robot":
    fail(f"reward asset_cfg.name unexpected: {getattr(asset_cfg, 'name', None)!r}")

# v4-B direct-yaw semantics must remain.
if bool(j_env.commands.base_velocity.heading_command):
    fail("v4-J unexpectedly enabled heading_command")

# Official/v4-B command ranges must remain untouched.
ranges = j_env.commands.base_velocity.ranges
if tuple(ranges.lin_vel_x) != (0.0, 1.0):
    fail(f"unexpected vx command range: {ranges.lin_vel_x}")
if tuple(ranges.lin_vel_y) != (-0.5, 0.5):
    fail(f"unexpected vy command range: {ranges.lin_vel_y}")
if tuple(ranges.ang_vel_z) != (-1.0, 1.0):
    fail(f"unexpected wz command range: {ranges.ang_vel_z}")

if not math.isclose(float(j_env.commands.base_velocity.rel_standing_envs), 0.02, abs_tol=1e-12):
    fail(
        "v4-J unexpectedly changed rel_standing_envs: "
        f"{j_env.commands.base_velocity.rel_standing_envs}"
    )

# -------------------------------------------------------------------------------------------------
# PPO: sole difference must be experiment_name.
# -------------------------------------------------------------------------------------------------
b_ppo_cmp = copy.deepcopy(b_ppo_dict)
j_ppo_cmp = copy.deepcopy(j_ppo_dict)

b_exp = b_ppo_cmp.pop("experiment_name", None)
j_exp = j_ppo_cmp.pop("experiment_name", None)

if j_exp != EXPECTED_EXPERIMENT:
    fail(f"unexpected v4-J experiment_name: {j_exp!r}")
if b_exp == j_exp:
    fail("v4-J experiment_name did not differ from v4-B")

if j_ppo_cmp != b_ppo_cmp:
    differing = sorted(
        key for key in (set(j_ppo_cmp) & set(b_ppo_cmp))
        if j_ppo_cmp[key] != b_ppo_cmp[key]
    )
    fail(
        "v4-J has unexpected PPO differences versus v4-B after removing experiment_name. "
        f"differing_top_level_keys={differing}"
    )

print("=" * 108)
print("v4-J RESOLVED-CONFIG AUDIT")
print("=" * 108)
print(f"Task ID expected              : {EXPECTED_TASK_ID}")
print("Scientific parent             : v4-B Direct-Yaw")
print("Resolved env sole delta       : PASS")
print(f"Added reward                   : {EXPECTED_REWARD_NAME}")
print(f"Reward class                   : {reward.func.__name__}")
print(f"Reward weight                  : {float(reward.weight):+.2f}")
print(f"EMA tau                        : {float(reward.params['tau_s']):.2f} s")
print(f"Bounded sigma                  : {float(reward.params['sigma']):.3f} m/s")
print(
    "Straight gate                  : "
    f"vx>{float(reward.params['min_forward_command']):.2f}, "
    f"|vy|<{float(reward.params['lateral_command_threshold']):.2f}, "
    f"|wz|<{float(reward.params['yaw_command_threshold']):.2f}"
)
print(f"heading_command                : {j_env.commands.base_velocity.heading_command}")
print(
    "command ranges                 : "
    f"vx={tuple(ranges.lin_vel_x)}, vy={tuple(ranges.lin_vel_y)}, wz={tuple(ranges.ang_vel_z)}"
)
print(f"rel_standing_envs              : {j_env.commands.base_velocity.rel_standing_envs}")
print("PPO sole delta experiment_name : PASS")
print(f"v4-J experiment_name           : {j_exp}")
print()
print("V4J_RESOLVED_CONFIG_AUDIT_PASS")
print("=" * 108)
