from __future__ import annotations

from copy import deepcopy

from humanoid_rl_lab.tasks.manager_based.humanoid_rl_lab.v4_env_cfg import (
    HumanoidRlLabG1V4BDirectYawEnvCfg,
    HumanoidRlLabG1V4GZeroCommandFullMotionEnvCfg,
)
from humanoid_rl_lab.tasks.manager_based.humanoid_rl_lab.agents.v4b_ppo_cfg import (
    V4BDirectYawPPORunnerCfg,
)
from humanoid_rl_lab.tasks.manager_based.humanoid_rl_lab.agents.v4g_ppo_cfg import (
    V4GZeroCommandFullMotionPPORunnerCfg,
)
from humanoid_rl_lab.tasks.manager_based.humanoid_rl_lab.mdp.v4_rewards import (
    zero_command_full_motion_l2,
)


def fail(msg: str) -> None:
    raise SystemExit(f"ABORT: {msg}")


def check(cond: bool, msg: str) -> None:
    if not cond:
        fail(msg)


def obj_to_dict(obj):
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return {
            k: obj_to_dict(v)
            for k, v in vars(obj).items()
            if not k.startswith("_")
        }
    if isinstance(obj, dict):
        return {k: obj_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(obj_to_dict(v) for v in obj)
    return obj


b = HumanoidRlLabG1V4BDirectYawEnvCfg()
g = HumanoidRlLabG1V4GZeroCommandFullMotionEnvCfg()
pb = V4BDirectYawPPORunnerCfg()
pg = V4GZeroCommandFullMotionPPORunnerCfg()

check(
    HumanoidRlLabG1V4GZeroCommandFullMotionEnvCfg.__bases__[0]
    is HumanoidRlLabG1V4BDirectYawEnvCfg,
    "v4-G env is not a DIRECT v4-B child",
)
check(
    V4GZeroCommandFullMotionPPORunnerCfg.__bases__[0]
    is V4BDirectYawPPORunnerCfg,
    "v4-G PPO is not a DIRECT v4-B child",
)

# Commands must remain exactly v4-B.
check(g.commands.base_velocity.rel_standing_envs == 0.02, "rel_standing_envs != 0.02")
check(g.commands.base_velocity.heading_command is False, "heading_command != False")
check(
    obj_to_dict(b.commands.base_velocity) == obj_to_dict(g.commands.base_velocity),
    "base_velocity command config differs from v4-B",
)

# Rewards must be exactly v4-B + one term.
b_rewards = obj_to_dict(b.rewards)
g_rewards = obj_to_dict(g.rewards)

check("zero_command_full_motion" not in b_rewards, "v4-B already has zero_command_full_motion")
check("zero_command_full_motion" in g_rewards, "v4-G zero_command_full_motion missing")

g_rewards_minus_new = deepcopy(g_rewards)
g_rewards_minus_new.pop("zero_command_full_motion")
check(
    g_rewards_minus_new == b_rewards,
    "reward config differs beyond zero_command_full_motion",
)

active_b = [
    name for name, term in vars(b.rewards).items()
    if not name.startswith("_") and term is not None
]
active_g = [
    name for name, term in vars(g.rewards).items()
    if not name.startswith("_") and term is not None
]
check(len(active_b) == 16, f"expected 16 active v4-B rewards, found {len(active_b)}")
check(len(active_g) == 17, f"expected 17 active v4-G rewards, found {len(active_g)}")
check(
    set(active_g) == set(active_b) | {"zero_command_full_motion"},
    "active reward set is not v4-B + zero_command_full_motion",
)

term = g.rewards.zero_command_full_motion
check(term.func is zero_command_full_motion_l2, "reward function mismatch")
check(term.weight == -5.0, f"reward weight mismatch: {term.weight!r}")

params = term.params
expected_params = {
    "command_name": "base_velocity",
    "command_threshold": 0.06,
    "base_planar_scale": 1.0,
    "yaw_rate_scale": 1.0,
    "foot_xy_scale": 0.25,
}
for key, expected in expected_params.items():
    check(params[key] == expected, f"{key} mismatch: {params[key]!r}")

asset_cfg = params["asset_cfg"]
check(asset_cfg.name == "robot", f"asset_cfg.name mismatch: {asset_cfg.name!r}")
check(
    list(asset_cfg.body_names) == ["left_ankle_roll_link", "right_ankle_roll_link"],
    f"asset_cfg.body_names mismatch: {asset_cfg.body_names!r}",
)
check(getattr(asset_cfg, "preserve_order", None) is True, "asset_cfg.preserve_order is not True")

# Critical v4-B settings.
check(
    g.rewards.track_lin_vel_xy_exp.params["std"] == 0.5,
    "track_lin_vel_xy_exp std != 0.5",
)
for attr in ("observations", "events", "actions", "scene", "terminations"):
    check(
        obj_to_dict(getattr(b, attr)) == obj_to_dict(getattr(g, attr)),
        f"{attr} config differs from v4-B",
    )

check(
    getattr(g.observations.policy, "enable_corruption", None) is True,
    "observation corruption unexpectedly disabled",
)
check(
    getattr(g.events, "base_external_force_torque", None) is not None,
    "external-force event unexpectedly disabled",
)
check(
    getattr(g.events, "push_robot", None) is not None,
    "push event unexpectedly disabled",
)

action_scale = None
for name, term_cfg in vars(g.actions).items():
    if name.startswith("_") or term_cfg is None:
        continue
    if hasattr(term_cfg, "scale"):
        action_scale = term_cfg.scale
        break
check(action_scale == 0.5, f"joint-position action scale mismatch: {action_scale!r}")

# PPO must differ only in experiment_name.
pb_dict = obj_to_dict(pb)
pg_dict = obj_to_dict(pg)

check(
    pb.experiment_name == "humanoid_rl_lab_g1_v4b_direct_yaw_scratch",
    f"unexpected v4-B experiment name: {pb.experiment_name!r}",
)
check(
    pg.experiment_name == "humanoid_rl_lab_g1_v4g_zero_command_full_motion_scratch",
    f"unexpected v4-G experiment name: {pg.experiment_name!r}",
)

pb_cmp = deepcopy(pb_dict)
pb_cmp["experiment_name"] = pg.experiment_name
check(pb_cmp == pg_dict, "PPO differs beyond experiment_name")

print("=" * 116)
print("V4-G RESOLVED CONFIG AUDIT")
print("=" * 116)
print("env inheritance              : v4-G -> v4-B DIRECT")
print("PPO inheritance              : v4-G -> v4-B DIRECT")
print(f"rel_standing_envs            : {g.commands.base_velocity.rel_standing_envs}")
print(f"heading_command              : {g.commands.base_velocity.heading_command}")
print(f"lin_vel_x range              : {g.commands.base_velocity.ranges.lin_vel_x}")
print(f"lin_vel_y range              : {g.commands.base_velocity.ranges.lin_vel_y}")
print(f"ang_vel_z range              : {g.commands.base_velocity.ranges.ang_vel_z}")
print(f"track_lin_vel_xy std         : {g.rewards.track_lin_vel_xy_exp.params['std']}")
print(f"active rewards v4-B -> v4-G : {len(active_b)} -> {len(active_g)}")
print("new reward                   : zero_command_full_motion")
print(f"new reward weight            : {term.weight}")
print(f"full-command threshold       : {params['command_threshold']}")
print(f"base planar scale            : {params['base_planar_scale']}")
print(f"yaw-rate scale               : {params['yaw_rate_scale']}")
print(f"foot XY scale                : {params['foot_xy_scale']}")
print(f"reward feet                  : {list(asset_cfg.body_names)}")
print(f"observation corruption       : {g.observations.policy.enable_corruption}")
print(f"external force event         : {g.events.base_external_force_torque is not None}")
print(f"push event                   : {g.events.push_robot is not None}")
print(f"joint-position action scale  : {action_scale}")
print(f"PPO experiment               : {pg.experiment_name}")
print("commands diff                : NONE")
print("reward diff beyond new term  : NONE")
print("observations/events/actions  : IDENTICAL")
print("scene/terminations           : IDENTICAL")
print("PPO diff beyond experiment   : NONE")
print()
print("V4G_RESOLVED_CONFIG_AUDIT_PASS")
print("=" * 116)
