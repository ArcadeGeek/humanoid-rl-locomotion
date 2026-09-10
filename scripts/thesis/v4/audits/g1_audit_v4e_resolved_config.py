from __future__ import annotations

from copy import deepcopy

from humanoid_rl_lab.tasks.manager_based.humanoid_rl_lab.v4_env_cfg import (
    HumanoidRlLabG1V4BDirectYawEnvCfg,
    HumanoidRlLabG1V4EZeroCommandMotionEnvCfg,
)
from humanoid_rl_lab.tasks.manager_based.humanoid_rl_lab.agents.v4b_ppo_cfg import (
    V4BDirectYawPPORunnerCfg,
)
from humanoid_rl_lab.tasks.manager_based.humanoid_rl_lab.agents.v4e_ppo_cfg import (
    V4EZeroCommandMotionPPORunnerCfg,
)
from humanoid_rl_lab.tasks.manager_based.humanoid_rl_lab.mdp.v4_rewards import (
    zero_command_motion_l2,
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
e = HumanoidRlLabG1V4EZeroCommandMotionEnvCfg()
pb = V4BDirectYawPPORunnerCfg()
pe = V4EZeroCommandMotionPPORunnerCfg()

check(
    issubclass(HumanoidRlLabG1V4EZeroCommandMotionEnvCfg, HumanoidRlLabG1V4BDirectYawEnvCfg),
    "v4-E env does not inherit directly from v4-B",
)
check(
    issubclass(V4EZeroCommandMotionPPORunnerCfg, V4BDirectYawPPORunnerCfg),
    "v4-E PPO does not inherit directly from v4-B",
)

# Commands must remain exactly v4-B / Official-direct-yaw.
check(e.commands.base_velocity.rel_standing_envs == 0.02, "v4-E rel_standing_envs is not 0.02")
check(e.commands.base_velocity.heading_command is False, "v4-E heading_command is not False")
check(
    obj_to_dict(b.commands.base_velocity) == obj_to_dict(e.commands.base_velocity),
    "resolved base_velocity command cfg differs between v4-B and v4-E",
)

# Rewards: v4-E must equal v4-B plus exactly one new term.
b_rewards = obj_to_dict(b.rewards)
e_rewards = obj_to_dict(e.rewards)

check("zero_command_motion" not in b_rewards, "v4-B unexpectedly already has zero_command_motion")
check("zero_command_motion" in e_rewards, "v4-E zero_command_motion reward missing")

e_rewards_without_new = deepcopy(e_rewards)
new_reward_dict = e_rewards_without_new.pop("zero_command_motion")
check(
    b_rewards == e_rewards_without_new,
    "resolved reward config differs beyond the single zero_command_motion term",
)

active_b = [
    k for k, v in vars(b.rewards).items()
    if not k.startswith("_") and v is not None
]
active_e = [
    k for k, v in vars(e.rewards).items()
    if not k.startswith("_") and v is not None
]
check(len(active_b) == 16, f"expected 16 active v4-B rewards, found {len(active_b)}")
check(len(active_e) == 17, f"expected 17 active v4-E rewards, found {len(active_e)}")
check(set(active_e) == set(active_b) | {"zero_command_motion"}, "v4-E active reward set is not v4-B + zero_command_motion")

term = e.rewards.zero_command_motion
check(term.func is zero_command_motion_l2, "v4-E zero_command_motion func mismatch")
check(term.weight == -5.0, f"v4-E zero_command_motion weight mismatch: {term.weight!r}")

params = term.params
check(params["command_name"] == "base_velocity", "command_name mismatch")
check(params["command_threshold"] == 0.06, "command_threshold mismatch")
check(params["base_planar_scale"] == 1.0, "base_planar_scale mismatch")
check(params["foot_xy_scale"] == 0.25, "foot_xy_scale mismatch")

asset_cfg = params["asset_cfg"]
check(asset_cfg.name == "robot", f"asset_cfg.name mismatch: {asset_cfg.name!r}")
check(
    list(asset_cfg.body_names) == ["left_ankle_roll_link", "right_ankle_roll_link"],
    f"asset_cfg.body_names mismatch: {asset_cfg.body_names!r}",
)
check(getattr(asset_cfg, "preserve_order", None) is True, "asset_cfg.preserve_order is not True")

# Critical inherited Official/v4-B settings remain unchanged.
check(
    e.rewards.track_lin_vel_xy_exp.params["std"] == 0.5,
    "v4-E accidentally inherited/introduced v4-C std != 0.5",
)
for attr in ("observations", "events", "actions", "scene", "terminations"):
    check(
        obj_to_dict(getattr(b, attr)) == obj_to_dict(getattr(e, attr)),
        f"resolved {attr} config differs between v4-B and v4-E",
    )

check(
    getattr(e.observations.policy, "enable_corruption", None) is True,
    "training observation corruption unexpectedly disabled",
)
check(
    getattr(e.events, "base_external_force_torque", None) is not None,
    "external-force event unexpectedly disabled",
)
check(
    getattr(e.events, "push_robot", None) is not None,
    "push event unexpectedly disabled",
)

action_scale = None
for name, term_cfg in vars(e.actions).items():
    if name.startswith("_") or term_cfg is None:
        continue
    if hasattr(term_cfg, "scale"):
        action_scale = term_cfg.scale
        break
check(action_scale == 0.5, f"joint-position action scale mismatch: {action_scale!r}")

# PPO must be identical except experiment_name.
pb_dict = obj_to_dict(pb)
pe_dict = obj_to_dict(pe)
check(
    pb.experiment_name == "humanoid_rl_lab_g1_v4b_direct_yaw_scratch",
    f"unexpected v4-B experiment name: {pb.experiment_name!r}",
)
check(
    pe.experiment_name == "humanoid_rl_lab_g1_v4e_zero_command_motion_scratch",
    f"unexpected v4-E experiment name: {pe.experiment_name!r}",
)

pb_cmp = deepcopy(pb_dict)
pb_cmp["experiment_name"] = pe.experiment_name
check(pb_cmp == pe_dict, "resolved PPO differs beyond experiment_name")

print("=" * 112)
print("V4-E RESOLVED CONFIG AUDIT")
print("=" * 112)
print("env inheritance              : v4-E -> v4-B DIRECT")
print("PPO inheritance              : v4-E -> v4-B DIRECT")
print(f"rel_standing_envs            : {e.commands.base_velocity.rel_standing_envs}")
print(f"heading_command              : {e.commands.base_velocity.heading_command}")
print(f"lin_vel_x range              : {e.commands.base_velocity.ranges.lin_vel_x}")
print(f"lin_vel_y range              : {e.commands.base_velocity.ranges.lin_vel_y}")
print(f"ang_vel_z range              : {e.commands.base_velocity.ranges.ang_vel_z}")
print(f"track_lin_vel_xy std         : {e.rewards.track_lin_vel_xy_exp.params['std']}")
print(f"active rewards v4-B -> v4-E : {len(active_b)} -> {len(active_e)}")
print("new reward                   : zero_command_motion")
print(f"new reward weight            : {term.weight}")
print(f"full-command threshold       : {params['command_threshold']}")
print(f"base planar scale            : {params['base_planar_scale']}")
print(f"foot XY scale                : {params['foot_xy_scale']}")
print(f"reward feet                  : {list(asset_cfg.body_names)}")
print(f"observation corruption       : {e.observations.policy.enable_corruption}")
print(f"external force event         : {e.events.base_external_force_torque is not None}")
print(f"push event                   : {e.events.push_robot is not None}")
print(f"joint-position action scale  : {action_scale}")
print(f"PPO experiment               : {pe.experiment_name}")
print("commands diff                : NONE")
print("reward diff beyond new term  : NONE")
print("observations/events/actions  : IDENTICAL")
print("scene/terminations           : IDENTICAL")
print("PPO diff beyond experiment   : NONE")
print()
print("V4E_RESOLVED_CONFIG_AUDIT_PASS")
print("=" * 112)
