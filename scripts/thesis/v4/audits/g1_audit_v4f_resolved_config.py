from __future__ import annotations

from copy import deepcopy

from humanoid_rl_lab.tasks.manager_based.humanoid_rl_lab.v4_env_cfg import (
    HumanoidRlLabG1V4BDirectYawEnvCfg,
    HumanoidRlLabG1V4FStandStillJointDeviationEnvCfg,
)
from humanoid_rl_lab.tasks.manager_based.humanoid_rl_lab.agents.v4b_ppo_cfg import (
    V4BDirectYawPPORunnerCfg,
)
from humanoid_rl_lab.tasks.manager_based.humanoid_rl_lab.agents.v4f_ppo_cfg import (
    V4FStandStillJointDeviationPPORunnerCfg,
)
from humanoid_rl_lab.tasks.manager_based.humanoid_rl_lab.mdp.v4_rewards import (
    stand_still_joint_deviation_full_command_l1,
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
f = HumanoidRlLabG1V4FStandStillJointDeviationEnvCfg()
pb = V4BDirectYawPPORunnerCfg()
pf = V4FStandStillJointDeviationPPORunnerCfg()

check(
    HumanoidRlLabG1V4FStandStillJointDeviationEnvCfg.__bases__[0]
    is HumanoidRlLabG1V4BDirectYawEnvCfg,
    "v4-F env is not a DIRECT v4-B child",
)
check(
    V4FStandStillJointDeviationPPORunnerCfg.__bases__[0]
    is V4BDirectYawPPORunnerCfg,
    "v4-F PPO is not a DIRECT v4-B child",
)

# Commands must remain exactly v4-B.
check(f.commands.base_velocity.rel_standing_envs == 0.02, "rel_standing_envs != 0.02")
check(f.commands.base_velocity.heading_command is False, "heading_command != False")
check(
    obj_to_dict(b.commands.base_velocity) == obj_to_dict(f.commands.base_velocity),
    "base_velocity command config differs from v4-B",
)

# v4-F rewards must be exactly v4-B + one term.
b_rewards = obj_to_dict(b.rewards)
f_rewards = obj_to_dict(f.rewards)

check("stand_still_full_command" not in b_rewards, "v4-B already has stand_still_full_command")
check("stand_still_full_command" in f_rewards, "v4-F stand_still_full_command missing")

f_rewards_minus_new = deepcopy(f_rewards)
f_rewards_minus_new.pop("stand_still_full_command")
check(
    f_rewards_minus_new == b_rewards,
    "reward config differs beyond stand_still_full_command",
)

active_b = [
    name for name, term in vars(b.rewards).items()
    if not name.startswith("_") and term is not None
]
active_f = [
    name for name, term in vars(f.rewards).items()
    if not name.startswith("_") and term is not None
]
check(len(active_b) == 16, f"expected 16 active v4-B rewards, found {len(active_b)}")
check(len(active_f) == 17, f"expected 17 active v4-F rewards, found {len(active_f)}")
check(
    set(active_f) == set(active_b) | {"stand_still_full_command"},
    "active reward set is not v4-B + stand_still_full_command",
)

term = f.rewards.stand_still_full_command
check(term.func is stand_still_joint_deviation_full_command_l1, "reward function mismatch")
check(term.weight == -0.4, f"reward weight mismatch: {term.weight!r}")

params = term.params
check(params["command_name"] == "base_velocity", "command_name mismatch")
check(params["command_threshold"] == 0.06, "command_threshold mismatch")

asset_cfg = params["asset_cfg"]
check(asset_cfg.name == "robot", f"asset_cfg.name mismatch: {asset_cfg.name!r}")
expected_patterns = [
    ".*_hip_.*",
    ".*_knee_joint",
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
]
check(
    list(asset_cfg.joint_names) == expected_patterns,
    f"joint regex list mismatch: {asset_cfg.joint_names!r}",
)

# Critical v4-B settings.
check(
    f.rewards.track_lin_vel_xy_exp.params["std"] == 0.5,
    "track_lin_vel_xy_exp std != 0.5",
)
for attr in ("observations", "events", "actions", "scene", "terminations"):
    check(
        obj_to_dict(getattr(b, attr)) == obj_to_dict(getattr(f, attr)),
        f"{attr} config differs from v4-B",
    )

check(
    getattr(f.observations.policy, "enable_corruption", None) is True,
    "observation corruption unexpectedly disabled",
)
check(
    getattr(f.events, "base_external_force_torque", None) is not None,
    "external-force event unexpectedly disabled",
)
check(
    getattr(f.events, "push_robot", None) is not None,
    "push event unexpectedly disabled",
)

action_scale = None
for name, term_cfg in vars(f.actions).items():
    if name.startswith("_") or term_cfg is None:
        continue
    if hasattr(term_cfg, "scale"):
        action_scale = term_cfg.scale
        break
check(action_scale == 0.5, f"joint-position action scale mismatch: {action_scale!r}")

# PPO must differ only in experiment_name.
pb_dict = obj_to_dict(pb)
pf_dict = obj_to_dict(pf)
check(
    pb.experiment_name == "humanoid_rl_lab_g1_v4b_direct_yaw_scratch",
    f"unexpected v4-B experiment name: {pb.experiment_name!r}",
)
check(
    pf.experiment_name == "humanoid_rl_lab_g1_v4f_stand_still_joint_deviation_scratch",
    f"unexpected v4-F experiment name: {pf.experiment_name!r}",
)
pb_cmp = deepcopy(pb_dict)
pb_cmp["experiment_name"] = pf.experiment_name
check(pb_cmp == pf_dict, "PPO differs beyond experiment_name")

print("=" * 116)
print("V4-F RESOLVED CONFIG AUDIT")
print("=" * 116)
print("env inheritance              : v4-F -> v4-B DIRECT")
print("PPO inheritance              : v4-F -> v4-B DIRECT")
print(f"rel_standing_envs            : {f.commands.base_velocity.rel_standing_envs}")
print(f"heading_command              : {f.commands.base_velocity.heading_command}")
print(f"lin_vel_x range              : {f.commands.base_velocity.ranges.lin_vel_x}")
print(f"lin_vel_y range              : {f.commands.base_velocity.ranges.lin_vel_y}")
print(f"ang_vel_z range              : {f.commands.base_velocity.ranges.ang_vel_z}")
print(f"track_lin_vel_xy std         : {f.rewards.track_lin_vel_xy_exp.params['std']}")
print(f"active rewards v4-B -> v4-F : {len(active_b)} -> {len(active_f)}")
print("new reward                   : stand_still_full_command")
print(f"new reward weight            : {term.weight}")
print(f"full-command threshold       : {params['command_threshold']}")
print(f"joint regexes                : {list(asset_cfg.joint_names)}")
print("expected resolved leg joints : 12 (verified in runtime smoke after scene resolution)")
print(f"observation corruption       : {f.observations.policy.enable_corruption}")
print(f"external force event         : {f.events.base_external_force_torque is not None}")
print(f"push event                   : {f.events.push_robot is not None}")
print(f"joint-position action scale  : {action_scale}")
print(f"PPO experiment               : {pf.experiment_name}")
print("commands diff                : NONE")
print("reward diff beyond new term  : NONE")
print("observations/events/actions  : IDENTICAL")
print("scene/terminations           : IDENTICAL")
print("PPO diff beyond experiment   : NONE")
print()
print("V4F_RESOLVED_CONFIG_AUDIT_PASS")
print("=" * 116)
