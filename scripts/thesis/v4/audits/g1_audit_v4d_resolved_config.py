from __future__ import annotations

import math
from copy import deepcopy

from humanoid_rl_lab.tasks.manager_based.humanoid_rl_lab.v4_env_cfg import (
    HumanoidRlLabG1V4BDirectYawEnvCfg,
    HumanoidRlLabG1V4DStandingExposureEnvCfg,
)
from humanoid_rl_lab.tasks.manager_based.humanoid_rl_lab.agents.v4b_ppo_cfg import (
    V4BDirectYawPPORunnerCfg,
)
from humanoid_rl_lab.tasks.manager_based.humanoid_rl_lab.agents.v4d_ppo_cfg import (
    V4DStandingExposurePPORunnerCfg,
)


def fail(msg: str) -> None:
    raise SystemExit(f"ABORT: {msg}")


def check(cond: bool, msg: str) -> None:
    if not cond:
        fail(msg)


def same(a, b, label: str) -> None:
    if a != b:
        fail(f"{label} differs: v4-B={a!r} v4-D={b!r}")


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
d = HumanoidRlLabG1V4DStandingExposureEnvCfg()
pb = V4BDirectYawPPORunnerCfg()
pd = V4DStandingExposurePPORunnerCfg()

check(
    issubclass(HumanoidRlLabG1V4DStandingExposureEnvCfg, HumanoidRlLabG1V4BDirectYawEnvCfg),
    "v4-D env does not inherit directly from v4-B",
)
check(
    issubclass(V4DStandingExposurePPORunnerCfg, V4BDirectYawPPORunnerCfg),
    "v4-D PPO does not inherit directly from v4-B",
)

# --- exact intended scientific delta ---
same(b.commands.base_velocity.rel_standing_envs, 0.02, "v4-B rel_standing_envs baseline")
same(d.commands.base_velocity.rel_standing_envs, 0.05, "v4-D rel_standing_envs")
same(b.commands.base_velocity.heading_command, False, "v4-B heading_command")
same(d.commands.base_velocity.heading_command, False, "v4-D heading_command")

# --- commands unchanged apart from rel_standing_envs ---
bc = obj_to_dict(b.commands.base_velocity)
dc = obj_to_dict(d.commands.base_velocity)
bc_cmp = deepcopy(bc)
dc_cmp = deepcopy(dc)
bc_cmp["rel_standing_envs"] = 0.05
check(bc_cmp == dc_cmp, "resolved base_velocity command cfg differs beyond rel_standing_envs")

# --- rewards must be exactly identical ---
br = obj_to_dict(b.rewards)
dr = obj_to_dict(d.rewards)
check(br == dr, "resolved reward config differs between v4-B and v4-D")

reward_names = [
    k for k, v in vars(d.rewards).items()
    if not k.startswith("_") and v is not None
]
check(len(reward_names) == 16, f"expected 16 active rewards, found {len(reward_names)}: {reward_names}")

same(
    d.rewards.track_lin_vel_xy_exp.params["std"],
    0.5,
    "v4-D track_lin_vel_xy_exp std",
)

# --- observations, events, actions, scene, terminations unchanged ---
for attr in ("observations", "events", "actions", "scene", "terminations"):
    bv = obj_to_dict(getattr(b, attr))
    dv = obj_to_dict(getattr(d, attr))
    check(bv == dv, f"resolved {attr} config differs between v4-B and v4-D")

# --- training randomization/perturbation still enabled ---
check(
    getattr(d.observations.policy, "enable_corruption", None) is True,
    "training observation corruption is not enabled",
)
check(
    getattr(d.events, "base_external_force_torque", None) is not None,
    "base external force/torque event unexpectedly disabled",
)
check(
    getattr(d.events, "push_robot", None) is not None,
    "push_robot event unexpectedly disabled",
)

# --- action semantics: same as v4-B, and expected project/Official scale ---
same(
    obj_to_dict(b.actions),
    obj_to_dict(d.actions),
    "resolved actions",
)

# Try to expose the common joint-position action scale without assuming the field name.
action_scale = None
for name, term in vars(d.actions).items():
    if name.startswith("_") or term is None:
        continue
    if hasattr(term, "scale"):
        action_scale = term.scale
        break
check(action_scale is not None, "could not locate action scale in resolved actions")
same(action_scale, 0.5, "v4-D joint-position action scale")

# --- PPO: identical except experiment_name ---
pbd = obj_to_dict(pb)
pdd = obj_to_dict(pd)
same(pb.experiment_name, "humanoid_rl_lab_g1_v4b_direct_yaw_scratch", "v4-B experiment name")
same(pd.experiment_name, "humanoid_rl_lab_g1_v4d_standing_exposure_scratch", "v4-D experiment name")

pbd_cmp = deepcopy(pbd)
pbd_cmp["experiment_name"] = pd.experiment_name
check(pbd_cmp == pdd, "resolved PPO config differs beyond experiment_name")

print("=" * 108)
print("V4-D RESOLVED CONFIG AUDIT")
print("=" * 108)
print("env inheritance             : v4-D -> v4-B DIRECT")
print("PPO inheritance             : v4-D -> v4-B DIRECT")
print(f"rel_standing_envs           : {b.commands.base_velocity.rel_standing_envs:.2f} -> {d.commands.base_velocity.rel_standing_envs:.2f}")
print(f"heading_command             : {d.commands.base_velocity.heading_command}")
print(f"lin_vel_x range             : {d.commands.base_velocity.ranges.lin_vel_x}")
print(f"lin_vel_y range             : {d.commands.base_velocity.ranges.lin_vel_y}")
print(f"ang_vel_z range             : {d.commands.base_velocity.ranges.ang_vel_z}")
print(f"track_lin_vel_xy std        : {d.rewards.track_lin_vel_xy_exp.params['std']}")
print(f"active rewards              : {len(reward_names)}")
print(f"observation corruption      : {d.observations.policy.enable_corruption}")
print(f"external force event        : {d.events.base_external_force_torque is not None}")
print(f"push event                  : {d.events.push_robot is not None}")
print(f"joint-position action scale : {action_scale}")
print(f"PPO experiment              : {pd.experiment_name}")
print("commands diff beyond standing exposure : NONE")
print("rewards diff                : NONE")
print("observations/events/actions : IDENTICAL")
print("scene/terminations          : IDENTICAL")
print("PPO diff beyond experiment  : NONE")
print()
print("V4D_RESOLVED_CONFIG_AUDIT_PASS")
print("=" * 108)
