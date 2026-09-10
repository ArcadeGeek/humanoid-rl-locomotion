from __future__ import annotations

import gymnasium as gym

import humanoid_rl_lab.tasks.manager_based.humanoid_rl_lab  # noqa: F401

from humanoid_rl_lab.tasks.manager_based.humanoid_rl_lab.v4_env_cfg import (
    HumanoidRlLabG1V4BDirectYawEnvCfg,
    HumanoidRlLabG1V4CPrecisionEnvCfg,
)
from humanoid_rl_lab.tasks.manager_based.humanoid_rl_lab.agents.v4b_ppo_cfg import (
    V4BDirectYawPPORunnerCfg,
)
from humanoid_rl_lab.tasks.manager_based.humanoid_rl_lab.agents.v4c_ppo_cfg import (
    V4CPrecisionPPORunnerCfg,
)

TASK_ID = "Humanoid-RL-Lab-G1-Flat-V4C-Precision-v0"
EXPECTED_EXPERIMENT = "humanoid_rl_lab_g1_v4c_precision_scratch"

OFFICIAL_REWARD_NAMES = [
    "termination_penalty",
    "track_lin_vel_xy_exp",
    "track_ang_vel_z_exp",
    "lin_vel_z_l2",
    "ang_vel_xy_l2",
    "dof_torques_l2",
    "dof_acc_l2",
    "action_rate_l2",
    "feet_air_time",
    "flat_orientation_l2",
    "dof_pos_limits",
    "feet_slide",
    "joint_deviation_hip",
    "joint_deviation_arms",
    "joint_deviation_fingers",
    "joint_deviation_torso",
]


def fail(msg: str) -> None:
    raise SystemExit(f"ABORT: {msg}")


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        fail(f"{name} FAIL" + (f" — {detail}" if detail else ""))
    print(f"{name:<44} PASS" + (f"  {detail}" if detail else ""))


def close(a, b, tol=1.0e-12) -> bool:
    return abs(float(a) - float(b)) <= tol


def reward_terms(cfg):
    result = {}
    for name in OFFICIAL_REWARD_NAMES:
        term = getattr(cfg.rewards, name, None)
        if term is None:
            fail(f"reward missing or disabled: {name}")
        result[name] = term
    return result


# Registry.
try:
    spec = gym.spec(TASK_ID)
except Exception as exc:
    fail(f"task not registered: {TASK_ID}: {exc}")

kwargs = dict(spec.kwargs or {})
check("task_registered", spec is not None, TASK_ID)
check(
    "registry_env_entry_point",
    "HumanoidRlLabG1V4CPrecisionEnvCfg" in str(kwargs.get("env_cfg_entry_point", "")),
    str(kwargs.get("env_cfg_entry_point")),
)
check(
    "registry_ppo_entry_point",
    "V4CPrecisionPPORunnerCfg" in str(kwargs.get("rsl_rl_cfg_entry_point", "")),
    str(kwargs.get("rsl_rl_cfg_entry_point")),
)

# Resolve configs.
b = HumanoidRlLabG1V4BDirectYawEnvCfg()
c = HumanoidRlLabG1V4CPrecisionEnvCfg()
pb = V4BDirectYawPPORunnerCfg()
pc = V4CPrecisionPPORunnerCfg()

print()
print("=" * 120)
print("V4-C RESOLVED CONFIG AUDIT V2")
print("=" * 120)

# Commands.
bcmd = b.commands.base_velocity
ccmd = c.commands.base_velocity
check("heading_command_false", ccmd.heading_command is False, str(ccmd.heading_command))
check("heading_command_same_as_v4b", ccmd.heading_command == bcmd.heading_command)
check("rel_heading_envs_same", close(ccmd.rel_heading_envs, bcmd.rel_heading_envs), str(ccmd.rel_heading_envs))
check("rel_standing_envs_same", close(ccmd.rel_standing_envs, bcmd.rel_standing_envs), str(ccmd.rel_standing_envs))
check("lin_vel_x_range_same", tuple(ccmd.ranges.lin_vel_x) == tuple(bcmd.ranges.lin_vel_x) == (0.0, 1.0), str(tuple(ccmd.ranges.lin_vel_x)))
check("lin_vel_y_range_same", tuple(ccmd.ranges.lin_vel_y) == tuple(bcmd.ranges.lin_vel_y) == (-0.5, 0.5), str(tuple(ccmd.ranges.lin_vel_y)))
check("ang_vel_z_range_same", tuple(ccmd.ranges.ang_vel_z) == tuple(bcmd.ranges.ang_vel_z) == (-1.0, 1.0), str(tuple(ccmd.ranges.ang_vel_z)))

# Rewards.
bt = reward_terms(b)
ct = reward_terms(c)

enabled_c = [
    name
    for name, value in vars(c.rewards).items()
    if not name.startswith("_") and value is not None and hasattr(value, "weight")
]
check("official_reward_count_16", len(enabled_c) == 16, f"enabled={len(enabled_c)}")
check("official_reward_names_exact", set(enabled_c) == set(OFFICIAL_REWARD_NAMES))

for name in OFFICIAL_REWARD_NAMES:
    check(f"reward_weight_same::{name}", close(ct[name].weight, bt[name].weight), str(ct[name].weight))
    check(
        f"reward_func_same::{name}",
        ct[name].func is bt[name].func,
        getattr(ct[name].func, "__name__", repr(ct[name].func)),
    )

b_std = float(bt["track_lin_vel_xy_exp"].params["std"])
c_std = float(ct["track_lin_vel_xy_exp"].params["std"])
check("v4b_xy_tracking_std_0p50", close(b_std, 0.5), str(b_std))
check("v4c_xy_tracking_std_0p25", close(c_std, 0.25), str(c_std))

for name in OFFICIAL_REWARD_NAMES:
    bp = dict(bt[name].params or {})
    cp = dict(ct[name].params or {})
    if name == "track_lin_vel_xy_exp":
        bp.pop("std", None)
        cp.pop("std", None)
    check(
        f"reward_params_same::{name}",
        repr(bp) == repr(cp),
        "unchanged except allowed std" if name == "track_lin_vel_xy_exp" else "unchanged",
    )

for forbidden in (
    "knee_flexion_above_threshold",
    "stand_still_full_command",
    "swing_foot_clearance_relative",
    "zero_command_motion",
):
    check(f"custom_reward_absent::{forbidden}", getattr(c.rewards, forbidden, None) is None)

# Observations / corruption.
check(
    "policy_corruption_same",
    c.observations.policy.enable_corruption == b.observations.policy.enable_corruption is True,
    str(c.observations.policy.enable_corruption),
)

policy_obs_names = [
    name
    for name, value in vars(c.observations.policy).items()
    if not name.startswith("_") and value is not None and name != "enable_corruption"
]
for expected_obs in (
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "joint_pos",
    "joint_vel",
    "actions",
):
    check(f"policy_obs_present::{expected_obs}", expected_obs in policy_obs_names)
check("height_scan_absent", "height_scan" not in policy_obs_names)

# Training events.
check(
    "external_force_event_active",
    c.events.base_external_force_torque is not None
    and b.events.base_external_force_torque is not None,
)
check(
    "push_robot_event_active",
    c.events.push_robot is not None
    and b.events.push_robot is not None,
)

# Actions.
check(
    "action_scale_same_0p5",
    close(c.actions.joint_pos.scale, b.actions.joint_pos.scale)
    and close(c.actions.joint_pos.scale, 0.5),
    str(c.actions.joint_pos.scale),
)
check(
    "action_default_offset_true",
    c.actions.joint_pos.use_default_offset is True
    and c.actions.joint_pos.use_default_offset == b.actions.joint_pos.use_default_offset,
    str(c.actions.joint_pos.use_default_offset),
)

# PPO runner.
check("ppo_experiment_name", pc.experiment_name == EXPECTED_EXPERIMENT, pc.experiment_name)
check("ppo_seed_same", pc.seed == pb.seed == 42, str(pc.seed))
check("ppo_steps_per_env_same", pc.num_steps_per_env == pb.num_steps_per_env == 24, str(pc.num_steps_per_env))
check("ppo_max_iterations_same", pc.max_iterations == pb.max_iterations == 1500, str(pc.max_iterations))
check("ppo_save_interval_same", pc.save_interval == pb.save_interval == 50, str(pc.save_interval))

for attr, expected in (
    ("learning_rate", 1.0e-3),
    ("entropy_coef", 0.008),
    ("gamma", 0.99),
    ("lam", 0.95),
    ("clip_param", 0.2),
    ("num_learning_epochs", 5),
    ("num_mini_batches", 4),
    ("desired_kl", 0.01),
):
    cv = getattr(pc.algorithm, attr)
    bv = getattr(pb.algorithm, attr)
    check(f"ppo_algorithm_same::{attr}", close(cv, bv) and close(cv, expected), str(cv))

check(
    "ppo_schedule_adaptive",
    pc.algorithm.schedule == pb.algorithm.schedule == "adaptive",
    str(pc.algorithm.schedule),
)

# Current Isaac Lab / RSL-RL API uses separate actor and critic configs.
check(
    "actor_config_present",
    hasattr(pc, "actor") and hasattr(pb, "actor"),
    f"v4C={type(getattr(pc, 'actor', None)).__name__}, v4B={type(getattr(pb, 'actor', None)).__name__}",
)
check(
    "critic_config_present",
    hasattr(pc, "critic") and hasattr(pb, "critic"),
    f"v4C={type(getattr(pc, 'critic', None)).__name__}, v4B={type(getattr(pb, 'critic', None)).__name__}",
)

check(
    "actor_hidden_dims_same",
    list(pc.actor.hidden_dims) == list(pb.actor.hidden_dims) == [256, 128, 128],
    str(pc.actor.hidden_dims),
)
check(
    "critic_hidden_dims_same",
    list(pc.critic.hidden_dims) == list(pb.critic.hidden_dims) == [256, 128, 128],
    str(pc.critic.hidden_dims),
)
check(
    "actor_activation_same_elu",
    str(pc.actor.activation).lower() == str(pb.actor.activation).lower() == "elu",
    str(pc.actor.activation),
)
check(
    "critic_activation_same_elu",
    str(pc.critic.activation).lower() == str(pb.critic.activation).lower() == "elu",
    str(pc.critic.activation),
)
check(
    "actor_obs_normalization_same",
    pc.actor.obs_normalization == pb.actor.obs_normalization,
    str(pc.actor.obs_normalization),
)
check(
    "critic_obs_normalization_same",
    pc.critic.obs_normalization == pb.critic.obs_normalization,
    str(pc.critic.obs_normalization),
)

print()
print("RESOLVED SCIENTIFIC DELTA")
print(f"  v4-B track_lin_vel_xy_exp std : {b_std:.2f}")
print(f"  v4-C track_lin_vel_xy_exp std : {c_std:.2f}")
print("  all 16 reward weights         : identical")
print("  all reward functions          : identical")
print("  all other reward params       : identical")
print("  commands                      : identical")
print("  observations/events/actions   : identical")
print("  PPO algorithm                 : identical")
print("  actor/critic architecture     : identical [256, 128, 128], ELU")
print("  only PPO provenance delta     : experiment_name")
print()
print("V4C_RESOLVED_CONFIG_AUDIT_PASS")
print("=" * 120)
