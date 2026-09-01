import inspect
from pathlib import Path

TASK_ID = "Humanoid-RL-Lab-G1-Flat-V4A-Official-Baseline-v0"

import humanoid_rl_lab.tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

def fn_name(obj):
    if obj is None:
        return "None"
    f = getattr(obj, "func", None)
    if f is None:
        return type(obj).__name__
    return getattr(f, "__name__", str(f))

def get_weight(term):
    return getattr(term, "weight", None)

env_cfg = load_cfg_from_registry(TASK_ID, "env_cfg_entry_point")
agent_cfg = load_cfg_from_registry(TASK_ID, "rsl_rl_cfg_entry_point")

print("=" * 112)
print("V4-A RESOLVED CONFIG AUDIT")
print(f"Task ID                             : {TASK_ID}")
print(f"Env cfg class                       : {type(env_cfg).__module__}.{type(env_cfg).__name__}")
print(f"Agent cfg class                     : {type(agent_cfg).__module__}.{type(agent_cfg).__name__}")
print("=" * 112)

# Commands
cmd = env_cfg.commands.base_velocity
print("COMMANDS")
print(f"  heading_command                   : {cmd.heading_command}")
print(f"  rel_heading_envs                  : {cmd.rel_heading_envs}")
print(f"  rel_standing_envs                 : {cmd.rel_standing_envs}")
print(f"  lin_vel_x                         : {cmd.ranges.lin_vel_x}")
print(f"  lin_vel_y                         : {cmd.ranges.lin_vel_y}")
print(f"  ang_vel_z                         : {cmd.ranges.ang_vel_z}")

# Rewards
print("REWARDS")
reward_names = []
for name, term in vars(env_cfg.rewards).items():
    if name.startswith("_") or term is None:
        continue
    if hasattr(term, "weight"):
        reward_names.append(name)
        print(f"  {name:<34} weight={get_weight(term)!r:<12} func={fn_name(term)}")

print(f"  reward_term_count                 : {len(reward_names)}")
print(f"  knee_custom_present               : {'knee_flexion_above_threshold' in reward_names}")

# Observation policy group
policy_obs = env_cfg.observations.policy
print("OBSERVATIONS")
print(f"  enable_corruption                 : {getattr(policy_obs, 'enable_corruption', None)}")
obs_names = []
for name, term in vars(policy_obs).items():
    if name.startswith("_") or term is None:
        continue
    if hasattr(term, "func"):
        obs_names.append(name)
        print(f"  {name:<34} func={fn_name(term)}")
print(f"  policy_obs_term_count             : {len(obs_names)}")

# Events
print("EVENTS")
for ev_name in ("base_external_force_torque", "push_robot", "reset_base", "reset_robot_joints"):
    ev = getattr(env_cfg.events, ev_name, None)
    print(f"  {ev_name:<34}: {'ACTIVE' if ev is not None else 'NONE'}")

# Actions
print("ACTIONS")
for name, term in vars(env_cfg.actions).items():
    if name.startswith("_") or term is None:
        continue
    print(f"  {name:<34}: {type(term).__module__}.{type(term).__name__}")
    for attr in ("scale", "use_default_offset"):
        if hasattr(term, attr):
            print(f"    {attr:<31}: {getattr(term, attr)}")

# PPO
alg = agent_cfg.algorithm
actor = agent_cfg.actor
critic = agent_cfg.critic
print("PPO")
for key in (
    "num_steps_per_env", "max_iterations", "save_interval", "seed", "experiment_name"
):
    print(f"  {key:<34}: {getattr(agent_cfg, key, None)}")
for key in (
    "learning_rate", "entropy_coef", "schedule", "gamma", "lam",
    "clip_param", "num_learning_epochs", "num_mini_batches", "desired_kl"
):
    print(f"  algorithm.{key:<24}: {getattr(alg, key, None)}")
print(f"  actor.hidden_dims                  : {getattr(actor, 'hidden_dims', None)}")
print(f"  actor.activation                   : {getattr(actor, 'activation', None)}")
print(f"  critic.hidden_dims                 : {getattr(critic, 'hidden_dims', None)}")
print(f"  critic.activation                  : {getattr(critic, 'activation', None)}")

# Hard scientific checks for v4-A provenance.
checks = {
    "env_is_v4a_class": type(env_cfg).__name__ == "HumanoidRlLabG1V4AOfficialBaselineEnvCfg",
    "agent_is_v4a_class": type(agent_cfg).__name__ == "V4AOfficialBaselinePPORunnerCfg",
    "no_custom_knee_reward": "knee_flexion_above_threshold" not in reward_names,
    "official_flat_orientation": abs(float(env_cfg.rewards.flat_orientation_l2.weight) - (-1.0)) < 1e-12,
    "official_feet_air_time": abs(float(env_cfg.rewards.feet_air_time.weight) - 0.75) < 1e-12,
    "official_feet_slide": abs(float(env_cfg.rewards.feet_slide.weight) - (-0.1)) < 1e-12,
    "official_lr": abs(float(alg.learning_rate) - 1.0e-3) < 1e-15,
    "official_entropy": abs(float(alg.entropy_coef) - 0.008) < 1e-15,
    "official_schedule": str(alg.schedule) == "adaptive",
    "scratch_experiment_name": str(agent_cfg.experiment_name) == "humanoid_rl_lab_g1_v4a_official_baseline_scratch",
}
print("CHECKS")
failed = []
for name, ok in checks.items():
    print(f"  {name:<34}: {'PASS' if ok else 'FAIL'}")
    if not ok:
        failed.append(name)

if failed:
    print("=" * 112)
    print("V4A_RESOLVED_CONFIG_AUDIT_FAIL")
    print("FAILED:", ", ".join(failed))
    raise SystemExit(2)

print("=" * 112)
print("V4A_RESOLVED_CONFIG_AUDIT_PASS")
print("=" * 112)
