import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Resolved-config audit for v4-B direct-yaw genuine-scratch task.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

TASK_ID = "Humanoid-RL-Lab-G1-Flat-V4B-Direct-Yaw-v0"
EXPECTED_EXPERIMENT = "humanoid_rl_lab_g1_v4b_direct_yaw_scratch"

EXPECTED_REWARDS = {
    "track_lin_vel_xy_exp": 1.0,
    "track_ang_vel_z_exp": 1.0,
    "lin_vel_z_l2": -0.2,
    "ang_vel_xy_l2": -0.05,
    "dof_torques_l2": -2.0e-6,
    "dof_acc_l2": -1.0e-7,
    "action_rate_l2": -0.005,
    "feet_air_time": 0.75,
    "flat_orientation_l2": -1.0,
    "dof_pos_limits": -1.0,
    "termination_penalty": -200.0,
    "feet_slide": -0.1,
    "joint_deviation_hip": -0.1,
    "joint_deviation_arms": -0.1,
    "joint_deviation_fingers": -0.05,
    "joint_deviation_torso": -0.1,
}

def close(a, b, atol=1.0e-12):
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=atol)

def check(name, cond, detail=""):
    status = "PASS" if bool(cond) else "FAIL"
    suffix = f" | {detail}" if detail else ""
    print(f"{name:<42} {status}{suffix}")
    if not cond:
        raise SystemExit(f"ABORT: {name} failed{suffix}")

try:
    import humanoid_rl_lab.tasks  # noqa: F401

    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    env_cfg = parse_env_cfg(TASK_ID, device="cuda:0", num_envs=4)
    agent_cfg = load_cfg_from_registry(TASK_ID, "rsl_rl_cfg_entry_point")

    cmd = env_cfg.commands.base_velocity

    print("=" * 112)
    print("V4-B RESOLVED CONFIG AUDIT")
    print("=" * 112)
    print(f"Task ID                              : {TASK_ID}")
    print()

    print("COMMANDS")
    print("-" * 112)
    print(f"heading_command                      : {cmd.heading_command}")
    print(f"rel_heading_envs                     : {cmd.rel_heading_envs}")
    print(f"rel_standing_envs                    : {cmd.rel_standing_envs}")
    print(f"lin_vel_x                            : {cmd.ranges.lin_vel_x}")
    print(f"lin_vel_y                            : {cmd.ranges.lin_vel_y}")
    print(f"ang_vel_z                            : {cmd.ranges.ang_vel_z}")
    print()

    check("heading_command_false", cmd.heading_command is False)
    check("rel_heading_envs_1", close(cmd.rel_heading_envs, 1.0), str(cmd.rel_heading_envs))
    check("rel_standing_envs_002", close(cmd.rel_standing_envs, 0.02), str(cmd.rel_standing_envs))
    check("vx_range_0_1", tuple(cmd.ranges.lin_vel_x) == (0.0, 1.0), str(cmd.ranges.lin_vel_x))
    check("vy_range_pm_05", tuple(cmd.ranges.lin_vel_y) == (-0.5, 0.5), str(cmd.ranges.lin_vel_y))
    check("wz_range_pm_1", tuple(cmd.ranges.ang_vel_z) == (-1.0, 1.0), str(cmd.ranges.ang_vel_z))

    print()
    print("REWARDS")
    print("-" * 112)
    active_rewards = {}
    for name, expected_weight in EXPECTED_REWARDS.items():
        term = getattr(env_cfg.rewards, name, None)
        check(f"reward_present_{name}", term is not None)
        actual_weight = float(term.weight)
        active_rewards[name] = actual_weight
        print(f"{name:<36} {actual_weight:+.12g}")
        check(f"reward_weight_{name}", close(actual_weight, expected_weight), f"{actual_weight} vs {expected_weight}")

    reward_terms = {
        name
        for name, value in vars(env_cfg.rewards).items()
        if value is not None and hasattr(value, "weight")
    }
    check("reward_count_16", len(reward_terms) == 16, str(sorted(reward_terms)))
    check("reward_set_exact", reward_terms == set(EXPECTED_REWARDS), str(sorted(reward_terms)))
    check(
        "custom_knee_reward_absent",
        not any("knee_flexion_above_threshold" in x for x in reward_terms),
        str(sorted(reward_terms)),
    )

    print()
    print("OBSERVATIONS / EVENTS / ACTION")
    print("-" * 112)
    policy_obs = env_cfg.observations.policy
    check("policy_corruption_enabled", policy_obs.enable_corruption is True, str(policy_obs.enable_corruption))

    expected_obs_terms = (
        "base_lin_vel",
        "base_ang_vel",
        "projected_gravity",
        "velocity_commands",
        "joint_pos",
        "joint_vel",
        "actions",
    )
    for name in expected_obs_terms:
        check(f"obs_term_{name}", getattr(policy_obs, name, None) is not None)

    check("height_scan_absent", getattr(policy_obs, "height_scan", None) is None)
    check("event_external_force_active", env_cfg.events.base_external_force_torque is not None)
    check("event_push_robot_active", env_cfg.events.push_robot is not None)

    action = env_cfg.actions.joint_pos
    check("action_scale_05", close(action.scale, 0.5), str(action.scale))
    check("action_default_offset_true", action.use_default_offset is True, str(action.use_default_offset))

    print()
    print("PPO")
    print("-" * 112)
    print(f"experiment_name                      : {agent_cfg.experiment_name}")
    print(f"seed                                 : {agent_cfg.seed}")
    print(f"num_steps_per_env                    : {agent_cfg.num_steps_per_env}")
    print(f"max_iterations                       : {agent_cfg.max_iterations}")
    print(f"save_interval                        : {agent_cfg.save_interval}")
    print(f"learning_rate                        : {agent_cfg.algorithm.learning_rate}")
    print(f"entropy_coef                         : {agent_cfg.algorithm.entropy_coef}")
    print(f"schedule                             : {agent_cfg.algorithm.schedule}")
    print(f"gamma                                : {agent_cfg.algorithm.gamma}")
    print(f"lam                                  : {agent_cfg.algorithm.lam}")
    print(f"clip_param                           : {agent_cfg.algorithm.clip_param}")
    print(f"num_learning_epochs                  : {agent_cfg.algorithm.num_learning_epochs}")
    print(f"num_mini_batches                     : {agent_cfg.algorithm.num_mini_batches}")
    print(f"desired_kl                           : {agent_cfg.algorithm.desired_kl}")
    print(f"actor_hidden_dims                    : {agent_cfg.policy.actor_hidden_dims}")
    print(f"critic_hidden_dims                   : {agent_cfg.policy.critic_hidden_dims}")
    print(f"activation                           : {agent_cfg.policy.activation}")

    check("ppo_experiment_name", agent_cfg.experiment_name == EXPECTED_EXPERIMENT, agent_cfg.experiment_name)
    check("ppo_seed_42", int(agent_cfg.seed) == 42, str(agent_cfg.seed))
    check("ppo_steps_24", int(agent_cfg.num_steps_per_env) == 24, str(agent_cfg.num_steps_per_env))
    check("ppo_iterations_1500", int(agent_cfg.max_iterations) == 1500, str(agent_cfg.max_iterations))
    check("ppo_save_interval_50", int(agent_cfg.save_interval) == 50, str(agent_cfg.save_interval))
    check("ppo_lr_1e3", close(agent_cfg.algorithm.learning_rate, 1.0e-3), str(agent_cfg.algorithm.learning_rate))
    check("ppo_entropy_008", close(agent_cfg.algorithm.entropy_coef, 0.008), str(agent_cfg.algorithm.entropy_coef))
    check("ppo_schedule_adaptive", str(agent_cfg.algorithm.schedule) == "adaptive", str(agent_cfg.algorithm.schedule))
    check("ppo_gamma_099", close(agent_cfg.algorithm.gamma, 0.99), str(agent_cfg.algorithm.gamma))
    check("ppo_lambda_095", close(agent_cfg.algorithm.lam, 0.95), str(agent_cfg.algorithm.lam))
    check("ppo_clip_02", close(agent_cfg.algorithm.clip_param, 0.2), str(agent_cfg.algorithm.clip_param))
    check("ppo_epochs_5", int(agent_cfg.algorithm.num_learning_epochs) == 5, str(agent_cfg.algorithm.num_learning_epochs))
    check("ppo_minibatches_4", int(agent_cfg.algorithm.num_mini_batches) == 4, str(agent_cfg.algorithm.num_mini_batches))
    check("ppo_desired_kl_001", close(agent_cfg.algorithm.desired_kl, 0.01), str(agent_cfg.algorithm.desired_kl))
    check("actor_dims_256_128_128", list(agent_cfg.policy.actor_hidden_dims) == [256, 128, 128], str(agent_cfg.policy.actor_hidden_dims))
    check("critic_dims_256_128_128", list(agent_cfg.policy.critic_hidden_dims) == [256, 128, 128], str(agent_cfg.policy.critic_hidden_dims))
    check("activation_elu", str(agent_cfg.policy.activation).lower() == "elu", str(agent_cfg.policy.activation))

    print()
    print("=" * 112)
    print("V4B_RESOLVED_CONFIG_AUDIT_PASS")
    print("Only intended behavioral delta from v4-A: heading_command True -> False.")
    print("No checkpoint/resume path is configured by this audit.")
    print("=" * 112)

finally:
    simulation_app.close()
