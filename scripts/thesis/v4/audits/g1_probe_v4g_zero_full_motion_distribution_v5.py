from __future__ import annotations

import argparse
import sys
import traceback
from importlib import metadata
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Read-only distribution probe for the v4-G zero-command full-motion reward."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from rsl_rl.runners import OnPolicyRunner

import humanoid_rl_lab.tasks  # noqa: F401


TASK = "Humanoid-RL-Lab-G1-Flat-V4G-Zero-Command-Full-Motion-v0"
RUN_DIR = Path(
    r"C:\humanoid_rl_lab\logs\rsl_rl\humanoid_rl_lab_g1_v4g_zero_command_full_motion_scratch"
    r"\2026-09-02_23-59-15_v4g_genuine_scratch_seed42_official1500"
)
CHECKPOINT_NAME = "model_0.pt"

NUM_ENVS = 4096
ROLLOUT_STEPS = 24
SEED = 42

COMMAND_THRESHOLD = 0.06
BASE_PLANAR_SCALE = 1.0
YAW_RATE_SCALE = 1.0
FOOT_XY_SCALE = 0.25
REWARD_WEIGHT = -5.0

FOOT_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]


def stage(name: str) -> None:
    print(f"[STAGE] {name}", flush=True)


def fail(msg: str) -> None:
    raise RuntimeError(msg)


def as_tensor(value):
    if hasattr(value, "torch"):
        value = value.torch
    if not torch.is_tensor(value):
        fail(f"expected torch tensor, got {type(value)!r}")
    return value


def summarize(name: str, values: torch.Tensor) -> None:
    values = values.detach().float().reshape(-1).cpu()

    if values.numel() == 0:
        print(f"{name:28s}: NO ACTIVE SAMPLES")
        return

    if not torch.isfinite(values).all():
        fail(f"non-finite values in {name}")

    q_levels = torch.tensor(
        [0.50, 0.90, 0.95, 0.99, 0.999],
        dtype=values.dtype,
    )
    q = torch.quantile(values, q_levels)

    print(
        f"{name:28s}: "
        f"n={values.numel():7d}  "
        f"mean={values.mean().item():12.6f}  "
        f"std={values.std(unbiased=False).item():12.6f}  "
        f"median={q[0].item():12.6f}  "
        f"p90={q[1].item():12.6f}  "
        f"p95={q[2].item():12.6f}  "
        f"p99={q[3].item():12.6f}  "
        f"p99.9={q[4].item():12.6f}  "
        f"max={values.max().item():12.6f}"
    )


env = None

try:
    stage("ENTER_TRY")
    checkpoint_path = RUN_DIR / CHECKPOINT_NAME
    if not checkpoint_path.is_file():
        fail(f"checkpoint not found: {checkpoint_path}")

    stage("CHECKPOINT_EXISTS")
    env_cfg = parse_env_cfg(
        TASK,
        device=args_cli.device,
        num_envs=NUM_ENVS,
    )
    env_cfg.seed = SEED

    stage("ENV_CFG_PARSED")
    agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    agent_cfg.device = args_cli.device

    installed_version = metadata.version("rsl-rl-lib")
    print(f"[INFO] installed rsl-rl-lib version: {installed_version}", flush=True)
    stage("ABOUT_TO_MIGRATE_RSL_RL_CFG")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    stage("RSL_RL_CFG_MIGRATED")

    term = env_cfg.rewards.zero_command_full_motion
    if term.weight != REWARD_WEIGHT:
        fail(f"reward weight mismatch: {term.weight!r}")
    params = term.params

    exact_expected = {
        "command_threshold": COMMAND_THRESHOLD,
        "base_planar_scale": BASE_PLANAR_SCALE,
        "yaw_rate_scale": YAW_RATE_SCALE,
        "foot_xy_scale": FOOT_XY_SCALE,
    }
    for key, expected in exact_expected.items():
        if params[key] != expected:
            fail(f"{key} mismatch: {params[key]!r} != {expected!r}")

    if env_cfg.commands.base_velocity.rel_standing_envs != 0.02:
        fail(
            "rel_standing_envs mismatch: "
            f"{env_cfg.commands.base_velocity.rel_standing_envs!r}"
        )
    if env_cfg.commands.base_velocity.heading_command is not False:
        fail(
            "heading_command mismatch: "
            f"{env_cfg.commands.base_velocity.heading_command!r}"
        )

    stage("CONFIG_GUARDS_PASS")
    raw_env = gym.make(TASK, cfg=env_cfg)
    stage("GYM_ENV_CREATED")
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)

    stage("RSL_WRAPPER_CREATED")
    runner_cfg = agent_cfg.to_dict()

    actor_cfg = runner_cfg.get("actor")
    if not isinstance(actor_cfg, dict):
        fail(f"post-migration runner actor config is not a dict: {type(actor_cfg)!r}")

    forbidden_legacy = {
        "stochastic",
        "init_noise_std",
        "noise_std_type",
        "state_dependent_std",
    }
    legacy_remaining = sorted(forbidden_legacy.intersection(actor_cfg))
    if legacy_remaining:
        fail(
            "deprecated actor fields remain after handle_deprecated_rsl_rl_cfg: "
            f"{legacy_remaining}"
        )

    distribution_cfg = actor_cfg.get("distribution_cfg")
    if not isinstance(distribution_cfg, dict):
        fail(
            "post-migration actor distribution_cfg missing/not dict: "
            f"{distribution_cfg!r}"
        )

    print(
        "[INFO] migrated actor distribution_cfg: "
        f"{distribution_cfg}",
        flush=True,
    )

    runner = OnPolicyRunner(
        env,
        runner_cfg,
        log_dir=None,
        device=agent_cfg.device,
    )
    stage("RUNNER_CREATED")
    stage("ABOUT_TO_LOAD_CHECKPOINT")
    runner.load(str(checkpoint_path))
    stage("CHECKPOINT_LOADED")

    if not hasattr(runner, "alg"):
        fail("runner has no alg attribute")
    if not hasattr(runner.alg, "get_policy"):
        fail("runner.alg has no get_policy() method")

    actor = runner.alg.get_policy()
    if not callable(actor):
        fail(f"runner.alg.get_policy() is not callable: {type(actor)!r}")

    stage("POLICY_ACCESS_GUARDS_PASS")
    print(
        f"[INFO] stochastic action path: {type(runner.alg).__name__}.get_policy()"
        "(obs, stochastic_output=True)",
        flush=True,
    )
    stage("POLICY_ACQUIRED")
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    body_names = list(robot.data.body_names)

    missing = [name for name in FOOT_NAMES if name not in body_names]
    if missing:
        fail(f"foot bodies missing from robot: {missing}")

    foot_ids = [body_names.index(name) for name in FOOT_NAMES]

    # Installed PPO samples training actions via actor(obs, stochastic_output=True).
    # Seed Torch immediately before the diagnostic rollout for reproducibility.
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    stage("FOOT_RESOLUTION_PASS")
    obs = env.get_observations()
    stage("OBSERVATIONS_ACQUIRED")

    active_flags = []
    base_contrib_samples = []
    yaw_contrib_samples = []
    foot_contrib_samples = []
    raw_cost_samples = []
    weighted_reward_samples = []

    active_counts_per_step = []
    raw_max_per_step = []

    stage("ABOUT_TO_ROLLOUT")
    with torch.inference_mode():
        for step_idx in range(ROLLOUT_STEPS):
            actions = actor(obs, stochastic_output=True)
            obs, _, _, _ = env.step(actions)

            command = as_tensor(
                unwrapped.command_manager.get_command("base_velocity")
            )
            root_lin_vel_b = as_tensor(robot.data.root_lin_vel_b)
            root_ang_vel_b = as_tensor(robot.data.root_ang_vel_b)
            body_lin_vel_w = as_tensor(robot.data.body_lin_vel_w)

            if command.shape[0] != NUM_ENVS:
                fail(
                    f"command env count mismatch at step {step_idx}: "
                    f"{command.shape[0]} != {NUM_ENVS}"
                )

            base_planar_sq = torch.sum(
                torch.square(root_lin_vel_b[:, :2]),
                dim=1,
            )
            yaw_rate_sq = torch.square(root_ang_vel_b[:, 2])

            foot_xy_sq = torch.sum(
                torch.square(body_lin_vel_w[:, foot_ids, :2]),
                dim=-1,
            )
            mean_foot_xy_sq = torch.mean(foot_xy_sq, dim=1)

            base_contrib = BASE_PLANAR_SCALE * base_planar_sq
            yaw_contrib = YAW_RATE_SCALE * yaw_rate_sq
            foot_contrib = FOOT_XY_SCALE * mean_foot_xy_sq

            raw_cost = base_contrib + yaw_contrib + foot_contrib
            weighted_reward = REWARD_WEIGHT * raw_cost

            full_command_norm = torch.linalg.norm(command[:, :3], dim=1)
            active = full_command_norm < COMMAND_THRESHOLD

            for label, tensor in (
                ("base_contrib", base_contrib),
                ("yaw_contrib", yaw_contrib),
                ("foot_contrib", foot_contrib),
                ("raw_cost", raw_cost),
                ("weighted_reward", weighted_reward),
            ):
                if not torch.isfinite(tensor).all():
                    fail(f"non-finite {label} at rollout step {step_idx}")

            active_flags.append(active.detach().cpu())
            active_counts_per_step.append(int(active.sum().item()))

            if bool(active.any().item()):
                base_contrib_samples.append(base_contrib[active].detach().cpu())
                yaw_contrib_samples.append(yaw_contrib[active].detach().cpu())
                foot_contrib_samples.append(foot_contrib[active].detach().cpu())
                raw_cost_samples.append(raw_cost[active].detach().cpu())
                weighted_reward_samples.append(weighted_reward[active].detach().cpu())
                raw_max_per_step.append(float(raw_cost[active].max().item()))
            else:
                raw_max_per_step.append(0.0)

    stage("ROLLOUT_COMPLETE")
    all_active = torch.cat(active_flags)
    total_samples = int(all_active.numel())
    active_samples = int(all_active.sum().item())
    active_fraction = active_samples / total_samples

    if active_samples == 0:
        fail("zero active near-zero-command samples collected")

    base_values = torch.cat(base_contrib_samples)
    yaw_values = torch.cat(yaw_contrib_samples)
    foot_values = torch.cat(foot_contrib_samples)
    raw_values = torch.cat(raw_cost_samples)
    weighted_values = torch.cat(weighted_reward_samples)

    if not (
        base_values.numel()
        == yaw_values.numel()
        == foot_values.numel()
        == raw_values.numel()
        == weighted_values.numel()
        == active_samples
    ):
        fail("active-sample aggregation length mismatch")

    reconstruction_error = torch.max(
        torch.abs(raw_values - (base_values + yaw_values + foot_values))
    ).item()
    if reconstruction_error > 1.0e-6:
        fail(
            "raw-cost reconstruction mismatch: "
            f"max abs error={reconstruction_error}"
        )

    print("=" * 180)
    stage("AGGREGATION_COMPLETE")
    print("V4-G ZERO-COMMAND FULL-MOTION DISTRIBUTION PROBE")
    print("=" * 180)
    print(f"task                     : {TASK}")
    print(f"run                      : {RUN_DIR}")
    print(f"checkpoint               : {checkpoint_path}")
    print(f"num envs                 : {NUM_ENVS}")
    print(f"rollout steps            : {ROLLOUT_STEPS}")
    print(f"total env-step samples   : {total_samples}")
    print(f"seed                     : {SEED}")
    print("policy actions           : STOCHASTIC actor(obs, stochastic_output=True)")
    print(f"command threshold        : {COMMAND_THRESHOLD}")
    print(f"reward weight            : {REWARD_WEIGHT}")
    print(f"base planar scale        : {BASE_PLANAR_SCALE}")
    print(f"yaw-rate scale           : {YAW_RATE_SCALE}")
    print(f"foot XY scale            : {FOOT_XY_SCALE}")
    print(f"resolved foot ids        : {foot_ids}")
    print()
    print(f"active samples           : {active_samples}")
    print(f"active fraction          : {active_fraction:.6f} ({100.0 * active_fraction:.3f}%)")
    print(
        "active count/step       : "
        f"min={min(active_counts_per_step)}  "
        f"mean={sum(active_counts_per_step)/len(active_counts_per_step):.2f}  "
        f"max={max(active_counts_per_step)}"
    )
    print()
    print("ACTIVE-SAMPLE DISTRIBUTIONS")
    print("Each component below is its contribution to raw_cost after its configured scale.")
    summarize("base_planar contribution", base_values)
    summarize("yaw_rate contribution", yaw_values)
    summarize("foot_xy contribution", foot_values)
    summarize("raw_cost total", raw_values)
    summarize("weighted reward (-5*x)", weighted_values)
    print()
    print(
        "raw_cost max per step   : "
        f"min={min(raw_max_per_step):.6f}  "
        f"mean={sum(raw_max_per_step)/len(raw_max_per_step):.6f}  "
        f"max={max(raw_max_per_step):.6f}"
    )
    print(f"reconstruction max error : {reconstruction_error:.3e}")
    print()
    print("No training update performed. Repository files were not modified.")
    print()
    print("V4G_ZERO_FULL_MOTION_DISTRIBUTION_PROBE_PASS")
    print("=" * 180)

except BaseException as exc:
    print("", flush=True)
    print("=" * 120, flush=True)
    print("V4G_ZERO_FULL_MOTION_PROBE_V5_EXCEPTION", flush=True)
    print(f"type    : {type(exc).__name__}", flush=True)
    print(f"message : {exc}", flush=True)
    print("traceback:", flush=True)
    traceback.print_exc(file=sys.stdout)
    print("=" * 120, flush=True)
    raise
finally:
    stage("FINALLY_ENTER")
    if env is not None:
        stage("CLOSING_ENV")
        env.close()
    stage("CLOSING_SIMULATION_APP")
    simulation_app.close()
    stage("FINALLY_EXIT")
