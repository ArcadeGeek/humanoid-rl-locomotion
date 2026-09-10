from __future__ import annotations

import argparse
import sys
import traceback
from importlib import metadata
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description=(
        "Read-only offline sizing of a bounded zero-command yaw-rate penalty "
        "on the healthy v4-B model_0 scratch transient."
    )
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from rsl_rl.runners import OnPolicyRunner

import humanoid_rl_lab.tasks  # noqa: F401


TASK = "Humanoid-RL-Lab-G1-Flat-V4G-Zero-Command-Full-Motion-v0"

REFERENCE_CHECKPOINT = Path(
    r"C:\humanoid_rl_lab\logs\rsl_rl\humanoid_rl_lab_g1_v4b_direct_yaw_scratch"
    r"\2026-09-01_20-08-18_v4b_genuine_scratch_seed42_official1500\model_0.pt"
)

NUM_ENVS = 4096
ROLLOUT_STEPS = 24
SEED = 42

COMMAND_THRESHOLD = 0.06
BASE_PLANAR_SCALE = 1.0
FOOT_XY_SCALE = 0.25
HYPOTHETICAL_WEIGHT = -5.0

# Candidate saturation scales in rad/s.
SIGMAS = (0.10, 0.20, 0.30, 0.50, 1.00)

# Useful physical reference rates for reading the transfer function.
REFERENCE_WZ = (0.05, 0.10, 0.176, 0.25, 0.50, 1.00, 5.00, 15.00)

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


def stats(values: torch.Tensor) -> dict[str, float]:
    x = values.detach().float().reshape(-1).cpu()
    if x.numel() == 0:
        fail("cannot summarize empty tensor")
    if not torch.isfinite(x).all():
        fail("non-finite statistics input")

    q = torch.quantile(
        x,
        torch.tensor([0.50, 0.90, 0.95, 0.99, 0.999], dtype=x.dtype),
    )
    return {
        "n": int(x.numel()),
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "median": float(q[0].item()),
        "p90": float(q[1].item()),
        "p95": float(q[2].item()),
        "p99": float(q[3].item()),
        "p999": float(q[4].item()),
        "max": float(x.max().item()),
    }


def print_stats(name: str, s: dict[str, float]) -> None:
    print(
        f"{name:28s}: "
        f"n={s['n']:7d}  "
        f"mean={s['mean']:11.6f}  "
        f"std={s['std']:11.6f}  "
        f"median={s['median']:11.6f}  "
        f"p90={s['p90']:11.6f}  "
        f"p95={s['p95']:11.6f}  "
        f"p99={s['p99']:11.6f}  "
        f"p99.9={s['p999']:11.6f}  "
        f"max={s['max']:11.6f}"
    )


env = None

try:
    stage("ENTER_TRY")

    if not REFERENCE_CHECKPOINT.is_file():
        fail(f"reference checkpoint not found: {REFERENCE_CHECKPOINT}")
    stage("CHECKPOINT_EXISTS")

    env_cfg = parse_env_cfg(
        TASK,
        device=args_cli.device,
        num_envs=NUM_ENVS,
    )
    env_cfg.seed = SEED

    agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    agent_cfg.device = args_cli.device

    installed_version = metadata.version("rsl-rl-lib")
    print(f"[INFO] installed rsl-rl-lib version: {installed_version}", flush=True)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    stage("RSL_RL_CFG_MIGRATED")

    term = env_cfg.rewards.zero_command_full_motion
    if term.weight != HYPOTHETICAL_WEIGHT:
        fail(f"v4-G weight mismatch: {term.weight!r}")
    if term.params["command_threshold"] != COMMAND_THRESHOLD:
        fail(
            "command_threshold mismatch: "
            f"{term.params['command_threshold']!r}"
        )
    if term.params["base_planar_scale"] != BASE_PLANAR_SCALE:
        fail(
            "base_planar_scale mismatch: "
            f"{term.params['base_planar_scale']!r}"
        )
    if term.params["foot_xy_scale"] != FOOT_XY_SCALE:
        fail(
            "foot_xy_scale mismatch: "
            f"{term.params['foot_xy_scale']!r}"
        )
    if term.params["yaw_rate_scale"] != 1.0:
        fail(f"canonical v4-G yaw_rate_scale mismatch: {term.params['yaw_rate_scale']!r}")

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
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    stage("ENV_CREATED")

    runner_cfg = agent_cfg.to_dict()
    actor_cfg = runner_cfg.get("actor")
    if not isinstance(actor_cfg, dict):
        fail(f"post-migration actor config is not dict: {type(actor_cfg)!r}")

    forbidden_legacy = {
        "stochastic",
        "init_noise_std",
        "noise_std_type",
        "state_dependent_std",
    }
    remaining = sorted(forbidden_legacy.intersection(actor_cfg))
    if remaining:
        fail(f"legacy actor fields remain after migration: {remaining}")
    if not isinstance(actor_cfg.get("distribution_cfg"), dict):
        fail("post-migration actor distribution_cfg missing/not dict")

    runner = OnPolicyRunner(
        env,
        runner_cfg,
        log_dir=None,
        device=agent_cfg.device,
    )
    stage("RUNNER_CREATED")

    runner.load(str(REFERENCE_CHECKPOINT))
    stage("CHECKPOINT_LOADED")

    if not hasattr(runner, "alg") or not hasattr(runner.alg, "get_policy"):
        fail("runner algorithm does not expose get_policy()")
    actor = runner.alg.get_policy()
    if not callable(actor):
        fail(f"runner policy is not callable: {type(actor)!r}")

    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    body_names = list(robot.data.body_names)

    missing = [name for name in FOOT_NAMES if name not in body_names]
    if missing:
        fail(f"foot bodies missing: {missing}")
    foot_ids = [body_names.index(name) for name in FOOT_NAMES]
    stage("POLICY_AND_FEET_READY")

    # Reproduce seed-42 scratch-transient sampling without state injection.
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    reset_result = raw_env.reset(seed=SEED)
    if not isinstance(reset_result, tuple) or len(reset_result) != 2:
        fail("raw_env.reset(seed=SEED) did not return (observations, info)")

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    obs = env.get_observations()

    active_masks = []
    base_values = []
    foot_values = []
    wz_values = []
    wz_sq_values = []
    active_counts = []

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

            base_planar_sq = torch.sum(
                torch.square(root_lin_vel_b[:, :2]),
                dim=1,
            )
            abs_wz = torch.abs(root_ang_vel_b[:, 2])
            wz_sq = torch.square(root_ang_vel_b[:, 2])

            foot_xy_sq = torch.sum(
                torch.square(body_lin_vel_w[:, foot_ids, :2]),
                dim=-1,
            )
            mean_foot_xy_sq = torch.mean(foot_xy_sq, dim=1)

            base_contrib = BASE_PLANAR_SCALE * base_planar_sq
            foot_contrib = FOOT_XY_SCALE * mean_foot_xy_sq

            active = (
                torch.linalg.norm(command[:, :3], dim=1)
                < COMMAND_THRESHOLD
            )

            for name, tensor in (
                ("base_contrib", base_contrib),
                ("foot_contrib", foot_contrib),
                ("abs_wz", abs_wz),
                ("wz_sq", wz_sq),
            ):
                if not torch.isfinite(tensor).all():
                    fail(f"non-finite {name} at rollout step {step_idx}")

            active_masks.append(active.detach().cpu())
            active_counts.append(int(active.sum().item()))

            if bool(active.any().item()):
                base_values.append(base_contrib[active].detach().cpu())
                foot_values.append(foot_contrib[active].detach().cpu())
                wz_values.append(abs_wz[active].detach().cpu())
                wz_sq_values.append(wz_sq[active].detach().cpu())

    stage("ROLLOUT_COMPLETE")

    pooled_mask = torch.cat(active_masks)
    active_samples = int(pooled_mask.sum().item())
    total_samples = int(pooled_mask.numel())
    if active_samples == 0:
        fail("no active zero-command samples")

    base = torch.cat(base_values)
    foot = torch.cat(foot_values)
    abs_wz = torch.cat(wz_values)
    wz_sq = torch.cat(wz_sq_values)

    if not (
        base.numel()
        == foot.numel()
        == abs_wz.numel()
        == wz_sq.numel()
        == active_samples
    ):
        fail("pooled active-sample length mismatch")

    translational_only = base + foot
    original_total = translational_only + wz_sq

    active_fraction = active_samples / total_samples

    print("=" * 190)
    print("BOUNDED ZERO-COMMAND YAW PENALTY SIZING — HEALTHY v4-B model_0 SCRATCH TRANSIENT")
    print("=" * 190)
    print(f"environment task           : {TASK}")
    print(f"reference checkpoint       : {REFERENCE_CHECKPOINT}")
    print(f"num envs                   : {NUM_ENVS}")
    print(f"rollout steps              : {ROLLOUT_STEPS}")
    print(f"seed                       : {SEED}")
    print("policy actions             : STOCHASTIC actor(obs, stochastic_output=True)")
    print(f"command threshold          : {COMMAND_THRESHOLD}")
    print(f"hypothetical weight        : {HYPOTHETICAL_WEIGHT}")
    print(f"foot ids                   : {foot_ids}")
    print()
    print(f"active samples             : {active_samples}")
    print(
        f"active fraction            : {active_fraction:.6f} "
        f"({100.0 * active_fraction:.3f}%)"
    )
    print(
        "active count/step         : "
        f"min={min(active_counts)}  "
        f"mean={sum(active_counts)/len(active_counts):.2f}  "
        f"max={max(active_counts)}"
    )
    print()

    print("REFERENCE DISTRIBUTIONS")
    print_stats("abs yaw rate |wz| rad/s", stats(abs_wz))
    print_stats("original yaw wz^2", stats(wz_sq))
    print_stats("base + foot only", stats(translational_only))
    print_stats("original total", stats(original_total))
    print(
        "original pre-dt all-sample weighted mean: "
        f"{HYPOTHETICAL_WEIGHT * original_total.mean().item() * active_fraction:.6f}"
    )
    print()

    print("TRANSFER FUNCTION VALUES  f=1-exp(-(wz/sigma)^2)")
    header = "sigma  " + "  ".join(f"|wz|={v:>6.3f}" for v in REFERENCE_WZ)
    print(header)
    for sigma in SIGMAS:
        vals = [
            1.0 - torch.exp(
                -torch.tensor((wz / sigma) ** 2, dtype=torch.float64)
            ).item()
            for wz in REFERENCE_WZ
        ]
        print(
            f"{sigma:5.2f}  "
            + "  ".join(f"{v:10.6f}" for v in vals)
        )
    print()

    print("EMPIRICAL CANDIDATES ON ACTIVE model_0 SAMPLES")
    print(
        "For each sigma: bounded_yaw is in [0,1]. total = base + foot + bounded_yaw."
    )
    print(
        "all-sample weighted mean includes inactive samples as zero; "
        "it is pre any RewardManager timestep scaling."
    )
    print()

    candidate_rows = []

    for sigma in SIGMAS:
        bounded_yaw = 1.0 - torch.exp(-wz_sq / (sigma * sigma))
        bounded_total = translational_only + bounded_yaw

        yaw_s = stats(bounded_yaw)
        total_s = stats(bounded_total)

        active_weighted_mean = HYPOTHETICAL_WEIGHT * total_s["mean"]
        all_sample_weighted_mean = active_weighted_mean * active_fraction
        yaw_all_sample_weighted_mean = (
            HYPOTHETICAL_WEIGHT * yaw_s["mean"] * active_fraction
        )

        candidate_rows.append(
            (
                sigma,
                yaw_s,
                total_s,
                active_weighted_mean,
                all_sample_weighted_mean,
                yaw_all_sample_weighted_mean,
            )
        )

        print("-" * 190)
        print(f"sigma = {sigma:.2f} rad/s")
        print_stats("bounded yaw", yaw_s)
        print_stats("bounded total", total_s)
        print(
            f"active weighted mean       : {active_weighted_mean:.6f}"
        )
        print(
            f"all-sample weighted mean   : {all_sample_weighted_mean:.6f}"
        )
        print(
            f"yaw-only all-sample mean   : {yaw_all_sample_weighted_mean:.6f}"
        )

    print()
    print("=" * 190)
    print("COMPACT SUMMARY")
    print("=" * 190)
    print(
        " sigma | yaw_mean | yaw_p95 | yaw_p99 | yaw_max | "
        "total_mean | total_p95 | total_p99 | total_max | "
        "all_sample_weighted_mean"
    )
    for (
        sigma,
        yaw_s,
        total_s,
        _,
        all_sample_weighted_mean,
        _,
    ) in candidate_rows:
        print(
            f"{sigma:6.2f} | "
            f"{yaw_s['mean']:8.5f} | "
            f"{yaw_s['p95']:7.5f} | "
            f"{yaw_s['p99']:7.5f} | "
            f"{yaw_s['max']:7.5f} | "
            f"{total_s['mean']:10.5f} | "
            f"{total_s['p95']:9.5f} | "
            f"{total_s['p99']:9.5f} | "
            f"{total_s['max']:9.5f} | "
            f"{all_sample_weighted_mean:24.6f}"
        )

    print()
    print("No PPO update performed. No repository files modified.")
    print()
    print("BOUNDED_YAW_PENALTY_SIZING_PASS")
    print("=" * 190)

except BaseException as exc:
    print("", flush=True)
    print("=" * 120, flush=True)
    print("BOUNDED_YAW_PENALTY_SIZING_EXCEPTION", flush=True)
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
