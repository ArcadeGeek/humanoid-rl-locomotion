from __future__ import annotations

import argparse
import sys
import traceback
from importlib import metadata
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description=(
        "Read-only v4-B model_0 vs v4-G model_0 comparison using the same "
        "hypothetical zero-command full-motion cost."
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

V4B_CHECKPOINT = Path(
    r"C:\humanoid_rl_lab\logs\rsl_rl\humanoid_rl_lab_g1_v4b_direct_yaw_scratch"
    r"\2026-09-01_20-08-18_v4b_genuine_scratch_seed42_official1500\model_0.pt"
)
V4G_CHECKPOINT = Path(
    r"C:\humanoid_rl_lab\logs\rsl_rl\humanoid_rl_lab_g1_v4g_zero_command_full_motion_scratch"
    r"\2026-09-02_23-59-15_v4g_genuine_scratch_seed42_official1500\model_0.pt"
)

NUM_ENVS = 4096
ROLLOUT_STEPS = 24
SEED = 42

COMMAND_THRESHOLD = 0.06
BASE_PLANAR_SCALE = 1.0
YAW_RATE_SCALE = 1.0
FOOT_XY_SCALE = 0.25
HYPOTHETICAL_WEIGHT = -5.0

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


def tensor_stats(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().reshape(-1).cpu()
    if values.numel() == 0:
        fail("cannot summarize an empty tensor")
    if not torch.isfinite(values).all():
        fail("non-finite values in statistics input")

    q = torch.quantile(
        values,
        torch.tensor([0.50, 0.90, 0.95, 0.99, 0.999], dtype=values.dtype),
    )
    return {
        "n": float(values.numel()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "median": float(q[0].item()),
        "p90": float(q[1].item()),
        "p95": float(q[2].item()),
        "p99": float(q[3].item()),
        "p999": float(q[4].item()),
        "max": float(values.max().item()),
    }


def print_stats(name: str, stats: dict[str, float]) -> None:
    print(
        f"{name:26s}: "
        f"n={int(stats['n']):7d}  "
        f"mean={stats['mean']:12.6f}  "
        f"std={stats['std']:12.6f}  "
        f"median={stats['median']:12.6f}  "
        f"p90={stats['p90']:12.6f}  "
        f"p95={stats['p95']:12.6f}  "
        f"p99={stats['p99']:12.6f}  "
        f"p99.9={stats['p999']:12.6f}  "
        f"max={stats['max']:12.6f}"
    )


env = None

try:
    stage("ENTER_TRY")

    for label, path in (("v4-B", V4B_CHECKPOINT), ("v4-G", V4G_CHECKPOINT)):
        if not path.is_file():
            fail(f"{label} checkpoint not found: {path}")
    stage("CHECKPOINTS_EXIST")

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
        fail(f"v4-G reward weight mismatch: {term.weight!r}")

    exact_expected = {
        "command_threshold": COMMAND_THRESHOLD,
        "base_planar_scale": BASE_PLANAR_SCALE,
        "yaw_rate_scale": YAW_RATE_SCALE,
        "foot_xy_scale": FOOT_XY_SCALE,
    }
    for key, expected in exact_expected.items():
        actual = term.params[key]
        if actual != expected:
            fail(f"{key} mismatch: {actual!r} != {expected!r}")

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
        fail(f"post-migration actor config is not a dict: {type(actor_cfg)!r}")

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
        fail(f"foot bodies missing from robot: {missing}")
    foot_ids = [body_names.index(name) for name in FOOT_NAMES]
    stage("POLICY_AND_FEET_READY")

    def reset_same_seed() -> None:
        # Reproduce the same reset seed and the same stochastic-action RNG stream
        # before each checkpoint rollout. No state injection or replay is used.
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

        reset_result = raw_env.reset(seed=SEED)
        if not isinstance(reset_result, tuple) or len(reset_result) != 2:
            fail(
                "raw_env.reset(seed=SEED) did not return the expected "
                "(observations, info) tuple"
            )

        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

    def rollout(label: str, checkpoint: Path) -> dict:
        stage(f"{label}_LOAD_CHECKPOINT")
        runner.load(str(checkpoint))
        stage(f"{label}_CHECKPOINT_LOADED")

        # MLP policy: no recurrent hidden state. Use the exact installed PPO
        # stochastic forward path without calling process_env_step/update.
        reset_same_seed()
        obs = env.get_observations()

        active_masks = []
        base_values = []
        yaw_values = []
        abs_wz_values = []
        foot_values = []
        raw_values = []
        active_counts = []
        raw_max_per_step = []

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
                yaw_rate_sq = torch.square(root_ang_vel_b[:, 2])
                abs_wz = torch.abs(root_ang_vel_b[:, 2])

                foot_xy_sq = torch.sum(
                    torch.square(body_lin_vel_w[:, foot_ids, :2]),
                    dim=-1,
                )
                mean_foot_xy_sq = torch.mean(foot_xy_sq, dim=1)

                base_contrib = BASE_PLANAR_SCALE * base_planar_sq
                yaw_contrib = YAW_RATE_SCALE * yaw_rate_sq
                foot_contrib = FOOT_XY_SCALE * mean_foot_xy_sq
                raw_cost = base_contrib + yaw_contrib + foot_contrib

                full_command_norm = torch.linalg.norm(command[:, :3], dim=1)
                active = full_command_norm < COMMAND_THRESHOLD

                for name, tensor in (
                    ("base", base_contrib),
                    ("yaw", yaw_contrib),
                    ("abs_wz", abs_wz),
                    ("foot", foot_contrib),
                    ("raw", raw_cost),
                ):
                    if not torch.isfinite(tensor).all():
                        fail(f"{label}: non-finite {name} at step {step_idx}")

                active_masks.append(active.detach().cpu())
                active_counts.append(int(active.sum().item()))

                if bool(active.any().item()):
                    base_values.append(base_contrib[active].detach().cpu())
                    yaw_values.append(yaw_contrib[active].detach().cpu())
                    abs_wz_values.append(abs_wz[active].detach().cpu())
                    foot_values.append(foot_contrib[active].detach().cpu())
                    raw_values.append(raw_cost[active].detach().cpu())
                    raw_max_per_step.append(float(raw_cost[active].max().item()))
                else:
                    raw_max_per_step.append(0.0)

        pooled_mask = torch.cat(active_masks)
        active_samples = int(pooled_mask.sum().item())
        total_samples = int(pooled_mask.numel())
        if active_samples == 0:
            fail(f"{label}: no active zero-command samples")

        base = torch.cat(base_values)
        yaw = torch.cat(yaw_values)
        abs_wz = torch.cat(abs_wz_values)
        foot = torch.cat(foot_values)
        raw = torch.cat(raw_values)

        if not (
            base.numel()
            == yaw.numel()
            == abs_wz.numel()
            == foot.numel()
            == raw.numel()
            == active_samples
        ):
            fail(f"{label}: pooled active-sample length mismatch")

        reconstruction_error = torch.max(
            torch.abs(raw - (base + yaw + foot))
        ).item()
        if reconstruction_error > 1.0e-6:
            fail(
                f"{label}: raw reconstruction error too large: "
                f"{reconstruction_error}"
            )

        return {
            "label": label,
            "checkpoint": str(checkpoint),
            "active_masks": active_masks,
            "active_samples": active_samples,
            "total_samples": total_samples,
            "active_fraction": active_samples / total_samples,
            "active_counts": active_counts,
            "raw_max_per_step": raw_max_per_step,
            "base": tensor_stats(base),
            "yaw": tensor_stats(yaw),
            "abs_wz": tensor_stats(abs_wz),
            "foot": tensor_stats(foot),
            "raw": tensor_stats(raw),
            "reconstruction_error": reconstruction_error,
        }

    stage("ABOUT_TO_ROLLOUT_V4B")
    result_b = rollout("V4B", V4B_CHECKPOINT)
    stage("V4B_ROLLOUT_COMPLETE")

    stage("ABOUT_TO_ROLLOUT_V4G")
    result_g = rollout("V4G", V4G_CHECKPOINT)
    stage("V4G_ROLLOUT_COMPLETE")

    mask_mismatches = []
    for b_mask, g_mask in zip(
        result_b["active_masks"],
        result_g["active_masks"],
        strict=True,
    ):
        mask_mismatches.append(int(torch.count_nonzero(b_mask != g_mask).item()))

    print("=" * 180)
    print("V4-B model_0 vs V4-G model_0 — SAME HYPOTHETICAL ZERO-MOTION COST")
    print("=" * 180)
    print(f"environment task          : {TASK}")
    print(f"num envs                  : {NUM_ENVS}")
    print(f"rollout steps             : {ROLLOUT_STEPS}")
    print(f"seed                      : {SEED}")
    print("policy action path        : actor(obs, stochastic_output=True)")
    print(
        "hypothetical raw cost     : "
        "1.0*(vx^2+vy^2) + 1.0*wz^2 + 0.25*mean(foot_vx^2+foot_vy^2)"
    )
    print(f"hypothetical weight       : {HYPOTHETICAL_WEIGHT}")
    print(f"command gate              : ||cmd[:3]|| < {COMMAND_THRESHOLD}")
    print(f"foot ids                  : {foot_ids}")
    print()
    print(f"V4B checkpoint            : {result_b['checkpoint']}")
    print(f"V4G checkpoint            : {result_g['checkpoint']}")
    print()
    print(
        "active-mask mismatch/step: "
        f"min={min(mask_mismatches)}  "
        f"mean={sum(mask_mismatches)/len(mask_mismatches):.2f}  "
        f"max={max(mask_mismatches)}"
    )
    print()

    for result in (result_b, result_g):
        print("-" * 180)
        print(result["label"])
        print("-" * 180)
        print(
            f"active fraction           : {result['active_fraction']:.6f} "
            f"({100.0 * result['active_fraction']:.3f}%)"
        )
        print(
            "active count/step         : "
            f"min={min(result['active_counts'])}  "
            f"mean={sum(result['active_counts'])/len(result['active_counts']):.2f}  "
            f"max={max(result['active_counts'])}"
        )
        print_stats("base contribution", result["base"])
        print_stats("yaw contribution (wz^2)", result["yaw"])
        print_stats("abs yaw rate |wz|", result["abs_wz"])
        print_stats("foot contribution", result["foot"])
        print_stats("raw cost total", result["raw"])
        print(
            "hypothetical weighted mean: "
            f"{HYPOTHETICAL_WEIGHT * result['raw']['mean']:.6f}"
        )
        print(
            "raw max per step          : "
            f"min={min(result['raw_max_per_step']):.6f}  "
            f"mean={sum(result['raw_max_per_step'])/len(result['raw_max_per_step']):.6f}  "
            f"max={max(result['raw_max_per_step']):.6f}"
        )
        print(
            f"reconstruction max error  : {result['reconstruction_error']:.3e}"
        )
        print()

    def safe_ratio(g: float, b: float) -> float:
        return float("inf") if b == 0.0 else g / b

    print("=" * 180)
    print("G / B COMPARISON")
    print("=" * 180)
    print(
        f"base mean ratio G/B       : "
        f"{safe_ratio(result_g['base']['mean'], result_b['base']['mean']):.6f}"
    )
    print(
        f"yaw mean ratio G/B        : "
        f"{safe_ratio(result_g['yaw']['mean'], result_b['yaw']['mean']):.6f}"
    )
    print(
        f"yaw p95 ratio G/B         : "
        f"{safe_ratio(result_g['yaw']['p95'], result_b['yaw']['p95']):.6f}"
    )
    print(
        f"yaw p99 ratio G/B         : "
        f"{safe_ratio(result_g['yaw']['p99'], result_b['yaw']['p99']):.6f}"
    )
    print(
        f"|wz| mean ratio G/B       : "
        f"{safe_ratio(result_g['abs_wz']['mean'], result_b['abs_wz']['mean']):.6f}"
    )
    print(
        f"foot mean ratio G/B       : "
        f"{safe_ratio(result_g['foot']['mean'], result_b['foot']['mean']):.6f}"
    )
    print(
        f"raw mean ratio G/B        : "
        f"{safe_ratio(result_g['raw']['mean'], result_b['raw']['mean']):.6f}"
    )
    print()
    print("No PPO update performed. No repository files modified.")
    print()
    print("V4B_V4G_MODEL0_ZERO_MOTION_COMPARISON_PASS")
    print("=" * 180)

except BaseException as exc:
    print("", flush=True)
    print("=" * 120, flush=True)
    print("V4B_V4G_MODEL0_ZERO_MOTION_COMPARISON_EXCEPTION", flush=True)
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
