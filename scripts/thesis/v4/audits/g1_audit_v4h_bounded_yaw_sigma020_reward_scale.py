from __future__ import annotations

import argparse
import sys
import traceback
from importlib import metadata
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description=(
        "Read-only reward-scale audit for the candidate bounded zero-command "
        "yaw penalty with sigma=0.20 rad/s, evaluated on healthy v4-B model_0."
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
SIGMA = 0.20
WEIGHT = -5.0

FOOT_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]


def fail(msg: str) -> None:
    raise RuntimeError(msg)


def as_tensor(value):
    if hasattr(value, "torch"):
        value = value.torch
    if not torch.is_tensor(value):
        fail(f"expected torch tensor, got {type(value)!r}")
    return value


def summarize(x: torch.Tensor) -> dict[str, float]:
    x = x.detach().float().reshape(-1).cpu()
    if x.numel() == 0:
        fail("cannot summarize empty tensor")
    if not torch.isfinite(x).all():
        fail("non-finite values in summary input")
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
        "min": float(x.min().item()),
    }


def print_summary(name: str, s: dict[str, float]) -> None:
    print(
        f"{name:32s}: "
        f"mean={s['mean']:12.6f}  "
        f"median={s['median']:12.6f}  "
        f"p95={s['p95']:12.6f}  "
        f"p99={s['p99']:12.6f}  "
        f"p99.9={s['p999']:12.6f}  "
        f"min={s['min']:12.6f}  "
        f"max={s['max']:12.6f}"
    )


env = None

try:
    if not REFERENCE_CHECKPOINT.is_file():
        fail(f"checkpoint not found: {REFERENCE_CHECKPOINT}")

    env_cfg = parse_env_cfg(
        TASK,
        device=args_cli.device,
        num_envs=NUM_ENVS,
    )
    env_cfg.seed = SEED

    agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    agent_cfg.device = args_cli.device

    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    term = env_cfg.rewards.zero_command_full_motion
    if term.weight != WEIGHT:
        fail(f"canonical v4-G weight mismatch: {term.weight!r}")
    if term.params["command_threshold"] != COMMAND_THRESHOLD:
        fail(
            "canonical command_threshold mismatch: "
            f"{term.params['command_threshold']!r}"
        )
    if term.params["base_planar_scale"] != BASE_PLANAR_SCALE:
        fail(
            "canonical base_planar_scale mismatch: "
            f"{term.params['base_planar_scale']!r}"
        )
    if term.params["foot_xy_scale"] != FOOT_XY_SCALE:
        fail(
            "canonical foot_xy_scale mismatch: "
            f"{term.params['foot_xy_scale']!r}"
        )
    if term.params["yaw_rate_scale"] != 1.0:
        fail(
            "canonical v4-G yaw_rate_scale mismatch: "
            f"{term.params['yaw_rate_scale']!r}"
        )
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

    raw_env = gym.make(TASK, cfg=env_cfg)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)

    unwrapped = env.unwrapped

    step_dt = getattr(unwrapped, "step_dt", None)
    if step_dt is None:
        fail("environment does not expose step_dt")
    step_dt = float(step_dt)
    if not (step_dt > 0.0):
        fail(f"invalid step_dt: {step_dt!r}")

    sim_dt = float(unwrapped.cfg.sim.dt)
    decimation = int(unwrapped.cfg.decimation)
    reconstructed_step_dt = sim_dt * decimation
    if abs(step_dt - reconstructed_step_dt) > 1.0e-12:
        fail(
            "step_dt mismatch: "
            f"runtime={step_dt}, sim_dt*decimation={reconstructed_step_dt}"
        )

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
    runner.load(str(REFERENCE_CHECKPOINT))

    if not hasattr(runner, "alg") or not hasattr(runner.alg, "get_policy"):
        fail("runner algorithm does not expose get_policy()")
    actor = runner.alg.get_policy()
    if not callable(actor):
        fail(f"runner policy is not callable: {type(actor)!r}")

    robot = unwrapped.scene["robot"]
    body_names = list(robot.data.body_names)
    missing = [name for name in FOOT_NAMES if name not in body_names]
    if missing:
        fail(f"foot bodies missing: {missing}")
    foot_ids = [body_names.index(name) for name in FOOT_NAMES]

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
    active_counts = []

    base_active = []
    foot_active = []
    wz_sq_active = []
    bounded_yaw_active = []

    original_all = []
    translational_all = []
    bounded_all = []

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
            wz_sq = torch.square(root_ang_vel_b[:, 2])

            foot_xy_sq = torch.sum(
                torch.square(body_lin_vel_w[:, foot_ids, :2]),
                dim=-1,
            )
            mean_foot_xy_sq = torch.mean(foot_xy_sq, dim=1)

            base = BASE_PLANAR_SCALE * base_planar_sq
            foot = FOOT_XY_SCALE * mean_foot_xy_sq
            bounded_yaw = 1.0 - torch.exp(-wz_sq / (SIGMA * SIGMA))

            active = (
                torch.linalg.norm(command[:, :3], dim=1)
                < COMMAND_THRESHOLD
            )
            active_f = active.to(dtype=base.dtype)

            original_raw = (base + foot + wz_sq) * active_f
            translational_raw = (base + foot) * active_f
            bounded_raw = (base + foot + bounded_yaw) * active_f

            for name, tensor in (
                ("base", base),
                ("foot", foot),
                ("wz_sq", wz_sq),
                ("bounded_yaw", bounded_yaw),
                ("original_raw", original_raw),
                ("translational_raw", translational_raw),
                ("bounded_raw", bounded_raw),
            ):
                if not torch.isfinite(tensor).all():
                    fail(f"non-finite {name} at step {step_idx}")

            active_masks.append(active.detach().cpu())
            active_counts.append(int(active.sum().item()))

            if bool(active.any().item()):
                base_active.append(base[active].detach().cpu())
                foot_active.append(foot[active].detach().cpu())
                wz_sq_active.append(wz_sq[active].detach().cpu())
                bounded_yaw_active.append(bounded_yaw[active].detach().cpu())

            original_all.append(original_raw.detach().cpu())
            translational_all.append(translational_raw.detach().cpu())
            bounded_all.append(bounded_raw.detach().cpu())

    pooled_mask = torch.cat(active_masks)
    active_samples = int(pooled_mask.sum().item())
    total_samples = int(pooled_mask.numel())
    if active_samples == 0:
        fail("no active zero-command samples")

    active_fraction = active_samples / total_samples

    base = torch.cat(base_active)
    foot = torch.cat(foot_active)
    wz_sq = torch.cat(wz_sq_active)
    bounded_yaw = torch.cat(bounded_yaw_active)

    original_raw_all = torch.cat(original_all)
    translational_raw_all = torch.cat(translational_all)
    bounded_raw_all = torch.cat(bounded_all)

    if not (
        original_raw_all.numel()
        == translational_raw_all.numel()
        == bounded_raw_all.numel()
        == total_samples
    ):
        fail("all-sample length mismatch")

    # Exact RewardManager semantics:
    # value = func(...) * weight * dt
    original_effective = original_raw_all * WEIGHT * step_dt
    translational_effective = translational_raw_all * WEIGHT * step_dt
    bounded_effective = bounded_raw_all * WEIGHT * step_dt

    # _step_reward semantics:
    # self._step_reward = value / dt = func(...) * weight
    original_logged = original_raw_all * WEIGHT
    translational_logged = translational_raw_all * WEIGHT
    bounded_logged = bounded_raw_all * WEIGHT

    print("=" * 188)
    print("SIGMA=0.20 BOUNDED YAW REWARD-SCALE AUDIT — v4-B model_0")
    print("=" * 188)
    print(f"task                       : {TASK}")
    print(f"reference checkpoint       : {REFERENCE_CHECKPOINT}")
    print(f"num envs                   : {NUM_ENVS}")
    print(f"rollout steps              : {ROLLOUT_STEPS}")
    print(f"total env-step samples     : {total_samples}")
    print(f"seed                       : {SEED}")
    print(f"sim dt                     : {sim_dt:.9f} s")
    print(f"decimation                 : {decimation}")
    print(f"runtime step_dt            : {step_dt:.9f} s")
    print(f"command threshold          : {COMMAND_THRESHOLD}")
    print(f"weight                     : {WEIGHT}")
    print(f"sigma                      : {SIGMA:.2f} rad/s")
    print(f"foot ids                   : {foot_ids}")
    print(
        f"active fraction            : {active_fraction:.6f} "
        f"({100.0 * active_fraction:.3f}%)"
    )
    print(
        "active count/step          : "
        f"min={min(active_counts)}  "
        f"mean={sum(active_counts)/len(active_counts):.2f}  "
        f"max={max(active_counts)}"
    )
    print()

    print("ACTIVE-SAMPLE RAW COMPONENTS")
    print_summary("base contribution", summarize(base))
    print_summary("foot contribution", summarize(foot))
    print_summary("old yaw contribution wz^2", summarize(wz_sq))
    print_summary("bounded yaw sigma=.20", summarize(bounded_yaw))
    print()

    print("ALL-SAMPLE PRE-dt VALUES  (inactive envs are zero)")
    print_summary("old v4-G raw", summarize(original_raw_all))
    print_summary("translational-only raw", summarize(translational_raw_all))
    print_summary("bounded candidate raw", summarize(bounded_raw_all))
    print()

    print("ALL-SAMPLE LOGGED TERM = func * weight  (_step_reward semantics)")
    print_summary("old v4-G logged", summarize(original_logged))
    print_summary("trans-only logged", summarize(translational_logged))
    print_summary("bounded logged", summarize(bounded_logged))
    print()

    print("ALL-SAMPLE EFFECTIVE PPO REWARD = func * weight * step_dt")
    old_eff = summarize(original_effective)
    trans_eff = summarize(translational_effective)
    bounded_eff = summarize(bounded_effective)
    print_summary("old v4-G effective", old_eff)
    print_summary("trans-only effective", trans_eff)
    print_summary("bounded effective", bounded_eff)
    print()

    # Active-only effective values are useful for understanding per-standing-env severity.
    old_active_effective = (base + foot + wz_sq) * WEIGHT * step_dt
    trans_active_effective = (base + foot) * WEIGHT * step_dt
    bounded_active_effective = (base + foot + bounded_yaw) * WEIGHT * step_dt

    print("ACTIVE-ONLY EFFECTIVE PPO REWARD")
    old_active_s = summarize(old_active_effective)
    trans_active_s = summarize(trans_active_effective)
    bounded_active_s = summarize(bounded_active_effective)
    print_summary("old v4-G active", old_active_s)
    print_summary("trans-only active", trans_active_s)
    print_summary("bounded active", bounded_active_s)
    print()

    print("KEY RATIOS")
    if old_eff["mean"] == 0.0:
        fail("old effective mean is zero; cannot compute ratio")
    if old_active_s["mean"] == 0.0:
        fail("old active effective mean is zero; cannot compute ratio")

    print(
        "bounded/old |all-sample mean| : "
        f"{abs(bounded_eff['mean']) / abs(old_eff['mean']):.6f}"
    )
    print(
        "trans/old   |all-sample mean| : "
        f"{abs(trans_eff['mean']) / abs(old_eff['mean']):.6f}"
    )
    print(
        "bounded/old |active mean|     : "
        f"{abs(bounded_active_s['mean']) / abs(old_active_s['mean']):.6f}"
    )
    print(
        "bounded minus trans active mean: "
        f"{bounded_active_s['mean'] - trans_active_s['mean']:.6f}"
    )
    print(
        "max bounded yaw-only effective penalty per active step: "
        f"{abs(WEIGHT) * step_dt:.6f}"
    )
    print()

    print("RewardManager formula audited:")
    print("  effective policy reward contribution = func * weight * step_dt")
    print("  logged _step_reward contribution     = func * weight")
    print()
    print("No PPO update performed. No repository files modified.")
    print()
    print("V4H_SIGMA020_REWARD_SCALE_AUDIT_PASS")
    print("=" * 188)

except BaseException as exc:
    print("", flush=True)
    print("=" * 120, flush=True)
    print("V4H_SIGMA020_REWARD_SCALE_AUDIT_EXCEPTION", flush=True)
    print(f"type    : {type(exc).__name__}", flush=True)
    print(f"message : {exc}", flush=True)
    print("traceback:", flush=True)
    traceback.print_exc(file=sys.stdout)
    print("=" * 120, flush=True)
    raise
finally:
    if env is not None:
        env.close()
    simulation_app.close()
