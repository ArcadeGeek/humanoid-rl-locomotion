from __future__ import annotations

from pathlib import Path
from math import isfinite

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


RUN_B = Path(
    r"C:\humanoid_rl_lab\logs\rsl_rl\humanoid_rl_lab_g1_v4b_direct_yaw_scratch"
    r"\2026-09-01_20-08-18_v4b_genuine_scratch_seed42_official1500"
)
RUN_G = Path(
    r"C:\humanoid_rl_lab\logs\rsl_rl\humanoid_rl_lab_g1_v4g_zero_command_full_motion_scratch"
    r"\2026-09-02_23-59-15_v4g_genuine_scratch_seed42_official1500"
)

COMMON_TAGS = {
    "mean_reward": "Train/mean_reward",
    "episode_length": "Train/mean_episode_length",
    "action_std": "Policy/mean_std",
    "success_rate": "Metrics/success_rate",
    "base_contact": "Episode_Termination/base_contact",
    "time_out": "Episode_Termination/time_out",
    "value_loss": "Loss/value",
    "surrogate_loss": "Loss/surrogate",
    "entropy": "Loss/entropy",
    "error_xy": "Metrics/base_velocity/error_vel_xy",
    "error_yaw": "Metrics/base_velocity/error_vel_yaw",
}
G_ONLY_TAG = "Episode_Reward/zero_command_full_motion"

SAMPLE_ITERS = list(range(0, 201, 10))


def fail(msg: str) -> None:
    raise SystemExit(f"ABORT: {msg}")


def load_acc(run_dir: Path) -> tuple[EventAccumulator, Path]:
    event_files = sorted(run_dir.glob("events.out.tfevents*"))
    if len(event_files) != 1:
        fail(
            f"expected exactly one TensorBoard event file in {run_dir}, "
            f"found {len(event_files)}"
        )
    event_file = event_files[0]
    acc = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    acc.Reload()
    return acc, event_file


def scalar_map(acc: EventAccumulator, tag: str, label: str) -> dict[int, float]:
    tags = acc.Tags().get("scalars", [])
    if tag not in tags:
        print(f"[MISSING] {label}: {tag}")
        print("Available scalar tags:")
        for available in sorted(tags):
            print(f"  {available}")
        fail(f"required scalar tag missing for {label}: {tag}")

    events = acc.Scalars(tag)
    if not events:
        fail(f"no scalar events for {label}: {tag}")

    values: dict[int, float] = {}
    for event in events:
        step = int(event.step)
        value = float(event.value)
        if not isfinite(value):
            fail(f"non-finite value for {label} at iteration {step}: {value}")
        values[step] = value
    return values


def require_exact_samples(series: dict[int, float], label: str) -> None:
    missing = [iteration for iteration in SAMPLE_ITERS if iteration not in series]
    if missing:
        fail(f"{label} missing exact sampled iterations: {missing}")


acc_b, event_b = load_acc(RUN_B)
acc_g, event_g = load_acc(RUN_G)

series_b = {}
series_g = {}

for label, tag in COMMON_TAGS.items():
    series_b[label] = scalar_map(acc_b, tag, f"v4-B/{label}")
    series_g[label] = scalar_map(acc_g, tag, f"v4-G/{label}")
    require_exact_samples(series_b[label], f"v4-B/{label}")
    require_exact_samples(series_g[label], f"v4-G/{label}")

zero_g = scalar_map(acc_g, G_ONLY_TAG, "v4-G/zero_command_full_motion")
require_exact_samples(zero_g, "v4-G/zero_command_full_motion")

# v4-B must not contain the v4-G experimental reward tag.
tags_b = acc_b.Tags().get("scalars", [])
if G_ONLY_TAG in tags_b:
    fail("v4-B unexpectedly contains the v4-G zero_command_full_motion scalar tag")

print("=" * 170)
print("EARLY TRAINING COMPARISON: v4-B SUCCESSFUL BASELINE vs v4-G FULL-MOTION")
print("=" * 170)
print(f"v4-B run   : {RUN_B}")
print(f"v4-B event : {event_b.name}")
print(f"v4-G run   : {RUN_G}")
print(f"v4-G event : {event_g.name}")
print()
print("Exact sampled iterations: 0..200 every 10")
print("No interpolation / nearest-neighbor sampling.")
print()

print(
    f"{'iter':>5s} | "
    f"{'B_reward':>9s} {'G_reward':>9s} | "
    f"{'B_len':>7s} {'G_len':>7s} | "
    f"{'B_std':>7s} {'G_std':>7s} | "
    f"{'B_base':>7s} {'G_base':>7s} | "
    f"{'B_tout':>7s} {'G_tout':>7s} | "
    f"{'B_succ':>7s} {'G_succ':>7s} | "
    f"{'B_vloss':>8s} {'G_vloss':>8s} | "
    f"{'G_zero':>9s}"
)
print("-" * 170)

for iteration in SAMPLE_ITERS:
    print(
        f"{iteration:5d} | "
        f"{series_b['mean_reward'][iteration]:9.3f} "
        f"{series_g['mean_reward'][iteration]:9.3f} | "
        f"{series_b['episode_length'][iteration]:7.1f} "
        f"{series_g['episode_length'][iteration]:7.1f} | "
        f"{series_b['action_std'][iteration]:7.3f} "
        f"{series_g['action_std'][iteration]:7.3f} | "
        f"{series_b['base_contact'][iteration]:7.3f} "
        f"{series_g['base_contact'][iteration]:7.3f} | "
        f"{series_b['time_out'][iteration]:7.3f} "
        f"{series_g['time_out'][iteration]:7.3f} | "
        f"{series_b['success_rate'][iteration]:7.3f} "
        f"{series_g['success_rate'][iteration]:7.3f} | "
        f"{series_b['value_loss'][iteration]:8.3f} "
        f"{series_g['value_loss'][iteration]:8.3f} | "
        f"{zero_g[iteration]:9.5f}"
    )

print()
print("Velocity tracking diagnostics:")
print(
    f"{'iter':>5s} | "
    f"{'B_err_xy':>9s} {'G_err_xy':>9s} | "
    f"{'B_err_yaw':>10s} {'G_err_yaw':>10s} | "
    f"{'B_surr':>8s} {'G_surr':>8s} | "
    f"{'B_entropy':>9s} {'G_entropy':>9s}"
)
print("-" * 100)

for iteration in SAMPLE_ITERS:
    print(
        f"{iteration:5d} | "
        f"{series_b['error_xy'][iteration]:9.4f} "
        f"{series_g['error_xy'][iteration]:9.4f} | "
        f"{series_b['error_yaw'][iteration]:10.4f} "
        f"{series_g['error_yaw'][iteration]:10.4f} | "
        f"{series_b['surrogate_loss'][iteration]:8.4f} "
        f"{series_g['surrogate_loss'][iteration]:8.4f} | "
        f"{series_b['entropy'][iteration]:9.4f} "
        f"{series_g['entropy'][iteration]:9.4f}"
    )

# Diagnostic transition locator on every logged iteration, not just 10-step samples.
def first_iter_where(series: dict[int, float], predicate) -> int | None:
    for iteration in sorted(series):
        if predicate(series[iteration]):
            return iteration
    return None


def first_sustained_base_contact(series: dict[int, float], threshold: float, consecutive: int) -> int | None:
    steps = sorted(series)
    for i in range(0, len(steps) - consecutive + 1):
        window = steps[i : i + consecutive]
        if any(window[j + 1] != window[j] + 1 for j in range(len(window) - 1)):
            continue
        if all(series[step] >= threshold for step in window):
            return window[0]
    return None


b_first_090 = first_iter_where(series_b["base_contact"], lambda x: x >= 0.90)
g_first_090 = first_iter_where(series_g["base_contact"], lambda x: x >= 0.90)
b_sustained_090 = first_sustained_base_contact(series_b["base_contact"], 0.90, 5)
g_sustained_090 = first_sustained_base_contact(series_g["base_contact"], 0.90, 5)

print()
print("Mechanical transition diagnostics (not scientific acceptance thresholds):")
print(f"v4-B first base_contact >= 0.90              : {b_first_090}")
print(f"v4-G first base_contact >= 0.90              : {g_first_090}")
print(f"v4-B first 5 consecutive iters >= 0.90      : {b_sustained_090}")
print(f"v4-G first 5 consecutive iters >= 0.90      : {g_sustained_090}")

print()
print("No simulation launched. No checkpoint loaded. No repository files modified.")
print()
print("V4B_V4G_EARLY_TRAINING_COMPARISON_PASS")
print("=" * 170)
