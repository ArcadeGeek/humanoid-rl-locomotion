from __future__ import annotations

from pathlib import Path
from math import isfinite

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


RUN_DIR = Path(
    r"C:\humanoid_rl_lab\logs\rsl_rl\humanoid_rl_lab_g1_v4g_zero_command_full_motion_scratch"
    r"\2026-09-02_23-59-15_v4g_genuine_scratch_seed42_official1500"
)

TARGETS = {
    "mean_reward": ("Train/mean_reward",),
    "mean_episode_length": ("Train/mean_episode_length",),
    "mean_action_std": ("Policy/mean_std",),
    "success_rate": ("Metrics/success_rate",),
    "base_contact": ("Episode_Termination/base_contact",),
    "time_out": ("Episode_Termination/time_out",),
    "zero_command_full_motion": ("Episode_Reward/zero_command_full_motion",),
}


def fail(msg: str) -> None:
    raise SystemExit(f"ABORT: {msg}")


def resolve_tag(all_tags: list[str], aliases: tuple[str, ...], label: str) -> str:
    for alias in aliases:
        if alias in all_tags:
            return alias

    lower_map = {tag.lower(): tag for tag in all_tags}
    for alias in aliases:
        found = lower_map.get(alias.lower())
        if found is not None:
            return found

    print(f"[MISSING] {label}")
    print("Available scalar tags:")
    for tag in all_tags:
        print(f"  {tag}")
    fail(f"unable to resolve TensorBoard tag for {label}")


def load_series(acc: EventAccumulator, tag: str) -> list[tuple[int, float]]:
    events = acc.Scalars(tag)
    if not events:
        fail(f"tag has no scalar events: {tag}")

    values = [(int(e.step), float(e.value)) for e in events]
    for step, value in values:
        if not isfinite(value):
            fail(f"non-finite scalar in {tag} at step {step}: {value}")
    return values


def nearest_value(series: list[tuple[int, float]], target_step: int) -> tuple[int, float]:
    return min(series, key=lambda item: abs(item[0] - target_step))


event_files = sorted(RUN_DIR.glob("events.out.tfevents*"))
if len(event_files) != 1:
    fail(
        f"expected exactly one TensorBoard event file, "
        f"found {len(event_files)} in {RUN_DIR}"
    )

event_file = event_files[0]
acc = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
acc.Reload()

scalar_tags = sorted(acc.Tags().get("scalars", []))
if not scalar_tags:
    fail("no scalar tags found in TensorBoard event file")

resolved = {
    label: resolve_tag(scalar_tags, aliases, label)
    for label, aliases in TARGETS.items()
}

series = {
    label: load_series(acc, tag)
    for label, tag in resolved.items()
}

# Confirm that every target really spans the canonical run.
last_steps = {label: values[-1][0] for label, values in series.items()}
min_final_step = min(last_steps.values())
if min_final_step < 1498:
    fail(
        "one or more scalar series are incomplete; "
        f"minimum final step is {min_final_step}"
    )

checkpoint_iters = list(range(0, 1500, 50))
checkpoint_iters.append(1499)

rows = []
for iteration in checkpoint_iters:
    row = {"iteration": iteration}
    nearest_steps = []

    for label, values in series.items():
        actual_step, value = nearest_value(values, iteration)
        row[label] = value
        nearest_steps.append(actual_step)

    max_distance = max(abs(step - iteration) for step in nearest_steps)
    if max_distance > 2:
        fail(
            f"no sufficiently close TensorBoard sample around iteration {iteration}; "
            f"nearest steps={sorted(set(nearest_steps))}"
        )

    rows.append(row)

print("=" * 146)
print("V4-G TENSORBOARD COLLAPSE ANALYSIS")
print("=" * 146)
print(f"run   : {RUN_DIR}")
print(f"event : {event_file.name}")
print()
print("Resolved scalar tags:")
for label, tag in resolved.items():
    print(f"  {label:24s} -> {tag}")

print()
print(
    f"{'iter':>5s} "
    f"{'mean_reward':>12s} "
    f"{'ep_len':>10s} "
    f"{'act_std':>9s} "
    f"{'success':>9s} "
    f"{'base_ct':>9s} "
    f"{'timeout':>9s} "
    f"{'zero_full':>11s}"
)
print("-" * 94)

for row in rows:
    print(
        f"{row['iteration']:5d} "
        f"{row['mean_reward']:12.4f} "
        f"{row['mean_episode_length']:10.2f} "
        f"{row['mean_action_std']:9.4f} "
        f"{row['success_rate']:9.4f} "
        f"{row['base_contact']:9.4f} "
        f"{row['time_out']:9.4f} "
        f"{row['zero_command_full_motion']:11.5f}"
    )

# Mechanical locator only: these thresholds are diagnostic, not acceptance criteria.
collapse_rows = [
    row
    for row in rows
    if (
        row["base_contact"] >= 0.90
        and row["time_out"] <= 0.10
        and row["mean_episode_length"] <= 100.0
    )
]

print()
if collapse_rows:
    first = collapse_rows[0]
    idx = rows.index(first)
    previous = rows[idx - 1] if idx > 0 else None

    print(
        "first unequivocally collapsed sampled checkpoint : "
        f"{first['iteration']}"
    )
    if previous is not None:
        print(
            "previous sampled checkpoint                       : "
            f"{previous['iteration']}"
        )
        print(
            "transition bracket                               : "
            f"[{previous['iteration']}, {first['iteration']}]"
        )
else:
    print(
        "first unequivocally collapsed sampled checkpoint : "
        "NOT FOUND by mechanical locator"
    )

print()
print(
    "NOTE: collapse locator thresholds are diagnostic only, "
    "not scientific acceptance criteria."
)
print("No simulation launched. No checkpoint loaded. No repository files modified.")
print()
print("V4G_TENSORBOARD_ANALYSIS_PASS")
print("=" * 146)
