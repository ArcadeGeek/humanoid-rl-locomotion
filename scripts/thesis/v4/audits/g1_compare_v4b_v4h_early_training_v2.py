from __future__ import annotations

import math
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator


B_RUN = Path(
    r"C:\humanoid_rl_lab\logs\rsl_rl"
    r"\humanoid_rl_lab_g1_v4b_direct_yaw_scratch"
    r"\2026-09-01_20-08-18_v4b_genuine_scratch_seed42_official1500"
)

H_EXPERIMENT_DIR = Path(
    r"C:\humanoid_rl_lab\logs\rsl_rl"
    r"\humanoid_rl_lab_g1_v4h_bounded_zero_command_full_motion_scratch"
)

H_RUN_NAME = "v4h_early_recovery_seed42_200"

ITER_MAX = 199
SAMPLE_STEPS = list(range(0, 200, 10))
if SAMPLE_STEPS[-1] != ITER_MAX:
    SAMPLE_STEPS.append(ITER_MAX)


TAG_CANDIDATES = {
    "value_loss": (
        "Loss/value",
        "Loss/value_function",
        "Loss/value_function_loss",
        "Train/value_function_loss",
    ),
    "surrogate_loss": (
        "Loss/surrogate",
        "Loss/surrogate_loss",
        "Train/surrogate_loss",
    ),
    "entropy": (
        "Loss/entropy",
        "Loss/entropy_loss",
        "Train/entropy_loss",
    ),
    "action_std": (
        "Policy/mean_std",
        "Train/mean_action_noise_std",
    ),
    "reward": (
        "Train/mean_reward",
    ),
    "ep_len": (
        "Train/mean_episode_length",
    ),
    "success": (
        "Metrics/success_rate",
    ),
    "xy_err": (
        "Metrics/base_velocity/error_vel_xy",
    ),
    "yaw_err": (
        "Metrics/base_velocity/error_vel_yaw",
    ),
    "base_contact": (
        "Episode_Termination/base_contact",
    ),
    "timeout": (
        "Episode_Termination/time_out",
    ),
}

H_REWARD_CANDIDATES = (
    "Episode_Reward/zero_command_bounded_full_motion",
)

B_OPTIONAL_REWARD_CANDIDATES = (
    "Episode_Reward/zero_command_bounded_full_motion",
    "Episode_Reward/zero_command_full_motion",
    "Episode_Reward/zero_command_motion",
)


def fail(msg: str) -> None:
    raise RuntimeError(msg)


def discover_h_run() -> Path:
    if not H_EXPERIMENT_DIR.is_dir():
        fail(f"v4-H experiment directory not found: {H_EXPERIMENT_DIR}")

    matches = []
    for run_dir in H_EXPERIMENT_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        if H_RUN_NAME in run_dir.name:
            matches.append(run_dir)
            continue

        # RSL-RL commonly stores run_name in the timestamped folder suffix.
        # If not, accept exactly one folder containing an event file and a model_199.pt
        # only as a fallback when its name is the requested pilot's timestamped run.
        event_files = list(run_dir.glob("events.out.tfevents.*"))
        if event_files and (run_dir / "model_199.pt").is_file():
            # This fallback is intentionally collected but resolved below only if unique.
            matches.append(run_dir)

    # Prefer an explicit name match if available.
    explicit = [p for p in matches if H_RUN_NAME in p.name]
    if len(explicit) == 1:
        return explicit[0]
    if len(explicit) > 1:
        fail(f"multiple explicit v4-H pilot runs found: {explicit}")

    # Otherwise require exactly one model_199/event candidate.
    unique = sorted(set(matches))
    if len(unique) != 1:
        fail(
            "unable to identify exactly one v4-H early-recovery run. "
            f"candidates={unique}"
        )
    return unique[0]


def find_event_file(run_dir: Path) -> Path:
    files = sorted(
        run_dir.glob("events.out.tfevents.*"),
        key=lambda p: p.stat().st_mtime,
    )
    if not files:
        fail(f"no TensorBoard event file found in: {run_dir}")
    # RSL-RL can create more than one event file after resume/reopen. Read the newest.
    return files[-1]


def load_scalars(event_file: Path):
    ea = event_accumulator.EventAccumulator(
        str(event_file),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    return ea, tags


def resolve_tag(tags: list[str], candidates: tuple[str, ...], label: str, required: bool = True):
    for candidate in candidates:
        if candidate in tags:
            return candidate

    # Conservative suffix fallback for small version-dependent naming changes.
    normalized = {}
    for tag in tags:
        normalized[tag.lower().replace("-", "_")] = tag

    for candidate in candidates:
        c = candidate.lower().replace("-", "_")
        c_suffix = c.split("/")[-1]
        hits = [
            original
            for norm, original in normalized.items()
            if norm.endswith("/" + c_suffix) or norm == c_suffix
        ]
        if len(hits) == 1:
            return hits[0]

    if required:
        fail(
            f"required scalar tag not found for {label}. "
            f"candidates={candidates}; available={tags}"
        )
    return None


def scalar_series(ea, tag: str) -> dict[int, float]:
    values = {}
    for ev in ea.Scalars(tag):
        step = int(ev.step)
        values[step] = float(ev.value)
    return values


def exact_value(series: dict[int, float], step: int, label: str) -> float:
    if step not in series:
        fail(f"missing exact iteration {step} for {label}")
    return series[step]


def first_step_where(series: dict[int, float], predicate, start: int = 0, end: int = ITER_MAX):
    for step in range(start, end + 1):
        if step in series and predicate(series[step]):
            return step, series[step]
    return None


def first_consecutive_where(
    series: dict[int, float],
    predicate,
    count: int,
    start: int = 0,
    end: int = ITER_MAX,
):
    run = 0
    run_start = None
    for step in range(start, end + 1):
        if step not in series:
            run = 0
            run_start = None
            continue
        if predicate(series[step]):
            if run == 0:
                run_start = step
            run += 1
            if run >= count:
                return run_start, step
        else:
            run = 0
            run_start = None
    return None


def fmt(v: float, digits: int = 4) -> str:
    return f"{v:.{digits}f}"


def pct_delta(h: float, b: float):
    if b == 0.0:
        return math.nan
    return 100.0 * (h - b) / abs(b)


H_RUN = discover_h_run()
B_EVENT = find_event_file(B_RUN)
H_EVENT = find_event_file(H_RUN)

b_ea, b_tags = load_scalars(B_EVENT)
h_ea, h_tags = load_scalars(H_EVENT)

resolved = {"B": {}, "H": {}}
series = {"B": {}, "H": {}}

for key, candidates in TAG_CANDIDATES.items():
    b_tag = resolve_tag(b_tags, candidates, f"B:{key}", required=True)
    h_tag = resolve_tag(h_tags, candidates, f"H:{key}", required=True)
    resolved["B"][key] = b_tag
    resolved["H"][key] = h_tag
    series["B"][key] = scalar_series(b_ea, b_tag)
    series["H"][key] = scalar_series(h_ea, h_tag)

h_reward_tag = resolve_tag(
    h_tags,
    H_REWARD_CANDIDATES,
    "H:bounded_zero_reward",
    required=True,
)
h_zero_reward = scalar_series(h_ea, h_reward_tag)

b_optional_reward_tag = resolve_tag(
    b_tags,
    B_OPTIONAL_REWARD_CANDIDATES,
    "B:optional_zero_reward",
    required=False,
)
b_optional_reward = (
    scalar_series(b_ea, b_optional_reward_tag)
    if b_optional_reward_tag is not None
    else None
)

# Ensure the compared horizon is complete for all required tags.
for run_label in ("B", "H"):
    for key in TAG_CANDIDATES:
        s = series[run_label][key]
        missing = [step for step in range(0, ITER_MAX + 1) if step not in s]
        if missing:
            fail(
                f"{run_label}:{key} missing {len(missing)} iterations "
                f"in 0..{ITER_MAX}; first missing={missing[:10]}"
            )

missing_h_reward = [
    step for step in range(0, ITER_MAX + 1)
    if step not in h_zero_reward
]
if missing_h_reward:
    fail(
        "H bounded reward missing iterations in 0..199; "
        f"first missing={missing_h_reward[:10]}"
    )

# Mechanical collapse/recovery diagnostics.
diagnostics = {}
for run_label in ("B", "H"):
    bc = series[run_label]["base_contact"]
    to = series[run_label]["timeout"]
    ep = series[run_label]["ep_len"]

    diagnostics[run_label] = {
        "first_bc_ge_090": first_step_where(bc, lambda x: x >= 0.90),
        "first_5_bc_ge_090": first_consecutive_where(bc, lambda x: x >= 0.90, 5),
        "first_bc_lt_099_after10": first_step_where(
            bc, lambda x: x < 0.99, start=10
        ),
        "first_5_bc_lt_090_after10": first_consecutive_where(
            bc, lambda x: x < 0.90, 5, start=10
        ),
        "first_timeout_gt_001_after10": first_step_where(
            to, lambda x: x > 0.01, start=10
        ),
        "first_timeout_gt_050_after10": first_step_where(
            to, lambda x: x > 0.50, start=10
        ),
        "first_ep_len_gt_100_after10": first_step_where(
            ep, lambda x: x > 100.0, start=10
        ),
        "first_ep_len_gt_500_after10": first_step_where(
            ep, lambda x: x > 500.0, start=10
        ),
    }

# Locate earliest sustained divergence in base-contact recovery.
bc_gap_first = None
bc_gap_5 = None
gap_series = {}
for step in range(0, ITER_MAX + 1):
    gap = series["H"]["base_contact"][step] - series["B"]["base_contact"][step]
    gap_series[step] = gap

bc_gap_first = first_step_where(gap_series, lambda x: x >= 0.10, start=10)
bc_gap_5 = first_consecutive_where(gap_series, lambda x: x >= 0.10, 5, start=10)

print("=" * 220)
print("V4-B vs V4-H EARLY TRAINING COMPARISON — ITERATIONS 0..199")
print("=" * 220)
print(f"v4-B run                  : {B_RUN}")
print(f"v4-B event                : {B_EVENT}")
print(f"v4-H run                  : {H_RUN}")
print(f"v4-H event                : {H_EVENT}")
print(f"v4-H bounded reward tag   : {h_reward_tag}")
print()
print("Resolved required tags:")
for key in TAG_CANDIDATES:
    print(
        f"  {key:16s} B={resolved['B'][key]} | H={resolved['H'][key]}"
    )
print()

header = (
    "iter | "
    "B_len H_len | "
    "B_bc H_bc | "
    "B_to H_to | "
    "B_std H_std | "
    "B_val H_val | "
    "B_sur H_sur | "
    "B_ent H_ent | "
    "B_xy H_xy | "
    "B_yaw H_yaw | "
    "B_succ H_succ | "
    "H_zero"
)
print(header)
print("-" * len(header))

for step in SAMPLE_STEPS:
    b = {k: exact_value(series["B"][k], step, f"B:{k}") for k in TAG_CANDIDATES}
    h = {k: exact_value(series["H"][k], step, f"H:{k}") for k in TAG_CANDIDATES}
    hz = exact_value(h_zero_reward, step, "H:bounded_zero_reward")

    print(
        f"{step:4d} | "
        f"{b['ep_len']:5.1f} {h['ep_len']:5.1f} | "
        f"{b['base_contact']:4.3f} {h['base_contact']:4.3f} | "
        f"{b['timeout']:4.3f} {h['timeout']:4.3f} | "
        f"{b['action_std']:5.3f} {h['action_std']:5.3f} | "
        f"{b['value_loss']:6.3f} {h['value_loss']:6.3f} | "
        f"{b['surrogate_loss']:6.3f} {h['surrogate_loss']:6.3f} | "
        f"{b['entropy']:6.2f} {h['entropy']:6.2f} | "
        f"{b['xy_err']:5.3f} {h['xy_err']:5.3f} | "
        f"{b['yaw_err']:5.3f} {h['yaw_err']:5.3f} | "
        f"{b['success']:5.3f} {h['success']:5.3f} | "
        f"{hz: .5f}"
    )

print()
print("=" * 220)
print("RECOVERY DIAGNOSTICS")
print("=" * 220)

def print_diag(name: str, key: str):
    print(
        f"{name:38s}: "
        f"B={diagnostics['B'][key]} | H={diagnostics['H'][key]}"
    )

print_diag("first base_contact >= 0.90", "first_bc_ge_090")
print_diag("first 5 consecutive base_contact >= 0.90", "first_5_bc_ge_090")
print_diag("first base_contact < 0.99 after iter 10", "first_bc_lt_099_after10")
print_diag("first 5 consecutive base_contact < 0.90", "first_5_bc_lt_090_after10")
print_diag("first timeout > 0.01 after iter 10", "first_timeout_gt_001_after10")
print_diag("first timeout > 0.50 after iter 10", "first_timeout_gt_050_after10")
print_diag("first episode length > 100", "first_ep_len_gt_100_after10")
print_diag("first episode length > 500", "first_ep_len_gt_500_after10")
print(f"first H-B base_contact gap >= 0.10 : {bc_gap_first}")
print(f"first 5-consecutive H-B gap >= .10: {bc_gap_5}")

print()
print("=" * 220)
print("SELECTED DELTAS H - B")
print("=" * 220)
for step in (0, 50, 100, 120, 130, 140, 150, 160, 170, 180, 190, 199):
    vals = {}
    for key in (
        "ep_len",
        "base_contact",
        "timeout",
        "action_std",
        "value_loss",
        "surrogate_loss",
        "entropy",
        "xy_err",
        "yaw_err",
        "success",
    ):
        vals[key] = series["H"][key][step] - series["B"][key][step]

    print(
        f"{step:4d}: "
        f"d_len={vals['ep_len']:+8.2f}  "
        f"d_bc={vals['base_contact']:+7.4f}  "
        f"d_to={vals['timeout']:+7.4f}  "
        f"d_std={vals['action_std']:+7.4f}  "
        f"d_val={vals['value_loss']:+8.4f}  "
        f"d_sur={vals['surrogate_loss']:+8.4f}  "
        f"d_ent={vals['entropy']:+8.3f}  "
        f"d_xy={vals['xy_err']:+7.4f}  "
        f"d_yaw={vals['yaw_err']:+7.4f}  "
        f"d_succ={vals['success']:+7.4f}"
    )

print()
print("=" * 220)
print("SUMMARY AT ITERATION 199")
print("=" * 220)
for key in (
    "ep_len",
    "base_contact",
    "timeout",
    "action_std",
    "value_loss",
    "surrogate_loss",
    "entropy",
    "reward",
    "xy_err",
    "yaw_err",
    "success",
):
    b = series["B"][key][ITER_MAX]
    h = series["H"][key][ITER_MAX]
    pd = pct_delta(h, b)
    pd_text = "nan" if math.isnan(pd) else f"{pd:+.2f}%"
    print(
        f"{key:16s}: B={b:12.6f}  H={h:12.6f}  "
        f"H-B={h-b:+12.6f}  pct={pd_text}"
    )

print(
    f"H bounded zero reward @199: {h_zero_reward[ITER_MAX]:.6f}"
)
if b_optional_reward_tag is not None:
    print(
        f"B optional zero reward tag : {b_optional_reward_tag}"
    )

print()
print("No simulation started. No training update performed. Repository files were not modified.")
print()
print("V4B_V4H_EARLY_TRAINING_COMPARISON_V2_PASS")
print("=" * 220)
