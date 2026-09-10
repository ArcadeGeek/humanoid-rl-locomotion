from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


REPO = Path(r"C:\humanoid_rl_lab")
EVALUATOR = REPO / "scripts" / "rsl_rl" / "evaluate_g1.py"

TASK = "Humanoid-RL-Lab-G1-Flat-V4I-Delayed-Bounded-Zero-Command-Full-Motion-v0"
EXPERIMENT = "humanoid_rl_lab_g1_v4i_delayed_bounded_zero_command_full_motion_scratch"
RUN_SUFFIX = "v4i_early_recovery_seed42_400"
CHECKPOINT_NAME = "model_399.pt"

SEED = 42
WARMUP_S = 3.0
DURATION_S = 20.0
VX = 0.4
VY = 0.0
WZ = 0.0

OUT_DIR = REPO / "logs" / "evaluations" / "g1" / "v4i_model399_playeq_aligned_cmd040_seed42_20s_v1"


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def require_once(source: str, anchor: str, label: str) -> None:
    count = source.count(anchor)
    if count != 1:
        fail(f"{label}: expected anchor exactly once, found {count}")


def col(rows: list[dict[str, str]], name: str) -> np.ndarray:
    if not rows:
        fail("samples.csv contains no rows")
    if name not in rows[0]:
        fail(f"samples.csv missing required column {name!r}; columns={list(rows[0])}")
    return np.asarray([float(row[name]) for row in rows], dtype=np.float64)


def touchdown_count(contact: np.ndarray) -> int:
    mask = contact > 0.5
    if len(mask) < 2:
        return 0
    return int(np.count_nonzero((~mask[:-1]) & mask[1:]))


def path_length_xyz(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    delta = np.stack((np.diff(x), np.diff(y), np.diff(z)), axis=1)
    return float(np.sum(np.linalg.norm(delta, axis=1)))


if not EVALUATOR.is_file():
    fail(f"repository evaluator missing: {EVALUATOR}")

experiment_dir = REPO / "logs" / "rsl_rl" / EXPERIMENT
if not experiment_dir.is_dir():
    fail(f"experiment directory missing: {experiment_dir}")

matching_runs = sorted(
    p for p in experiment_dir.iterdir()
    if p.is_dir() and p.name.endswith(RUN_SUFFIX)
)
if len(matching_runs) != 1:
    fail(
        f"expected exactly one run ending {RUN_SUFFIX!r}, found {len(matching_runs)}: "
        f"{[p.name for p in matching_runs]}"
    )

run_dir = matching_runs[0]
checkpoint = run_dir / CHECKPOINT_NAME
if not checkpoint.is_file():
    fail(f"checkpoint missing: {checkpoint}")

if OUT_DIR.exists():
    fail(f"output directory already exists: {OUT_DIR}\nRefusing to overwrite prior evidence.")

checkpoint_sha = sha256(checkpoint)
evaluator_sha = sha256(EVALUATOR)
source = EVALUATOR.read_text(encoding="utf-8")

# Patch 1: Play-equivalent evaluation + aligned yaw reset.
play_anchor = "        checkpoint = retrieve_file_path(args_cli.checkpoint)"
require_once(source, play_anchor, "Play/aligned-reset insertion")

play_patch = r'''        # v4-I standing probe: Play-equivalent + aligned yaw reset.
        env_cfg.observations.policy.enable_corruption = False
        if not hasattr(env_cfg, "events"):
            raise RuntimeError("Standing probe requires env_cfg.events")
        env_cfg.events.base_external_force_torque = None
        env_cfg.events.push_robot = None

        reset_base_cfg = env_cfg.events.reset_base
        if reset_base_cfg is None:
            raise RuntimeError("Standing probe requires events.reset_base")

        reset_base_cfg.params = dict(reset_base_cfg.params)
        pose_range = dict(reset_base_cfg.params["pose_range"])
        velocity_range = dict(reset_base_cfg.params["velocity_range"])
        pose_range["yaw"] = (0.0, 0.0)
        velocity_range["yaw"] = (0.0, 0.0)
        reset_base_cfg.params["pose_range"] = pose_range
        reset_base_cfg.params["velocity_range"] = velocity_range

'''
source = source.replace(play_anchor, play_patch + play_anchor, 1)

# Patch 2: resolve ankle-roll body IDs explicitly, then sample XYZ and XY velocity.
# The repository evaluator intentionally discards the IDs returned by robot.find_bodies(...)
# for the two feet, so the probe resolves them again in the temporary evaluator only.
foot_resolution_anchor = '''            if resolved_foot_body_names != foot_body_names:
                raise RuntimeError(
                    f"Unexpected foot body resolution: {resolved_foot_body_names}"
                )
'''
require_once(source, foot_resolution_anchor, "foot-body resolution guard")

foot_resolution_patch = foot_resolution_anchor + r'''
            probe_foot_body_ids, probe_resolved_foot_body_names = robot.find_bodies(
                foot_body_names, preserve_order=True
            )
            if probe_resolved_foot_body_names != foot_body_names:
                raise RuntimeError(
                    "Standing probe foot-body resolution mismatch: "
                    f"{probe_resolved_foot_body_names}"
                )
            if len(probe_foot_body_ids) != 2:
                raise RuntimeError(
                    "Standing probe expected exactly two foot body IDs, got "
                    f"{probe_foot_body_ids}"
                )
'''
source = source.replace(foot_resolution_anchor, foot_resolution_patch, 1)

root_anchor = "                root_pos = _cpu_values(_torch(robot.data.root_pos_w)[0])"
require_once(source, root_anchor, "root sampling")
root_patch = root_anchor + r'''
                probe_foot_body_pos_w = np.asarray(
                    _cpu_values(_torch(robot.data.body_pos_w)[0, probe_foot_body_ids]),
                    dtype=np.float64,
                ).reshape(2, 3)
                probe_foot_body_lin_vel_w = np.asarray(
                    _cpu_values(_torch(robot.data.body_lin_vel_w)[0, probe_foot_body_ids]),
                    dtype=np.float64,
                ).reshape(2, 3)'''
source = source.replace(root_anchor, root_patch, 1)

# Patch 3: add explicit standing-analysis columns.
record_anchor = '                        "left_foot_normal_force_n": float(foot_normal_forces[0]),'
require_once(source, record_anchor, "CSV record")
record_patch = r'''                        "probe_root_z_world_m": float(root_pos[2]),
                        "probe_heading_world_rad": float(heading),
                        "probe_left_foot_x_world_m": float(probe_foot_body_pos_w[0, 0]),
                        "probe_left_foot_y_world_m": float(probe_foot_body_pos_w[0, 1]),
                        "probe_left_foot_z_world_m": float(probe_foot_body_pos_w[0, 2]),
                        "probe_right_foot_x_world_m": float(probe_foot_body_pos_w[1, 0]),
                        "probe_right_foot_y_world_m": float(probe_foot_body_pos_w[1, 1]),
                        "probe_right_foot_z_world_m": float(probe_foot_body_pos_w[1, 2]),
                        "probe_left_foot_xy_speed_mps": float(np.linalg.norm(probe_foot_body_lin_vel_w[0, :2])),
                        "probe_right_foot_xy_speed_mps": float(np.linalg.norm(probe_foot_body_lin_vel_w[1, :2])),
                        "probe_left_knee_rad": float(posture[2]),
                        "probe_right_knee_rad": float(posture[3]),
''' + record_anchor
source = source.replace(record_anchor, record_patch, 1)

# Validate fully instrumented evaluator before launching Isaac Sim.
compile(source, "evaluate_g1_v4i_standing_instrumented.py", "exec")

print("=" * 128)
print("G1 v4-I MODEL_399 CONTACT-AWARE GAIT PROBE — PLAY-EQUIVALENT / ALIGNED RESET")
print("=" * 128)
print("Task                    :", TASK)
print("Run                     :", run_dir)
print("Checkpoint              :", checkpoint)
print("Checkpoint SHA256       :", checkpoint_sha)
print("Evaluator SHA256        :", evaluator_sha)
print("Seed                    :", SEED)
print("Command                 :", [VX, VY, WZ])
print("Warmup / measurement    :", f"{WARMUP_S:.1f} s / {DURATION_S:.1f} s")
print("Observation corruption  : OFF")
print("External force / push   : OFF / OFF")
print("Reset yaw pose / vel    : 0 / 0")
print("Other reset randomness  : PRESERVED")
print("Policy                  : deterministic")
print("Output                  :", OUT_DIR)
print()

with tempfile.TemporaryDirectory(prefix="g1_v4i_standing_") as tmp:
    temp_dir = Path(tmp)

    sibling_py_files = sorted(EVALUATOR.parent.glob("*.py"))
    if not sibling_py_files:
        fail(f"no Python helper files found beside evaluator: {EVALUATOR.parent}")

    for helper in sibling_py_files:
        shutil.copy2(helper, temp_dir / helper.name)

    cli_args_copy = temp_dir / "cli_args.py"
    if not cli_args_copy.is_file():
        fail(
            "temporary evaluator context is incomplete: cli_args.py was not copied "
            f"from {EVALUATOR.parent}"
        )

    instrumented = temp_dir / "evaluate_g1_v4i_standing_instrumented.py"
    instrumented.write_text(source, encoding="utf-8", newline="\n")

    command = [
        sys.executable,
        str(instrumented),
        "--task", TASK,
        "--checkpoint", str(checkpoint),
        "--num_envs", "1",
        "--seed", str(SEED),
        "--warmup_s", str(WARMUP_S),
        "--duration_s", str(DURATION_S),
        "--print_every_s", "5",
        "--command_vx_mps", str(VX),
        "--command_vy_mps", str(VY),
        "--command_wz_radps", str(WZ),
        "--deterministic",
        "--headless",
        "--output_dir", str(OUT_DIR),
        "--label", "v4i_model399_playeq_aligned_cmd040_gait",
    ]

    proc = subprocess.Popen(
        command,
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
    rc = proc.wait()
    if rc != 0:
        fail(f"instrumented evaluator failed with exit code {rc}")

samples_path = OUT_DIR / "samples.csv"
if not samples_path.is_file():
    fail(f"samples.csv missing: {samples_path}")

with samples_path.open("r", encoding="utf-8-sig", newline="") as stream:
    rows = list(csv.DictReader(stream))
if not rows:
    fail("samples.csv is empty")

time_s = col(rows, "time_s")
cmd_vx = col(rows, "cmd_vx_mps")
cmd_vy = col(rows, "cmd_vy_mps")
cmd_wz = col(rows, "cmd_wz_radps")
base_vx = col(rows, "base_vx_mps")
base_vy = col(rows, "base_vy_mps")
base_wz = col(rows, "base_wz_radps")
world_x = col(rows, "world_x_m")
world_y = col(rows, "world_y_m")
left_contact = col(rows, "left_foot_contact") > 0.5
right_contact = col(rows, "right_foot_contact") > 0.5

root_z = col(rows, "probe_root_z_world_m")
heading = col(rows, "probe_heading_world_rad")
lfx = col(rows, "probe_left_foot_x_world_m")
lfy = col(rows, "probe_left_foot_y_world_m")
lfz = col(rows, "probe_left_foot_z_world_m")
rfx = col(rows, "probe_right_foot_x_world_m")
rfy = col(rows, "probe_right_foot_y_world_m")
rfz = col(rows, "probe_right_foot_z_world_m")
lf_xy_speed = col(rows, "probe_left_foot_xy_speed_mps")
rf_xy_speed = col(rows, "probe_right_foot_xy_speed_mps")
left_knee = col(rows, "probe_left_knee_rad")
right_knee = col(rows, "probe_right_knee_rad")

expected_command = np.asarray([VX, VY, WZ], dtype=np.float64)
command_matrix = np.column_stack((cmd_vx, cmd_vy, cmd_wz))
command_max_abs_error = float(np.max(np.abs(command_matrix - expected_command[None, :])))
if command_max_abs_error > 1.0e-9:
    fail(
        "fixed command was not preserved; "
        f"max abs command error={command_max_abs_error:.9e}"
    )

dt = float(np.median(np.diff(time_s))) if len(time_s) >= 2 else float("nan")
expected_samples = int(round(DURATION_S / 0.02))
full_window = len(rows) >= expected_samples
measured_span_s = float(time_s[-1] - time_s[0]) if len(time_s) >= 2 else 0.0

if not np.isfinite(dt) or abs(dt - 0.02) > 1.0e-6:
    fail(f"unexpected evaluator dt: {dt}")

def rising_edges(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if len(mask) < 2:
        return np.empty(0, dtype=np.int64)
    return np.flatnonzero((~mask[:-1]) & mask[1:]) + 1

def falling_edges(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if len(mask) < 2:
        return np.empty(0, dtype=np.int64)
    return np.flatnonzero(mask[:-1] & (~mask[1:])) + 1

def finite_mean(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")

def finite_median(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")

def pct_cv(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan")
    mean = float(np.mean(arr))
    return float(100.0 * np.std(arr) / mean) if mean > 0.0 else float("nan")

def path_length_2d(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    return float(np.sum(np.hypot(np.diff(x), np.diff(y))))

def path_length_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    d = np.column_stack((np.diff(x), np.diff(y), np.diff(z)))
    return float(np.sum(np.linalg.norm(d, axis=1)))

def same_foot_stride_metrics(
    td: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[list[float], list[float]]:
    periods: list[float] = []
    lengths: list[float] = []
    for a, b in zip(td[:-1], td[1:]):
        if b <= a:
            continue
        periods.append(float((b - a) * dt))
        lengths.append(float(math.hypot(x[b] - x[a], y[b] - y[a])))
    return periods, lengths

def swing_metrics(
    contact: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    other_z: np.ndarray,
) -> tuple[list[float], list[float], list[float]]:
    liftoffs = falling_edges(contact)
    touchdowns = rising_edges(contact)
    durations: list[float] = []
    apex_rel: list[float] = []
    travels: list[float] = []
    td_cursor = 0
    for lo in liftoffs:
        while td_cursor < len(touchdowns) and touchdowns[td_cursor] <= lo:
            td_cursor += 1
        if td_cursor >= len(touchdowns):
            break
        td = int(touchdowns[td_cursor])
        if td <= lo + 1:
            continue
        sl = slice(int(lo), td + 1)
        durations.append(float((td - lo) * dt))
        apex_rel.append(float(np.max(z[sl] - other_z[sl])))
        travels.append(path_length_3d(x[sl], y[sl], z[sl]))
        td_cursor += 1
    return durations, apex_rel, travels

left_td = rising_edges(left_contact)
right_td = rising_edges(right_contact)

left_periods, left_stride_lengths = same_foot_stride_metrics(left_td, lfx, lfy)
right_periods, right_stride_lengths = same_foot_stride_metrics(right_td, rfx, rfy)
all_periods = left_periods + right_periods
all_stride_lengths = left_stride_lengths + right_stride_lengths

left_swing_durations, left_apex_rel, left_swing_travel = swing_metrics(
    left_contact, lfx, lfy, lfz, rfz
)
right_swing_durations, right_apex_rel, right_swing_travel = swing_metrics(
    right_contact, rfx, rfy, rfz, lfz
)
all_swing_durations = left_swing_durations + right_swing_durations
all_apex_rel = left_apex_rel + right_apex_rel
all_swing_travel = left_swing_travel + right_swing_travel

# Touchdown separation and lateral width in the initial measured-heading frame.
initial_heading = float(heading[0])
c0 = math.cos(initial_heading)
s0 = math.sin(initial_heading)

td_separations: list[float] = []
td_widths: list[float] = []
for idx in np.concatenate((left_td, right_td)):
    idx = int(idx)
    dx_feet = float(lfx[idx] - rfx[idx])
    dy_feet = float(lfy[idx] - rfy[idx])
    td_separations.append(math.hypot(dx_feet, dy_feet))
    lateral_sep = -s0 * dx_feet + c0 * dy_feet
    td_widths.append(abs(lateral_sep))

double_support = left_contact & right_contact
single_support = left_contact ^ right_contact
flight = (~left_contact) & (~right_contact)

contact_slip_samples = np.concatenate(
    (
        lf_xy_speed[left_contact],
        rf_xy_speed[right_contact],
    )
)
mean_contact_slip = (
    float(np.mean(contact_slip_samples))
    if contact_slip_samples.size
    else float("nan")
)

mean_stride_period = finite_mean(all_periods)
median_stride_period = finite_median(all_periods)
cadence_spm = (
    120.0 / mean_stride_period
    if np.isfinite(mean_stride_period) and mean_stride_period > 0.0
    else float("nan")
)
stride_cv_pct = pct_cv(all_periods)

mean_stride_length = finite_mean(all_stride_lengths)
mean_swing_duration = finite_mean(all_swing_durations)
mean_swing_apex_rel = finite_mean(all_apex_rel)
mean_swing_travel = finite_mean(all_swing_travel)
mean_td_separation = finite_mean(td_separations)
mean_width = finite_mean(td_widths)

mean_vx = float(np.mean(base_vx))
mean_vy = float(np.mean(base_vy))
mean_wz = float(np.mean(base_wz))

dx_world = float(world_x[-1] - world_x[0])
dy_world = float(world_y[-1] - world_y[0])
net_displacement = float(math.hypot(dx_world, dy_world))
root_path = path_length_2d(world_x, world_y)
straightness = net_displacement / root_path if root_path > 1.0e-12 else float("nan")

forward_initial = c0 * dx_world + s0 * dy_world
cross_track_initial = -s0 * dx_world + c0 * dy_world

heading_wrapped = np.arctan2(np.sin(heading), np.cos(heading))
heading_deg = np.degrees(heading_wrapped)
heading_abs_deg = np.abs(heading_deg)

left_knee_stance = float(np.mean(left_knee[left_contact])) if np.any(left_contact) else float("nan")
left_knee_swing = float(np.mean(left_knee[~left_contact])) if np.any(~left_contact) else float("nan")
right_knee_stance = float(np.mean(right_knee[right_contact])) if np.any(right_contact) else float("nan")
right_knee_swing = float(np.mean(right_knee[~right_contact])) if np.any(~right_contact) else float("nan")

analysis = {
    "protocol": {
        "task": TASK,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "evaluator_sha256": evaluator_sha,
        "seed": SEED,
        "command": [VX, VY, WZ],
        "warmup_s": WARMUP_S,
        "duration_s": DURATION_S,
        "aligned_reset_yaw_pose": [0.0, 0.0],
        "aligned_reset_yaw_velocity": [0.0, 0.0],
        "other_reset_randomization_preserved": True,
        "observation_corruption": False,
        "external_force_event": False,
        "push_event": False,
        "deterministic_policy": True,
    },
    "samples": len(rows),
    "dt_s": dt,
    "measured_span_s": measured_span_s,
    "full_requested_window": full_window,
    "command_max_abs_error": command_max_abs_error,
    "body_velocity": {
        "mean_vx_mps": mean_vx,
        "mean_vy_mps": mean_vy,
        "mean_wz_radps": mean_wz,
    },
    "gait": {
        "left_touchdowns": int(len(left_td)),
        "right_touchdowns": int(len(right_td)),
        "median_stride_period_s": median_stride_period,
        "mean_stride_period_s": mean_stride_period,
        "cadence_spm": cadence_spm,
        "stride_period_cv_pct": stride_cv_pct,
        "mean_touchdown_separation_m": mean_td_separation,
        "mean_stride_length_m": mean_stride_length,
        "mean_swing_duration_s": mean_swing_duration,
        "mean_relative_swing_apex_m": mean_swing_apex_rel,
        "mean_swing_travel_m": mean_swing_travel,
        "mean_width_m": mean_width,
        "single_support_pct": float(100.0 * np.mean(single_support)),
        "double_support_pct": float(100.0 * np.mean(double_support)),
        "flight_pct": float(100.0 * np.mean(flight)),
        "mean_contact_slip_xy_mps": mean_contact_slip,
    },
    "knees_rad": {
        "left_stance": left_knee_stance,
        "left_swing": left_knee_swing,
        "right_stance": right_knee_stance,
        "right_swing": right_knee_swing,
    },
    "trajectory": {
        "world_dx_m": dx_world,
        "world_dy_m": dy_world,
        "net_displacement_m": net_displacement,
        "root_path_length_m": root_path,
        "straightness_net_over_path": straightness,
        "initial_heading_forward_m": forward_initial,
        "initial_heading_cross_track_m": cross_track_initial,
    },
    "heading": {
        "first_deg": float(heading_deg[0]),
        "final_error_deg": float(heading_deg[-1]),
        "mean_abs_deg": float(np.mean(heading_abs_deg)),
        "rms_deg": float(np.sqrt(np.mean(np.square(heading_deg)))),
        "p95_abs_deg": float(np.percentile(heading_abs_deg, 95.0)),
        "max_abs_deg": float(np.max(heading_abs_deg)),
    },
    "root_height": {
        "mean_m": float(np.mean(root_z)),
        "std_m": float(np.std(root_z)),
    },
}

analysis_path = OUT_DIR / "gait_analysis.json"
with analysis_path.open("w", encoding="utf-8") as stream:
    json.dump(analysis, stream, indent=2, sort_keys=True)
    stream.write("\n")

g = analysis["gait"]
k = analysis["knees_rad"]
tr = analysis["trajectory"]
h = analysis["heading"]
rh = analysis["root_height"]

print()
print("=" * 128)
print("V4-I MODEL_399 — CONTACT-AWARE GAIT RESULTS @ [0.40, 0, 0]")
print("=" * 128)
print(f"samples / dt                         : {len(rows)} / {dt:.5f} s")
print(f"measured span / requested            : {measured_span_s:.3f} / {DURATION_S:.3f} s")
print(f"full requested window                : {'YES' if full_window else 'NO'}")
print(f"command max abs error                : {command_max_abs_error:.3e}")
print()
print(f"mean body vx                         : {mean_vx:.6f} m/s")
print(f"mean body vy                         : {mean_vy:+.6f} m/s")
print(f"mean body wz                         : {mean_wz:+.6f} rad/s")
print()
print(f"median / mean stride period          : {g['median_stride_period_s']:.3f} / {g['mean_stride_period_s']:.3f} s")
print(f"cadence                              : {g['cadence_spm']:.1f} spm")
print(f"stride-period CV                     : {g['stride_period_cv_pct']:.2f}%")
print(f"touchdowns L / R                     : {g['left_touchdowns']} / {g['right_touchdowns']}")
print(f"mean touchdown separation            : {100.0*g['mean_touchdown_separation_m']:.2f} cm")
print(f"mean stride length                   : {100.0*g['mean_stride_length_m']:.2f} cm")
print(f"mean swing duration                  : {g['mean_swing_duration_s']:.3f} s")
print(f"mean relative swing apex             : {100.0*g['mean_relative_swing_apex_m']:.2f} cm")
print(f"mean swing travel                    : {100.0*g['mean_swing_travel_m']:.2f} cm")
print(f"mean gait width                      : {100.0*g['mean_width_m']:.2f} cm")
print(f"single / double / flight             : {g['single_support_pct']:.2f}% / {g['double_support_pct']:.2f}% / {g['flight_pct']:.2f}%")
print(f"mean contact slip XY                 : {g['mean_contact_slip_xy_mps']:.4f} m/s")
print()
print(f"knee L stance -> swing               : {k['left_stance']:.4f} -> {k['left_swing']:.4f} rad")
print(f"knee R stance -> swing               : {k['right_stance']:.4f} -> {k['right_swing']:.4f} rad")
print()
print(f"world displacement dx / dy           : {tr['world_dx_m']:+.4f} / {tr['world_dy_m']:+.4f} m")
print(f"initial-heading forward / cross-track: {tr['initial_heading_forward_m']:+.4f} / {tr['initial_heading_cross_track_m']:+.4f} m")
print(f"root horizontal path                 : {tr['root_path_length_m']:.4f} m")
print(f"straightness net/path                : {tr['straightness_net_over_path']:.6f}")
print()
print(f"heading first                        : {h['first_deg']:+.3f} deg")
print(f"heading final error                  : {h['final_error_deg']:+.3f} deg")
print(f"heading mean abs                     : {h['mean_abs_deg']:.3f} deg")
print(f"heading RMS                          : {h['rms_deg']:.3f} deg")
print(f"heading p95 abs                      : {h['p95_abs_deg']:.3f} deg")
print(f"heading max abs                      : {h['max_abs_deg']:.3f} deg")
print(f"root height mean +/- std             : {rh['mean_m']:.4f} +/- {rh['std_m']:.4f} m")
print()
print("Analysis JSON                        :", analysis_path)
print("Repository evaluator modified        : NO")
print("Checkpoint modified                  : NO")
print()

if not full_window:
    print("V4I_MODEL399_CONTACT_GAIT_PROBE_FAIL_INCOMPLETE_WINDOW")
    fail(
        f"policy did not complete the requested {DURATION_S:.1f}s measurement window; "
        f"samples={len(rows)}, expected={expected_samples}"
    )

if len(left_td) < 2 or len(right_td) < 2:
    print("V4I_MODEL399_CONTACT_GAIT_PROBE_FAIL_INSUFFICIENT_TOUCHDOWNS")
    fail(
        "insufficient touchdown cycles for gait characterization: "
        f"L={len(left_td)}, R={len(right_td)}"
    )

print("V4I_MODEL399_CONTACT_GAIT_PROBE_PASS")
print("=" * 128)
