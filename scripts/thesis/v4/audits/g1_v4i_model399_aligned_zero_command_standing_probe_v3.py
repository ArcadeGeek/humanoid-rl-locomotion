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
VX = 0.0
VY = 0.0
WZ = 0.0

OUT_DIR = REPO / "logs" / "evaluations" / "g1" / "v4i_model399_aligned_zero_command_seed42_20s_v3"


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
''' + record_anchor
source = source.replace(record_anchor, record_patch, 1)

# Validate fully instrumented evaluator before launching Isaac Sim.
compile(source, "evaluate_g1_v4i_standing_instrumented.py", "exec")

print("=" * 128)
print("G1 v4-I MODEL_399 ALIGNED ZERO-COMMAND STANDING PROBE")
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
        "--label", "v4i_model399_aligned_zero_command_standing",
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
left_contact = col(rows, "left_foot_contact")
right_contact = col(rows, "right_foot_contact")

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

command_max_abs = float(max(
    np.max(np.abs(cmd_vx)),
    np.max(np.abs(cmd_vy)),
    np.max(np.abs(cmd_wz)),
))
if command_max_abs > 1.0e-9:
    fail(f"fixed zero command was not preserved; max abs command={command_max_abs}")

dt = float(np.median(np.diff(time_s))) if len(time_s) >= 2 else float("nan")
planar_speed = np.hypot(base_vx, base_vy)
abs_wz = np.abs(base_wz)

heading_wrapped = np.arctan2(np.sin(heading), np.cos(heading))
heading_deg = np.degrees(heading_wrapped)
heading_abs_deg = np.abs(heading_deg)

double_support = (left_contact > 0.5) & (right_contact > 0.5)
single_support = (left_contact > 0.5) ^ (right_contact > 0.5)
flight = (left_contact <= 0.5) & (right_contact <= 0.5)

dx = float(world_x[-1] - world_x[0])
dy = float(world_y[-1] - world_y[0])
net_drift = float(math.hypot(dx, dy))

expected_samples = int(round(DURATION_S / 0.02))
measured_span_s = float(time_s[-1] - time_s[0]) if len(time_s) >= 2 else 0.0
full_window = len(rows) >= expected_samples

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
    "command_max_abs": command_max_abs,
    "planar": {
        "mean_mps": float(np.mean(planar_speed)),
        "rms_mps": float(np.sqrt(np.mean(np.square(planar_speed)))),
        "p95_mps": float(np.percentile(planar_speed, 95.0)),
        "max_mps": float(np.max(planar_speed)),
        "mean_abs_vx_mps": float(np.mean(np.abs(base_vx))),
        "mean_abs_vy_mps": float(np.mean(np.abs(base_vy))),
    },
    "yaw_rate": {
        "mean_radps": float(np.mean(base_wz)),
        "mean_abs_radps": float(np.mean(abs_wz)),
        "rms_radps": float(np.sqrt(np.mean(np.square(base_wz)))),
        "p95_abs_radps": float(np.percentile(abs_wz, 95.0)),
        "max_abs_radps": float(np.max(abs_wz)),
    },
    "heading": {
        "first_deg": float(heading_deg[0]),
        "final_error_deg": float(heading_deg[-1]),
        "mean_abs_deg": float(np.mean(heading_abs_deg)),
        "rms_deg": float(np.sqrt(np.mean(np.square(heading_deg)))),
        "p95_abs_deg": float(np.percentile(heading_abs_deg, 95.0)),
        "max_abs_deg": float(np.max(heading_abs_deg)),
    },
    "drift": {"dx_m": dx, "dy_m": dy, "net_m": net_drift},
    "root_height": {
        "mean_m": float(np.mean(root_z)),
        "std_m": float(np.std(root_z)),
        "min_m": float(np.min(root_z)),
        "max_m": float(np.max(root_z)),
    },
    "support": {
        "double_pct": float(100.0 * np.mean(double_support)),
        "single_pct": float(100.0 * np.mean(single_support)),
        "flight_pct": float(100.0 * np.mean(flight)),
        "left_touchdowns": touchdown_count(left_contact),
        "right_touchdowns": touchdown_count(right_contact),
    },
    "foot_motion": {
        "left_path_m": path_length_xyz(lfx, lfy, lfz),
        "right_path_m": path_length_xyz(rfx, rfy, rfz),
        "left_mean_xy_speed_mps": float(np.mean(lf_xy_speed)),
        "right_mean_xy_speed_mps": float(np.mean(rf_xy_speed)),
    },
}

analysis_path = OUT_DIR / "standing_analysis.json"
with analysis_path.open("w", encoding="utf-8") as stream:
    json.dump(analysis, stream, indent=2, sort_keys=True)
    stream.write("\n")

p = analysis["planar"]
y = analysis["yaw_rate"]
h = analysis["heading"]
rh = analysis["root_height"]
s = analysis["support"]
fm = analysis["foot_motion"]

print()
print("=" * 128)
print("V4-I MODEL_399 — ALIGNED ZERO-COMMAND STANDING RESULTS")
print("=" * 128)
print(f"samples / dt                         : {len(rows)} / {dt:.5f} s")
print(f"measured span / requested            : {measured_span_s:.3f} / {DURATION_S:.3f} s")
print(f"full requested window                 : {'YES' if full_window else 'NO'}")
print(f"command max abs                       : {command_max_abs:.3e}")
print()
print(f"mean planar speed                     : {p['mean_mps']:.6f} m/s")
print(f"RMS planar speed                      : {p['rms_mps']:.6f} m/s")
print(f"p95 planar speed                      : {p['p95_mps']:.6f} m/s")
print(f"max planar speed                      : {p['max_mps']:.6f} m/s")
print(f"mean |vx| / |vy|                      : {p['mean_abs_vx_mps']:.6f} / {p['mean_abs_vy_mps']:.6f} m/s")
print()
print(f"mean wz                               : {y['mean_radps']:+.6f} rad/s")
print(f"mean |wz|                             : {y['mean_abs_radps']:.6f} rad/s")
print(f"RMS wz                                : {y['rms_radps']:.6f} rad/s")
print(f"p95 |wz|                              : {y['p95_abs_radps']:.6f} rad/s")
print(f"max |wz|                              : {y['max_abs_radps']:.6f} rad/s")
print()
print(f"heading first                         : {h['first_deg']:+.3f} deg")
print(f"heading final error                   : {h['final_error_deg']:+.3f} deg")
print(f"heading mean abs                      : {h['mean_abs_deg']:.3f} deg")
print(f"heading RMS                           : {h['rms_deg']:.3f} deg")
print(f"heading p95 abs                       : {h['p95_abs_deg']:.3f} deg")
print(f"heading max abs                       : {h['max_abs_deg']:.3f} deg")
print()
print(f"drift dx / dy                         : {dx:+.4f} / {dy:+.4f} m")
print(f"net drift                             : {net_drift:.4f} m")
print(f"root height mean +/- std              : {rh['mean_m']:.4f} +/- {rh['std_m']:.4f} m")
print()
print(f"double / single / flight              : {s['double_pct']:.2f}% / {s['single_pct']:.2f}% / {s['flight_pct']:.2f}%")
print(f"touchdowns L / R                      : {s['left_touchdowns']} / {s['right_touchdowns']}")
print(f"foot path L / R                       : {fm['left_path_m']:.3f} / {fm['right_path_m']:.3f} m")
print(f"foot mean XY speed L / R              : {fm['left_mean_xy_speed_mps']:.4f} / {fm['right_mean_xy_speed_mps']:.4f} m/s")
print()
print("Analysis JSON                         :", analysis_path)
print("Repository evaluator modified         : NO")
print("Checkpoint modified                   : NO")
print()
print("V4I_MODEL399_ALIGNED_ZERO_COMMAND_STANDING_PROBE_PASS")
print("=" * 128)
