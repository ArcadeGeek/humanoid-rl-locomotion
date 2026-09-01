#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(r"C:\humanoid_rl_lab")
EVALUATOR = REPO / "scripts" / "rsl_rl" / "evaluate_g1.py"
TASK = "Humanoid-RL-Lab-G1-Flat-V4B-Direct-Yaw-v0"
RUN_DIR = (
    REPO
    / "logs"
    / "rsl_rl"
    / "humanoid_rl_lab_g1_v4b_direct_yaw_scratch"
    / "2026-09-01_20-08-18_v4b_genuine_scratch_seed42_official1500"
)
CHECKPOINT = RUN_DIR / "model_1499.pt"
OUT_DIR = (
    REPO
    / "logs"
    / "evaluations"
    / "g1"
    / "v4b_model1499_play_equivalent_seed42_vx040_20s_canonical_v1"
)

SEED = 42
WARMUP_S = 3.0
DURATION_S = 20.0
VX = 0.40
VY = 0.0
WZ = 0.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def median_or_nan(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def mean_or_nan(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def pct(mask):
    mask = np.asarray(mask, dtype=bool)
    return 100.0 * float(np.mean(mask)) if mask.size else float("nan")


def rising_edges(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    if m.size < 2:
        return np.asarray([], dtype=np.int64)
    return np.flatnonzero((~m[:-1]) & m[1:]) + 1


def falling_edges(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    if m.size < 2:
        return np.asarray([], dtype=np.int64)
    return np.flatnonzero(m[:-1] & (~m[1:])) + 1


def match_swings(contact: np.ndarray):
    los = falling_edges(contact)
    tds = rising_edges(contact)
    pairs = []
    j = 0
    for lo in los:
        while j < len(tds) and tds[j] <= lo:
            j += 1
        if j < len(tds):
            pairs.append((int(lo), int(tds[j])))
            j += 1
    return pairs


def fmt(v, scale=1.0, digits=4):
    if not np.isfinite(v):
        return "n/a"
    return f"{v * scale:.{digits}f}"


if not EVALUATOR.is_file():
    raise SystemExit(f"ABORT: evaluator missing: {EVALUATOR}")
if not CHECKPOINT.is_file():
    raise SystemExit(f"ABORT: v4-B final checkpoint missing: {CHECKPOINT}")

checkpoint_sha = sha256(CHECKPOINT)
evaluator_sha = sha256(EVALUATOR)

print("=" * 120)
print("G1 v4-B CONTACT-AWARE GAIT PROBE — PLAY-EQUIVALENT")
print("=" * 120)
print("Task            :", TASK)
print("Checkpoint      :", CHECKPOINT)
print("Checkpoint SHA  :", checkpoint_sha)
print("Evaluator       :", EVALUATOR)
print("Evaluator SHA   :", evaluator_sha)
print("Command         :", [VX, VY, WZ])
print("Seed            :", SEED)
print("Warmup / probe  :", f"{WARMUP_S:.1f} s / {DURATION_S:.1f} s")
print("Output          :", OUT_DIR)
print()

if OUT_DIR.exists():
    raise SystemExit(
        f"ABORT: canonical output directory already exists: {OUT_DIR}\n"
        "Refusing to overwrite or delete any prior evidence. Use a new output label for a rerun."
    )

source = EVALUATOR.read_text(encoding="utf-8")

play_anchor = "        checkpoint = retrieve_file_path(args_cli.checkpoint)"
play_patch = r"""        # v4-B gait probe: Official G1 Flat PLAY-equivalent semantics.
        env_cfg.observations.policy.enable_corruption = False
        if hasattr(env_cfg, "events"):
            if hasattr(env_cfg.events, "base_external_force_torque"):
                env_cfg.events.base_external_force_torque = None
            if hasattr(env_cfg.events, "push_robot"):
                env_cfg.events.push_robot = None

"""
if source.count(play_anchor) != 1:
    raise SystemExit(
        f"ABORT: expected PLAY patch anchor exactly once, found {source.count(play_anchor)}."
    )
source = source.replace(play_anchor, play_patch + play_anchor, 1)

# -------------------------------------------------------------------------------------------------
# PATCH B — independently resolve the two ankle-roll links in the temporary evaluator.
# This deliberately does NOT depend on any internal foot-ID variable name used by evaluate_g1.py.
# -------------------------------------------------------------------------------------------------
probe_ids_anchor = '            robot = base_env.scene["robot"]'
probe_ids_patch = probe_ids_anchor + r"""
            probe_foot_body_names = [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
            ]
            probe_foot_body_ids, probe_resolved_foot_body_names = robot.find_bodies(
                probe_foot_body_names,
                preserve_order=True,
            )
            if probe_resolved_foot_body_names != probe_foot_body_names:
                raise RuntimeError(
                    f"Unexpected probe foot body resolution: {probe_resolved_foot_body_names}"
                )"""
if source.count(probe_ids_anchor) != 1:
    raise SystemExit(
        f"ABORT: expected robot anchor exactly once, found {source.count(probe_ids_anchor)}."
    )
source = source.replace(probe_ids_anchor, probe_ids_patch, 1)

root_anchor = "                root_pos = _cpu_values(_torch(robot.data.root_pos_w)[0])"
root_patch = root_anchor + r"""
                foot_body_pos_w = np.asarray(
                    _cpu_values(_torch(robot.data.body_pos_w)[0, probe_foot_body_ids]),
                    dtype=np.float64,
                ).reshape(2, 3)
                foot_body_lin_vel_w = np.asarray(
                    _cpu_values(_torch(robot.data.body_lin_vel_w)[0, probe_foot_body_ids]),
                    dtype=np.float64,
                ).reshape(2, 3)"""
if source.count(root_anchor) != 1:
    raise SystemExit(
        f"ABORT: expected root sampling anchor exactly once, found {source.count(root_anchor)}."
    )
source = source.replace(root_anchor, root_patch, 1)

record_anchor = '                        "left_foot_normal_force_n": float(foot_normal_forces[0]),'
record_patch = r"""                        "left_foot_x_world_m": float(foot_body_pos_w[0, 0]),
                        "left_foot_y_world_m": float(foot_body_pos_w[0, 1]),
                        "left_foot_z_world_m": float(foot_body_pos_w[0, 2]),
                        "right_foot_x_world_m": float(foot_body_pos_w[1, 0]),
                        "right_foot_y_world_m": float(foot_body_pos_w[1, 1]),
                        "right_foot_z_world_m": float(foot_body_pos_w[1, 2]),
                        "left_foot_vx_world_mps": float(foot_body_lin_vel_w[0, 0]),
                        "left_foot_vy_world_mps": float(foot_body_lin_vel_w[0, 1]),
                        "left_foot_vz_world_mps": float(foot_body_lin_vel_w[0, 2]),
                        "right_foot_vx_world_mps": float(foot_body_lin_vel_w[1, 0]),
                        "right_foot_vy_world_mps": float(foot_body_lin_vel_w[1, 1]),
                        "right_foot_vz_world_mps": float(foot_body_lin_vel_w[1, 2]),
                        "left_foot_xy_speed_mps": float(np.linalg.norm(foot_body_lin_vel_w[0, :2])),
                        "right_foot_xy_speed_mps": float(np.linalg.norm(foot_body_lin_vel_w[1, :2])),
                        "left_knee_probe_rad": float(posture[2]),
                        "right_knee_probe_rad": float(posture[3]),
"""
if source.count(record_anchor) != 1:
    raise SystemExit(
        f"ABORT: expected foot-record anchor exactly once, found {source.count(record_anchor)}."
    )
source = source.replace(record_anchor, record_patch + record_anchor, 1)

# Final structural preflight before launching Isaac.
if source.count("probe_foot_body_ids") != 3:
    raise SystemExit(
        "ABORT: patched evaluator does not contain the expected probe-owned foot-ID structure."
    )
if source.count('"left_knee_probe_rad"') != 1 or source.count('"right_knee_probe_rad"') != 1:
    raise SystemExit("ABORT: patched evaluator knee instrumentation is incomplete.")
try:
    compile(source, str(EVALUATOR), "exec")
except SyntaxError as exc:
    raise SystemExit(f"ABORT: patched evaluator syntax invalid: {exc}") from exc
print("PATCHED_EVALUATOR_PREFLIGHT_PASS")

with tempfile.TemporaryDirectory(prefix="g1_v4a_gait_probe_") as td:
    temp_dir = Path(td)
    instrumented = temp_dir / "evaluate_g1_v4a_gait_instrumented.py"
    instrumented.write_text(source, encoding="utf-8")

    env = os.environ.copy()
    original_script_dir = str(EVALUATOR.parent)
    env["PYTHONPATH"] = (
        original_script_dir
        if not env.get("PYTHONPATH")
        else original_script_dir + os.pathsep + env["PYTHONPATH"]
    )

    cmd = [
        sys.executable,
        str(instrumented),
        "--task", TASK,
        "--checkpoint", str(CHECKPOINT),
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
        "--label", "v4a_model1499_play_equivalent_gait_probe",
    ]

    print("RUNNING TEMPORARY INSTRUMENTED COPY OF EXISTING EVALUATOR")
    print("-" * 120)
    print("Repository evaluator is NOT modified.")
    print("Training task/checkpoint are NOT modified.")
    print("Temporary evaluator patch: PLAY-equivalent + foot XYZ/velocity + explicit knee columns.")
    print()

    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO,
            env=env,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        raise SystemExit(
            "ABORT: instrumented evaluator exceeded 300 s. "
            "This is not normal for a 20 s single-env probe."
        )
    if proc.returncode != 0:
        raise SystemExit(f"ABORT: instrumented evaluator failed with code {proc.returncode}.")

samples_path = OUT_DIR / "samples.csv"
if not samples_path.is_file():
    raise SystemExit(f"ABORT: samples.csv missing: {samples_path}")

rows = list(csv.DictReader(samples_path.open("r", encoding="utf-8-sig", newline="")))
if not rows:
    raise SystemExit("ABORT: samples.csv is empty.")


def col(name: str) -> np.ndarray:
    if name not in rows[0]:
        raise SystemExit(f"ABORT: expected CSV column missing: {name}")
    return np.asarray([float(r[name]) for r in rows], dtype=np.float64)


t = col("time_s")
if len(t) < 3:
    raise SystemExit("ABORT: too few samples.")
dt = float(np.median(np.diff(t)))

lc = col("left_foot_contact") > 0.5
rc = col("right_foot_contact") > 0.5

lx, ly, lz = col("left_foot_x_world_m"), col("left_foot_y_world_m"), col("left_foot_z_world_m")
rx, ry, rz = col("right_foot_x_world_m"), col("right_foot_y_world_m"), col("right_foot_z_world_m")
lxy, rxy = col("left_foot_xy_speed_mps"), col("right_foot_xy_speed_mps")
lk, rk = col("left_knee_probe_rad"), col("right_knee_probe_rad")

base_vx = col("base_vx_mps")
world_x = col("world_x_m")
world_y = col("world_y_m")

l_td = rising_edges(lc)
r_td = rising_edges(rc)
l_swings = match_swings(lc)
r_swings = match_swings(rc)

stride_periods = []
stride_lengths = []
for td, x in ((l_td, lx), (r_td, rx)):
    if len(td) >= 2:
        stride_periods.extend(np.diff(t[td]).tolist())
        stride_lengths.extend(np.diff(x[td]).tolist())

events = [(int(i), "L") for i in l_td] + [(int(i), "R") for i in r_td]
events.sort(key=lambda p: (p[0], p[1]))

alt_intervals = []
touchdown_forward_sep = []
touchdown_width = []
for idx, side in events:
    touchdown_forward_sep.append(abs(lx[idx] - rx[idx]))
    touchdown_width.append(abs(ly[idx] - ry[idx]))

for (i0, s0), (i1, s1) in zip(events[:-1], events[1:]):
    if s0 != s1 and i1 > i0:
        alt_intervals.append(t[i1] - t[i0])

cadence_spm = 60.0 / median_or_nan(alt_intervals) if alt_intervals else float("nan")

swing_durations = []
swing_apices = []
swing_forward_travel = []
for pairs, own_x, own_z, other_z in (
    (l_swings, lx, lz, rz),
    (r_swings, rx, rz, lz),
):
    for lo, td in pairs:
        if td <= lo:
            continue
        swing_durations.append(t[td] - t[lo])
        segment = slice(lo, td + 1)
        swing_apices.append(float(np.max(own_z[segment] - other_z[segment])))
        swing_forward_travel.append(float(own_x[td] - own_x[lo]))

single_support = np.logical_xor(lc, rc)
double_support = np.logical_and(lc, rc)
flight = np.logical_and(~lc, ~rc)

contact_slip_samples = np.concatenate([lxy[lc], rxy[rc]])
mean_contact_slip = mean_or_nan(contact_slip_samples)

stride_lengths_arr = np.asarray(stride_lengths, dtype=np.float64)
stride_lengths_arr = stride_lengths_arr[np.isfinite(stride_lengths_arr)]
stride_cv_pct = (
    100.0 * float(np.std(stride_lengths_arr, ddof=0) / np.mean(stride_lengths_arr))
    if stride_lengths_arr.size >= 2 and abs(float(np.mean(stride_lengths_arr))) > 1e-12
    else float("nan")
)

lk_stance = mean_or_nan(lk[lc])
lk_swing = mean_or_nan(lk[~lc])
rk_stance = mean_or_nan(rk[rc])
rk_swing = mean_or_nan(rk[~rc])

mean_vx = mean_or_nan(base_vx)
forward = float(world_x[-1] - world_x[0])
lateral = float(world_y[-1] - world_y[0])
dx = np.diff(world_x)
dy = np.diff(world_y)
path_length = float(np.sum(np.sqrt(dx * dx + dy * dy)))
straightness = forward / path_length if path_length > 1e-12 else float("nan")

report = {
    "protocol": {
        "task": TASK,
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": checkpoint_sha,
        "evaluator": str(EVALUATOR),
        "evaluator_sha256": evaluator_sha,
        "seed": SEED,
        "warmup_s": WARMUP_S,
        "duration_s": DURATION_S,
        "command": [VX, VY, WZ],
        "deterministic": True,
        "play_equivalent": {
            "observation_corruption": False,
            "base_external_force_torque": None,
            "push_robot": None,
        },
        "dt_s": dt,
        "samples": len(rows),
    },
    "gait": {
        "same_foot_stride_period_median_s": median_or_nan(stride_periods),
        "cadence_touchdowns_spm": cadence_spm,
        "touchdown_forward_foot_separation_median_m": median_or_nan(touchdown_forward_sep),
        "stride_length_median_m": median_or_nan(stride_lengths),
        "swing_duration_median_s": median_or_nan(swing_durations),
        "relative_swing_apex_median_m": median_or_nan(swing_apices),
        "swing_forward_travel_median_m": median_or_nan(swing_forward_travel),
        "step_width_touchdown_median_m": median_or_nan(touchdown_width),
        "single_support_pct": pct(single_support),
        "double_support_pct": pct(double_support),
        "flight_pct": pct(flight),
        "mean_foot_slip_during_contact_mps": mean_contact_slip,
        "stride_length_cv_pct": stride_cv_pct,
        "left_touchdowns": int(len(l_td)),
        "right_touchdowns": int(len(r_td)),
        "complete_left_swings": int(len(l_swings)),
        "complete_right_swings": int(len(r_swings)),
    },
    "knees_rad": {
        "left_stance_mean": lk_stance,
        "left_swing_mean": lk_swing,
        "right_stance_mean": rk_stance,
        "right_swing_mean": rk_swing,
    },
    "tracking_context": {
        "mean_base_vx_mps": mean_vx,
        "forward_displacement_m": forward,
        "final_lateral_drift_m": lateral,
        "path_length_m": path_length,
        "straightness_forward_over_path": straightness,
    },
}

analysis_path = OUT_DIR / "gait_probe_analysis.json"
with analysis_path.open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, sort_keys=True)
    f.write("\n")

print()
print("=" * 120)
print("V4-B MODEL_1499 — CONTACT-AWARE GAIT RESULTS")
print("=" * 120)
print(f"samples / dt                         : {len(rows)} / {dt:.5f} s")
print(f"mean body vx                         : {fmt(mean_vx)} m/s")
print(f"same-foot stride period median       : {fmt(report['gait']['same_foot_stride_period_median_s'])} s")
print(f"cadence from alternating touchdowns  : {fmt(cadence_spm, digits=1)} spm")
print(f"forward foot separation @ TD median  : {fmt(report['gait']['touchdown_forward_foot_separation_median_m'], 100.0, 2)} cm")
print(f"stride length median                  : {fmt(report['gait']['stride_length_median_m'], 100.0, 2)} cm")
print(f"swing duration median                 : {fmt(report['gait']['swing_duration_median_s'])} s")
print(f"relative swing apex median            : {fmt(report['gait']['relative_swing_apex_median_m'], 100.0, 2)} cm")
print(f"swing forward travel median           : {fmt(report['gait']['swing_forward_travel_median_m'], 100.0, 2)} cm")
print(f"step width @ touchdown median         : {fmt(report['gait']['step_width_touchdown_median_m'], 100.0, 2)} cm")
print(f"single support                        : {fmt(report['gait']['single_support_pct'], digits=1)} %")
print(f"double support                        : {fmt(report['gait']['double_support_pct'], digits=1)} %")
print(f"flight                                : {fmt(report['gait']['flight_pct'], digits=1)} %")
print(f"mean foot slip during contact         : {fmt(mean_contact_slip)} m/s")
print(f"stride length CV                      : {fmt(stride_cv_pct, digits=1)} %")
print(f"knees L stance -> swing              : {fmt(lk_stance)} -> {fmt(lk_swing)} rad")
print(f"knees R stance -> swing              : {fmt(rk_stance)} -> {fmt(rk_swing)} rad")
print(f"forward / lateral drift               : {forward:+.4f} / {lateral:+.4f} m")
print(f"straightness                          : {fmt(straightness, digits=6)}")
print(f"touchdowns L/R                        : {len(l_td)} / {len(r_td)}")
print(f"analysis JSON                         : {analysis_path}")
print()
print("V4B_CONTACT_GAIT_PROBE_CANONICAL_PASS")
print("Original evaluator/source/checkpoint were not modified.")
print("=" * 120)
