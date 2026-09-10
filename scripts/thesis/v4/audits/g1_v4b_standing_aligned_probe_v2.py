from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(r"C:\humanoid_rl_lab")
CANONICAL_PROBE = REPO / "scripts" / "thesis" / "v4" / "g1_v4b_contact_gait_probe.py"
EXPECTED_PROBE_SHA = "40948E95BA457FCC1AFD2708619054FE734CCEFF515130A997AFFF3958A84848"

CHECKPOINT = (
    REPO
    / "logs"
    / "rsl_rl"
    / "humanoid_rl_lab_g1_v4b_direct_yaw_scratch"
    / "2026-09-01_20-08-18_v4b_genuine_scratch_seed42_official1500"
    / "model_1499.pt"
)
EXPECTED_CHECKPOINT_SHA = "B6B9E94DC222F21AF698BA158B9E91C18071FE90ABC255B9C4311874E106EC7F"

OUT_TOKEN_OLD = "v4b_model1499_play_equivalent_seed42_vx040_20s_canonical_v1"
OUT_TOKEN_NEW = "v4b_model1499_play_equivalent_seed42_zero_cmd_aligned_yaw_20s_standing_v2"
OUT_DIR = REPO / "logs" / "evaluations" / "g1" / OUT_TOKEN_NEW

PASS_OLD = "V4B_CONTACT_GAIT_PROBE_CANONICAL_PASS"
PASS_DERIVED = "V4B_STANDING_ALIGNED_INNER_PROBE_PASS"

PLAY_EVENT_ANCHOR = '''        if hasattr(env_cfg, "events"):
            if hasattr(env_cfg.events, "base_external_force_torque"):'''

PLAY_EVENT_REPLACEMENT = '''        if hasattr(env_cfg, "events"):
            # Evaluation-only aligned reset for the standing diagnostic.
            # Training/config/checkpoint are NOT modified.
            if hasattr(env_cfg.events, "reset_base") and env_cfg.events.reset_base is not None:
                _standing_pose_range = dict(env_cfg.events.reset_base.params.get("pose_range", {}))
                _standing_velocity_range = dict(env_cfg.events.reset_base.params.get("velocity_range", {}))
                _standing_pose_range["yaw"] = (0.0, 0.0)
                _standing_velocity_range["yaw"] = (0.0, 0.0)
                env_cfg.events.reset_base.params["pose_range"] = _standing_pose_range
                env_cfg.events.reset_base.params["velocity_range"] = _standing_velocity_range
            if hasattr(env_cfg.events, "base_external_force_torque"):'''


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def wrap_pi(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def percentile(values, q):
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    return float(np.percentile(a, q))


def pct(mask):
    m = np.asarray(mask, dtype=bool)
    return float(100.0 * np.mean(m)) if m.size else float("nan")


def rising_edges(mask):
    m = np.asarray(mask, dtype=bool)
    if m.size < 2:
        return np.asarray([], dtype=np.int64)
    return np.flatnonzero((~m[:-1]) & m[1:]) + 1


def recursive_resetish(obj, prefix=""):
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            kl = str(k).lower()
            if any(token in kl for token in ("reset", "termination", "terminated", "fall")):
                if isinstance(v, (str, int, float, bool)) or v is None:
                    found.append((p, v))
            found.extend(recursive_resetish(v, p))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            found.extend(recursive_resetish(v, f"{prefix}[{i}]"))
    return found


if not CANONICAL_PROBE.is_file():
    raise SystemExit(f"ABORT: canonical v4-B probe missing: {CANONICAL_PROBE}")
probe_sha = sha256(CANONICAL_PROBE)
if probe_sha != EXPECTED_PROBE_SHA:
    raise SystemExit(
        "ABORT: canonical v4-B probe SHA mismatch.\n"
        f"expected={EXPECTED_PROBE_SHA}\nactual  ={probe_sha}"
    )

if not CHECKPOINT.is_file():
    raise SystemExit(f"ABORT: frozen v4-B checkpoint missing: {CHECKPOINT}")
checkpoint_sha = sha256(CHECKPOINT)
if checkpoint_sha != EXPECTED_CHECKPOINT_SHA:
    raise SystemExit(
        "ABORT: v4-B checkpoint SHA mismatch.\n"
        f"expected={EXPECTED_CHECKPOINT_SHA}\nactual  ={checkpoint_sha}"
    )

if OUT_DIR.exists():
    raise SystemExit(
        f"ABORT: standing output already exists: {OUT_DIR}\n"
        "Refusing to overwrite/delete prior evidence."
    )

source = CANONICAL_PROBE.read_text(encoding="utf-8")

replacements = [
    ("VX = 0.40", "VX = 0.00", "zero forward command"),
    (OUT_TOKEN_OLD, OUT_TOKEN_NEW, "standing output token"),
    ("v4a_model1499_play_equivalent_gait_probe", "v4b_model1499_aligned_standing_probe", "label"),
    (PASS_OLD, PASS_DERIVED, "inner PASS marker"),
    (PLAY_EVENT_ANCHOR, PLAY_EVENT_REPLACEMENT, "aligned reset patch"),
]

derived = source
for old, new, label in replacements:
    count = derived.count(old)
    if count != 1:
        raise SystemExit(
            f"ABORT: {label} anchor occurrence mismatch: expected 1, found {count}"
        )
    derived = derived.replace(old, new, 1)

reversed_text = derived
for old, new, label in reversed(replacements):
    if reversed_text.count(new) != 1:
        raise SystemExit(f"ABORT: reverse-check failed for {label}")
    reversed_text = reversed_text.replace(new, old, 1)
if reversed_text != source:
    raise SystemExit(
        "ABORT: reversibility guard failed: derived execution probe differs "
        "outside the explicitly allowed standing-test changes."
    )

for token in (
    "SEED = 42",
    "WARMUP_S = 3.0",
    "DURATION_S = 20.0",
    "VY = 0.0",
    "WZ = 0.0",
):
    if derived.count(token) != 1:
        raise SystemExit(f"ABORT: expected unchanged protocol token missing/non-unique: {token}")

if derived.count("VX = 0.00") != 1:
    raise SystemExit("ABORT: standing VX=0 token missing/non-unique")

compile(derived, "<v4b_standing_aligned_derived_probe>", "exec")

print("=" * 124)
print("V4-B STANDING BALANCE — ALIGNED-RESET DIAGNOSTIC")
print("=" * 124)
print(f"canonical probe      : {CANONICAL_PROBE}")
print(f"canonical probe SHA  : {probe_sha}")
print(f"checkpoint           : {CHECKPOINT}")
print(f"checkpoint SHA       : {checkpoint_sha}")
print("seed                 : 42")
print("warmup / measure     : 3.0 s / 20.0 s")
print("command              : [0.00, 0.00, 0.00]")
print("Play-equivalent      : YES")
print("reset yaw pose       : evaluation-only (0.0, 0.0) rad")
print("reset yaw velocity   : evaluation-only (0.0, 0.0) rad/s")
print("other reset ranges   : unchanged")
print(f"output               : {OUT_DIR}")
print("reversibility guard  : PASS")
print("=" * 124)

with tempfile.TemporaryDirectory(prefix="g1_v4b_standing_aligned_") as td:
    temp_probe = Path(td) / "g1_v4b_standing_aligned_derived_probe.py"
    temp_probe.write_text(derived, encoding="utf-8", newline="\n")

    print()
    print("STARTING INNER PROBE WITH LIVE OUTPUT")
    print("All Isaac/evaluator output below is streamed directly to this terminal.")
    print("-" * 124)
    sys.stdout.flush()

    try:
        proc = subprocess.run(
            [sys.executable, str(temp_probe)],
            cwd=str(REPO),
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        raise SystemExit(
            "ABORT: derived standing probe exceeded 300 s. "
            "Any output produced before timeout was shown live above."
        )

    if proc.returncode != 0:
        raise SystemExit(f"ABORT: derived standing probe failed with code {proc.returncode}")

    print("-" * 124)
    print("INNER PROBE RETURNED SUCCESSFULLY; validating generated evidence...")
    sys.stdout.flush()

samples_path = OUT_DIR / "samples.csv"
summary_path = OUT_DIR / "summary.json"

if not samples_path.is_file():
    raise SystemExit(
        f"ABORT: inner probe returned 0 but standing samples.csv is missing: {samples_path}"
    )

with samples_path.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

if len(rows) != 1000:
    raise SystemExit(f"ABORT: expected 1000 standing samples, found {len(rows)}")

def col(name):
    if name not in rows[0]:
        raise SystemExit(f"ABORT: expected CSV column missing: {name}")
    a = np.asarray([float(r[name]) for r in rows], dtype=np.float64)
    if not np.all(np.isfinite(a)):
        raise SystemExit(f"ABORT: non-finite values in CSV column: {name}")
    return a

t = col("time_s")
cmd_vx = col("cmd_vx_mps")
cmd_vy = col("cmd_vy_mps")
cmd_wz = col("cmd_wz_radps")
vx = col("base_vx_mps")
vy = col("base_vy_mps")
wz = col("base_wz_radps")
x = col("world_x_m")
y = col("world_y_m")
z = col("root_height_m")
heading = col("heading_rad")

lc = col("left_foot_contact") > 0.5
rc = col("right_foot_contact") > 0.5
lxy = col("left_foot_xy_speed_mps")
rxy = col("right_foot_xy_speed_mps")
lx = col("left_foot_x_world_m")
ly = col("left_foot_y_world_m")
rx = col("right_foot_x_world_m")
ry = col("right_foot_y_world_m")
lk = col("left_knee_probe_rad")
rk = col("right_knee_probe_rad")

tol = 1.0e-12
if np.max(np.abs(cmd_vx)) > tol or np.max(np.abs(cmd_vy)) > tol or np.max(np.abs(cmd_wz)) > tol:
    raise SystemExit("ABORT: measured command is not exactly [0,0,0] throughout")

dt = float(np.median(np.diff(t)))
planar_speed = np.hypot(vx, vy)
abs_wz = np.abs(wz)

dx = float(x[-1] - x[0])
dy = float(y[-1] - y[0])
net_drift = float(math.hypot(dx, dy))

h0 = float(heading[0])
herr = wrap_pi(heading - h0)
abs_herr_deg = np.abs(np.degrees(herr))

double = lc & rc
single = np.logical_xor(lc, rc)
flight = (~lc) & (~rc)

l_contact_slip = lxy[lc]
r_contact_slip = rxy[rc]

l_path = float(np.sum(np.hypot(np.diff(lx), np.diff(ly))))
r_path = float(np.sum(np.hypot(np.diff(rx), np.diff(ry))))
l_net = float(math.hypot(lx[-1] - lx[0], ly[-1] - ly[0]))
r_net = float(math.hypot(rx[-1] - rx[0], ry[-1] - ry[0]))

l_td = rising_edges(lc)
r_td = rising_edges(rc)

resetish = []
if summary_path.is_file():
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        resetish = recursive_resetish(summary)
    except Exception as exc:
        resetish = [("summary_parse_error", repr(exc))]

print()
print("=" * 124)
print("V4-B STANDING BALANCE RESULTS — ALIGNED RESET, ZERO COMMAND")
print("=" * 124)
print(f"samples / dt                         : {len(rows)} / {dt:.5f} s")
print(f"command max |vx/vy/wz|               : {np.max(np.abs(cmd_vx)):.6f} / {np.max(np.abs(cmd_vy)):.6f} / {np.max(np.abs(cmd_wz)):.6f}")
print()
print(f"mean planar speed                    : {np.mean(planar_speed):.6f} m/s")
print(f"RMS planar speed                     : {math.sqrt(np.mean(planar_speed**2)):.6f} m/s")
print(f"p95 planar speed                     : {percentile(planar_speed, 95):.6f} m/s")
print(f"max planar speed                     : {np.max(planar_speed):.6f} m/s")
print(f"mean |body vx|                       : {np.mean(np.abs(vx)):.6f} m/s")
print(f"mean |body vy|                       : {np.mean(np.abs(vy)):.6f} m/s")
print()
print(f"mean body wz                         : {np.mean(wz):+.6f} rad/s")
print(f"mean |body wz|                       : {np.mean(abs_wz):.6f} rad/s")
print(f"RMS body wz                          : {math.sqrt(np.mean(wz**2)):.6f} rad/s")
print(f"p95 |body wz|                        : {percentile(abs_wz, 95):.6f} rad/s")
print(f"max |body wz|                        : {np.max(abs_wz):.6f} rad/s")
print()
print(f"heading at first measured sample     : {math.degrees(h0):+.3f} deg")
print(f"final heading error vs first sample  : {math.degrees(float(herr[-1])):+.3f} deg")
print(f"mean |heading error|                 : {np.mean(abs_herr_deg):.3f} deg")
print(f"RMS heading error                    : {math.sqrt(np.mean(np.degrees(herr)**2)):.3f} deg")
print(f"p95 |heading error|                  : {percentile(abs_herr_deg, 95):.3f} deg")
print(f"max |heading error|                  : {np.max(abs_herr_deg):.3f} deg")
print()
print(f"world displacement dx / dy           : {dx:+.4f} / {dy:+.4f} m")
print(f"net planar drift                     : {net_drift:.4f} m")
print(f"root height mean / std               : {np.mean(z):.4f} / {np.std(z, ddof=1):.4f} m")
print(f"root height min / max                : {np.min(z):.4f} / {np.max(z):.4f} m")
print()
print(f"double support                       : {pct(double):.1f} %")
print(f"single support                       : {pct(single):.1f} %")
print(f"flight                               : {pct(flight):.1f} %")
print(f"touchdowns L/R                       : {len(l_td)} / {len(r_td)}")
print(f"mean contact slip L/R                : {np.mean(l_contact_slip) if l_contact_slip.size else float('nan'):.6f} / {np.mean(r_contact_slip) if r_contact_slip.size else float('nan'):.6f} m/s")
print(f"foot planar path L/R                 : {l_path:.4f} / {r_path:.4f} m")
print(f"foot net planar displacement L/R     : {l_net:.4f} / {r_net:.4f} m")
print(f"knee mean L/R                        : {np.mean(lk):.4f} / {np.mean(rk):.4f} rad")
print(f"knee std L/R                         : {np.std(lk, ddof=1):.4f} / {np.std(rk, ddof=1):.4f} rad")
print()

if resetish:
    print("summary reset/termination fields")
    for k, v in resetish[:30]:
        print(f"  {k}: {v}")
else:
    print("summary reset/termination fields     : none exposed by summary.json")

print()
print(f"samples CSV                          : {samples_path}")
print(f"summary JSON                         : {summary_path if summary_path.is_file() else 'missing'}")
print()
print("V4B_STANDING_ALIGNED_PROBE_PASS")
print("=" * 124)
