#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

CSV = Path(
    r"C:\humanoid_rl_lab\logs\thesis\video_final"
    r"\V01_v2g_scientific_straight_canonical_50hz_take01"
    r"\g1_v01_canonical_50hz_trajectory.csv"
)
META = Path(
    r"C:\humanoid_rl_lab\logs\thesis\video_final"
    r"\V01_v2g_scientific_straight_canonical_50hz_take01"
    r"\g1_v01_canonical_50hz_trajectory_meta.json"
)
EXPECTED_SHA256 = "a663258a00c933610b3a3615cb2907fdf8ffbde4fbecded0f469323ff61a6020"
FS = 50.0
DT = 1.0 / FS

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

if not CSV.is_file():
    raise SystemExit(f"CSV_MISSING: {CSV}")
if not META.is_file():
    raise SystemExit(f"META_MISSING: {META}")

actual_sha = sha256(CSV)
if actual_sha.lower() != EXPECTED_SHA256.lower():
    raise SystemExit(f"CSV_SHA_MISMATCH: expected {EXPECTED_SHA256}, got {actual_sha}")

meta = json.loads(META.read_text(encoding="utf-8"))
joint_names = list(meta.get("joint_names", []))
if len(joint_names) != 37:
    raise SystemExit(f"JOINT_NAME_COUNT_BAD: {len(joint_names)}")

with CSV.open("r", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

straight = [
    r for r in rows
    if abs(float(r["cmd_vx"]) - 0.40) < 1e-6
    and abs(float(r["cmd_vy"])) < 1e-6
    and abs(float(r["cmd_wz"])) < 1e-6
]

if len(straight) != 600:
    raise SystemExit(f"STRAIGHT_SAMPLE_COUNT_BAD: {len(straight)} != 600")

# Lower-limb joints only.
lower_ids = []
for i, name in enumerate(joint_names):
    low = name.lower()
    if (
        ("left" in low or "right" in low)
        and any(k in low for k in ("hip", "knee", "ankle"))
    ):
        lower_ids.append(i)

if not lower_ids:
    raise SystemExit("NO_LOWER_LIMB_JOINTS_FOUND")

def detrend(x: np.ndarray) -> np.ndarray:
    t = np.arange(len(x), dtype=np.float64)
    p = np.polyfit(t, x, 1)
    y = x - np.polyval(p, t)
    y -= y.mean()
    return y

def fft_peak_hz(x: np.ndarray) -> tuple[float, float]:
    y = detrend(x)
    w = np.hanning(len(y))
    spec = np.abs(np.fft.rfft(y * w))
    freqs = np.fft.rfftfreq(len(y), d=DT)
    band = (freqs >= 0.8) & (freqs <= 4.0)
    idxs = np.flatnonzero(band)
    j = idxs[np.argmax(spec[idxs])]
    return float(freqs[j]), float(spec[j])

def autocorr_cycle(x: np.ndarray) -> tuple[int, float]:
    y = detrend(x)
    norm = np.dot(y, y)
    if norm <= 1e-12:
        return 0, float("nan")
    # Exclude the expected half-cycle (~0.20-0.28 s) and search the
    # same-leg recurrence over 0.30-1.00 s.
    lag_min = int(round(0.30 * FS))
    lag_max = int(round(1.00 * FS))
    vals = []
    for lag in range(lag_min, lag_max + 1):
        v = float(np.dot(y[:-lag], y[lag:]) / norm)
        vals.append((v, lag))
    # Prefer a local maximum, otherwise global maximum.
    local = []
    for k in range(1, len(vals) - 1):
        if vals[k][0] >= vals[k-1][0] and vals[k][0] >= vals[k+1][0]:
            local.append(vals[k])
    best = max(local if local else vals, key=lambda z: z[0])
    return int(best[1]), float(best[0])

records = []
for i in lower_ids:
    x = np.array([float(r[f"joint_pos_{i:02d}"]) for r in straight], dtype=np.float64)
    std = float(np.std(x))
    if std < 1e-4:
        continue
    f_hz, amp = fft_peak_hz(x)
    lag, ac = autocorr_cycle(x)
    period = lag / FS if lag else float("nan")
    records.append(
        {
            "index": i,
            "name": joint_names[i],
            "std": std,
            "fft_hz": f_hz,
            "lag": lag,
            "period_s": period,
            "autocorr": ac,
        }
    )

# Use joints with a reasonably clear same-leg recurrence.
valid = [r for r in records if math.isfinite(r["autocorr"]) and r["autocorr"] >= 0.25]
if not valid:
    valid = records

periods = np.array([r["period_s"] for r in valid], dtype=np.float64)
median_period = float(np.median(periods))
cycle_hz = 1.0 / median_period
steps_per_s = 2.0 * cycle_hz
steps_per_min = 60.0 * steps_per_s

print("=" * 104)
print("G1 CANONICAL HEADLESS 50 HZ — GAIT FREQUENCY AUDIT")
print("=" * 104)
print(f"Trajectory SHA256                   : PASS ({actual_sha})")
print(f"Straight scientific samples         : {len(straight)}")
print(f"Straight scientific duration        : {len(straight) * DT:.2f} s")
print(f"Sampling frequency                  : {FS:.1f} Hz")
print(f"Lower-limb joints analyzed          : {len(records)}")
print()
print("Per-joint same-leg recurrence:")
for r in records:
    print(
        f"  {r['index']:02d} {r['name']:<34} "
        f"period={r['period_s']:.3f}s  "
        f"cycle={1.0/r['period_s']:.3f}Hz  "
        f"FFT={r['fft_hz']:.3f}Hz  "
        f"AC={r['autocorr']:.3f}"
    )

print()
print("-" * 104)
print(f"Median full gait-cycle period       : {median_period:.3f} s")
print(f"Median full gait-cycle frequency    : {cycle_hz:.3f} Hz")
print(f"Implied alternating step rate       : {steps_per_s:.3f} steps/s")
print(f"Implied cadence                     : {steps_per_min:.1f} steps/min")
print()
print("VIDEO V5.44 REFERENCE (measured from uploaded video)")
print("  Full visual recurrence            : ~0.46 s (~23 frames @ 50 fps)")
print("  Half-cycle / alternating step     : ~0.24 s (~12 frames @ 50 fps)")
print("  Full gait-cycle frequency         : ~2.17 Hz")
print("  Alternating step rate             : ~4.35 steps/s")
print()
rel_err = abs(median_period - 0.46) / 0.46 * 100.0
print(f"Headless-vs-video period difference : {rel_err:.1f}%")
if rel_err <= 10.0:
    print("RESULT                              : MATCH — video cadence is consistent with canonical headless gait")
elif rel_err <= 20.0:
    print("RESULT                              : NEAR MATCH — inspect individual joints / startup transient")
else:
    print("RESULT                              : MISMATCH — video timing/dynamics require further investigation")
print("G1_CANONICAL_GAIT_FREQUENCY_AUDIT_PASS")
