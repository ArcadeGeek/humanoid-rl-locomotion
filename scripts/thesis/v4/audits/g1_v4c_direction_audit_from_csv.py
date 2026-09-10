from __future__ import annotations

import csv
import math
from pathlib import Path

CSV_PATH = Path(
    r"C:\humanoid_rl_lab\logs\evaluations\g1"
    r"\v4c_model1499_play_equivalent_seed42_vx040_20s_canonical_v1"
    r"\samples.csv"
)

EXPECTED_N = 1000
EXPECTED_CMD = (0.4, 0.0, 0.0)


def wrap_pi(x: float) -> float:
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def percentile_linear(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("empty values")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    w = pos - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w


if not CSV_PATH.is_file():
    raise SystemExit(f"ABORT: samples.csv missing: {CSV_PATH}")

with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

if len(rows) != EXPECTED_N:
    raise SystemExit(f"ABORT: expected {EXPECTED_N} rows, found {len(rows)}")

required = {
    "time_s",
    "cmd_vx_mps",
    "cmd_vy_mps",
    "cmd_wz_radps",
    "base_vx_mps",
    "base_vy_mps",
    "base_wz_radps",
    "world_x_m",
    "world_y_m",
    "heading_rad",
}
missing = required.difference(rows[0].keys())
if missing:
    raise SystemExit(f"ABORT: missing required CSV columns: {sorted(missing)}")

def col(name: str) -> list[float]:
    return [float(r[name]) for r in rows]

t = col("time_s")
cmd_vx = col("cmd_vx_mps")
cmd_vy = col("cmd_vy_mps")
cmd_wz = col("cmd_wz_radps")
vx = col("base_vx_mps")
vy = col("base_vy_mps")
wz = col("base_wz_radps")
x = col("world_x_m")
y = col("world_y_m")
heading = col("heading_rad")

for name, vals in [
    ("time", t), ("cmd_vx", cmd_vx), ("cmd_vy", cmd_vy), ("cmd_wz", cmd_wz),
    ("vx", vx), ("vy", vy), ("wz", wz), ("x", x), ("y", y), ("heading", heading),
]:
    if not all(math.isfinite(v) for v in vals):
        raise SystemExit(f"ABORT: non-finite values in {name}")

tol = 1.0e-12
if any(abs(v - EXPECTED_CMD[0]) > tol for v in cmd_vx):
    raise SystemExit("ABORT: cmd_vx is not exactly 0.4 throughout.")
if any(abs(v - EXPECTED_CMD[1]) > tol for v in cmd_vy):
    raise SystemExit("ABORT: cmd_vy is not exactly 0.0 throughout.")
if any(abs(v - EXPECTED_CMD[2]) > tol for v in cmd_wz):
    raise SystemExit("ABORT: cmd_wz is not exactly 0.0 throughout.")

mean = lambda a: sum(a) / len(a)

mean_vx = mean(vx)
mean_vy = mean(vy)
mean_wz = mean(wz)

h0 = heading[0]
herr = [wrap_pi(h - h0) for h in heading]
abs_herr_deg = [abs(math.degrees(v)) for v in herr]
final_herr_deg = math.degrees(herr[-1])
mean_abs_herr_deg = mean(abs_herr_deg)
rms_herr_deg = math.sqrt(mean([v * v for v in [math.degrees(z) for z in herr]]))
p95_herr_deg = percentile_linear(abs_herr_deg, 0.95)
max_herr_deg = max(abs_herr_deg)

dx = x[-1] - x[0]
dy = y[-1] - y[0]

c = math.cos(h0)
s = math.sin(h0)
along = c * dx + s * dy
cross = -s * dx + c * dy

travel_heading = math.atan2(dy, dx)
travel_heading_error = wrap_pi(travel_heading - h0)

# Expected world-y change if motion were perfectly aligned with the initial heading,
# using the observed along-track displacement.
expected_world_y_from_initial_heading = along * math.sin(h0)
excess_world_y = dy - expected_world_y_from_initial_heading

print("=" * 116)
print("V4-C OFFLINE DIRECTION AUDIT — CANONICAL SEED42")
print("=" * 116)
print(f"samples / duration                  : {len(rows)} / {t[-1]:.2f} s")
print(f"command mean vx/vy/wz               : {mean(cmd_vx):+.6f} / {mean(cmd_vy):+.6f} / {mean(cmd_wz):+.6f}")
print(f"base vx mean                        : {mean_vx:+.6f} m/s")
print(f"base vy mean                        : {mean_vy:+.6f} m/s")
print(f"base wz mean                        : {mean_wz:+.6f} rad/s")
print()
print(f"initial heading                     : {math.degrees(h0):+.3f} deg")
print(f"final heading error                 : {final_herr_deg:+.3f} deg")
print(f"mean |heading error|                : {mean_abs_herr_deg:.3f} deg")
print(f"RMS heading error                   : {rms_herr_deg:.3f} deg")
print(f"p95 |heading error|                 : {p95_herr_deg:.3f} deg")
print(f"max |heading error|                 : {max_herr_deg:.3f} deg")
print()
print(f"global dx / dy                      : {dx:+.4f} / {dy:+.4f} m")
print(f"initial-heading-frame along-track   : {along:+.4f} m")
print(f"initial-heading-frame cross-track   : {cross:+.4f} m")
print(f"cross-track / |along-track|         : {100.0 * abs(cross) / max(abs(along), 1e-12):.3f} %")
print(f"expected world-y from initial head. : {expected_world_y_from_initial_heading:+.4f} m")
print(f"excess world-y                      : {excess_world_y:+.4f} m")
print(f"travel heading                      : {math.degrees(travel_heading):+.3f} deg")
print(f"travel-heading error vs initial     : {math.degrees(travel_heading_error):+.3f} deg")
print()
print("V4C_OFFLINE_DIRECTION_AUDIT_PASS")
print("=" * 116)
