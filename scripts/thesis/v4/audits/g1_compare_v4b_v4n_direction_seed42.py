from __future__ import annotations

import csv
import math
from pathlib import Path


REPO = Path(r"C:\humanoid_rl_lab")

B_CSV = (
    REPO
    / "logs"
    / "evaluations"
    / "g1"
    / "v4b_model1499_play_equivalent_seed42_vx040_20s_canonical_v1"
    / "samples.csv"
)

N_CSV = (
    REPO
    / "logs"
    / "evaluations"
    / "g1"
    / "v4n_model1499_play_equivalent_seed42_vx040_20s_canonical_v1"
    / "samples.csv"
)

EXPECTED_COMMAND = (0.4, 0.0, 0.0)


def wrap_pi(x: float) -> float:
    while x > math.pi:
        x -= 2.0 * math.pi
    while x < -math.pi:
        x += 2.0 * math.pi
    return x


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"ABORT: missing samples.csv: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if len(rows) < 2:
        raise SystemExit(f"ABORT: insufficient rows in {path}")

    required = {
        "cmd_vx_mps",
        "cmd_vy_mps",
        "cmd_wz_radps",
        "base_vx_mps",
        "base_vy_mps",
        "base_wz_radps",
        "heading_rad",
        "world_x_m",
        "world_y_m",
    }

    missing = required.difference(rows[0].keys())
    if missing:
        raise SystemExit(
            f"ABORT: missing columns in {path}: {sorted(missing)}"
        )

    return rows


def analyse(path: Path) -> dict[str, float]:
    rows = read_rows(path)

    def col(name: str) -> list[float]:
        return [float(r[name]) for r in rows]

    cmd_vx = col("cmd_vx_mps")
    cmd_vy = col("cmd_vy_mps")
    cmd_wz = col("cmd_wz_radps")

    if max(abs(v - EXPECTED_COMMAND[0]) for v in cmd_vx) > 1.0e-9:
        raise SystemExit(f"ABORT: cmd_vx is not fixed at 0.4 in {path}")
    if max(abs(v - EXPECTED_COMMAND[1]) for v in cmd_vy) > 1.0e-9:
        raise SystemExit(f"ABORT: cmd_vy is not fixed at 0.0 in {path}")
    if max(abs(v - EXPECTED_COMMAND[2]) for v in cmd_wz) > 1.0e-9:
        raise SystemExit(f"ABORT: cmd_wz is not fixed at 0.0 in {path}")

    vx = col("base_vx_mps")
    vy = col("base_vy_mps")
    wz = col("base_wz_radps")
    heading = col("heading_rad")
    world_x = col("world_x_m")
    world_y = col("world_y_m")

    h0 = heading[0]
    heading_errors = [wrap_pi(h - h0) for h in heading]
    abs_heading_errors = [abs(e) for e in heading_errors]

    dx = world_x[-1] - world_x[0]
    dy = world_y[-1] - world_y[0]

    c = math.cos(h0)
    s = math.sin(h0)

    along = c * dx + s * dy
    cross = -s * dx + c * dy

    path_length = 0.0
    for i in range(1, len(rows)):
        ddx = world_x[i] - world_x[i - 1]
        ddy = world_y[i] - world_y[i - 1]
        path_length += math.hypot(ddx, ddy)

    return {
        "samples": float(len(rows)),
        "mean_vx": mean(vx),
        "mean_vy": mean(vy),
        "mean_wz": mean(wz),
        "initial_heading_deg": math.degrees(h0),
        "final_heading_error_deg": math.degrees(heading_errors[-1]),
        "mean_abs_heading_error_deg": math.degrees(mean(abs_heading_errors)),
        "rms_heading_error_deg": math.degrees(
            math.sqrt(mean([e * e for e in heading_errors]))
        ),
        "p95_abs_heading_error_deg": math.degrees(
            sorted(abs_heading_errors)[
                min(
                    len(abs_heading_errors) - 1,
                    math.ceil(0.95 * len(abs_heading_errors)) - 1,
                )
            ]
        ),
        "max_abs_heading_error_deg": math.degrees(max(abs_heading_errors)),
        "world_dx": dx,
        "world_dy": dy,
        "along": along,
        "cross": cross,
        "cross_ratio_pct": 100.0 * abs(cross) / max(abs(along), 1.0e-12),
        "path_length": path_length,
        "initial_frame_straightness": (
            along / path_length if path_length > 1.0e-12 else float("nan")
        ),
    }


b = analyse(B_CSV)
n = analyse(N_CSV)

print("=" * 122)
print("v4-B vs v4-N — SEED42 DIRECTIONAL CLOSURE")
print("=" * 122)
print(f"v4-B CSV : {B_CSV}")
print(f"v4-N CSV : {N_CSV}")
print("command  : [0.40, 0.00, 0.00]")
print()

print(
    "metric                               | v4-B              | v4-N              | delta N-B"
)
print("-" * 122)

rows = [
    ("mean body vx [m/s]", "mean_vx", "{:+.6f}"),
    ("mean body vy [m/s]", "mean_vy", "{:+.6f}"),
    ("mean body wz [rad/s]", "mean_wz", "{:+.6f}"),
    ("initial heading [deg]", "initial_heading_deg", "{:+.3f}"),
    ("final heading error [deg]", "final_heading_error_deg", "{:+.3f}"),
    ("mean |heading error| [deg]", "mean_abs_heading_error_deg", "{:.3f}"),
    ("RMS heading error [deg]", "rms_heading_error_deg", "{:.3f}"),
    ("p95 |heading error| [deg]", "p95_abs_heading_error_deg", "{:.3f}"),
    ("max |heading error| [deg]", "max_abs_heading_error_deg", "{:.3f}"),
    ("initial-frame along [m]", "along", "{:+.4f}"),
    ("initial-frame cross [m]", "cross", "{:+.4f}"),
    ("|cross| / |along| [%]", "cross_ratio_pct", "{:.3f}"),
    ("path length [m]", "path_length", "{:.4f}"),
    ("initial-frame straightness", "initial_frame_straightness", "{:.6f}"),
]

for label, key, fmt in rows:
    bv = b[key]
    nv = n[key]
    dv = nv - bv
    print(
        f"{label:<36s} | "
        f"{fmt.format(bv):>17s} | "
        f"{fmt.format(nv):>17s} | "
        f"{fmt.format(dv):>17s}"
    )

b_abs_cross = abs(b["cross"])
n_abs_cross = abs(n["cross"])

if b_abs_cross > 1.0e-12:
    cross_change_pct = 100.0 * (n_abs_cross - b_abs_cross) / b_abs_cross
else:
    cross_change_pct = float("nan")

print()
print("-" * 122)
print("CLOSURE METRICS")
print("-" * 122)
print(f"|cross| v4-B                         : {b_abs_cross:.4f} m")
print(f"|cross| v4-N                         : {n_abs_cross:.4f} m")
print(f"|cross| relative change N vs B       : {cross_change_pct:+.2f} %")
print(
    f"|mean vy| relative change N vs B     : "
    f"{100.0 * (abs(n['mean_vy']) - abs(b['mean_vy'])) / max(abs(b['mean_vy']), 1.0e-12):+.2f} %"
)
print(
    f"mean |heading error| delta N-B        : "
    f"{n['mean_abs_heading_error_deg'] - b['mean_abs_heading_error_deg']:+.3f} deg"
)
print(
    f"final |heading error| delta N-B       : "
    f"{abs(n['final_heading_error_deg']) - abs(b['final_heading_error_deg']):+.3f} deg"
)
print(
    f"initial-frame straightness delta N-B  : "
    f"{n['initial_frame_straightness'] - b['initial_frame_straightness']:+.6f}"
)
print()
print("V4B_V4N_SEED42_DIRECTIONAL_CLOSURE_PASS")
print("=" * 122)
