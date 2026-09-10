from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np


REPO = Path(r"C:\humanoid_rl_lab")

CSV_B = (
    REPO / "logs" / "evaluations" / "g1"
    / "v4b_model1499_play_equivalent_seed42_vx040_20s_canonical_v1"
    / "samples.csv"
)

CSV_K = (
    REPO / "logs" / "evaluations" / "g1"
    / "v4k_model1499_play_equivalent_seed42_vx040_20s_canonical_v1"
    / "samples.csv"
)

EXPECTED_SAMPLES = 1000
EXPECTED_DT = 0.02
WINDOW_SAMPLES = 500  # first 10 s at verified dt=0.02
EXPECTED_COMMAND = np.asarray([0.40, 0.0, 0.0], dtype=np.float64)

REQUIRED = (
    "time_s",
    "cmd_vx_mps",
    "cmd_vy_mps",
    "cmd_wz_radps",
    "world_x_m",
    "world_y_m",
    "heading_rad",
)

SIGMA_X_M = 0.20
SIGMA_HEADING_DEG = (2.0, 3.0, 4.0, 5.0, 6.0)
WEIGHTS = (-0.10, -0.15, -0.20)


def load(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise RuntimeError(f"missing CSV: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"missing CSV header: {path}")

        missing = [c for c in REQUIRED if c not in reader.fieldnames]
        if missing:
            raise RuntimeError(f"missing columns {missing}: {path}")

        rows = list(reader)

    if len(rows) != EXPECTED_SAMPLES:
        raise RuntimeError(
            f"expected {EXPECTED_SAMPLES} samples, got {len(rows)}: {path}"
        )

    data = {
        c: np.asarray([float(row[c]) for row in rows], dtype=np.float64)
        for c in REQUIRED
    }

    dt = np.diff(data["time_s"])
    if not np.allclose(dt, EXPECTED_DT, rtol=0.0, atol=1e-9):
        raise RuntimeError(f"unexpected/nonuniform dt: {path}")

    command = np.column_stack(
        (data["cmd_vx_mps"], data["cmd_vy_mps"], data["cmd_wz_radps"])
    )
    if not np.allclose(
        command, EXPECTED_COMMAND[None, :], rtol=0.0, atol=1e-12
    ):
        raise RuntimeError(f"command is not fixed [0.40,0,0]: {path}")

    return data


def first_10s_errors(data: dict[str, np.ndarray]):
    sl = slice(0, WINDOW_SAMPLES)

    x = data["world_x_m"][sl]
    y = data["world_y_m"][sl]

    heading = np.unwrap(data["heading_rad"][sl])
    heading0 = float(heading[0])

    dx = x - x[0]
    dy = y - y[0]

    cross = (
        -math.sin(heading0) * dx
        + math.cos(heading0) * dy
    )

    # For the fixed straight command [0.40, 0, 0], the commanded body heading
    # is exactly the segment-start heading, so the heading reference is heading0.
    heading_error = heading - heading0

    return cross, heading_error


b = load(CSV_B)
k = load(CSV_K)

b_cross, b_heading = first_10s_errors(b)
k_cross, k_heading = first_10s_errors(k)

# Established 10 s sanity checks from the previous horizon audit.
if abs(float(b_cross[-1]) - (-0.1422)) > 5e-4:
    raise RuntimeError(
        f"v4-B 10 s cross sanity failed: {float(b_cross[-1])}"
    )
if abs(math.degrees(float(b_heading[-1])) - (-0.860)) > 0.01:
    raise RuntimeError(
        f"v4-B 10 s heading sanity failed: {math.degrees(float(b_heading[-1]))}"
    )
if abs(float(k_cross[-1]) - (-0.0871)) > 5e-4:
    raise RuntimeError(
        f"v4-K 10 s cross sanity failed: {float(k_cross[-1])}"
    )
if abs(math.degrees(float(k_heading[-1])) - (-6.676)) > 0.01:
    raise RuntimeError(
        f"v4-K 10 s heading sanity failed: {math.degrees(float(k_heading[-1]))}"
    )

print("=" * 126)
print("v4-L PATH-POSE REWARD SIZING — FIRST 10 s — FROZEN v4-B / v4-K TRACES")
print("=" * 126)
print("Composite bounded cost:")
print("  1 - exp(-[(cross_track / 0.20 m)^2 + (heading_error / sigma_heading)^2])")
print("Straight-command heading ref : segment-start heading")
print("Scope                        : sizing only; NO training")
print()

print("-" * 126)
print("FIRST-10s ERROR DISTRIBUTIONS")
print("-" * 126)
for label, cross, heading in (
    ("v4-B", b_cross, b_heading),
    ("v4-K", k_cross, k_heading),
):
    abs_cross = np.abs(cross)
    abs_heading_deg = np.degrees(np.abs(heading))
    print(label)
    print(
        f"  final cross / heading       : "
        f"{float(cross[-1]):+.4f} m / "
        f"{math.degrees(float(heading[-1])):+.3f} deg"
    )
    print(
        f"  mean |cross| / p95          : "
        f"{float(np.mean(abs_cross)):.4f} / "
        f"{float(np.percentile(abs_cross,95)):.4f} m"
    )
    print(
        f"  mean |heading| / p95        : "
        f"{float(np.mean(abs_heading_deg)):.3f} / "
        f"{float(np.percentile(abs_heading_deg,95)):.3f} deg"
    )

print()
print("-" * 126)
print("COMPOSITE COST SIZING")
print("-" * 126)
print(
    "sigma_h | B mean | B p95 | K mean | K p95 | "
    "K/B mean ratio | B frac>0.9 | K frac>0.9"
)

for sigma_deg in SIGMA_HEADING_DEG:
    sigma_rad = math.radians(sigma_deg)

    b_exponent = (
        np.square(b_cross / SIGMA_X_M)
        + np.square(b_heading / sigma_rad)
    )
    k_exponent = (
        np.square(k_cross / SIGMA_X_M)
        + np.square(k_heading / sigma_rad)
    )

    b_cost = 1.0 - np.exp(-b_exponent)
    k_cost = 1.0 - np.exp(-k_exponent)

    ratio = (
        float(np.mean(k_cost)) / float(np.mean(b_cost))
        if float(np.mean(b_cost)) > 1e-12
        else float("inf")
    )

    print(
        f"{sigma_deg:7.1f} | "
        f"{float(np.mean(b_cost)):6.4f} | "
        f"{float(np.percentile(b_cost,95)):6.4f} | "
        f"{float(np.mean(k_cost)):6.4f} | "
        f"{float(np.percentile(k_cost,95)):6.4f} | "
        f"{ratio:14.3f} | "
        f"{float(np.mean(b_cost > 0.90)):10.4f} | "
        f"{float(np.mean(k_cost > 0.90)):10.4f}"
    )

print()
print("-" * 126)
print("EFFECTIVE MEAN REWARD PER STEP")
print("-" * 126)
print(
    "sigma_h | weight | B mean reward | K mean reward | "
    "B p95 |reward| | K p95 |reward|"
)

for sigma_deg in SIGMA_HEADING_DEG:
    sigma_rad = math.radians(sigma_deg)

    b_cost = 1.0 - np.exp(
        -(
            np.square(b_cross / SIGMA_X_M)
            + np.square(b_heading / sigma_rad)
        )
    )
    k_cost = 1.0 - np.exp(
        -(
            np.square(k_cross / SIGMA_X_M)
            + np.square(k_heading / sigma_rad)
        )
    )

    for weight in WEIGHTS:
        b_reward = weight * b_cost
        k_reward = weight * k_cost

        print(
            f"{sigma_deg:7.1f} | "
            f"{weight:6.2f} | "
            f"{float(np.mean(b_reward)):+13.6f} | "
            f"{float(np.mean(k_reward)):+13.6f} | "
            f"{float(np.percentile(np.abs(b_reward),95)):.6f} | "
            f"{float(np.percentile(np.abs(k_reward),95)):.6f}"
        )

print()
print("V4L_PATH_POSE_REWARD_SIZING_PASS")
print("=" * 126)
