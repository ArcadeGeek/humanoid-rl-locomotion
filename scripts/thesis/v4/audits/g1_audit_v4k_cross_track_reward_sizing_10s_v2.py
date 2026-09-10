from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np


REPO = Path(r"C:\humanoid_rl_lab")

CASES = [
    (
        "seed42",
        REPO
        / "logs"
        / "evaluations"
        / "g1"
        / "v4b_model1499_play_equivalent_seed42_vx040_20s_canonical_v1"
        / "samples.csv",
    ),
]

for seed in range(100, 110):
    CASES.append(
        (
            f"seed{seed}",
            REPO
            / "logs"
            / "evaluations"
            / "g1"
            / f"v4b_model1499_play_equivalent_seed{seed}_vx040_20s_direction_dev"
            / "samples.csv",
        )
    )


REQUIRED_COLUMNS = (
    "time_s",
    "cmd_vx_mps",
    "cmd_vy_mps",
    "cmd_wz_radps",
    "world_x_m",
    "world_y_m",
    "heading_rad",
)

EXPECTED_TOTAL_SAMPLES = 1000
EXPECTED_DT_S = 0.02
WINDOW_S = 10.0
WINDOW_SAMPLES = int(round(WINDOW_S / EXPECTED_DT_S))

EXPECTED_COMMAND = np.asarray([0.40, 0.0, 0.0], dtype=np.float64)

SIGMAS_M = (0.15, 0.20, 0.25, 0.30)
WEIGHTS = (-0.10, -0.15, -0.25)


def load_csv(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise RuntimeError(f"missing CSV: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)

        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {path}")

        missing = [
            column
            for column in REQUIRED_COLUMNS
            if column not in reader.fieldnames
        ]
        if missing:
            raise RuntimeError(
                f"missing required columns in {path}: {missing}"
            )

        rows = list(reader)

    if len(rows) != EXPECTED_TOTAL_SAMPLES:
        raise RuntimeError(
            f"unexpected sample count in {path}: "
            f"{len(rows)} != {EXPECTED_TOTAL_SAMPLES}"
        )

    data = {
        column: np.asarray(
            [float(row[column]) for row in rows],
            dtype=np.float64,
        )
        for column in REQUIRED_COLUMNS
    }

    dt = np.diff(data["time_s"])
    if not np.allclose(
        dt,
        EXPECTED_DT_S,
        rtol=0.0,
        atol=1.0e-9,
    ):
        raise RuntimeError(
            f"unexpected/nonuniform dt in {path}: "
            f"median={float(np.median(dt)):.12f}"
        )

    command = np.column_stack(
        (
            data["cmd_vx_mps"],
            data["cmd_vy_mps"],
            data["cmd_wz_radps"],
        )
    )

    if not np.allclose(
        command,
        EXPECTED_COMMAND[None, :],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError(
            f"command is not fixed [0.40,0,0] in {path}"
        )

    return data


def cross_track_first_window(
    data: dict[str, np.ndarray],
) -> np.ndarray:
    if WINDOW_SAMPLES != 500:
        raise RuntimeError(
            f"internal window sizing error: "
            f"WINDOW_SAMPLES={WINDOW_SAMPLES}"
        )

    if len(data["time_s"]) < WINDOW_SAMPLES:
        raise RuntimeError(
            f"not enough samples for {WINDOW_S:.1f}s window"
        )

    # The evaluator records one sample per 0.02 s step.
    # Therefore the first 500 samples are exactly 10.0 s of measured data,
    # regardless of whether the first timestamp is 0.00 s or 0.02 s.
    sl = slice(0, WINDOW_SAMPLES)

    x = data["world_x_m"][sl]
    y = data["world_y_m"][sl]

    heading0 = float(data["heading_rad"][0])

    dx = x - x[0]
    dy = y - y[0]

    lateral_axis_x = -math.sin(heading0)
    lateral_axis_y = math.cos(heading0)

    return lateral_axis_x * dx + lateral_axis_y * dy


all_abs_cross = []
per_run = []

print("=" * 118)
print("v4-K GEOMETRIC CROSS-TRACK SIZING — FIRST 10 s OF FROZEN v4-B TRACES")
print("=" * 118)
print("Window    : first 500 samples = 10.0 s at verified dt=0.02 s")
print("Reason    : match the nominal 10 s command-resampling horizon")
print("Cost form : 1 - exp(-(cross_track / sigma)^2)")
print("Scope     : seed42 canonical + dev seeds 100-109; NOT holdout")
print()

for label, path in CASES:
    data = load_csv(path)
    cross = cross_track_first_window(data)
    abs_cross = np.abs(cross)

    all_abs_cross.append(abs_cross)

    per_run.append(
        (
            label,
            float(cross[-1]),
            float(np.mean(abs_cross)),
            float(np.percentile(abs_cross, 95.0)),
        )
    )

print(
    "case    | cross @ 500th sample [m] | "
    "mean|x| [m] | p95|x| [m]"
)

for label, final_cross, mean_abs, p95_abs in per_run:
    print(
        f"{label:7s} | "
        f"{final_cross:+22.4f} | "
        f"{mean_abs:11.4f} | "
        f"{p95_abs:10.4f}"
    )

pooled = np.concatenate(all_abs_cross)

print()
print("POOLED FIRST-10s")
print(f"samples            : {pooled.size}")
print(
    f"mean |cross-track| : "
    f"{float(np.mean(pooled)):.5f} m"
)
print(
    f"RMS cross-track    : "
    f"{float(np.sqrt(np.mean(np.square(pooled)))):.5f} m"
)

for percentile in (50, 75, 90, 95, 99):
    print(
        f"p{percentile:02d} |cross-track|  : "
        f"{float(np.percentile(pooled, percentile)):.5f} m"
    )

print(
    f"max |cross-track|  : "
    f"{float(np.max(pooled)):.5f} m"
)

print()
print("BOUNDED COST — FIRST 10s")
print(
    "sigma [m] | mean cost | p50 | p90 | p95 | p99 | frac>0.90"
)

for sigma in SIGMAS_M:
    cost = 1.0 - np.exp(
        -np.square(pooled / sigma)
    )

    print(
        f"{sigma:9.2f} | "
        f"{float(np.mean(cost)):9.5f} | "
        f"{float(np.percentile(cost, 50.0)):.5f} | "
        f"{float(np.percentile(cost, 90.0)):.5f} | "
        f"{float(np.percentile(cost, 95.0)):.5f} | "
        f"{float(np.percentile(cost, 99.0)):.5f} | "
        f"{float(np.mean(cost > 0.90)):.5f}"
    )

print()
print("EFFECTIVE REWARD — CONDITIONED ON STRAIGHT GATE")
print(
    "sigma [m] | weight | mean reward/step | p95 |reward|"
)

for sigma in SIGMAS_M:
    cost = 1.0 - np.exp(
        -np.square(pooled / sigma)
    )

    for weight in WEIGHTS:
        reward = weight * cost

        print(
            f"{sigma:9.2f} | "
            f"{weight:6.2f} | "
            f"{float(np.mean(reward)):+16.7f} | "
            f"{float(np.percentile(np.abs(reward), 95.0)):.7f}"
        )

print()
print("Repository/source modification : NO")
print("V4K_CROSS_TRACK_REWARD_SIZING_10S_PASS")
print("=" * 118)
