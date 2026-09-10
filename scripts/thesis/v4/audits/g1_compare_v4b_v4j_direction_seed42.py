from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np


REPO = Path(r"C:\humanoid_rl_lab")

CASES = {
    "v4-B model1499": (
        REPO
        / "logs"
        / "evaluations"
        / "g1"
        / "v4b_model1499_play_equivalent_seed42_vx040_20s_canonical_v1"
        / "samples.csv"
    ),
    "v4-J model1499": (
        REPO
        / "logs"
        / "evaluations"
        / "g1"
        / "v4j_model1499_play_equivalent_seed42_vx040_20s_canonical_v1"
        / "samples.csv"
    ),
}

REQUIRED_COLUMNS = (
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
)

EXPECTED_SAMPLES = 1000
EXPECTED_DT = 0.02
EXPECTED_COMMAND = np.array([0.40, 0.0, 0.0], dtype=np.float64)


def load_csv(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise RuntimeError(f"missing CSV: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {path}")

        missing = [name for name in REQUIRED_COLUMNS if name not in reader.fieldnames]
        if missing:
            raise RuntimeError(f"missing required columns in {path}: {missing}")

        rows = list(reader)

    if len(rows) != EXPECTED_SAMPLES:
        raise RuntimeError(
            f"unexpected sample count in {path}: {len(rows)} != {EXPECTED_SAMPLES}"
        )

    data = {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        for name in REQUIRED_COLUMNS
    }

    dt = np.diff(data["time_s"])
    if not np.allclose(dt, EXPECTED_DT, rtol=0.0, atol=1.0e-9):
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
    if not np.allclose(command, EXPECTED_COMMAND[None, :], rtol=0.0, atol=1.0e-12):
        raise RuntimeError(f"command is not fixed [0.40,0,0] in {path}")

    return data


def metrics(data: dict[str, np.ndarray]) -> dict[str, float]:
    x = data["world_x_m"]
    y = data["world_y_m"]

    dx = float(x[-1] - x[0])
    dy = float(y[-1] - y[0])

    heading = np.unwrap(data["heading_rad"])
    h0 = float(heading[0])
    heading_error = heading - h0

    c = math.cos(h0)
    s = math.sin(h0)

    forward_initial = c * dx + s * dy
    cross_track_initial = -s * dx + c * dy

    segment_dx = np.diff(x)
    segment_dy = np.diff(y)
    path_length = float(np.sum(np.hypot(segment_dx, segment_dy)))

    return {
        "mean_vx": float(np.mean(data["base_vx_mps"])),
        "mean_vy": float(np.mean(data["base_vy_mps"])),
        "mean_wz": float(np.mean(data["base_wz_radps"])),
        "initial_heading_deg": math.degrees(h0),
        "final_heading_error_deg": math.degrees(float(heading_error[-1])),
        "mean_abs_heading_error_deg": math.degrees(float(np.mean(np.abs(heading_error)))),
        "rms_heading_error_deg": math.degrees(
            float(np.sqrt(np.mean(np.square(heading_error))))
        ),
        "p95_abs_heading_error_deg": math.degrees(
            float(np.percentile(np.abs(heading_error), 95.0))
        ),
        "max_abs_heading_error_deg": math.degrees(
            float(np.max(np.abs(heading_error)))
        ),
        "world_dx": dx,
        "world_dy": dy,
        "initial_frame_forward": forward_initial,
        "initial_frame_cross_track": cross_track_initial,
        "initial_frame_cross_track_ratio_pct": (
            100.0 * abs(cross_track_initial) / abs(forward_initial)
            if abs(forward_initial) > 1.0e-12
            else float("nan")
        ),
        "path_length": path_length,
        "initial_frame_straightness": (
            forward_initial / path_length if path_length > 1.0e-12 else float("nan")
        ),
    }


loaded = {name: load_csv(path) for name, path in CASES.items()}
results = {name: metrics(data) for name, data in loaded.items()}

b = results["v4-B model1499"]
j = results["v4-J model1499"]

print("=" * 116)
print("v4-B vs v4-J DIRECTIONAL CLOSURE — SEED42 — [0.40, 0, 0]")
print("=" * 116)
print("Protocol                 : 1000 samples, dt=0.02 s, fixed [0.40,0,0]")
print("Cross-track frame        : initial measured heading of each run")
print()

for name in ("v4-B model1499", "v4-J model1499"):
    m = results[name]
    print("-" * 116)
    print(name)
    print("-" * 116)
    print(f"mean body vx                         : {m['mean_vx']:+.6f} m/s")
    print(f"mean body vy                         : {m['mean_vy']:+.6f} m/s")
    print(f"mean body wz                         : {m['mean_wz']:+.6f} rad/s")
    print(f"initial heading                      : {m['initial_heading_deg']:+.3f} deg")
    print(f"final heading error                  : {m['final_heading_error_deg']:+.3f} deg")
    print(f"mean |heading error|                 : {m['mean_abs_heading_error_deg']:.3f} deg")
    print(f"RMS heading error                    : {m['rms_heading_error_deg']:.3f} deg")
    print(f"p95 |heading error|                  : {m['p95_abs_heading_error_deg']:.3f} deg")
    print(f"max |heading error|                  : {m['max_abs_heading_error_deg']:.3f} deg")
    print(f"world dx / dy                        : {m['world_dx']:+.4f} / {m['world_dy']:+.4f} m")
    print(f"initial-frame forward                : {m['initial_frame_forward']:+.4f} m")
    print(f"initial-frame cross-track            : {m['initial_frame_cross_track']:+.4f} m")
    print(f"|cross-track| / |forward|            : {m['initial_frame_cross_track_ratio_pct']:.3f} %")
    print(f"path length                          : {m['path_length']:.4f} m")
    print(f"initial-frame straightness           : {m['initial_frame_straightness']:.6f}")

print()
print("-" * 116)
print("v4-J minus v4-B / improvement summary")
print("-" * 116)
print(f"delta mean vx                        : {j['mean_vx'] - b['mean_vx']:+.6f} m/s")
print(f"delta mean vy                        : {j['mean_vy'] - b['mean_vy']:+.6f} m/s")
print(f"delta mean wz                        : {j['mean_wz'] - b['mean_wz']:+.6f} rad/s")
print(
    "cross-track magnitude change         : "
    f"{abs(j['initial_frame_cross_track']) - abs(b['initial_frame_cross_track']):+.4f} m"
)
print(
    "cross-track magnitude reduction      : "
    f"{100.0 * (1.0 - abs(j['initial_frame_cross_track']) / abs(b['initial_frame_cross_track'])):+.2f} %"
)
print(
    "mean |heading error| change           : "
    f"{j['mean_abs_heading_error_deg'] - b['mean_abs_heading_error_deg']:+.3f} deg"
)
print(
    "final |heading error| change          : "
    f"{abs(j['final_heading_error_deg']) - abs(b['final_heading_error_deg']):+.3f} deg"
)
print(
    "initial-frame straightness change     : "
    f"{j['initial_frame_straightness'] - b['initial_frame_straightness']:+.6f}"
)
print()
print("V4BJ_DIRECTIONAL_CLOSURE_PASS")
print("=" * 116)
