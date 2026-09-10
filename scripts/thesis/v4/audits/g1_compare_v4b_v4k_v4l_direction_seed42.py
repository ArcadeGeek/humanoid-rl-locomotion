from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np


REPO = Path(r"C:\humanoid_rl_lab")

CSV = {
    "v4-B": (
        REPO / "logs" / "evaluations" / "g1"
        / "v4b_model1499_play_equivalent_seed42_vx040_20s_canonical_v1"
        / "samples.csv"
    ),
    "v4-K": (
        REPO / "logs" / "evaluations" / "g1"
        / "v4k_model1499_play_equivalent_seed42_vx040_20s_canonical_v1"
        / "samples.csv"
    ),
    "v4-L": (
        REPO / "logs" / "evaluations" / "g1"
        / "v4l_model1499_play_equivalent_seed42_vx040_20s_canonical_v1"
        / "samples.csv"
    ),
}

EXPECTED_SAMPLES = 1000
EXPECTED_DT = 0.02
EXPECTED_COMMAND = np.asarray([0.40, 0.0, 0.0], dtype=np.float64)

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


def load_csv(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise RuntimeError(f"missing CSV: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)

        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {path}")

        missing = [
            name
            for name in REQUIRED_COLUMNS
            if name not in reader.fieldnames
        ]
        if missing:
            raise RuntimeError(
                f"missing required columns in {path}: {missing}"
            )

        rows = list(reader)

    if len(rows) != EXPECTED_SAMPLES:
        raise RuntimeError(
            f"unexpected sample count in {path}: "
            f"{len(rows)} != {EXPECTED_SAMPLES}"
        )

    data = {
        name: np.asarray(
            [float(row[name]) for row in rows],
            dtype=np.float64,
        )
        for name in REQUIRED_COLUMNS
    }

    dt = np.diff(data["time_s"])
    if not np.allclose(
        dt,
        EXPECTED_DT,
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


def metrics(data: dict[str, np.ndarray]) -> dict[str, float]:
    x = data["world_x_m"]
    y = data["world_y_m"]

    dx = float(x[-1] - x[0])
    dy = float(y[-1] - y[0])

    heading = np.unwrap(data["heading_rad"])
    heading0 = float(heading[0])
    heading_error = heading - heading0

    c0 = math.cos(heading0)
    s0 = math.sin(heading0)

    forward = c0 * dx + s0 * dy
    cross = -s0 * dx + c0 * dy

    path_length = float(
        np.sum(
            np.sqrt(
                np.square(np.diff(x))
                + np.square(np.diff(y))
            )
        )
    )

    abs_heading_deg = np.degrees(
        np.abs(heading_error)
    )

    return {
        "mean_vx": float(np.mean(data["base_vx_mps"])),
        "mean_vy": float(np.mean(data["base_vy_mps"])),
        "mean_wz": float(np.mean(data["base_wz_radps"])),
        "heading0_deg": math.degrees(heading0),
        "final_heading_error_deg": math.degrees(float(heading_error[-1])),
        "mean_abs_heading_error_deg": float(np.mean(abs_heading_deg)),
        "rms_heading_error_deg": math.degrees(
            float(np.sqrt(np.mean(np.square(heading_error))))
        ),
        "p95_abs_heading_error_deg": float(np.percentile(abs_heading_deg, 95.0)),
        "max_abs_heading_error_deg": float(np.max(abs_heading_deg)),
        "world_dx": dx,
        "world_dy": dy,
        "forward": forward,
        "cross": cross,
        "cross_ratio_pct": (
            100.0 * abs(cross) / abs(forward)
            if abs(forward) > 1.0e-12
            else float("nan")
        ),
        "path_length": path_length,
        "straightness": (
            forward / path_length
            if path_length > 1.0e-12
            else float("nan")
        ),
    }


results = {
    label: metrics(load_csv(path))
    for label, path in CSV.items()
}

# Established closures as scientific sanity guards.
checks = {
    ("v4-B", "mean_vx"): (0.382391, 5e-6),
    ("v4-B", "mean_vy"): (-0.011966, 5e-6),
    ("v4-B", "mean_wz"): (-0.007202, 5e-6),
    ("v4-B", "cross"): (-0.3243, 5e-4),
    ("v4-B", "final_heading_error_deg"): (-1.439, 0.01),
    ("v4-K", "cross"): (-0.5809, 5e-4),
    ("v4-K", "final_heading_error_deg"): (-13.145, 0.01),
}

for (label, key), (expected, tol) in checks.items():
    actual = results[label][key]
    if abs(actual - expected) > tol:
        raise RuntimeError(
            f"{label} sanity guard failed for {key}: "
            f"actual={actual}, expected≈{expected}"
        )


print("=" * 128)
print("v4-B vs v4-K vs v4-L DIRECTIONAL CLOSURE — SEED42 — [0.40,0,0]")
print("=" * 128)
print("Protocol           : 1000 samples, dt=0.02 s, fixed [0.40,0,0]")
print("Cross-track frame  : initial measured heading of each run")
print("v4-B/v4-K sanity   : PASS")
print()

for label in ("v4-B", "v4-K", "v4-L"):
    m = results[label]
    print("-" * 128)
    print(label)
    print("-" * 128)
    print(f"mean body vx                         : {m['mean_vx']:+.6f} m/s")
    print(f"mean body vy                         : {m['mean_vy']:+.6f} m/s")
    print(f"mean body wz                         : {m['mean_wz']:+.6f} rad/s")
    print(f"initial heading                      : {m['heading0_deg']:+.3f} deg")
    print(f"final heading error                  : {m['final_heading_error_deg']:+.3f} deg")
    print(f"mean |heading error|                 : {m['mean_abs_heading_error_deg']:.3f} deg")
    print(f"RMS heading error                    : {m['rms_heading_error_deg']:.3f} deg")
    print(f"p95 |heading error|                  : {m['p95_abs_heading_error_deg']:.3f} deg")
    print(f"max |heading error|                  : {m['max_abs_heading_error_deg']:.3f} deg")
    print(f"world dx / dy                        : {m['world_dx']:+.4f} / {m['world_dy']:+.4f} m")
    print(f"initial-frame forward                : {m['forward']:+.4f} m")
    print(f"initial-frame cross-track            : {m['cross']:+.4f} m")
    print(f"|cross-track| / |forward|            : {m['cross_ratio_pct']:.3f} %")
    print(f"path length                          : {m['path_length']:.4f} m")
    print(f"initial-frame straightness           : {m['straightness']:.6f}")

b = results["v4-B"]
k = results["v4-K"]
l = results["v4-L"]

print()
print("-" * 128)
print("v4-L improvement summary")
print("-" * 128)
print(f"delta mean vx vs v4-B                : {l['mean_vx'] - b['mean_vx']:+.6f} m/s")
print(f"delta mean vy vs v4-B                : {l['mean_vy'] - b['mean_vy']:+.6f} m/s")
print(f"delta mean wz vs v4-B                : {l['mean_wz'] - b['mean_wz']:+.6f} rad/s")
print(
    "cross magnitude change vs v4-B       : "
    f"{abs(l['cross']) - abs(b['cross']):+.4f} m"
)
print(
    "cross magnitude reduction vs v4-B    : "
    f"{100.0 * (abs(b['cross']) - abs(l['cross'])) / abs(b['cross']):+.2f} %"
)
print(
    "mean |heading| change vs v4-B         : "
    f"{l['mean_abs_heading_error_deg'] - b['mean_abs_heading_error_deg']:+.3f} deg"
)
print(
    "final |heading| change vs v4-B        : "
    f"{abs(l['final_heading_error_deg']) - abs(b['final_heading_error_deg']):+.3f} deg"
)
print(
    "straightness change vs v4-B           : "
    f"{l['straightness'] - b['straightness']:+.6f}"
)
print(
    "cross magnitude reduction vs v4-K    : "
    f"{100.0 * (abs(k['cross']) - abs(l['cross'])) / abs(k['cross']):+.2f} %"
)
print(
    "mean |heading| improvement vs v4-K    : "
    f"{k['mean_abs_heading_error_deg'] - l['mean_abs_heading_error_deg']:+.3f} deg"
)
print()
print("V4BKL_DIRECTIONAL_CLOSURE_PASS")
print("=" * 128)
