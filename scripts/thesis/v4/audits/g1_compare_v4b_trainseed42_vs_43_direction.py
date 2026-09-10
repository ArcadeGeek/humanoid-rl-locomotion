from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np


REPO = Path(r"C:\humanoid_rl_lab")

CSV = {
    "train-seed42": (
        REPO / "logs" / "evaluations" / "g1"
        / "v4b_model1499_play_equivalent_seed42_vx040_20s_canonical_v1"
        / "samples.csv"
    ),
    "train-seed43": (
        REPO / "logs" / "evaluations" / "g1"
        / "v4b_trainseed43_model1499_play_equivalent_evalseed42_vx040_20s_canonical_v1"
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

        missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise RuntimeError(f"missing columns in {path}: {missing}")

        rows = list(reader)

    if len(rows) != EXPECTED_SAMPLES:
        raise RuntimeError(
            f"unexpected sample count in {path}: {len(rows)} != {EXPECTED_SAMPLES}"
        )

    data = {
        c: np.asarray([float(row[c]) for row in rows], dtype=np.float64)
        for c in REQUIRED_COLUMNS
    }

    dt = np.diff(data["time_s"])
    if not np.allclose(dt, EXPECTED_DT, rtol=0.0, atol=1e-9):
        raise RuntimeError(
            f"unexpected/nonuniform dt in {path}: median={np.median(dt):.12f}"
        )

    cmd = np.column_stack(
        (data["cmd_vx_mps"], data["cmd_vy_mps"], data["cmd_wz_radps"])
    )
    if not np.allclose(cmd, EXPECTED_COMMAND[None, :], rtol=0.0, atol=1e-12):
        raise RuntimeError(f"command is not fixed [0.40,0,0] in {path}")

    return data


def metrics(data: dict[str, np.ndarray]) -> dict[str, float]:
    x = data["world_x_m"]
    y = data["world_y_m"]

    dx = float(x[-1] - x[0])
    dy = float(y[-1] - y[0])

    heading = np.unwrap(data["heading_rad"])
    h0 = float(heading[0])
    herr = heading - h0

    c0 = math.cos(h0)
    s0 = math.sin(h0)

    forward = c0 * dx + s0 * dy
    cross = -s0 * dx + c0 * dy

    path_length = float(
        np.sum(np.hypot(np.diff(x), np.diff(y)))
    )

    abs_herr_deg = np.degrees(np.abs(herr))

    return {
        "mean_vx": float(np.mean(data["base_vx_mps"])),
        "mean_vy": float(np.mean(data["base_vy_mps"])),
        "mean_wz": float(np.mean(data["base_wz_radps"])),
        "heading0_deg": math.degrees(h0),
        "final_heading_error_deg": math.degrees(float(herr[-1])),
        "mean_abs_heading_error_deg": float(np.mean(abs_herr_deg)),
        "rms_heading_error_deg": math.degrees(
            float(np.sqrt(np.mean(np.square(herr))))
        ),
        "p95_abs_heading_error_deg": float(np.percentile(abs_herr_deg, 95.0)),
        "max_abs_heading_error_deg": float(np.max(abs_herr_deg)),
        "world_dx": dx,
        "world_dy": dy,
        "forward": forward,
        "cross": cross,
        "cross_ratio_pct": (
            100.0 * abs(cross) / abs(forward)
            if abs(forward) > 1e-12
            else float("nan")
        ),
        "path_length": path_length,
        "straightness": (
            forward / path_length
            if path_length > 1e-12
            else float("nan")
        ),
    }


results = {name: metrics(load_csv(path)) for name, path in CSV.items()}

# Canonical seed42 sanity guards.
b = results["train-seed42"]
checks = {
    "mean_vx": (0.382391, 5e-6),
    "mean_vy": (-0.011966, 5e-6),
    "mean_wz": (-0.007202, 5e-6),
    "cross": (-0.3243, 5e-4),
    "final_heading_error_deg": (-1.439, 0.01),
}
for key, (expected, tol) in checks.items():
    actual = b[key]
    if abs(actual - expected) > tol:
        raise RuntimeError(
            f"seed42 sanity guard failed for {key}: "
            f"actual={actual}, expected≈{expected}"
        )

print("=" * 126)
print("v4-B TRAIN-SEED42 vs TRAIN-SEED43 — DIRECTIONAL CLOSURE")
print("=" * 126)
print("Evaluation seed     : 42 for both")
print("Command             : [0.40, 0.0, 0.0]")
print("Samples / dt        : 1000 / 0.02 s")
print("Cross-track frame   : each run's own initial measured heading")
print("Seed42 sanity       : PASS")
print()

for label in ("train-seed42", "train-seed43"):
    m = results[label]
    print("-" * 126)
    print(label)
    print("-" * 126)
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

m42 = results["train-seed42"]
m43 = results["train-seed43"]

print()
print("-" * 126)
print("seed43 - seed42")
print("-" * 126)
print(f"delta mean vx                        : {m43['mean_vx'] - m42['mean_vx']:+.6f} m/s")
print(f"delta mean vy                        : {m43['mean_vy'] - m42['mean_vy']:+.6f} m/s")
print(f"delta mean wz                        : {m43['mean_wz'] - m42['mean_wz']:+.6f} rad/s")
print(f"cross magnitude change               : {abs(m43['cross']) - abs(m42['cross']):+.4f} m")
print(
    "cross magnitude reduction           : "
    f"{100.0 * (abs(m42['cross']) - abs(m43['cross'])) / abs(m42['cross']):+.2f} %"
)
print(
    "mean |heading| change               : "
    f"{m43['mean_abs_heading_error_deg'] - m42['mean_abs_heading_error_deg']:+.3f} deg"
)
print(
    "final |heading| change              : "
    f"{abs(m43['final_heading_error_deg']) - abs(m42['final_heading_error_deg']):+.3f} deg"
)
print(
    "straightness change                 : "
    f"{m43['straightness'] - m42['straightness']:+.6f}"
)
print()
print("V4B_TRAINSEED42_VS_43_DIRECTIONAL_CLOSURE_PASS")
print("=" * 126)
