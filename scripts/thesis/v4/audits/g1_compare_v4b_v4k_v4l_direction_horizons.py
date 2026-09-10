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

HORIZONS = (
    (5.0, 250),
    (10.0, 500),
    (15.0, 750),
    (20.0, 1000),
)

REQUIRED = (
    "time_s",
    "cmd_vx_mps",
    "cmd_vy_mps",
    "cmd_wz_radps",
    "world_x_m",
    "world_y_m",
    "heading_rad",
)


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
        command,
        EXPECTED_COMMAND[None, :],
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(f"command is not fixed [0.40,0,0]: {path}")

    return data


def horizon_metrics(data: dict[str, np.ndarray]):
    x = data["world_x_m"]
    y = data["world_y_m"]

    heading = np.unwrap(data["heading_rad"])
    h0 = float(heading[0])

    c0 = math.cos(h0)
    s0 = math.sin(h0)

    out_rows = []

    for nominal_s, sample_count in HORIZONS:
        i = sample_count - 1

        dx = float(x[i] - x[0])
        dy = float(y[i] - y[0])

        forward = c0 * dx + s0 * dy
        cross = -s0 * dx + c0 * dy
        herr_deg = math.degrees(float(heading[i] - h0))

        path_length = float(
            np.sum(
                np.sqrt(
                    np.square(np.diff(x[:sample_count]))
                    + np.square(np.diff(y[:sample_count]))
                )
            )
        )

        out_rows.append(
            {
                "t": nominal_s,
                "cross": cross,
                "forward": forward,
                "cross_pct": (
                    100.0 * abs(cross) / abs(forward)
                    if abs(forward) > 1e-12
                    else float("nan")
                ),
                "herr_deg": herr_deg,
                "straightness": (
                    forward / path_length
                    if path_length > 1e-12
                    else float("nan")
                ),
            }
        )

    return out_rows


results = {
    label: horizon_metrics(load(path))
    for label, path in CSV.items()
}

# Established final closure sanity guards.
expected_final = {
    "v4-B": (-0.3243, -1.439),
    "v4-K": (-0.5809, -13.145),
    "v4-L": (+0.8452, +7.800),
}

for label, (cross_expected, heading_expected) in expected_final.items():
    row = results[label][-1]

    if abs(row["cross"] - cross_expected) > 5e-4:
        raise RuntimeError(
            f"{label} final cross sanity failed: {row['cross']}"
        )

    if abs(row["herr_deg"] - heading_expected) > 0.01:
        raise RuntimeError(
            f"{label} final heading sanity failed: {row['herr_deg']}"
        )


print("=" * 142)
print("v4-B vs v4-K vs v4-L DIRECTIONAL EVOLUTION BY HORIZON — SEED42 — [0.40,0,0]")
print("=" * 142)
print("Frame                : initial measured heading of each run")
print("Horizons             : 250/500/750/1000 samples at dt=0.02")
print("Final closure sanity : v4-B PASS / v4-K PASS / v4-L PASS")
print()

print(
    "time | "
    "B cross [m] | K cross [m] | L cross [m] | "
    "B herr [deg] | K herr [deg] | L herr [deg] | "
    "B cross% | K cross% | L cross%"
)
print("-" * 142)

for i in range(len(HORIZONS)):
    b = results["v4-B"][i]
    k = results["v4-K"][i]
    l = results["v4-L"][i]

    print(
        f"{b['t']:4.0f}s | "
        f"{b['cross']:+11.4f} | "
        f"{k['cross']:+11.4f} | "
        f"{l['cross']:+11.4f} | "
        f"{b['herr_deg']:+12.3f} | "
        f"{k['herr_deg']:+12.3f} | "
        f"{l['herr_deg']:+12.3f} | "
        f"{b['cross_pct']:8.3f}% | "
        f"{k['cross_pct']:8.3f}% | "
        f"{l['cross_pct']:8.3f}%"
    )

print()
print("-" * 142)
print("v4-L INCREMENTAL 5 s BLOCKS")
print("-" * 142)
print(
    "block | dCross L [m] | dHeading L [deg] | "
    "dForward L [m] | L straightness"
)

prev_cross = 0.0
prev_heading = 0.0
prev_forward = 0.0
prev_t = 0

for row in results["v4-L"]:
    print(
        f"{prev_t:02d}-{int(row['t']):02d}s | "
        f"{row['cross'] - prev_cross:+12.4f} | "
        f"{row['herr_deg'] - prev_heading:+16.3f} | "
        f"{row['forward'] - prev_forward:+13.4f} | "
        f"{row['straightness']:.6f}"
    )
    prev_cross = row["cross"]
    prev_heading = row["herr_deg"]
    prev_forward = row["forward"]
    prev_t = int(row["t"])

print()
print("-" * 142)
print("10 s TRAINING-HORIZON CHECK")
print("-" * 142)

b10 = results["v4-B"][1]
k10 = results["v4-K"][1]
l10 = results["v4-L"][1]

print(f"v4-B cross / heading @10s : {b10['cross']:+.4f} m / {b10['herr_deg']:+.3f} deg")
print(f"v4-K cross / heading @10s : {k10['cross']:+.4f} m / {k10['herr_deg']:+.3f} deg")
print(f"v4-L cross / heading @10s : {l10['cross']:+.4f} m / {l10['herr_deg']:+.3f} deg")
print(
    "v4-L |cross| change vs B  : "
    f"{abs(l10['cross']) - abs(b10['cross']):+.4f} m"
)
print(
    "v4-L |heading| change vs B: "
    f"{abs(l10['herr_deg']) - abs(b10['herr_deg']):+.3f} deg"
)

if (
    abs(l10["cross"]) < abs(b10["cross"])
    and abs(l10["herr_deg"]) <= abs(b10["herr_deg"]) + 1.0
):
    verdict = (
        "TRAINING-HORIZON BENEFIT: v4-L is better than v4-B in cross-track "
        "without material heading degradation at ~10 s."
    )
elif abs(l10["cross"]) < abs(b10["cross"]):
    verdict = (
        "MIXED AT TRAINING HORIZON: v4-L cross-track is better at ~10 s, "
        "but heading is already materially worse."
    )
else:
    verdict = (
        "TRAINING-HORIZON FAIL: v4-L cross-track is already no better than "
        "v4-B by ~10 s."
    )

print(f"automatic horizon verdict      : {verdict}")
print()
print("V4BKL_DIRECTION_HORIZONS_PASS")
print("=" * 142)
