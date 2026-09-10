from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np


REPO = Path(r"C:\humanoid_rl_lab")

CSV_B = (
    REPO
    / "logs"
    / "evaluations"
    / "g1"
    / "v4b_model1499_play_equivalent_seed42_vx040_20s_canonical_v1"
    / "samples.csv"
)

CSV_K = (
    REPO
    / "logs"
    / "evaluations"
    / "g1"
    / "v4k_model1499_play_equivalent_seed42_vx040_20s_canonical_v1"
    / "samples.csv"
)

EXPECTED_SAMPLES = 1000
EXPECTED_DT = 0.02
EXPECTED_COMMAND = np.asarray([0.40, 0.0, 0.0], dtype=np.float64)

# Nominal protocol horizons by recorded sample count.
# 250/500/750/1000 samples at dt=0.02 correspond to the 5/10/15/20 s
# reporting horizons used by the canonical evaluator.
HORIZONS = (
    (5.0, 250),
    (10.0, 500),
    (15.0, 750),
    (20.0, 1000),
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


def horizon_metrics(
    data: dict[str, np.ndarray],
) -> list[dict[str, float]]:
    x = data["world_x_m"]
    y = data["world_y_m"]

    heading = np.unwrap(data["heading_rad"])
    heading0 = float(heading[0])

    c0 = math.cos(heading0)
    s0 = math.sin(heading0)

    rows = []

    for nominal_s, sample_count in HORIZONS:
        index = sample_count - 1

        dx = float(x[index] - x[0])
        dy = float(y[index] - y[0])

        forward = c0 * dx + s0 * dy
        cross = -s0 * dx + c0 * dy

        heading_error_deg = math.degrees(
            float(heading[index] - heading0)
        )

        path_length = float(
            np.sum(
                np.sqrt(
                    np.square(np.diff(x[:sample_count]))
                    + np.square(np.diff(y[:sample_count]))
                )
            )
        )

        rows.append(
            {
                "nominal_s": nominal_s,
                "sample_count": sample_count,
                "csv_time_s": float(data["time_s"][index]),
                "forward_m": forward,
                "cross_m": cross,
                "cross_ratio_pct": (
                    100.0 * abs(cross) / abs(forward)
                    if abs(forward) > 1.0e-12
                    else float("nan")
                ),
                "heading_error_deg": heading_error_deg,
                "path_length_m": path_length,
                "straightness": (
                    forward / path_length
                    if path_length > 1.0e-12
                    else float("nan")
                ),
            }
        )

    return rows


def segment_deltas(
    rows: list[dict[str, float]],
) -> list[dict[str, float]]:
    result = []

    previous_cross = 0.0
    previous_heading = 0.0
    previous_forward = 0.0

    for row in rows:
        result.append(
            {
                "nominal_s": row["nominal_s"],
                "delta_cross_m": row["cross_m"] - previous_cross,
                "delta_heading_deg": (
                    row["heading_error_deg"] - previous_heading
                ),
                "delta_forward_m": (
                    row["forward_m"] - previous_forward
                ),
            }
        )
        previous_cross = row["cross_m"]
        previous_heading = row["heading_error_deg"]
        previous_forward = row["forward_m"]

    return result


b = horizon_metrics(load_csv(CSV_B))
k = horizon_metrics(load_csv(CSV_K))

# Final-horizon scientific sanity guards must reproduce established closure.
b_final = b[-1]
k_final = k[-1]

if abs(b_final["cross_m"] - (-0.3243)) > 5.0e-4:
    raise RuntimeError(
        "v4-B final cross-track sanity guard failed: "
        f"{b_final['cross_m']}"
    )

if abs(b_final["heading_error_deg"] - (-1.439)) > 0.01:
    raise RuntimeError(
        "v4-B final heading sanity guard failed: "
        f"{b_final['heading_error_deg']}"
    )

if abs(k_final["cross_m"] - (-0.5809)) > 5.0e-4:
    raise RuntimeError(
        "v4-K final cross-track sanity guard failed: "
        f"{k_final['cross_m']}"
    )

if abs(k_final["heading_error_deg"] - (-13.145)) > 0.01:
    raise RuntimeError(
        "v4-K final heading sanity guard failed: "
        f"{k_final['heading_error_deg']}"
    )

b_segments = segment_deltas(b)
k_segments = segment_deltas(k)

print("=" * 138)
print("v4-B vs v4-K DIRECTIONAL EVOLUTION BY HORIZON — SEED42 — [0.40,0,0]")
print("=" * 138)
print("Frame                 : initial measured heading of each run")
print("Horizons              : first 250/500/750/1000 recorded samples at dt=0.02")
print("Purpose               : locate when v4-K directional degradation emerges")
print("Final closure sanity  : v4-B PASS / v4-K PASS")
print()

print("-" * 138)
print("CUMULATIVE METRICS")
print("-" * 138)
print(
    "time | B cross [m] | K cross [m] | "
    "|K|-|B| [m] | B herr [deg] | K herr [deg] | "
    "B cross% | K cross%"
)

for br, kr in zip(b, k):
    print(
        f"{br['nominal_s']:4.0f}s | "
        f"{br['cross_m']:+11.4f} | "
        f"{kr['cross_m']:+11.4f} | "
        f"{abs(kr['cross_m']) - abs(br['cross_m']):+11.4f} | "
        f"{br['heading_error_deg']:+12.3f} | "
        f"{kr['heading_error_deg']:+12.3f} | "
        f"{br['cross_ratio_pct']:8.3f}% | "
        f"{kr['cross_ratio_pct']:8.3f}%"
    )

print()
print("-" * 138)
print("CUMULATIVE FORWARD / STRAIGHTNESS")
print("-" * 138)
print(
    "time | B forward [m] | K forward [m] | "
    "B straightness | K straightness | CSV time B/K [s]"
)

for br, kr in zip(b, k):
    print(
        f"{br['nominal_s']:4.0f}s | "
        f"{br['forward_m']:+13.4f} | "
        f"{kr['forward_m']:+13.4f} | "
        f"{br['straightness']:14.6f} | "
        f"{kr['straightness']:14.6f} | "
        f"{br['csv_time_s']:.4f}/{kr['csv_time_s']:.4f}"
    )

print()
print("-" * 138)
print("INCREMENTAL CHANGE WITHIN EACH NOMINAL 5 s BLOCK")
print("-" * 138)
print(
    "block | dCross B [m] | dCross K [m] | "
    "dHeading B [deg] | dHeading K [deg] | "
    "dForward B [m] | dForward K [m]"
)

previous_t = 0
for bs, ks in zip(b_segments, k_segments):
    current_t = int(bs["nominal_s"])
    print(
        f"{previous_t:02d}-{current_t:02d}s | "
        f"{bs['delta_cross_m']:+12.4f} | "
        f"{ks['delta_cross_m']:+12.4f} | "
        f"{bs['delta_heading_deg']:+16.3f} | "
        f"{ks['delta_heading_deg']:+16.3f} | "
        f"{bs['delta_forward_m']:+13.4f} | "
        f"{ks['delta_forward_m']:+13.4f}"
    )
    previous_t = current_t

print()
print("-" * 138)
print("10 s TRAINING-HORIZON CHECK")
print("-" * 138)

b10 = b[1]
k10 = k[1]

print(f"v4-B cross-track @10s             : {b10['cross_m']:+.4f} m")
print(f"v4-K cross-track @10s             : {k10['cross_m']:+.4f} m")
print(
    "v4-K minus v4-B |cross| @10s     : "
    f"{abs(k10['cross_m']) - abs(b10['cross_m']):+.4f} m"
)
print(f"v4-B heading error @10s           : {b10['heading_error_deg']:+.3f} deg")
print(f"v4-K heading error @10s           : {k10['heading_error_deg']:+.3f} deg")
print(
    "v4-K minus v4-B |heading| @10s   : "
    f"{abs(k10['heading_error_deg']) - abs(b10['heading_error_deg']):+.3f} deg"
)

if (
    abs(k10["cross_m"]) < abs(b10["cross_m"])
    and abs(k10["heading_error_deg"]) <= abs(b10["heading_error_deg"]) + 1.0
):
    verdict = (
        "TRAINING-HORIZON BENEFIT: v4-K is better at ~10 s; "
        "later divergence deserves separate horizon investigation."
    )
elif abs(k10["cross_m"]) < abs(b10["cross_m"]):
    verdict = (
        "MIXED AT TRAINING HORIZON: cross-track is better at ~10 s, "
        "but heading is already materially worse."
    )
else:
    verdict = (
        "TRAINING-HORIZON FAIL: v4-K cross-track is already no better "
        "than v4-B by ~10 s."
    )

print(f"automatic horizon verdict           : {verdict}")
print()
print("V4BK_DIRECTION_HORIZONS_PASS")
print("=" * 138)
