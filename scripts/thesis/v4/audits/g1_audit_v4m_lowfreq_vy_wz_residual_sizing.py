from __future__ import annotations

import csv
import itertools
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
FIRST_10S_SAMPLES = 500
EXPECTED_COMMAND = np.asarray([0.40, 0.0, 0.0], dtype=np.float64)

REQUIRED = (
    "time_s",
    "cmd_vx_mps",
    "cmd_vy_mps",
    "cmd_wz_radps",
    "base_vy_mps",
    "base_wz_radps",
)

TAU_S = (0.50, 0.75, 1.00, 1.50)
SIGMA_VY_MPS = (0.010, 0.015, 0.020, 0.025, 0.030)
SIGMA_WZ_RADPS = (0.010, 0.015, 0.020, 0.025, 0.030)
WEIGHTS = (-0.05, -0.10, -0.15)


def load(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise RuntimeError(f"missing CSV: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise RuntimeError(f"missing CSV header: {path}")

        missing = [name for name in REQUIRED if name not in reader.fieldnames]
        if missing:
            raise RuntimeError(
                f"missing required columns in {path}: {missing}"
            )

        rows = list(reader)

    if len(rows) != EXPECTED_SAMPLES:
        raise RuntimeError(
            f"expected {EXPECTED_SAMPLES} samples, got {len(rows)}: {path}"
        )

    data = {
        name: np.asarray(
            [float(row[name]) for row in rows],
            dtype=np.float64,
        )
        for name in REQUIRED
    }

    dt = np.diff(data["time_s"])
    if not np.allclose(
        dt,
        EXPECTED_DT,
        rtol=0.0,
        atol=1.0e-9,
    ):
        raise RuntimeError(
            f"unexpected/nonuniform dt in {path}"
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


def ema_zero_initialized(
    values: np.ndarray,
    tau_s: float,
    dt_s: float,
) -> np.ndarray:
    if tau_s <= 0.0:
        raise ValueError("tau_s must be > 0")

    alpha = 1.0 - math.exp(-dt_s / tau_s)

    result = np.zeros_like(
        values,
        dtype=np.float64,
    )

    state = 0.0

    for index, value in enumerate(values):
        state = (
            (1.0 - alpha) * state
            + alpha * float(value)
        )
        result[index] = state

    return result


def bounded_cost(
    ema_vy: np.ndarray,
    ema_wz: np.ndarray,
    sigma_vy: float,
    sigma_wz: float,
) -> np.ndarray:
    exponent = (
        np.square(ema_vy / sigma_vy)
        + np.square(ema_wz / sigma_wz)
    )

    return 1.0 - np.exp(-exponent)


DATA = {
    label: load(path)
    for label, path in CSV.items()
}


known_means = {
    "v4-B": (-0.011966, -0.007202),
    "v4-K": (+0.012441, -0.018085),
    "v4-L": (+0.019652, +0.014104),
}

for label, (vy_expected, wz_expected) in known_means.items():
    vy_actual = float(np.mean(DATA[label]["base_vy_mps"]))
    wz_actual = float(np.mean(DATA[label]["base_wz_radps"]))

    if abs(vy_actual - vy_expected) > 5.0e-6:
        raise RuntimeError(
            f"{label} mean vy sanity failed: "
            f"{vy_actual} vs {vy_expected}"
        )

    if abs(wz_actual - wz_expected) > 5.0e-6:
        raise RuntimeError(
            f"{label} mean wz sanity failed: "
            f"{wz_actual} vs {wz_expected}"
        )


print("=" * 154)
print("v4-M LOW-FREQUENCY near-straight vy+wz RESIDUAL SIZING — FROZEN v4-B / v4-K / v4-L")
print("=" * 154)
print("Causal zero-initialized EMA:")
print("  alpha = 1 - exp(-dt/tau), dt=0.02 s")
print("Candidate bounded cost:")
print("  1 - exp(-[(EMA(vy)/sigma_vy)^2 + (EMA(wz)/sigma_wz)^2])")
print("Training-relevant window : first 500 samples = 10 s")
print("No geometric anchor / no path integration")
print("Scope                    : OFFLINE SIZING ONLY — NO TRAINING")
print()

print("-" * 154)
print("EMA LOW-FREQUENCY DISTRIBUTIONS — FIRST 10 s")
print("-" * 154)
print(
    "tau | model | "
    "vy mean signed | vy mean|.| | vy p95|.| | "
    "wz mean signed | wz mean|.| | wz p95|.|"
)

EMA_CACHE = {}

for tau_s in TAU_S:
    for label in ("v4-B", "v4-K", "v4-L"):
        vy = DATA[label]["base_vy_mps"][:FIRST_10S_SAMPLES]
        wz = DATA[label]["base_wz_radps"][:FIRST_10S_SAMPLES]

        ema_vy = ema_zero_initialized(
            vy,
            tau_s,
            EXPECTED_DT,
        )
        ema_wz = ema_zero_initialized(
            wz,
            tau_s,
            EXPECTED_DT,
        )

        EMA_CACHE[(tau_s, label)] = (
            ema_vy,
            ema_wz,
        )

        print(
            f"{tau_s:3.2f} | "
            f"{label:4s} | "
            f"{float(np.mean(ema_vy)):+14.6f} | "
            f"{float(np.mean(np.abs(ema_vy))):10.6f} | "
            f"{float(np.percentile(np.abs(ema_vy),95)):.6f} | "
            f"{float(np.mean(ema_wz)):+14.6f} | "
            f"{float(np.mean(np.abs(ema_wz))):10.6f} | "
            f"{float(np.percentile(np.abs(ema_wz),95)):.6f}"
        )

print()
print("-" * 154)
print("NON-SATURATING GRID SEARCH — FIRST 10 s")
print("-" * 154)
print(
    "Filter: B frac>0.9 <= 0.05, K frac>0.9 <= 0.10, L frac>0.9 <= 0.10"
)
print(
    "Score = min(K/B mean-cost ratio, L/B mean-cost ratio). "
    "Higher is better; tie-break lower B mean cost."
)

rows = []

for tau_s, sigma_vy, sigma_wz in itertools.product(
    TAU_S,
    SIGMA_VY_MPS,
    SIGMA_WZ_RADPS,
):
    costs = {}

    for label in ("v4-B", "v4-K", "v4-L"):
        ema_vy, ema_wz = EMA_CACHE[
            (tau_s, label)
        ]

        costs[label] = bounded_cost(
            ema_vy,
            ema_wz,
            sigma_vy,
            sigma_wz,
        )

    b_mean = float(np.mean(costs["v4-B"]))
    k_mean = float(np.mean(costs["v4-K"]))
    l_mean = float(np.mean(costs["v4-L"]))

    k_ratio = (
        k_mean / b_mean
        if b_mean > 1.0e-12
        else float("inf")
    )
    l_ratio = (
        l_mean / b_mean
        if b_mean > 1.0e-12
        else float("inf")
    )

    row = {
        "tau": tau_s,
        "sigma_vy": sigma_vy,
        "sigma_wz": sigma_wz,
        "b_mean": b_mean,
        "k_mean": k_mean,
        "l_mean": l_mean,
        "k_ratio": k_ratio,
        "l_ratio": l_ratio,
        "score": min(k_ratio, l_ratio),
        "b_p95": float(
            np.percentile(costs["v4-B"], 95.0)
        ),
        "k_p95": float(
            np.percentile(costs["v4-K"], 95.0)
        ),
        "l_p95": float(
            np.percentile(costs["v4-L"], 95.0)
        ),
        "b_sat": float(
            np.mean(costs["v4-B"] > 0.90)
        ),
        "k_sat": float(
            np.mean(costs["v4-K"] > 0.90)
        ),
        "l_sat": float(
            np.mean(costs["v4-L"] > 0.90)
        ),
    }

    rows.append(row)


eligible = [
    row
    for row in rows
    if (
        row["b_sat"] <= 0.05
        and row["k_sat"] <= 0.10
        and row["l_sat"] <= 0.10
    )
]

eligible.sort(
    key=lambda row: (
        row["score"],
        -row["b_mean"],
    ),
    reverse=True,
)

if not eligible:
    print("NO ELIGIBLE NON-SATURATING LOW-FREQUENCY CANDIDATE")
else:
    print(
        "rank | tau | sigma_vy | sigma_wz | "
        "B mean | K mean | L mean | "
        "K/B | L/B | min ratio | "
        "B p95 | K p95 | L p95 | "
        "B sat | K sat | L sat"
    )

    for rank, row in enumerate(
        eligible[:12],
        start=1,
    ):
        print(
            f"{rank:4d} | "
            f"{row['tau']:3.2f} | "
            f"{row['sigma_vy']:8.3f} | "
            f"{row['sigma_wz']:8.3f} | "
            f"{row['b_mean']:6.4f} | "
            f"{row['k_mean']:6.4f} | "
            f"{row['l_mean']:6.4f} | "
            f"{row['k_ratio']:5.2f} | "
            f"{row['l_ratio']:5.2f} | "
            f"{row['score']:9.3f} | "
            f"{row['b_p95']:6.4f} | "
            f"{row['k_p95']:6.4f} | "
            f"{row['l_p95']:6.4f} | "
            f"{row['b_sat']:5.3f} | "
            f"{row['k_sat']:5.3f} | "
            f"{row['l_sat']:5.3f}"
        )

print()
print("-" * 154)
print("TOP-CANDIDATE EFFECTIVE REWARD MAGNITUDES")
print("-" * 154)

if eligible:
    best = eligible[0]

    print(
        "selected-for-report "
        f"tau={best['tau']:.2f} s, "
        f"sigma_vy={best['sigma_vy']:.3f} m/s, "
        f"sigma_wz={best['sigma_wz']:.3f} rad/s"
    )

    print(
        "weight | B mean reward | K mean reward | L mean reward | "
        "B p95 |reward| | K p95 |reward| | L p95 |reward|"
    )

    for weight in WEIGHTS:
        values = {}

        for label in ("v4-B", "v4-K", "v4-L"):
            ema_vy, ema_wz = EMA_CACHE[
                (best["tau"], label)
            ]

            cost = bounded_cost(
                ema_vy,
                ema_wz,
                best["sigma_vy"],
                best["sigma_wz"],
            )

            reward = weight * cost

            values[label] = {
                "mean": float(
                    np.mean(reward)
                ),
                "p95_abs": float(
                    np.percentile(
                        np.abs(reward),
                        95.0,
                    )
                ),
            }

        print(
            f"{weight:6.2f} | "
            f"{values['v4-B']['mean']:+13.6f} | "
            f"{values['v4-K']['mean']:+13.6f} | "
            f"{values['v4-L']['mean']:+13.6f} | "
            f"{values['v4-B']['p95_abs']:.6f} | "
            f"{values['v4-K']['p95_abs']:.6f} | "
            f"{values['v4-L']['p95_abs']:.6f}"
        )

print()
print("V4M_LOWFREQ_VY_WZ_RESIDUAL_SIZING_PASS")
print("=" * 154)
