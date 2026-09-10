from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path


ROOT = Path(r"C:\humanoid_rl_lab\logs\evaluations\g1")

CANONICAL = ROOT / "v4b_model1499_play_equivalent_seed42_vx040_20s_canonical_v1" / "samples.csv"
DEV_PATTERN = "v4b_model1499_play_equivalent_seed*_vx040_20s_direction_dev"

EXPECTED_VX = 0.4
EXPECTED_VY = 0.0
EXPECTED_WZ = 0.0
DT_EXPECTED = 0.02

SIGMAS = (0.02, 0.04, 0.06, 0.08, 0.10)
WEIGHTS = (-0.10, -0.25, -0.50)


def fail(msg: str) -> None:
    raise RuntimeError(msg)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 0:
        fail(f"invalid moving-average window: {window}")
    if len(values) < window:
        return []
    out: list[float] = []
    acc = sum(values[:window])
    out.append(acc / window)
    for i in range(window, len(values)):
        acc += values[i] - values[i - window]
        out.append(acc / window)
    return out


def stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "mean": float("nan"),
            "mean_abs": float("nan"),
            "rms": float("nan"),
            "p95_abs": float("nan"),
            "max_abs": float("nan"),
            "negative_fraction": float("nan"),
        }
    abs_values = [abs(x) for x in values]
    return {
        "mean": statistics.fmean(values),
        "mean_abs": statistics.fmean(abs_values),
        "rms": math.sqrt(statistics.fmean(x * x for x in values)),
        "p95_abs": percentile(abs_values, 0.95),
        "max_abs": max(abs_values),
        "negative_fraction": sum(x < 0.0 for x in values) / len(values),
    }


def bounded_cost(error: float, sigma: float) -> float:
    return 1.0 - math.exp(-((error / sigma) ** 2))


def read_csv(path: Path) -> dict[str, object]:
    if not path.is_file():
        fail(f"missing CSV: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))

    if not rows:
        fail(f"empty CSV: {path}")

    required = {
        "time_s",
        "cmd_vx_mps",
        "cmd_vy_mps",
        "cmd_wz_radps",
        "base_vy_mps",
    }
    missing = sorted(required - set(rows[0].keys()))
    if missing:
        fail(f"{path}: missing required columns {missing}")

    t = [float(r["time_s"]) for r in rows]
    cmd_vx = [float(r["cmd_vx_mps"]) for r in rows]
    cmd_vy = [float(r["cmd_vy_mps"]) for r in rows]
    cmd_wz = [float(r["cmd_wz_radps"]) for r in rows]
    vy = [float(r["base_vy_mps"]) for r in rows]

    if len(t) < 2:
        fail(f"{path}: fewer than 2 samples")

    dts = [b - a for a, b in zip(t[:-1], t[1:])]
    dt = statistics.median(dts)
    if abs(dt - DT_EXPECTED) > 1.0e-6:
        fail(f"{path}: unexpected dt={dt}")

    command_error = max(
        max(abs(x - EXPECTED_VX) for x in cmd_vx),
        max(abs(x - EXPECTED_VY) for x in cmd_vy),
        max(abs(x - EXPECTED_WZ) for x in cmd_wz),
    )
    if command_error > 1.0e-9:
        fail(f"{path}: fixed command mismatch; max error={command_error:.3e}")

    return {
        "path": path,
        "rows": len(rows),
        "dt": dt,
        "vy": vy,
    }


def seed_from_dir_name(name: str) -> int:
    marker = "seed"
    start = name.find(marker)
    if start < 0:
        fail(f"cannot parse seed from directory name: {name}")
    start += len(marker)
    digits = []
    while start < len(name) and name[start].isdigit():
        digits.append(name[start])
        start += 1
    if not digits:
        fail(f"cannot parse numeric seed from directory name: {name}")
    return int("".join(digits))


if not ROOT.is_dir():
    fail(f"evaluation root missing: {ROOT}")

dev_dirs = sorted(
    p for p in ROOT.glob(DEV_PATTERN)
    if p.is_dir()
)

dev_csvs = []
for d in dev_dirs:
    seed = seed_from_dir_name(d.name)
    if 100 <= seed <= 109:
        dev_csvs.append((seed, d / "samples.csv"))

if len(dev_csvs) != 10:
    fail(
        "expected exactly development seeds 100..109; "
        f"found {[(seed, str(path)) for seed, path in dev_csvs]}"
    )

expected_seeds = list(range(100, 110))
actual_seeds = [seed for seed, _ in dev_csvs]
if actual_seeds != expected_seeds:
    fail(f"development seed set mismatch: expected {expected_seeds}, got {actual_seeds}")

runs = [(42, CANONICAL)] + dev_csvs
loaded = [(seed, read_csv(path)) for seed, path in runs]

print("=" * 126)
print("v4-J LATERAL-REWARD SIZING AUDIT — FROZEN v4-B, NO ISAAC SIM")
print("=" * 126)
print("Scientific source       : frozen v4-B model1499 fixed-command evaluations")
print("Canonical seed          : 42")
print("Development seeds       : 100..109")
print("Command                 : [0.40, 0.00, 0.00]")
print("Reward candidate        : 1 - exp(-(vy_error / sigma)^2)")
print("Repository modified     : NO")
print("Checkpoint modified     : NO")
print()

all_raw: list[float] = []
all_ma05: list[float] = []
all_ma10: list[float] = []
per_seed_means: list[float] = []

window_05 = int(round(0.5 / DT_EXPECTED))
window_10 = int(round(1.0 / DT_EXPECTED))

print("PER-SEED BASELINE")
print("-" * 126)
print("seed    N      mean vy      mean|vy|       RMS vy      p95|vy|     neg.frac    MA0.5 mean    MA1.0 mean")
for seed, data in loaded:
    vy = data["vy"]
    assert isinstance(vy, list)

    s = stats(vy)
    ma05 = moving_average(vy, window_05)
    ma10 = moving_average(vy, window_10)
    s05 = stats(ma05)
    s10 = stats(ma10)

    all_raw.extend(vy)
    all_ma05.extend(ma05)
    all_ma10.extend(ma10)
    per_seed_means.append(s["mean"])

    print(
        f"{seed:4d}  {len(vy):5d}  "
        f"{s['mean']:+.6f}    {s['mean_abs']:.6f}    {s['rms']:.6f}    "
        f"{s['p95_abs']:.6f}    {s['negative_fraction']:.3f}      "
        f"{s05['mean']:+.6f}      {s10['mean']:+.6f}"
    )

raw = stats(all_raw)
ma05 = stats(all_ma05)
ma10 = stats(all_ma10)
seed_means = stats(per_seed_means)

print()
print("POOLED / SLOW-BIAS DIAGNOSTIC")
print("-" * 126)
print(
    f"raw vy       : mean={raw['mean']:+.6f}, mean|.|={raw['mean_abs']:.6f}, "
    f"RMS={raw['rms']:.6f}, p95|.|={raw['p95_abs']:.6f}, "
    f"max|.|={raw['max_abs']:.6f}, neg.frac={raw['negative_fraction']:.3f}"
)
print(
    f"MA 0.5 s     : mean={ma05['mean']:+.6f}, mean|.|={ma05['mean_abs']:.6f}, "
    f"RMS={ma05['rms']:.6f}, p95|.|={ma05['p95_abs']:.6f}, "
    f"max|.|={ma05['max_abs']:.6f}, neg.frac={ma05['negative_fraction']:.3f}"
)
print(
    f"MA 1.0 s     : mean={ma10['mean']:+.6f}, mean|.|={ma10['mean_abs']:.6f}, "
    f"RMS={ma10['rms']:.6f}, p95|.|={ma10['p95_abs']:.6f}, "
    f"max|.|={ma10['max_abs']:.6f}, neg.frac={ma10['negative_fraction']:.3f}"
)
print(
    f"seed means   : mean={seed_means['mean']:+.6f}, "
    f"RMS={seed_means['rms']:.6f}, p95|.|={seed_means['p95_abs']:.6f}, "
    f"neg.frac={seed_means['negative_fraction']:.3f}"
)

oscillation_ratio = (
    raw["rms"] / abs(raw["mean"])
    if abs(raw["mean"]) > 1.0e-12
    else float("inf")
)
slow_ratio = (
    ma10["rms"] / abs(ma10["mean"])
    if abs(ma10["mean"]) > 1.0e-12
    else float("inf")
)
print(f"raw RMS / |mean bias|   : {oscillation_ratio:.3f}")
print(f"MA1.0 RMS / |mean bias| : {slow_ratio:.3f}")

print()
print("BOUNDED-COST SIZING ON RAW INSTANTANEOUS vy")
print("-" * 126)
print("sigma     mean cost    p95 cost    saturation>0.9    local slope @ pooled mean")
for sigma in SIGMAS:
    costs = [bounded_cost(v, sigma) for v in all_raw]
    mean_cost = statistics.fmean(costs)
    p95_cost = percentile(costs, 0.95)
    sat = sum(c > 0.9 for c in costs) / len(costs)
    mu = raw["mean"]
    slope = (
        2.0 * mu / (sigma * sigma) * math.exp(-((mu / sigma) ** 2))
    )
    print(
        f"{sigma:5.3f}     {mean_cost:8.5f}    {p95_cost:8.5f}       "
        f"{100.0*sat:7.3f}%            {slope:+.6f}"
    )

print()
print("REWARD-MANAGER SCALE (dt=0.02): mean effective value per step")
print("-" * 126)
header = "sigma   " + "  ".join(f"w={w:+.2f}" for w in WEIGHTS)
print(header)
for sigma in SIGMAS:
    mean_cost = statistics.fmean(bounded_cost(v, sigma) for v in all_raw)
    vals = [w * mean_cost * DT_EXPECTED for w in WEIGHTS]
    print(
        f"{sigma:5.3f}   "
        + "  ".join(f"{v:+.8f}" for v in vals)
    )

print()
print("INTERPRETATION GUARDS")
print("-" * 126)
if all(x < 0.0 for x in per_seed_means):
    print("PASS: lateral bias sign is negative in all 11 evaluated seeds.")
else:
    print("NOTE: lateral bias sign is not uniform across all evaluated seeds.")

if ma10["negative_fraction"] >= 0.90:
    print("PASS: 1.0 s filtered vy is negative >=90% of pooled samples -> persistent slow bias.")
else:
    print(
        "CAUTION: 1.0 s filtered vy is not predominantly negative; "
        "instantaneous penalty may mostly suppress gait-cycle sway."
    )

if oscillation_ratio >= 3.0:
    print(
        "CAUTION: raw vy RMS is >=3x mean bias -> do not choose a narrow sigma "
        "without considering gait-cycle interference."
    )
else:
    print("PASS: raw vy variation is not overwhelmingly larger than the mean bias.")

print()
print("V4J_LATERAL_REWARD_SIZING_AUDIT_PASS")
print("=" * 126)
