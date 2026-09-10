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

TAUS_S = (0.25, 0.50, 0.75, 1.00)
SIGMAS = (0.015, 0.020, 0.025, 0.030, 0.040)
WEIGHTS = (-0.10, -0.25, -0.50)
STEADY_START_S = 2.0


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


def ema(values: list[float], dt: float, tau_s: float) -> list[float]:
    if tau_s <= 0.0:
        fail(f"tau_s must be > 0, got {tau_s}")
    alpha = 1.0 - math.exp(-dt / tau_s)
    state = 0.0
    out: list[float] = []
    for x in values:
        state += alpha * (x - state)
        out.append(state)
    return out


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

    return {"path": path, "dt": dt, "vy": vy}


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

dev_dirs = sorted(p for p in ROOT.glob(DEV_PATTERN) if p.is_dir())
dev_csvs: list[tuple[int, Path]] = []
for d in dev_dirs:
    seed = seed_from_dir_name(d.name)
    if 100 <= seed <= 109:
        dev_csvs.append((seed, d / "samples.csv"))

if [seed for seed, _ in dev_csvs] != list(range(100, 110)):
    fail(f"development seed set mismatch: {[seed for seed, _ in dev_csvs]}")

runs = [(42, CANONICAL)] + dev_csvs
loaded = [(seed, read_csv(path)) for seed, path in runs]

steady_start_idx = int(round(STEADY_START_S / DT_EXPECTED))

raw_all: list[float] = []
for _, data in loaded:
    vy = data["vy"]
    assert isinstance(vy, list)
    raw_all.extend(vy[steady_start_idx:])

raw_s = stats(raw_all)

print("=" * 132)
print("v4-J EMA LATERAL-DRIFT REWARD SIZING — FROZEN v4-B, NO ISAAC SIM")
print("=" * 132)
print("Source                  : v4-B model1499 canonical seed42 + development seeds100..109")
print("Command                 : [0.40, 0.00, 0.00]")
print("EMA update              : alpha = 1 - exp(-dt/tau); state += alpha*(error-state)")
print("EMA reset               : state = 0 at episode reset")
print(f"Steady-state statistics : discard first {STEADY_START_S:.1f} s only for sizing tables")
print("Candidate cost          : 1 - exp(-(EMA(vy-vy_cmd)/sigma)^2)")
print("Repository modified     : NO")
print("Checkpoint modified     : NO")
print()

print("RAW REFERENCE AFTER STARTUP")
print("-" * 132)
print(
    f"mean={raw_s['mean']:+.6f}  RMS={raw_s['rms']:.6f}  "
    f"p95|.|={raw_s['p95_abs']:.6f}  neg.frac={raw_s['negative_fraction']:.3f}"
)
print()

ema_by_tau: dict[float, list[float]] = {}

print("EMA TAU DIAGNOSTIC")
print("-" * 132)
print("tau(s)   alpha       mean EMA     RMS EMA    p95|EMA|   neg.frac   RMS/rawRMS   |mean|/|rawmean|")
for tau in TAUS_S:
    pooled: list[float] = []
    for _, data in loaded:
        vy = data["vy"]
        dt = data["dt"]
        assert isinstance(vy, list)
        assert isinstance(dt, float)
        e = ema(vy, dt, tau)
        pooled.extend(e[steady_start_idx:])
    ema_by_tau[tau] = pooled
    s = stats(pooled)
    rms_ratio = s["rms"] / raw_s["rms"]
    mean_ratio = abs(s["mean"]) / abs(raw_s["mean"])
    alpha = 1.0 - math.exp(-DT_EXPECTED / tau)
    print(
        f"{tau:5.2f}   {alpha:8.6f}   {s['mean']:+.6f}   {s['rms']:.6f}   "
        f"{s['p95_abs']:.6f}     {s['negative_fraction']:.3f}       "
        f"{rms_ratio:.3f}          {mean_ratio:.3f}"
    )

print()
print("PER-SEED EMA MEANS")
print("-" * 132)
print("seed   " + "  ".join(f"tau={tau:.2f}" for tau in TAUS_S))
for seed, data in loaded:
    vy = data["vy"]
    dt = data["dt"]
    assert isinstance(vy, list)
    assert isinstance(dt, float)
    means = [
        statistics.fmean(ema(vy, dt, tau)[steady_start_idx:])
        for tau in TAUS_S
    ]
    print(f"{seed:4d}   " + "  ".join(f"{x:+.6f}" for x in means))

print()
print("BOUNDED COST ON EMA — STEADY-STATE")
print("-" * 132)
print("tau    sigma    mean cost    p95 cost    >0.9 frac    local slope @ EMA mean")
cost_table: dict[tuple[float, float], float] = {}
for tau in TAUS_S:
    vals = ema_by_tau[tau]
    mu = statistics.fmean(vals)
    for sigma in SIGMAS:
        costs = [bounded_cost(v, sigma) for v in vals]
        mean_cost = statistics.fmean(costs)
        p95_cost = percentile(costs, 0.95)
        sat = sum(c > 0.9 for c in costs) / len(costs)
        slope = 2.0 * mu / (sigma * sigma) * math.exp(-((mu / sigma) ** 2))
        cost_table[(tau, sigma)] = mean_cost
        print(
            f"{tau:4.2f}   {sigma:5.3f}     {mean_cost:8.5f}    {p95_cost:8.5f}      "
            f"{100.0*sat:7.3f}%       {slope:+.6f}"
        )

print()
print("REWARD-MANAGER MEAN EFFECTIVE VALUE PER STEP (dt=0.02)")
print("-" * 132)
print("tau    sigma    " + "  ".join(f"w={w:+.2f}" for w in WEIGHTS))
for tau in TAUS_S:
    for sigma in SIGMAS:
        mean_cost = cost_table[(tau, sigma)]
        values = [w * mean_cost * DT_EXPECTED for w in WEIGHTS]
        print(
            f"{tau:4.2f}   {sigma:5.3f}    "
            + "  ".join(f"{v:+.8f}" for v in values)
        )

print()
print("FILTER SELECTION GUARDS")
print("-" * 132)
eligible: list[float] = []
for tau in TAUS_S:
    s = stats(ema_by_tau[tau])
    rms_ratio = s["rms"] / raw_s["rms"]
    mean_ratio = abs(s["mean"]) / abs(raw_s["mean"])
    good = (
        s["negative_fraction"] >= 0.99
        and rms_ratio <= 0.50
        and mean_ratio >= 0.90
    )
    print(
        f"tau={tau:.2f}s : "
        f"neg.frac={s['negative_fraction']:.3f}, "
        f"RMS/raw={rms_ratio:.3f}, mean retention={mean_ratio:.3f} -> "
        f"{'ELIGIBLE' if good else 'NOT ELIGIBLE'}"
    )
    if good:
        eligible.append(tau)

if not eligible:
    fail("no tau satisfies the predeclared filter guards")

print(f"Eligible tau values     : {eligible}")
print()
print("V4J_EMA_LATERAL_REWARD_SIZING_AUDIT_PASS")
print("=" * 132)
