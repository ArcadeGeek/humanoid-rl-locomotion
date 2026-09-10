from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

DEFAULT_RESULT_DIR = Path(
    r"C:\humanoid_rl_lab\logs\thesis\handoff"
    r"\2026-09-05_03-55-44_v2g_v4n_brake_dwell_050_seed42"
)

parser = argparse.ArgumentParser(
    description=(
        "Offline audit of the 25 recorded v4-N zero-command brake steps. "
        "Selects data-supported candidate dwell times without rerunning Isaac Sim."
    )
)
parser.add_argument(
    "--result_dir",
    type=Path,
    default=DEFAULT_RESULT_DIR,
)
args = parser.parse_args()

CSV_NAME = "trajectory.csv"
EXPECTED_DT = 0.02
BRAKE_PHASE = "BRAKE_V4N"


def fail(message: str) -> None:
    raise RuntimeError(message)


def f(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except Exception as exc:
        raise RuntimeError(f"invalid numeric field {key!r}: {row.get(key)!r}") from exc


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        fail(f"trajectory CSV missing: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        fail("trajectory CSV is empty")
    return rows


def dominates(a: dict[str, float], b: dict[str, float]) -> bool:
    """Pareto dominance for lower vxy, tilt, |wz|, with double support preferred."""
    a_ds_penalty = 0.0 if a["double_support"] >= 0.5 else 1.0
    b_ds_penalty = 0.0 if b["double_support"] >= 0.5 else 1.0

    av = (a["vxy"], a["tilt"], a["abs_wz"], a_ds_penalty)
    bv = (b["vxy"], b["tilt"], b["abs_wz"], b_ds_penalty)

    no_worse = all(x <= y for x, y in zip(av, bv))
    strictly_better = any(x < y for x, y in zip(av, bv))
    return no_worse and strictly_better


result_dir = args.result_dir.resolve()
csv_path = result_dir / CSV_NAME
rows = read_rows(csv_path)

brake = [r for r in rows if r["phase"] == BRAKE_PHASE]
if len(brake) != 25:
    fail(f"expected 25 BRAKE_V4N rows, found {len(brake)}")

times = [f(r, "time_s") for r in brake]
for i in range(1, len(times)):
    dt = times[i] - times[i - 1]
    if not math.isclose(dt, EXPECTED_DT, rel_tol=0.0, abs_tol=1e-9):
        fail(f"unexpected brake dt at i={i}: {dt}")

t0 = times[0] - EXPECTED_DT

points: list[dict[str, float]] = []
for i, row in enumerate(brake, start=1):
    points.append(
        {
            "step": float(i),
            "dwell_s": i * EXPECTED_DT,
            "vxy": f(row, "root_vxy_mps"),
            "tilt": f(row, "root_tilt_deg"),
            "abs_wz": abs(f(row, "root_wz_radps")),
            "double_support": f(row, "double_support"),
            "x": f(row, "root_x_m"),
            "y": f(row, "root_y_m"),
        }
    )

pareto = [
    p for p in points
    if not any(dominates(q, p) for q in points if q is not p)
]

print("=" * 118)
print("v4-N ZERO-COMMAND BRAKE — OFFLINE DWELL-SELECTION AUDIT")
print("=" * 118)
print(f"result_dir            : {result_dir}")
print(f"brake_samples         : {len(points)}")
print(f"sample_period_s       : {EXPECTED_DT:.6f}")
print("rerun Isaac Sim       : NO")
print("source/checkpoint edit: NONE")
print()
print("Each row is the physical state AFTER that many seconds of v4-N with command [0,0,0].")
print()

print(
    f"{'DWELL(s)':>8} {'VXY(m/s)':>11} {'TILT(deg)':>11} "
    f"{'|WZ|(rad/s)':>13} {'DOUBLE':>8} {'XY_FROM_START(m)':>17}"
)
print("-" * 118)

x0, y0 = points[0]["x"], points[0]["y"]
for p in points:
    excursion = math.hypot(p["x"] - x0, p["y"] - y0)
    flag = "*" if p in pareto else " "
    print(
        f"{p['dwell_s']:8.2f} {p['vxy']:11.6f} {p['tilt']:11.6f} "
        f"{p['abs_wz']:13.6f} {int(p['double_support']):8d} "
        f"{excursion:17.6f} {flag}"
    )

print()
print("* = Pareto-nondominated among recorded brake states for")
print("    lower vxy, lower tilt, lower |wz|, and double support preferred.")
print()

print("-" * 118)
print("PARETO CANDIDATES")
print("-" * 118)
for p in sorted(pareto, key=lambda x: x["dwell_s"]):
    print(
        f"dwell={p['dwell_s']:.2f}s "
        f"vxy={p['vxy']:.6f} m/s "
        f"tilt={p['tilt']:.6f} deg "
        f"|wz|={p['abs_wz']:.6f} rad/s "
        f"double_support={int(p['double_support'])}"
    )

# Also expose threshold-crossing candidates, without defining them as acceptance criteria.
print()
print("-" * 118)
print("DESCRIPTIVE THRESHOLD CROSSINGS — NOT acceptance criteria")
print("-" * 118)

for threshold in (0.25, 0.20, 0.15, 0.10):
    matches = [p for p in points if p["vxy"] <= threshold]
    if matches:
        p = matches[0]
        print(
            f"first vxy <= {threshold:.2f} m/s : dwell={p['dwell_s']:.2f}s, "
            f"tilt={p['tilt']:.3f} deg, |wz|={p['abs_wz']:.3f}, "
            f"double_support={int(p['double_support'])}"
        )
    else:
        print(f"first vxy <= {threshold:.2f} m/s : NOT REACHED")

ds = [p for p in points if p["double_support"] >= 0.5]
if ds:
    p = ds[0]
    print(
        f"first double-support sample : dwell={p['dwell_s']:.2f}s, "
        f"vxy={p['vxy']:.3f} m/s, tilt={p['tilt']:.3f} deg, |wz|={p['abs_wz']:.3f}"
    )
else:
    print("first double-support sample : NOT REACHED")

print()
print("=" * 118)
print("INTERPRETATION_GUARDRAIL")
print("This audit selects candidate switch instants from the recorded v4-N brake trajectory only.")
print("It cannot predict the subsequent v2-G recovery without a new controlled simulation.")
print("No weighted score is used; Pareto candidates are reported to avoid arbitrary tuning weights.")
print("V2G_V4N_BRAKE_DWELL_SELECTION_AUDIT_PASS")
print("=" * 118)
