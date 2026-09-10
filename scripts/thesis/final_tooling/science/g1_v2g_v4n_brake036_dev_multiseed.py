from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

parser = argparse.ArgumentParser(
    description=(
        "Development multiseed robustness harness for the frozen v2-G/v4-N "
        "0.36 s braking-dwell handoff candidate."
    )
)
parser.add_argument(
    "--probe",
    type=Path,
    default=Path(os.environ["USERPROFILE"]) / "Desktop" / "g1_v2g_v4n_brake_dwell_036_smoke.py",
)
parser.add_argument(
    "--seeds",
    type=int,
    nargs="+",
    default=[100, 101, 102, 103, 104],
)
parser.add_argument(
    "--isaaclab_bat",
    type=Path,
    default=Path(r"C:\IsaacLab\isaaclab.bat"),
)
args = parser.parse_args()

HANDOFF_ROOT = Path(r"C:\humanoid_rl_lab\logs\thesis\handoff")
EXPECTED_PROBE_SHA256 = "10FF4DB312DC1F6B0F7334C85D5CFACF1C447C8645D99F0D087C063A67E506CE"
EXPECTED_MARKER = "V2G_V4N_BRAKE_DWELL_036_SMOKE_PASS"
DT = 0.02
SUSTAIN_S = 0.50

BANDS = {
    "STABILIZED": dict(vxy=0.020, tilt=2.0, wz=0.050, double_support=True),
    "NEAR_STAND": dict(vxy=0.005, tilt=1.3, wz=0.020, double_support=True),
}


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def fv(row: dict[str, str], key: str) -> float:
    return float(row[key])


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        fail(f"empty trajectory: {path}")
    return rows


def satisfies(row: dict[str, str], band: dict[str, float | bool]) -> bool:
    if fv(row, "root_vxy_mps") > float(band["vxy"]):
        return False
    if fv(row, "root_tilt_deg") > float(band["tilt"]):
        return False
    if abs(fv(row, "root_wz_radps")) > float(band["wz"]):
        return False
    if bool(band["double_support"]) and fv(row, "double_support") < 0.5:
        return False
    return True


def sustained_entry(rows: list[dict[str, str]], band: dict[str, float | bool]) -> float | None:
    need = int(round(SUSTAIN_S / DT))
    run = 0
    for i, row in enumerate(rows):
        run = run + 1 if satisfies(row, band) else 0
        if run >= need:
            return (i - need + 1) * DT
    return None


def first_phase_index(rows: list[dict[str, str]], phase: str) -> int:
    for i, row in enumerate(rows):
        if row["phase"] == phase:
            return i
    fail(f"phase missing: {phase}")


def xy_excursion(rows: list[dict[str, str]]) -> float:
    x0, y0 = fv(rows[0], "root_x_m"), fv(rows[0], "root_y_m")
    return max(
        math.hypot(fv(r, "root_x_m") - x0, fv(r, "root_y_m") - y0)
        for r in rows
    )


def summarize(seed: int, result_dir: Path) -> dict[str, float | int | str | None]:
    rows = read_csv(result_dir / "trajectory.csv")

    if any(int(round(fv(r, "done"))) != 0 for r in rows):
        fail(f"seed {seed}: trajectory contains done=1")

    phase_order: list[str] = []
    for row in rows:
        if not phase_order or phase_order[-1] != row["phase"]:
            phase_order.append(row["phase"])
    expected_order = ["STAND_PRE", "LOCOMOTION", "BRAKE_V4N", "STAND_POST"]
    if phase_order != expected_order:
        fail(f"seed {seed}: unexpected phase order {phase_order}")

    post_i = first_phase_index(rows, "STAND_POST")
    if post_i <= 0:
        fail(f"seed {seed}: no pre-switch row")

    pre = rows[post_i - 1]
    first_post = rows[post_i]
    post = rows[post_i:]
    first05 = post[:25]
    final20 = post[-100:]

    return {
        "seed": seed,
        "result_dir": str(result_dir),
        "pre_vxy": fv(pre, "root_vxy_mps"),
        "pre_tilt": fv(pre, "root_tilt_deg"),
        "pre_abs_wz": abs(fv(pre, "root_wz_radps")),
        "pre_double": int(round(fv(pre, "double_support"))),
        "actual_jump_rms": fv(first_post, "actual_action_jump_rms"),
        "actual_jump_max": fv(first_post, "actual_action_jump_max_abs"),
        "first05_mean_vxy": statistics.fmean(fv(r, "root_vxy_mps") for r in first05),
        "first05_max_vxy": max(fv(r, "root_vxy_mps") for r in first05),
        "first05_max_tilt": max(fv(r, "root_tilt_deg") for r in first05),
        "first05_max_abs_wz": max(abs(fv(r, "root_wz_radps")) for r in first05),
        "first05_ds_pct": 100.0 * statistics.fmean(fv(r, "double_support") for r in first05),
        "xy_excursion": xy_excursion(post),
        "stabilized_s": sustained_entry(post, BANDS["STABILIZED"]),
        "near_stand_s": sustained_entry(post, BANDS["NEAR_STAND"]),
        "final20_mean_vxy": statistics.fmean(fv(r, "root_vxy_mps") for r in final20),
        "final20_mean_tilt": statistics.fmean(fv(r, "root_tilt_deg") for r in final20),
        "final20_mean_abs_wz": statistics.fmean(abs(fv(r, "root_wz_radps")) for r in final20),
        "final20_ds_pct": 100.0 * statistics.fmean(fv(r, "double_support") for r in final20),
    }


def fmt_optional(value: float | None) -> str:
    return "N/R" if value is None else f"{value:.2f}"


probe = args.probe.resolve()
isaaclab_bat = args.isaaclab_bat.resolve()

if not probe.is_file():
    fail(f"probe missing: {probe}")
if not isaaclab_bat.is_file():
    fail(f"Isaac Lab launcher missing: {isaaclab_bat}")

actual_probe_sha = sha256(probe)
if actual_probe_sha != EXPECTED_PROBE_SHA256:
    fail(
        "probe SHA256 mismatch\n"
        f"Expected: {EXPECTED_PROBE_SHA256}\n"
        f"Actual  : {actual_probe_sha}"
    )

if len(set(args.seeds)) != len(args.seeds):
    fail(f"duplicate seeds are not allowed: {args.seeds}")

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
audit_dir = HANDOFF_ROOT / f"{timestamp}_brake036_dev_multiseed"
audit_dir.mkdir(parents=True, exist_ok=False)

print("=" * 120)
print("v2-G / v4-N BRAKE-DWELL 0.36 s — DEVELOPMENT MULTISEED ROBUSTNESS")
print("=" * 120)
print(f"probe              : {probe}")
print(f"probe SHA256       : {actual_probe_sha}")
print(f"development seeds  : {args.seeds}")
print("policy/checkpoints : FROZEN")
print("dwell              : 0.36 s FROZEN")
print("purpose            : development robustness only; NOT holdout evidence")
print(f"audit_dir          : {audit_dir}")
print("=" * 120)

records: list[dict[str, float | int | str | None]] = []

for seed in args.seeds:
    print()
    print("=" * 120)
    print(f"SEED {seed} BEGIN")
    print("=" * 120)

    pattern = f"*_v2g_v4n_brake_dwell_036_seed{seed}"
    before = {p.resolve() for p in HANDOFF_ROOT.glob(pattern) if p.is_dir()}

    log_path = audit_dir / f"seed_{seed}_console.txt"
    cmd = [
        str(isaaclab_bat),
        "-p",
        str(probe),
        "--seed",
        str(seed),
    ]

    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=r"C:\humanoid_rl_lab",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        marker_seen = False
        for line in proc.stdout:
            log.write(line)
            if (
                EXPECTED_MARKER in line
                or "BRAKE start state:" in line
                or "BRAKE end state" in line
                or "[SWITCH] v4-N BRAKE->v2-G STAND" in line
                or "FIRST 0.5 s AFTER BRAKE->STAND" in line
                or "termination    :" in line
            ):
                print(line.rstrip(), flush=True)
            if EXPECTED_MARKER in line:
                marker_seen = True
        return_code = proc.wait()

    if return_code != 0:
        fail(f"seed {seed}: probe returned code {return_code}; see {log_path}")
    if not marker_seen:
        fail(f"seed {seed}: PASS marker missing; see {log_path}")

    after = {p.resolve() for p in HANDOFF_ROOT.glob(pattern) if p.is_dir()}
    created = sorted(after - before, key=lambda p: p.stat().st_mtime)

    if len(created) != 1:
        fail(
            f"seed {seed}: expected exactly one new result directory, found {len(created)}: {created}"
        )

    result_dir = created[0]
    rec = summarize(seed, result_dir)
    records.append(rec)

    print(
        f"SEED {seed} PASS | "
        f"pre_vxy={rec['pre_vxy']:.4f} "
        f"jump_rms={rec['actual_jump_rms']:.4f} "
        f"0.5s_mean_vxy={rec['first05_mean_vxy']:.4f} "
        f"0.5s_max_tilt={rec['first05_max_tilt']:.3f} "
        f"0.5s_max|wz|={rec['first05_max_abs_wz']:.3f} "
        f"XYexc={rec['xy_excursion']:.4f} "
        f"STAB={fmt_optional(rec['stabilized_s'])} "
        f"NEAR={fmt_optional(rec['near_stand_s'])}"
    )

csv_path = audit_dir / "multiseed_summary.csv"
fieldnames = list(records[0].keys())
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records)

print()
print("=" * 140)
print("DEVELOPMENT MULTISEED SUMMARY")
print("=" * 140)
print(
    f"{'SEED':>5} {'PRE vxy':>9} {'PRE tilt':>9} {'JUMP RMS':>10} "
    f"{'0.5meanV':>10} {'0.5maxTilt':>11} {'0.5max|wz|':>11} "
    f"{'XYexc':>9} {'STAB':>7} {'NEAR':>7} {'FINAL vxy':>10} {'FINAL tilt':>11}"
)
print("-" * 140)

for r in records:
    print(
        f"{int(r['seed']):5d} "
        f"{r['pre_vxy']:9.4f} "
        f"{r['pre_tilt']:9.3f} "
        f"{r['actual_jump_rms']:10.4f} "
        f"{r['first05_mean_vxy']:10.4f} "
        f"{r['first05_max_tilt']:11.3f} "
        f"{r['first05_max_abs_wz']:11.3f} "
        f"{r['xy_excursion']:9.4f} "
        f"{fmt_optional(r['stabilized_s']):>7} "
        f"{fmt_optional(r['near_stand_s']):>7} "
        f"{r['final20_mean_vxy']:10.4f} "
        f"{r['final20_mean_tilt']:11.3f}"
    )

numeric_keys = [
    "pre_vxy",
    "pre_tilt",
    "actual_jump_rms",
    "first05_mean_vxy",
    "first05_max_tilt",
    "first05_max_abs_wz",
    "xy_excursion",
    "final20_mean_vxy",
    "final20_mean_tilt",
]

print()
print("-" * 140)
print("AGGREGATES (development seeds only)")
print("-" * 140)
for key in numeric_keys:
    vals = [float(r[key]) for r in records]
    mean = statistics.fmean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    print(f"{key:<24} mean={mean:.9f} std={std:.9f} min={min(vals):.9f} max={max(vals):.9f}")

for key in ("stabilized_s", "near_stand_s"):
    vals = [r[key] for r in records if r[key] is not None]
    if len(vals) != len(records):
        print(f"{key:<24} reached={len(vals)}/{len(records)}")
    else:
        vals_f = [float(v) for v in vals]
        mean = statistics.fmean(vals_f)
        std = statistics.stdev(vals_f) if len(vals_f) > 1 else 0.0
        print(
            f"{key:<24} reached={len(vals_f)}/{len(records)} "
            f"mean={mean:.6f} std={std:.6f} "
            f"min={min(vals_f):.6f} max={max(vals_f):.6f}"
        )

print()
print(f"summary CSV        : {csv_path}")
print(f"per-seed console   : {audit_dir}")
print("INTERPRETATION     : development robustness only; candidate dwell remains frozen.")
print("NO HOLDOUT SEEDS USED")
print("V2G_V4N_BRAKE036_DEV_MULTISEED_PASS")
print("=" * 140)
