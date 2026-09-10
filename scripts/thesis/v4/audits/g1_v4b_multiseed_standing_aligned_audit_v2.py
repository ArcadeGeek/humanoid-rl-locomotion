from __future__ import annotations

import csv
import hashlib
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(r"C:\humanoid_rl_lab")
BASE = Path(r"C:\Users\Mirko\Desktop\g1_v4b_standing_aligned_probe_v2.py")
EXPECTED_BASE_SHA = "52A50E9AE926D74595B7AB16833489F97634678A2992964A72848259192AB54F"

SEEDS = list(range(100, 110))
BASE_OUT_TOKEN = "v4b_model1499_play_equivalent_seed42_zero_cmd_aligned_yaw_20s_standing_v2"

ROOT_OUT = (
    REPO
    / "logs"
    / "evaluations"
    / "g1"
    / "v4b_multiseed_standing_dev_100_109_zero_cmd_aligned_yaw_20s_v2"
)
SUMMARY_CSV = ROOT_OUT / "multiseed_standing_summary.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def wrap_pi(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def pct(mask):
    m = np.asarray(mask, dtype=bool)
    return 100.0 * float(np.mean(m)) if m.size else float("nan")


def rising_edges(mask):
    m = np.asarray(mask, dtype=bool)
    if m.size < 2:
        return np.asarray([], dtype=np.int64)
    return np.flatnonzero((~m[:-1]) & m[1:]) + 1


def read_rows(samples_path: Path):
    with samples_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1000:
        raise SystemExit(
            f"ABORT: expected 1000 samples in {samples_path}, found {len(rows)}"
        )
    return rows


def analyze(samples_path: Path, seed: int):
    rows = read_rows(samples_path)

    def col(name):
        if name not in rows[0]:
            raise SystemExit(f"ABORT: seed {seed}: CSV column missing: {name}")
        a = np.asarray([float(r[name]) for r in rows], dtype=np.float64)
        if not np.all(np.isfinite(a)):
            raise SystemExit(f"ABORT: seed {seed}: non-finite column: {name}")
        return a

    t = col("time_s")
    cmd_vx = col("cmd_vx_mps")
    cmd_vy = col("cmd_vy_mps")
    cmd_wz = col("cmd_wz_radps")
    vx = col("base_vx_mps")
    vy = col("base_vy_mps")
    wz = col("base_wz_radps")
    x = col("world_x_m")
    y = col("world_y_m")
    heading = col("heading_rad")
    lc = col("left_foot_contact") > 0.5
    rc = col("right_foot_contact") > 0.5
    lxy = col("left_foot_xy_speed_mps")
    rxy = col("right_foot_xy_speed_mps")
    lx = col("left_foot_x_world_m")
    ly = col("left_foot_y_world_m")
    rx = col("right_foot_x_world_m")
    ry = col("right_foot_y_world_m")

    if max(
        float(np.max(np.abs(cmd_vx))),
        float(np.max(np.abs(cmd_vy))),
        float(np.max(np.abs(cmd_wz))),
    ) > 1.0e-12:
        raise SystemExit(f"ABORT: seed {seed}: command is not exactly [0,0,0]")

    speed = np.hypot(vx, vy)
    dx = float(x[-1] - x[0])
    dy = float(y[-1] - y[0])
    drift = math.hypot(dx, dy)
    h0 = float(heading[0])
    herr = wrap_pi(heading - h0)
    ahdeg = np.abs(np.degrees(herr))
    double = lc & rc
    single = np.logical_xor(lc, rc)
    flight = (~lc) & (~rc)

    return {
        "seed": seed,
        "samples": len(rows),
        "dt_s": float(np.median(np.diff(t))),
        "mean_planar_speed_mps": float(np.mean(speed)),
        "rms_planar_speed_mps": float(np.sqrt(np.mean(speed**2))),
        "p95_planar_speed_mps": float(np.percentile(speed, 95)),
        "mean_abs_vx_mps": float(np.mean(np.abs(vx))),
        "mean_abs_vy_mps": float(np.mean(np.abs(vy))),
        "mean_wz_radps": float(np.mean(wz)),
        "mean_abs_wz_radps": float(np.mean(np.abs(wz))),
        "heading_first_measured_deg": math.degrees(h0),
        "final_heading_error_deg": math.degrees(float(herr[-1])),
        "mean_abs_heading_error_deg": float(np.mean(ahdeg)),
        "dx_m": dx,
        "dy_m": dy,
        "net_drift_m": drift,
        "double_support_pct": pct(double),
        "single_support_pct": pct(single),
        "flight_pct": pct(flight),
        "touchdowns_left": int(len(rising_edges(lc))),
        "touchdowns_right": int(len(rising_edges(rc))),
        "mean_contact_slip_left_mps": float(np.mean(lxy[lc])) if np.any(lc) else float("nan"),
        "mean_contact_slip_right_mps": float(np.mean(rxy[rc])) if np.any(rc) else float("nan"),
        "foot_path_left_m": float(np.sum(np.hypot(np.diff(lx), np.diff(ly)))),
        "foot_path_right_m": float(np.sum(np.hypot(np.diff(rx), np.diff(ry)))),
    }


if not BASE.is_file():
    raise SystemExit(f"ABORT: base standing v2 script missing: {BASE}")

base_sha = sha256(BASE)
if base_sha != EXPECTED_BASE_SHA:
    raise SystemExit(
        "ABORT: base standing v2 SHA mismatch.\n"
        f"expected={EXPECTED_BASE_SHA}\nactual  ={base_sha}"
    )

base_text = BASE.read_text(encoding="utf-8")

# These are structural anchors in the validated v2 script.
VX_REPLACEMENT_LINE = '    ("VX = 0.40", "VX = 0.00", "zero forward command"),'
SEED_GUARD_LINE = '    "SEED = 42",'

for label, token in (
    ("VX replacement line", VX_REPLACEMENT_LINE),
    ("seed guard line", SEED_GUARD_LINE),
    ("base output token", BASE_OUT_TOKEN),
):
    if base_text.count(token) != 1:
        raise SystemExit(
            f"ABORT: validated base structure changed: {label} count={base_text.count(token)}"
        )

ROOT_OUT.mkdir(parents=True, exist_ok=True)

print("=" * 124)
print("V4-B MULTI-SEED STANDING AUDIT V2 — DEVELOPMENT SEEDS 100–109")
print("=" * 124)
print(f"base script      : {BASE}")
print(f"base SHA         : {base_sha}")
print("seeds            : 100..109 (DEVELOPMENT/TUNING; NOT HOLDOUT)")
print("command          : [0,0,0]")
print("aligned reset    : yaw=0, yaw-rate=0 evaluation-only")
print("warmup/measure   : 3 s / 20 s")
print(f"aggregate output : {ROOT_OUT}")
print("=" * 124)

results = []

for seed in SEEDS:
    run_token = (
        f"v4b_model1499_play_equivalent_seed{seed}_zero_cmd_"
        "aligned_yaw_20s_standing_dev_v2"
    )
    run_dir = REPO / "logs" / "evaluations" / "g1" / run_token
    samples_path = run_dir / "samples.csv"

    if run_dir.exists():
        if samples_path.is_file():
            print(f"[seed {seed}] complete existing output -> read-only reuse")
            results.append(analyze(samples_path, seed))
            continue
        raise SystemExit(
            f"ABORT: seed {seed} output dir exists but samples.csv is missing:\n"
            f"{run_dir}\nRefusing to overwrite/delete partial evidence."
        )

    derived = base_text

    # 1) Add the seed change to the v2 script's *canonical-probe replacement list*.
    seed_replacement_line = (
        f'    ("SEED = 42", "SEED = {seed}", "standing development seed"),'
    )
    replacement_block = VX_REPLACEMENT_LINE + "\n" + seed_replacement_line
    if derived.count(VX_REPLACEMENT_LINE) != 1:
        raise SystemExit(f"ABORT: seed {seed}: VX replacement anchor mismatch")
    derived = derived.replace(VX_REPLACEMENT_LINE, replacement_block, 1)

    # 2) Make the v2 protocol guard expect the newly derived seed.
    new_seed_guard_line = f'    "SEED = {seed}",'
    if derived.count(SEED_GUARD_LINE) != 1:
        raise SystemExit(f"ABORT: seed {seed}: seed guard anchor mismatch")
    derived = derived.replace(SEED_GUARD_LINE, new_seed_guard_line, 1)

    # 3) Give each seed a unique output namespace.
    if derived.count(BASE_OUT_TOKEN) != 1:
        raise SystemExit(f"ABORT: seed {seed}: output token anchor mismatch")
    derived = derived.replace(BASE_OUT_TOKEN, run_token, 1)

    # Guard against accidental extra scientific changes.
    if derived.count(seed_replacement_line) != 1:
        raise SystemExit(f"ABORT: seed {seed}: seed replacement line not unique")
    if derived.count(new_seed_guard_line) != 1:
        raise SystemExit(f"ABORT: seed {seed}: seed guard line not unique")
    if derived.count(run_token) != 1:
        raise SystemExit(f"ABORT: seed {seed}: run token not unique")

    compile(derived, f"<v4b_standing_seed{seed}_v2>", "exec")

    print()
    print("=" * 124)
    print(f"RUNNING DEVELOPMENT SEED {seed}")
    print(f"output: {run_dir}")
    print("=" * 124)
    sys.stdout.flush()

    with tempfile.TemporaryDirectory(prefix=f"g1_v4b_standing_seed{seed}_v2_") as td:
        temp_script = Path(td) / f"g1_v4b_standing_seed{seed}_v2.py"
        temp_script.write_text(derived, encoding="utf-8", newline="\n")

        try:
            proc = subprocess.run(
                [sys.executable, str(temp_script)],
                cwd=str(REPO),
                timeout=300,
            )
        except subprocess.TimeoutExpired:
            raise SystemExit(
                f"ABORT: seed {seed} exceeded 300 s; prior output was streamed live."
            )

        if proc.returncode != 0:
            raise SystemExit(
                f"ABORT: seed {seed} standing probe failed with code {proc.returncode}"
            )

    if not samples_path.is_file():
        raise SystemExit(
            f"ABORT: seed {seed} returned 0 but samples.csv is missing:\n{samples_path}"
        )

    r = analyze(samples_path, seed)
    results.append(r)
    print(
        f"[seed {seed}] mean_speed={r['mean_planar_speed_mps']:.4f} m/s | "
        f"drift={r['net_drift_m']:.3f} m | "
        f"single={r['single_support_pct']:.1f}% | "
        f"TD={r['touchdowns_left']}/{r['touchdowns_right']} | "
        f"mean|herr|={r['mean_abs_heading_error_deg']:.2f} deg"
    )

if len(results) != len(SEEDS):
    raise SystemExit(f"ABORT: expected 10 seed results, found {len(results)}")

fieldnames = list(results[0].keys())
with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(results)

def vals(key):
    return np.asarray([float(r[key]) for r in results], dtype=np.float64)

print()
print("=" * 124)
print("V4-B MULTI-SEED STANDING AGGREGATE V2 — DEVELOPMENT 100–109")
print("=" * 124)
print(f"mean planar speed   : {np.mean(vals('mean_planar_speed_mps')):.6f} m/s  std {np.std(vals('mean_planar_speed_mps'), ddof=1):.6f}")
print(f"mean net drift      : {np.mean(vals('net_drift_m')):.4f} m    std {np.std(vals('net_drift_m'), ddof=1):.4f}")
print(f"mean |body vx|      : {np.mean(vals('mean_abs_vx_mps')):.6f} m/s  std {np.std(vals('mean_abs_vx_mps'), ddof=1):.6f}")
print(f"mean |body vy|      : {np.mean(vals('mean_abs_vy_mps')):.6f} m/s  std {np.std(vals('mean_abs_vy_mps'), ddof=1):.6f}")
print(f"mean |body wz|      : {np.mean(vals('mean_abs_wz_radps')):.6f} rad/s std {np.std(vals('mean_abs_wz_radps'), ddof=1):.6f}")
print(f"mean first heading  : {np.mean(vals('heading_first_measured_deg')):+.3f} deg  std {np.std(vals('heading_first_measured_deg'), ddof=1):.3f}")
print(f"mean final herr     : {np.mean(vals('final_heading_error_deg')):+.3f} deg  std {np.std(vals('final_heading_error_deg'), ddof=1):.3f}")
print(f"mean per-run |herr| : {np.mean(vals('mean_abs_heading_error_deg')):.3f} deg  std {np.std(vals('mean_abs_heading_error_deg'), ddof=1):.3f}")
print(f"mean double support : {np.mean(vals('double_support_pct')):.2f}% std {np.std(vals('double_support_pct'), ddof=1):.2f}")
print(f"mean single support : {np.mean(vals('single_support_pct')):.2f}% std {np.std(vals('single_support_pct'), ddof=1):.2f}")
print(f"mean touchdowns L/R : {np.mean(vals('touchdowns_left')):.1f} / {np.mean(vals('touchdowns_right')):.1f}")
print(f"mean foot path L/R  : {np.mean(vals('foot_path_left_m')):.3f} / {np.mean(vals('foot_path_right_m')):.3f} m")
print(f"summary CSV         : {SUMMARY_CSV}")
print()
print("V4B_MULTISEED_STANDING_AUDIT_V2_PASS")
print("=" * 124)
