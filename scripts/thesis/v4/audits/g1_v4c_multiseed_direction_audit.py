from __future__ import annotations

import csv
import hashlib
import math
import subprocess
import sys
from pathlib import Path

REPO = Path(r"C:\humanoid_rl_lab")
CANONICAL_PROBE = REPO / "scripts" / "thesis" / "v4" / "g1_v4c_contact_gait_probe.py"
EXPECTED_PROBE_SHA = "C678B09DC772BE6B8EEE2A84B3FE6452101561693F7F2C4C56B8D383D16A78B3"

SEEDS = list(range(100, 110))
BASE_OUTPUT_TOKEN = "v4c_model1499_play_equivalent_seed42_vx040_20s_canonical_v1"
BASE_PASS = "V4C_CONTACT_GAIT_PROBE_CANONICAL_PASS"

AGG_DIR = (
    REPO
    / "logs"
    / "evaluations"
    / "g1"
    / "v4c_multiseed_direction_dev_100_109_vx040_20s_v2"
)
SUMMARY_CSV = AGG_DIR / "multiseed_direction_summary.csv"
TEMP_DIR = AGG_DIR / "_temp_probes"

EXPECTED_ROWS = 1000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def wrap_pi(x: float) -> float:
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def sample_std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def parse_run(seed: int, run_dir: Path) -> dict[str, float | int]:
    csv_path = run_dir / "samples.csv"
    if not csv_path.is_file():
        raise SystemExit(f"ABORT: seed {seed}: samples.csv missing: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != EXPECTED_ROWS:
        raise SystemExit(f"ABORT: seed {seed}: expected {EXPECTED_ROWS} rows, found {len(rows)}")

    required = {
        "cmd_vx_mps", "cmd_vy_mps", "cmd_wz_radps",
        "base_vx_mps", "base_vy_mps", "base_wz_radps",
        "world_x_m", "world_y_m", "heading_rad",
    }
    missing = required.difference(rows[0].keys())
    if missing:
        raise SystemExit(f"ABORT: seed {seed}: missing columns: {sorted(missing)}")

    def col(name: str) -> list[float]:
        vals = [float(r[name]) for r in rows]
        if not all(math.isfinite(v) for v in vals):
            raise SystemExit(f"ABORT: seed {seed}: non-finite {name}")
        return vals

    cmd_vx = col("cmd_vx_mps")
    cmd_vy = col("cmd_vy_mps")
    cmd_wz = col("cmd_wz_radps")
    vx = col("base_vx_mps")
    vy = col("base_vy_mps")
    wz = col("base_wz_radps")
    x = col("world_x_m")
    y = col("world_y_m")
    heading = col("heading_rad")

    tol = 1.0e-12
    if any(abs(v - 0.4) > tol for v in cmd_vx):
        raise SystemExit(f"ABORT: seed {seed}: cmd_vx is not exactly 0.4 throughout")
    if any(abs(v) > tol for v in cmd_vy):
        raise SystemExit(f"ABORT: seed {seed}: cmd_vy is not exactly 0.0 throughout")
    if any(abs(v) > tol for v in cmd_wz):
        raise SystemExit(f"ABORT: seed {seed}: cmd_wz is not exactly 0.0 throughout")

    h0 = heading[0]
    herr = [wrap_pi(h - h0) for h in heading]
    mean_abs_herr_deg = mean([abs(math.degrees(v)) for v in herr])

    dx = x[-1] - x[0]
    dy = y[-1] - y[0]
    c = math.cos(h0)
    s = math.sin(h0)
    along = c * dx + s * dy
    cross = -s * dx + c * dy

    travel_heading = math.atan2(dy, dx)
    travel_heading_error_deg = math.degrees(wrap_pi(travel_heading - h0))

    return {
        "seed": seed,
        "mean_base_vx_mps": mean(vx),
        "mean_base_vy_mps": mean(vy),
        "mean_base_wz_radps": mean(wz),
        "initial_heading_deg": math.degrees(h0),
        "mean_abs_heading_error_deg": mean_abs_herr_deg,
        "global_dx_m": dx,
        "global_dy_m": dy,
        "initial_heading_along_track_m": along,
        "initial_heading_cross_track_m": cross,
        "cross_track_ratio_pct": 100.0 * abs(cross) / max(abs(along), 1.0e-12),
        "travel_heading_error_deg": travel_heading_error_deg,
    }


def derive_probe(source: str, seed: int, out_token: str, marker: str) -> str:
    replacements = [
        ("SEED = 42", f"SEED = {seed}", "seed"),
        (BASE_OUTPUT_TOKEN, out_token, "output token"),
        (BASE_PASS, marker, "PASS marker"),
    ]

    derived = source
    for old, new, label in replacements:
        count = derived.count(old)
        if count != 1:
            raise SystemExit(
                f"ABORT: seed {seed}: {label} occurrence mismatch: expected 1, found {count}"
            )
        derived = derived.replace(old, new, 1)

    reversed_text = derived
    for old, new, label in reversed(replacements):
        if reversed_text.count(new) != 1:
            raise SystemExit(f"ABORT: seed {seed}: reverse-check failed for {label}")
        reversed_text = reversed_text.replace(new, old, 1)

    if reversed_text != source:
        raise SystemExit(f"ABORT: seed {seed}: reversibility guard failed")

    compile(derived, f"<v4c_seed_{seed}_probe>", "exec")
    return derived


if not CANONICAL_PROBE.is_file():
    raise SystemExit(f"ABORT: canonical v4-C probe missing: {CANONICAL_PROBE}")

actual_sha = sha256(CANONICAL_PROBE)
if actual_sha != EXPECTED_PROBE_SHA:
    raise SystemExit(
        "ABORT: canonical v4-C probe SHA mismatch.\n"
        f"expected={EXPECTED_PROBE_SHA}\n"
        f"actual  ={actual_sha}"
    )

source = CANONICAL_PROBE.read_text(encoding="utf-8")

# Canonical-source guards.
for token in ("SEED = 42", BASE_OUTPUT_TOKEN, BASE_PASS):
    if source.count(token) != 1:
        raise SystemExit(
            f"ABORT: canonical probe token mismatch for {token!r}: "
            f"expected 1, found {source.count(token)}"
        )

# Ensure aggregate directory is either fresh or a compatible resumable workspace.
if SUMMARY_CSV.exists():
    raise SystemExit(
        f"ABORT: final summary already exists: {SUMMARY_CSV}\n"
        "This v2 audit appears already completed; refusing to overwrite evidence."
    )

AGG_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 124)
print("V4-C MULTI-SEED DIRECTION AUDIT V2 — DEVELOPMENT SEEDS 100–109")
print("=" * 124)
print(f"canonical probe      : {CANONICAL_PROBE}")
print(f"canonical probe SHA  : {actual_sha}")
print("scientific command   : [0.40, 0.00, 0.00]")
print("duration             : 20 s")
print("classification       : DEVELOPMENT/TUNING EVIDENCE — NOT FUTURE UNTOUCHED HOLDOUT")
print("resume policy        : reuse only complete existing per-seed samples.csv; never delete/overwrite")
print("=" * 124)

results: list[dict[str, float | int]] = []

for seed in SEEDS:
    out_token = f"v4c_model1499_play_equivalent_seed{seed}_vx040_20s_dev_v2"
    per_seed_dir = REPO / "logs" / "evaluations" / "g1" / out_token
    marker = f"V4C_DEV_V2_SEED_{seed}_CONTACT_GAIT_PROBE_PASS"
    temp_probe = TEMP_DIR / f"g1_v4c_contact_gait_probe_seed{seed}.py"

    print()
    print(f"--- seed {seed} ---")

    if per_seed_dir.exists():
        # Resumable/read-only reuse is accepted only if the canonical samples.csv is complete and valid.
        print(f"existing output found: {per_seed_dir}")
        print("action               : READ-ONLY REUSE after validation")
        r = parse_run(seed, per_seed_dir)
    else:
        derived = derive_probe(source, seed, out_token, marker)
        temp_probe.write_text(derived, encoding="utf-8", newline="\n")

        proc = subprocess.run(
            [sys.executable, str(temp_probe)],
            cwd=str(REPO),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        if proc.returncode != 0:
            print(proc.stdout)
            raise SystemExit(f"ABORT: seed {seed}: probe failed with code {proc.returncode}")

        if marker not in proc.stdout:
            print(proc.stdout)
            raise SystemExit(f"ABORT: seed {seed}: PASS marker missing")

        r = parse_run(seed, per_seed_dir)

    results.append(r)

    print(
        f"seed {seed}: "
        f"vx {float(r['mean_base_vx_mps']):+.4f}  "
        f"vy {float(r['mean_base_vy_mps']):+.4f}  "
        f"wz {float(r['mean_base_wz_radps']):+.4f}  "
        f"cross {float(r['initial_heading_cross_track_m']):+.4f} m  "
        f"mean|herr| {float(r['mean_abs_heading_error_deg']):.3f} deg"
    )

fieldnames = list(results[0].keys())
with SUMMARY_CSV.open("x", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(results)

vx_vals = [float(r["mean_base_vx_mps"]) for r in results]
vy_vals = [float(r["mean_base_vy_mps"]) for r in results]
wz_vals = [float(r["mean_base_wz_radps"]) for r in results]
cross_vals = [float(r["initial_heading_cross_track_m"]) for r in results]
herr_vals = [float(r["mean_abs_heading_error_deg"]) for r in results]

neg = sum(v < 0.0 for v in vy_vals)
pos = sum(v > 0.0 for v in vy_vals)
zero = len(vy_vals) - neg - pos

print()
print("=" * 124)
print("V4-C MULTI-SEED AGGREGATE V2")
print("=" * 124)
print(f"mean base vx                         : {mean(vx_vals):+.6f} m/s  std {sample_std(vx_vals):.6f}")
print(f"mean base vy                         : {mean(vy_vals):+.6f} m/s  std {sample_std(vy_vals):.6f}")
print(f"base vy sign count                   : negative={neg} positive={pos} zero={zero}")
print(f"mean base wz                         : {mean(wz_vals):+.6f} rad/s std {sample_std(wz_vals):.6f}")
print(f"mean initial-heading cross-track     : {mean(cross_vals):+.4f} m    std {sample_std(cross_vals):.4f}")
print(f"mean per-run mean |heading error|    : {mean(herr_vals):.3f} deg  std {sample_std(herr_vals):.3f}")
print(f"summary CSV                          : {SUMMARY_CSV}")
print()
print("V4C_MULTISEED_DIRECTION_AUDIT_V2_PASS")
print("=" * 124)
