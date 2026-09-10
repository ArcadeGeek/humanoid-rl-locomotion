from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile

REPO = Path(r"C:\humanoid_rl_lab")
CANONICAL_PROBE = REPO / "scripts" / "thesis" / "v4" / "g1_v4b_contact_gait_probe.py"
EXPECTED_PROBE_SHA256 = "40948E95BA457FCC1AFD2708619054FE734CCEFF515130A997AFFF3958A84848"

SEEDS = list(range(100, 110))
BASE_OUT = REPO / "logs" / "evaluations" / "g1" / "v4b_multiseed_direction_dev_100_109_vx040_20s"
SUMMARY_CSV = BASE_OUT / "multiseed_direction_summary.csv"

CANONICAL_OUT_TOKEN = "v4b_model1499_play_equivalent_seed42_vx040_20s_canonical_v1"
CANONICAL_PASS = "V4B_CONTACT_GAIT_PROBE_CANONICAL_PASS"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def wrap_pi(x: float) -> float:
    while x > math.pi:
        x -= 2.0 * math.pi
    while x < -math.pi:
        x += 2.0 * math.pi
    return x


def avg(xs):
    return sum(xs) / len(xs)


def sample_std(xs):
    if len(xs) < 2:
        return 0.0
    m = avg(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def replace_seed_argument(source: str, seed: int) -> tuple[str, str]:
    """
    Replace only the evaluator CLI value immediately following --seed.
    Accepts forms such as:
        "--seed", "42"
        "--seed", str(SEED)
        "--seed", str(args.seed)
        '--seed', SOME_VAR
    The replacement is always a literal string containing the requested seed.
    """
    pattern = re.compile(
        r'(?P<prefix>["\']--seed["\']\s*,\s*)'
        r'(?P<value>'
        r'(?:"[^"\r\n]*"|\'[^\'\r\n]*\'|'
        r'str\s*\([^,\]\r\n]+\)|'
        r'[^,\]\r\n]+)'
        r')',
        flags=re.MULTILINE,
    )
    matches = list(pattern.finditer(source))
    if len(matches) != 1:
        diagnostics = []
        for i, line in enumerate(source.splitlines(), 1):
            if "seed" in line.lower():
                diagnostics.append(f"{i}: {line}")
        diag_text = "\n".join(diagnostics[:30]) if diagnostics else "(no lines containing 'seed')"
        raise RuntimeError(
            "Could not uniquely identify evaluator --seed argument.\n"
            f"matches={len(matches)}\n"
            "Seed-related canonical-probe lines:\n"
            f"{diag_text}"
        )

    m = matches[0]
    old_value = m.group("value").strip()
    replacement = m.group("prefix") + f'"{seed}"'
    transformed = source[:m.start()] + replacement + source[m.end():]
    return transformed, old_value


if not CANONICAL_PROBE.is_file():
    raise SystemExit(f"ABORT: canonical v4-B probe missing: {CANONICAL_PROBE}")

actual_probe_sha = sha256(CANONICAL_PROBE)
if actual_probe_sha != EXPECTED_PROBE_SHA256:
    raise SystemExit(
        "ABORT: canonical v4-B probe SHA mismatch.\n"
        f"expected={EXPECTED_PROBE_SHA256}\n"
        f"actual  ={actual_probe_sha}"
    )

source = CANONICAL_PROBE.read_text(encoding="utf-8")

if source.count(CANONICAL_OUT_TOKEN) != 1:
    raise SystemExit(
        "ABORT: canonical output token not found exactly once in v4-B probe."
    )
if source.count(CANONICAL_PASS) != 1:
    raise SystemExit(
        "ABORT: canonical PASS marker not found exactly once in v4-B probe."
    )

# Preflight seed-replacement discovery before creating any output directory.
try:
    _preflight, discovered_seed_expr = replace_seed_argument(source, 123456)
except Exception as exc:
    raise SystemExit(f"ABORT: seed preflight failed.\n{exc}")

print("=" * 124)
print("V4-B MULTI-SEED DIRECTION AUDIT V2 — DEVELOPMENT SEEDS 100..109")
print("=" * 124)
print("Canonical probe       :", CANONICAL_PROBE)
print("Canonical probe SHA   :", actual_probe_sha)
print("Discovered seed expr  :", discovered_seed_expr)
print("Checkpoint/protocol   : inherited unchanged from canonical v4-B probe")
print("Command               : [0.40, 0.00, 0.00]")
print("Play-equivalent       : unchanged")
print("Duration              : 20 s (+ canonical warmup)")
print("Seeds                 :", SEEDS)
print("Classification        : DEVELOPMENT/TUNING DIAGNOSTIC — NOT HOLDOUT")
print("=" * 124)

BASE_OUT.mkdir(parents=True, exist_ok=True)
summary_rows = []

with tempfile.TemporaryDirectory(prefix="v4b_multiseed_") as td:
    tmpdir = Path(td)

    for seed in SEEDS:
        seed_out_name = f"v4b_model1499_play_equivalent_seed{seed}_vx040_20s_direction_dev"
        expected_out = REPO / "logs" / "evaluations" / "g1" / seed_out_name

        if expected_out.exists():
            raise SystemExit(
                f"ABORT: output directory already exists for seed {seed}: {expected_out}\n"
                "Refusing to overwrite/reuse ambiguous evidence."
            )

        try:
            seeded_source, _old_expr = replace_seed_argument(source, seed)
        except Exception as exc:
            raise SystemExit(f"ABORT: seed replacement failed for seed {seed}.\n{exc}")

        seeded_source = seeded_source.replace(CANONICAL_OUT_TOKEN, seed_out_name, 1)
        seeded_source = seeded_source.replace(
            CANONICAL_PASS,
            f"V4B_CONTACT_GAIT_PROBE_SEED_{seed}_PASS",
            1,
        )

        # Guards: only provenance/seed tokens should differ in this temporary clone.
        if CANONICAL_OUT_TOKEN in seeded_source or CANONICAL_PASS in seeded_source:
            raise SystemExit(f"ABORT: stale canonical token survived for seed {seed}.")

        compile(seeded_source, f"<v4b-seed-{seed}>", "exec")
        temp_probe = tmpdir / f"g1_v4b_contact_gait_probe_seed{seed}.py"
        temp_probe.write_text(seeded_source, encoding="utf-8", newline="\n")

        print()
        print(f"[SEED {seed}] START", flush=True)
        p = subprocess.run(
            [sys.executable, str(temp_probe)],
            cwd=REPO,
            text=True,
        )
        if p.returncode != 0:
            raise SystemExit(
                f"ABORT: canonical-derived probe failed for seed {seed}, code {p.returncode}."
            )

        samples = expected_out / "samples.csv"
        if not samples.is_file():
            raise SystemExit(f"ABORT: samples.csv missing for seed {seed}: {samples}")

        with samples.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        if len(rows) < 2:
            raise SystemExit(f"ABORT: insufficient samples for seed {seed}.")

        required = {
            "cmd_vx_mps", "cmd_vy_mps", "cmd_wz_radps",
            "base_vx_mps", "base_vy_mps", "base_wz_radps",
            "heading_rad", "world_x_m", "world_y_m",
        }
        missing = required.difference(rows[0].keys())
        if missing:
            raise SystemExit(f"ABORT: seed {seed} CSV missing columns: {sorted(missing)}")

        def col(name):
            return [float(r[name]) for r in rows]

        cmd_vx = col("cmd_vx_mps")
        cmd_vy = col("cmd_vy_mps")
        cmd_wz = col("cmd_wz_radps")
        base_vx = col("base_vx_mps")
        base_vy = col("base_vy_mps")
        base_wz = col("base_wz_radps")
        heading = col("heading_rad")
        world_x = col("world_x_m")
        world_y = col("world_y_m")

        if max(abs(v - 0.4) for v in cmd_vx) > 1.0e-9:
            raise SystemExit(f"ABORT: seed {seed} cmd_vx not fixed at 0.4.")
        if max(abs(v) for v in cmd_vy) > 1.0e-9:
            raise SystemExit(f"ABORT: seed {seed} cmd_vy not fixed at 0.")
        if max(abs(v) for v in cmd_wz) > 1.0e-9:
            raise SystemExit(f"ABORT: seed {seed} cmd_wz not fixed at 0.")

        h0 = heading[0]
        heading_errors = [wrap_pi(h - h0) for h in heading]
        abs_heading_errors = [abs(e) for e in heading_errors]

        dx = world_x[-1] - world_x[0]
        dy = world_y[-1] - world_y[0]
        c = math.cos(h0)
        s = math.sin(h0)
        along = c * dx + s * dy
        cross = -s * dx + c * dy
        travel_heading = math.atan2(dy, dx)
        travel_heading_error = wrap_pi(travel_heading - h0)

        row = {
            "seed": seed,
            "samples": len(rows),
            "mean_base_vx_mps": avg(base_vx),
            "mean_base_vy_mps": avg(base_vy),
            "mean_base_wz_radps": avg(base_wz),
            "initial_heading_deg": math.degrees(h0),
            "final_heading_error_deg": math.degrees(heading_errors[-1]),
            "mean_abs_heading_error_deg": math.degrees(avg(abs_heading_errors)),
            "max_abs_heading_error_deg": math.degrees(max(abs_heading_errors)),
            "global_dx_m": dx,
            "global_dy_m": dy,
            "along_track_initial_heading_m": along,
            "cross_track_initial_heading_m": cross,
            "travel_heading_error_vs_initial_deg": math.degrees(travel_heading_error),
            "output_dir": str(expected_out),
        }
        summary_rows.append(row)

        print(
            f"[SEED {seed}] "
            f"vx={row['mean_base_vx_mps']:+.4f} "
            f"vy={row['mean_base_vy_mps']:+.4f} "
            f"wz={row['mean_base_wz_radps']:+.4f} "
            f"cross={row['cross_track_initial_heading_m']:+.4f} m "
            f"mean|herr|={row['mean_abs_heading_error_deg']:.3f} deg "
            f"PASS",
            flush=True,
        )

with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
    writer.writeheader()
    writer.writerows(summary_rows)

vy_means = [float(r["mean_base_vy_mps"]) for r in summary_rows]
vx_means = [float(r["mean_base_vx_mps"]) for r in summary_rows]
wz_means = [float(r["mean_base_wz_radps"]) for r in summary_rows]
crosses = [float(r["cross_track_initial_heading_m"]) for r in summary_rows]
heading_mae = [float(r["mean_abs_heading_error_deg"]) for r in summary_rows]

neg_vy_count = sum(v < 0.0 for v in vy_means)
pos_vy_count = sum(v > 0.0 for v in vy_means)

print()
print("=" * 124)
print("V4-B MULTI-SEED DIRECTION SUMMARY")
print("=" * 124)
print(f"seeds                                  : {SEEDS[0]}..{SEEDS[-1]} ({len(SEEDS)} runs)")
print(f"mean base vx                           : {avg(vx_means):+.6f} m/s  (std {sample_std(vx_means):.6f})")
print(f"mean base vy                           : {avg(vy_means):+.6f} m/s  (std {sample_std(vy_means):.6f})")
print(f"base vy sign count                     : negative={neg_vy_count}, positive={pos_vy_count}, zero={len(vy_means)-neg_vy_count-pos_vy_count}")
print(f"mean base wz                           : {avg(wz_means):+.6f} rad/s (std {sample_std(wz_means):.6f})")
print(f"mean initial-heading cross-track       : {avg(crosses):+.4f} m    (std {sample_std(crosses):.4f})")
print(f"mean per-run mean |heading error|      : {avg(heading_mae):.3f} deg  (std {sample_std(heading_mae):.3f})")
print(f"summary CSV                            : {SUMMARY_CSV}")
print()
print("INTERPRETATION GUARD")
print("  Seeds 100..109 are DEVELOPMENT/TUNING evidence.")
print("  They must not be reused as final untouched holdout if v4-C is designed from these results.")
print()
print("V4B_MULTISEED_DIRECTION_AUDIT_PASS")
print("=" * 124)
