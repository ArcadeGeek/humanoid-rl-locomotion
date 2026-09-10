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
SOURCE_PROBE = REPO / "scripts" / "thesis" / "v4" / "g1_v4b_contact_gait_probe.py"
EXPECTED_SOURCE_SHA = "40948E95BA457FCC1AFD2708619054FE734CCEFF515130A997AFFF3958A84848"

V4J_RUN = (
    REPO / "logs" / "rsl_rl"
    / "humanoid_rl_lab_g1_v4j_straight_line_precision_scratch"
    / "2026-09-03_14-13-41_v4j_genuine_scratch_seed42_official1500"
)
V4J_MODEL = V4J_RUN / "model_1499.pt"
EXPECTED_MODEL_SHA = "798EB4A1668B240F9FEE9097942C60DDD496959DF33F3A7C1BA969184201C728"

OLD_TASK = "Humanoid-RL-Lab-G1-Flat-V4B-Direct-Yaw-v0"
NEW_TASK = "Humanoid-RL-Lab-G1-Flat-V4J-Straight-Line-Precision-v0"
OLD_EXP = "humanoid_rl_lab_g1_v4b_direct_yaw_scratch"
NEW_EXP = "humanoid_rl_lab_g1_v4j_straight_line_precision_scratch"
OLD_RUN = "2026-09-01_20-08-18_v4b_genuine_scratch_seed42_official1500"
NEW_RUN = "2026-09-03_14-13-41_v4j_genuine_scratch_seed42_official1500"
OLD_OUT = "v4b_model1499_play_equivalent_seed42_vx040_20s_canonical_v1"
OLD_LABEL = '"v4a_model1499_play_equivalent_gait_probe"'
OLD_SEED = "SEED = 42"

SEEDS = tuple(range(100, 110))
REQ = (
    "time_s", "cmd_vx_mps", "cmd_vy_mps", "cmd_wz_radps",
    "base_vx_mps", "base_vy_mps", "base_wz_radps",
    "world_x_m", "world_y_m", "heading_rad",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def repl1(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise RuntimeError(f"{label}: expected 1 occurrence, found {n}: {old!r}")
    return text.replace(old, new, 1)


def v4b_csv(seed: int) -> Path:
    return (
        REPO / "logs" / "evaluations" / "g1"
        / f"v4b_model1499_play_equivalent_seed{seed}_vx040_20s_direction_dev"
        / "samples.csv"
    )


def v4j_dir(seed: int) -> Path:
    return (
        REPO / "logs" / "evaluations" / "g1"
        / f"v4j_model1499_play_equivalent_seed{seed}_vx040_20s_direction_dev"
    )


def load(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise RuntimeError(f"missing header: {path}")
        missing = [c for c in REQ if c not in r.fieldnames]
        if missing:
            raise RuntimeError(f"missing columns {missing}: {path}")
        rows = list(r)

    if len(rows) != 1000:
        raise RuntimeError(f"expected 1000 samples, got {len(rows)}: {path}")

    d = {
        c: np.asarray([float(row[c]) for row in rows], dtype=np.float64)
        for c in REQ
    }

    dt = np.diff(d["time_s"])
    if not np.allclose(dt, 0.02, rtol=0.0, atol=1e-9):
        raise RuntimeError(f"bad dt in {path}: median={np.median(dt)}")

    cmd = np.column_stack((d["cmd_vx_mps"], d["cmd_vy_mps"], d["cmd_wz_radps"]))
    if not np.allclose(cmd, np.asarray([0.4, 0.0, 0.0])[None, :], rtol=0.0, atol=1e-12):
        raise RuntimeError(f"command not fixed [0.4,0,0]: {path}")

    return d


def metrics(d: dict[str, np.ndarray]) -> dict[str, float]:
    x, y = d["world_x_m"], d["world_y_m"]
    dx, dy = float(x[-1] - x[0]), float(y[-1] - y[0])

    h = np.unwrap(d["heading_rad"])
    h0 = float(h[0])
    herr = h - h0

    c, s = math.cos(h0), math.sin(h0)
    forward = c * dx + s * dy
    cross = -s * dx + c * dy

    return {
        "vx": float(np.mean(d["base_vx_mps"])),
        "vy": float(np.mean(d["base_vy_mps"])),
        "wz": float(np.mean(d["base_wz_radps"])),
        "cross": cross,
        "forward": forward,
        "mean_abs_herr": math.degrees(float(np.mean(np.abs(herr)))),
        "final_herr": math.degrees(float(herr[-1])),
    }


def mean_std(values):
    a = np.asarray(values, dtype=np.float64)
    return float(np.mean(a)), float(np.std(a, ddof=0))


if not SOURCE_PROBE.is_file():
    raise SystemExit(f"ABORT: source probe missing: {SOURCE_PROBE}")
if sha256(SOURCE_PROBE) != EXPECTED_SOURCE_SHA:
    raise SystemExit(
        f"ABORT: source probe SHA mismatch\nExpected: {EXPECTED_SOURCE_SHA}\n"
        f"Actual  : {sha256(SOURCE_PROBE)}"
    )
if not V4J_MODEL.is_file():
    raise SystemExit(f"ABORT: v4-J model missing: {V4J_MODEL}")
if sha256(V4J_MODEL) != EXPECTED_MODEL_SHA:
    raise SystemExit(
        f"ABORT: v4-J model SHA mismatch\nExpected: {EXPECTED_MODEL_SHA}\n"
        f"Actual  : {sha256(V4J_MODEL)}"
    )

for seed in SEEDS:
    if not v4b_csv(seed).is_file():
        raise SystemExit(f"ABORT: missing existing v4-B dev CSV for seed {seed}: {v4b_csv(seed)}")
    if v4j_dir(seed).exists():
        raise SystemExit(
            f"ABORT: v4-J output already exists for seed {seed}: {v4j_dir(seed)}\n"
            "Refusing to overwrite prior evidence."
        )

base = SOURCE_PROBE.read_text(encoding="utf-8")

print("=" * 120)
print("v4-J MULTI-SEED DIRECTION DEV — SEEDS 100-109")
print("=" * 120)
print("Seeds             : 100-109 (development only; NOT holdout)")
print("Command           : [0.40, 0.0, 0.0]")
print("Protocol          : frozen v4-B canonical probe, Play-equivalent")
print("v4-B comparison   : existing seed-matched CSVs, read-only")
print("Repository source : unchanged")
print()

with tempfile.TemporaryDirectory(prefix="g1_v4j_dev_") as td:
    td = Path(td)

    for seed in SEEDS:
        new_out = f"v4j_model1499_play_equivalent_seed{seed}_vx040_20s_direction_dev"
        new_label = f'"v4j_model1499_seed{seed}_direction_dev"'

        src = base
        src = repl1(src, OLD_TASK, NEW_TASK, "task")
        src = repl1(src, OLD_EXP, NEW_EXP, "experiment")
        src = repl1(src, OLD_RUN, NEW_RUN, "run")
        src = repl1(src, OLD_OUT, new_out, "output")
        src = repl1(src, OLD_LABEL, new_label, "label")
        src = repl1(src, OLD_SEED, f"SEED = {seed}", "seed")

        for token in (
            '"--deterministic"',
            "env_cfg.observations.policy.enable_corruption = False",
            "env_cfg.events.base_external_force_torque = None",
            "env_cfg.events.push_robot = None",
        ):
            if token not in src:
                raise RuntimeError(f"seed {seed}: missing canonical token: {token}")

        compile(src, f"v4j_seed{seed}_derived.py", "exec")
        derived = td / f"v4j_seed{seed}_derived.py"
        derived.write_text(src, encoding="utf-8", newline="\n")

        print(f"[seed {seed}] running...")
        p = subprocess.run(
            [sys.executable, str(derived)],
            cwd=str(REPO),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if p.returncode != 0:
            print("\n".join(p.stdout.decode(errors="replace").splitlines()[-120:]))
            raise SystemExit(f"ABORT: seed {seed} failed with code {p.returncode}")

        csv_out = v4j_dir(seed) / "samples.csv"
        if not csv_out.is_file():
            raise SystemExit(f"ABORT: seed {seed} samples.csv missing: {csv_out}")
        print(f"[seed {seed}] PASS")

pairs = []
for seed in SEEDS:
    b = metrics(load(v4b_csv(seed)))
    j = metrics(load(v4j_dir(seed) / "samples.csv"))
    pairs.append((seed, b, j))

print()
print("=" * 146)
print("PAIRED RESULTS — DEVELOPMENT SEEDS 100-109")
print("=" * 146)
print("seed | B vy      J vy      | B vx      J vx      | |B cross| |J cross| | B mean|herr| J mean|herr|")
print("-" * 146)
for seed, b, j in pairs:
    print(
        f"{seed:4d} | {b['vy']:+.6f} {j['vy']:+.6f} | "
        f"{b['vx']:+.6f} {j['vx']:+.6f} | "
        f"{abs(b['cross']):9.4f} {abs(j['cross']):9.4f} | "
        f"{b['mean_abs_herr']:12.3f} {j['mean_abs_herr']:12.3f}"
    )

def side_values(which: str, key: str):
    idx = 1 if which == "B" else 2
    return [row[idx][key] for row in pairs]

b_vx = mean_std(side_values("B", "vx"))
j_vx = mean_std(side_values("J", "vx"))
b_vy = mean_std(side_values("B", "vy"))
j_vy = mean_std(side_values("J", "vy"))
b_wz = mean_std(side_values("B", "wz"))
j_wz = mean_std(side_values("J", "wz"))

b_cross = mean_std(side_values("B", "cross"))
j_cross = mean_std(side_values("J", "cross"))
b_abs_cross = mean_std([abs(v) for v in side_values("B", "cross")])
j_abs_cross = mean_std([abs(v) for v in side_values("J", "cross")])

b_h = mean_std(side_values("B", "mean_abs_herr"))
j_h = mean_std(side_values("J", "mean_abs_herr"))
b_final_abs = mean_std([abs(v) for v in side_values("B", "final_herr")])
j_final_abs = mean_std([abs(v) for v in side_values("J", "final_herr")])

d_vx = mean_std([j["vx"] - b["vx"] for _, b, j in pairs])
d_vy = mean_std([j["vy"] - b["vy"] for _, b, j in pairs])
d_abs_cross = mean_std([abs(j["cross"]) - abs(b["cross"]) for _, b, j in pairs])
d_h = mean_std([j["mean_abs_herr"] - b["mean_abs_herr"] for _, b, j in pairs])

negative_b = sum(v < 0.0 for v in side_values("B", "vy"))
negative_j = sum(v < 0.0 for v in side_values("J", "vy"))
cross_better = sum(abs(j["cross"]) < abs(b["cross"]) for _, b, j in pairs)
heading_better = sum(j["mean_abs_herr"] < b["mean_abs_herr"] for _, b, j in pairs)

print()
print("-" * 146)
print("AGGREGATE — mean ± population std across 10 development seeds")
print("-" * 146)
print(f"mean vx B                    : {b_vx[0]:+.6f} ± {b_vx[1]:.6f} m/s")
print(f"mean vx J                    : {j_vx[0]:+.6f} ± {j_vx[1]:.6f} m/s")
print(f"paired delta vx J-B          : {d_vx[0]:+.6f} ± {d_vx[1]:.6f} m/s")
print()
print(f"mean vy B                    : {b_vy[0]:+.6f} ± {b_vy[1]:.6f} m/s")
print(f"mean vy J                    : {j_vy[0]:+.6f} ± {j_vy[1]:.6f} m/s")
print(f"paired delta vy J-B          : {d_vy[0]:+.6f} ± {d_vy[1]:.6f} m/s")
print(f"negative mean-vy seeds B/J   : {negative_b}/10 / {negative_j}/10")
print()
print(f"mean wz B                    : {b_wz[0]:+.6f} ± {b_wz[1]:.6f} rad/s")
print(f"mean wz J                    : {j_wz[0]:+.6f} ± {j_wz[1]:.6f} rad/s")
print()
print(f"signed cross-track B         : {b_cross[0]:+.4f} ± {b_cross[1]:.4f} m")
print(f"signed cross-track J         : {j_cross[0]:+.4f} ± {j_cross[1]:.4f} m")
print(f"mean |cross-track| B         : {b_abs_cross[0]:.4f} ± {b_abs_cross[1]:.4f} m")
print(f"mean |cross-track| J         : {j_abs_cross[0]:.4f} ± {j_abs_cross[1]:.4f} m")
print(f"paired delta |cross| J-B     : {d_abs_cross[0]:+.4f} ± {d_abs_cross[1]:.4f} m")
print(f"|cross-track| improved seeds : {cross_better}/10")
print()
print(f"mean |heading error| B       : {b_h[0]:.3f} ± {b_h[1]:.3f} deg")
print(f"mean |heading error| J       : {j_h[0]:.3f} ± {j_h[1]:.3f} deg")
print(f"paired delta mean|herr| J-B  : {d_h[0]:+.3f} ± {d_h[1]:.3f} deg")
print(f"heading improved seeds       : {heading_better}/10")
print(f"mean final |heading| B       : {b_final_abs[0]:.3f} ± {b_final_abs[1]:.3f} deg")
print(f"mean final |heading| J       : {j_final_abs[0]:.3f} ± {j_final_abs[1]:.3f} deg")
print()
print("Scope                        : development comparison only; seeds 100-109 are NOT holdout")
print("Repository source modified   : NO")
print()
print("V4J_MULTI_SEED_DIRECTION_DEV_100_109_PASS")
print("=" * 146)
