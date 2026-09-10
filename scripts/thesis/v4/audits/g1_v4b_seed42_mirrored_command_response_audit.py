from __future__ import annotations

import csv
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path


REPO = Path(r"C:\humanoid_rl_lab")

SOURCE_PROBE = (
    REPO
    / "scripts"
    / "thesis"
    / "v4"
    / "g1_v4b_contact_gait_probe.py"
)
EXPECTED_SOURCE_SHA = (
    "40948E95BA457FCC1AFD2708619054FE734CCEFF515130A997AFFF3958A84848"
)

MODEL = (
    REPO
    / "logs"
    / "rsl_rl"
    / "humanoid_rl_lab_g1_v4b_direct_yaw_scratch"
    / "2026-09-01_20-08-18_v4b_genuine_scratch_seed42_official1500"
    / "model_1499.pt"
)
EXPECTED_MODEL_SHA = (
    "B6B9E94DC222F21AF698BA158B9E91C18071FE90ABC255B9C4311874E106EC7F"
)

OLD_OUT = "v4b_model1499_play_equivalent_seed42_vx040_20s_canonical_v1"

CASES = (
    ("baseline", 0.40, 0.00, 0.00),
    ("vy_p020", 0.40, +0.20, 0.00),
    ("vy_m020", 0.40, -0.20, 0.00),
    ("wz_p020", 0.40, 0.00, +0.20),
    ("wz_m020", 0.40, 0.00, -0.20),
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected exactly one occurrence of {old!r}, found {count}"
        )
    return text.replace(old, new, 1)


def output_name(case_name: str) -> str:
    return (
        "v4b_seed42_mirrored_command_response_"
        + case_name
        + "_play_equivalent_evalseed42_20s_v1"
    )


def read_means(csv_path: Path) -> dict[str, float]:
    required = (
        "cmd_vx_mps",
        "cmd_vy_mps",
        "cmd_wz_radps",
        "base_vx_mps",
        "base_vy_mps",
        "base_wz_radps",
    )

    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        reader = csv.DictReader(stream)

        if reader.fieldnames is None:
            raise RuntimeError(
                f"CSV has no header: {csv_path}"
            )

        missing = [
            name
            for name in required
            if name not in reader.fieldnames
        ]

        if missing:
            raise RuntimeError(
                f"missing columns in {csv_path}: {missing}"
            )

        rows = list(reader)

    if len(rows) != 1000:
        raise RuntimeError(
            f"expected 1000 samples in {csv_path}, got {len(rows)}"
        )

    sums = {
        name: 0.0
        for name in required
    }

    for row in rows:
        for name in required:
            sums[name] += float(row[name])

    return {
        name: sums[name] / len(rows)
        for name in required
    }


if not SOURCE_PROBE.is_file():
    raise SystemExit(
        f"ABORT: frozen v4-B canonical probe missing: {SOURCE_PROBE}"
    )

source_sha = sha256(SOURCE_PROBE)

if source_sha != EXPECTED_SOURCE_SHA:
    raise SystemExit(
        "ABORT: frozen v4-B probe SHA mismatch.\n"
        f"Expected: {EXPECTED_SOURCE_SHA}\n"
        f"Actual  : {source_sha}"
    )

if not MODEL.is_file():
    raise SystemExit(
        f"ABORT: frozen v4-B model missing: {MODEL}"
    )

model_sha = sha256(MODEL)

if model_sha != EXPECTED_MODEL_SHA:
    raise SystemExit(
        "ABORT: frozen v4-B model SHA mismatch.\n"
        f"Expected: {EXPECTED_MODEL_SHA}\n"
        f"Actual  : {model_sha}"
    )

base_source = SOURCE_PROBE.read_text(
    encoding="utf-8"
)

for token in (
    "SEED = 42",
    "VX = 0.40",
    "VY = 0.0",
    "WZ = 0.0",
    '"--deterministic"',
    "env_cfg.observations.policy.enable_corruption = False",
    "env_cfg.events.base_external_force_torque = None",
    "env_cfg.events.push_robot = None",
):
    if token not in base_source:
        raise RuntimeError(
            f"frozen protocol token missing: {token}"
        )

print("=" * 132)
print("v4-B SEED42 MIRRORED COMMAND RESPONSE AUDIT")
print("=" * 132)
print(f"Frozen probe SHA : {source_sha} — PASS")
print(f"Frozen model SHA : {model_sha} — PASS")
print("Evaluation seed  : 42 for every case")
print("Duration         : 20 s / 1000 samples per case")
print("Protocol         : frozen Play-equivalent evaluator")
print("Cases            : baseline, vy ±0.20, wz ±0.20")
print()

results = {}

with tempfile.TemporaryDirectory(
    prefix="g1_v4b_mirror_cmd_"
) as temp_dir:
    temp_dir = Path(temp_dir)

    for case_name, vx, vy, wz in CASES:
        out_name = output_name(case_name)
        output_dir = (
            REPO
            / "logs"
            / "evaluations"
            / "g1"
            / out_name
        )

        if output_dir.exists():
            raise SystemExit(
                f"ABORT: output already exists: {output_dir}"
            )

        source = base_source

        source = replace_once(
            source,
            OLD_OUT,
            out_name,
            f"{case_name} output",
        )

        if vx != 0.40:
            source = replace_once(
                source,
                "VX = 0.40",
                f"VX = {vx:.2f}",
                f"{case_name} VX",
            )

        if vy != 0.00:
            source = replace_once(
                source,
                "VY = 0.0",
                f"VY = {vy:+.2f}",
                f"{case_name} VY",
            )

        if wz != 0.00:
            source = replace_once(
                source,
                "WZ = 0.0",
                f"WZ = {wz:+.2f}",
                f"{case_name} WZ",
            )

        compile(
            source,
            f"derived_{case_name}.py",
            "exec",
        )

        derived = (
            temp_dir
            / f"derived_{case_name}.py"
        )

        derived.write_text(
            source,
            encoding="utf-8",
            newline="\n",
        )

        print(
            f"RUN {case_name:8s} : "
            f"[{vx:+.2f}, {vy:+.2f}, {wz:+.2f}]"
        )

        completed = subprocess.run(
            [sys.executable, str(derived)],
            cwd=str(REPO),
            check=False,
        )

        if completed.returncode != 0:
            raise SystemExit(
                f"ABORT: {case_name} evaluator failed "
                f"with exit code {completed.returncode}"
            )

        samples_csv = (
            output_dir
            / "samples.csv"
        )

        if not samples_csv.is_file():
            raise SystemExit(
                f"ABORT: samples.csv missing for {case_name}: "
                f"{samples_csv}"
            )

        means = read_means(
            samples_csv
        )

        tolerance = 1.0e-12

        if (
            abs(means["cmd_vx_mps"] - vx) > tolerance
            or abs(means["cmd_vy_mps"] - vy) > tolerance
            or abs(means["cmd_wz_radps"] - wz) > tolerance
        ):
            raise RuntimeError(
                f"{case_name}: measured command does not match requested command"
            )

        results[case_name] = means

print()
print("-" * 132)
print("MEAN BODY-FRAME RESPONSE")
print("-" * 132)
print(
    "case      | cmd vy | mean vy | vy error | "
    "cmd wz | mean wz | wz error | mean vx"
)

for case_name, _, _, _ in CASES:
    m = results[case_name]

    vy_error = (
        m["base_vy_mps"]
        - m["cmd_vy_mps"]
    )
    wz_error = (
        m["base_wz_radps"]
        - m["cmd_wz_radps"]
    )

    print(
        f"{case_name:9s} | "
        f"{m['cmd_vy_mps']:+.4f} | "
        f"{m['base_vy_mps']:+.6f} | "
        f"{vy_error:+.6f} | "
        f"{m['cmd_wz_radps']:+.4f} | "
        f"{m['base_wz_radps']:+.6f} | "
        f"{wz_error:+.6f} | "
        f"{m['base_vx_mps']:+.6f}"
    )

vp = results["vy_p020"]
vm = results["vy_m020"]
wp = results["wz_p020"]
wm = results["wz_m020"]
b = results["baseline"]

vy_response_center = 0.5 * (
    vp["base_vy_mps"]
    + vm["base_vy_mps"]
)

vy_response_halfspan = 0.5 * (
    vp["base_vy_mps"]
    - vm["base_vy_mps"]
)

vy_error_plus = (
    vp["base_vy_mps"]
    - vp["cmd_vy_mps"]
)

vy_error_minus = (
    vm["base_vy_mps"]
    - vm["cmd_vy_mps"]
)

vy_error_center = 0.5 * (
    vy_error_plus
    + vy_error_minus
)

wz_response_center = 0.5 * (
    wp["base_wz_radps"]
    + wm["base_wz_radps"]
)

wz_response_halfspan = 0.5 * (
    wp["base_wz_radps"]
    - wm["base_wz_radps"]
)

wz_error_plus = (
    wp["base_wz_radps"]
    - wp["cmd_wz_radps"]
)

wz_error_minus = (
    wm["base_wz_radps"]
    - wm["cmd_wz_radps"]
)

wz_error_center = 0.5 * (
    wz_error_plus
    + wz_error_minus
)

print()
print("-" * 132)
print("MIRROR-SYMMETRY CLOSURE")
print("-" * 132)
print(
    f"baseline mean vy                     : "
    f"{b['base_vy_mps']:+.6f} m/s"
)
print(
    f"vy mirrored response center          : "
    f"{vy_response_center:+.6f} m/s"
)
print(
    f"vy mirrored response half-span       : "
    f"{vy_response_halfspan:+.6f} m/s"
)
print(
    f"vy mirrored tracking-error center    : "
    f"{vy_error_center:+.6f} m/s"
)
print(
    f"baseline mean wz                     : "
    f"{b['base_wz_radps']:+.6f} rad/s"
)
print(
    f"wz mirrored response center          : "
    f"{wz_response_center:+.6f} rad/s"
)
print(
    f"wz mirrored response half-span       : "
    f"{wz_response_halfspan:+.6f} rad/s"
)
print(
    f"wz mirrored tracking-error center    : "
    f"{wz_error_center:+.6f} rad/s"
)
print()
print(
    "Interpretation: for a mirror-symmetric closed-loop response, "
    "the positive/negative response centers should be near zero. "
    "A persistent signed center indicates directional asymmetry "
    "under otherwise mirrored commands."
)
print()
print("V4B_SEED42_MIRRORED_COMMAND_RESPONSE_AUDIT_PASS")
print("=" * 132)
