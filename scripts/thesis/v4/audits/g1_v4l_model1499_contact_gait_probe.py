from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(r"C:\humanoid_rl_lab")
SOURCE_PROBE = REPO / "scripts" / "thesis" / "v4" / "g1_v4b_contact_gait_probe.py"
EXPECTED_SOURCE_SHA = "40948E95BA457FCC1AFD2708619054FE734CCEFF515130A997AFFF3958A84848"

RUN_DIR = (
    REPO / "logs" / "rsl_rl"
    / "humanoid_rl_lab_g1_v4l_path_pose_precision_scratch"
    / "2026-09-03_22-19-23_v4l_genuine_scratch_seed42_official1500"
)
MODEL = RUN_DIR / "model_1499.pt"
EXPECTED_MODEL_SHA = "9CE6C41956725116F08122CBBF4A8589EF645716A0C3509764472BAA8856802E"

OLD_TASK = "Humanoid-RL-Lab-G1-Flat-V4B-Direct-Yaw-v0"
NEW_TASK = "Humanoid-RL-Lab-G1-Flat-V4L-Path-Pose-Precision-v0"
OLD_EXP = "humanoid_rl_lab_g1_v4b_direct_yaw_scratch"
NEW_EXP = "humanoid_rl_lab_g1_v4l_path_pose_precision_scratch"
OLD_RUN = "2026-09-01_20-08-18_v4b_genuine_scratch_seed42_official1500"
NEW_RUN = "2026-09-03_22-19-23_v4l_genuine_scratch_seed42_official1500"
OLD_OUT = "v4b_model1499_play_equivalent_seed42_vx040_20s_canonical_v1"
NEW_OUT = "v4l_model1499_play_equivalent_seed42_vx040_20s_canonical_v1"
OLD_LABEL = '"v4a_model1499_play_equivalent_gait_probe"'
NEW_LABEL = '"v4l_model1499_play_equivalent_gait_probe"'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected exactly one occurrence of {old!r}, found {count}"
        )
    return text.replace(old, new, 1)


if not SOURCE_PROBE.is_file():
    raise SystemExit(f"ABORT: frozen v4-B probe missing: {SOURCE_PROBE}")

source_sha = sha256(SOURCE_PROBE)
if source_sha != EXPECTED_SOURCE_SHA:
    raise SystemExit(
        "ABORT: frozen v4-B probe SHA mismatch\n"
        f"Expected: {EXPECTED_SOURCE_SHA}\nActual  : {source_sha}"
    )

if not MODEL.is_file():
    raise SystemExit(f"ABORT: v4-L model missing: {MODEL}")

model_sha = sha256(MODEL)
if model_sha != EXPECTED_MODEL_SHA:
    raise SystemExit(
        "ABORT: v4-L model SHA mismatch\n"
        f"Expected: {EXPECTED_MODEL_SHA}\nActual  : {model_sha}"
    )

output_dir = REPO / "logs" / "evaluations" / "g1" / NEW_OUT
if output_dir.exists():
    raise SystemExit(
        f"ABORT: canonical v4-L output already exists: {output_dir}\n"
        "Refusing to overwrite prior evidence."
    )

source = SOURCE_PROBE.read_text(encoding="utf-8")
source = replace_once(source, OLD_TASK, NEW_TASK, "task")
source = replace_once(source, OLD_EXP, NEW_EXP, "experiment")
source = replace_once(source, OLD_RUN, NEW_RUN, "run")
source = replace_once(source, OLD_OUT, NEW_OUT, "output")
source = replace_once(source, OLD_LABEL, NEW_LABEL, "evaluator label")

source = source.replace(
    "G1 v4-B CONTACT-AWARE GAIT PROBE",
    "G1 v4-L CONTACT-AWARE GAIT PROBE",
)
source = source.replace("V4-B MODEL_1499", "V4-L MODEL_1499")
source = source.replace(
    "V4B_CONTACT_GAIT_PROBE_CANONICAL_PASS",
    "V4L_CONTACT_GAIT_PROBE_CANONICAL_PASS",
)
source = source.replace("v4-B gait probe", "v4-L gait probe")
source = source.replace("v4-B final checkpoint", "v4-L final checkpoint")

for stale in (OLD_TASK, OLD_EXP, OLD_RUN, OLD_OUT, OLD_LABEL):
    if stale in source:
        raise RuntimeError(f"stale v4-B runtime identity remains: {stale}")

for token in (
    "SEED = 42",
    "VX = 0.40",
    "VY = 0.0",
    "WZ = 0.0",
    '"--deterministic"',
    "env_cfg.observations.policy.enable_corruption = False",
    "env_cfg.events.base_external_force_torque = None",
    "env_cfg.events.push_robot = None",
    'print("Command         :", [VX, VY, WZ])',
):
    if token not in source:
        raise RuntimeError(f"frozen protocol token missing: {token}")

compile(source, "g1_v4l_model1499_contact_gait_probe_derived.py", "exec")

print("=" * 116)
print("v4-L CANONICAL CONTACT-AWARE GAIT PROBE WRAPPER")
print("=" * 116)
print(f"Frozen v4-B probe SHA : {source_sha} — PASS")
print(f"v4-L checkpoint SHA   : {model_sha} — PASS")
print(f"Task                  : {NEW_TASK}")
print(f"Run                   : {RUN_DIR}")
print(f"Output                : {output_dir}")
print("Seed                  : 42")
print("Command               : [0.40, 0.0, 0.0]")
print("Protocol              : frozen v4-B Play-equivalent evaluator")
print("Repository source     : unchanged")
print()

with tempfile.TemporaryDirectory(prefix="g1_v4l_canonical_") as td:
    derived = Path(td) / "g1_v4l_model1499_contact_gait_probe_derived.py"
    derived.write_text(source, encoding="utf-8", newline="\n")
    completed = subprocess.run(
        [sys.executable, str(derived)],
        cwd=str(REPO),
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(
            f"ABORT: derived v4-L probe failed with exit code {completed.returncode}"
        )

samples = output_dir / "samples.csv"
analysis = output_dir / "gait_probe_analysis.json"

if not samples.is_file():
    raise SystemExit(f"ABORT: samples.csv missing: {samples}")
if not analysis.is_file():
    raise SystemExit(f"ABORT: gait_probe_analysis.json missing: {analysis}")

print()
print("V4L_MODEL1499_CANONICAL_CONTACT_GAIT_WRAPPER_PASS")
