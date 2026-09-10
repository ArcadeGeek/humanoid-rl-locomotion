from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path


REPO = Path(r"C:\humanoid_rl_lab")
SOURCE_PROBE = REPO / "scripts" / "thesis" / "v4" / "g1_v4b_contact_gait_probe.py"
EXPECTED_SOURCE_SHA256 = "40948E95BA457FCC1AFD2708619054FE734CCEFF515130A997AFFF3958A84848"

NEW_CHECKPOINT = (
    REPO
    / "logs"
    / "rsl_rl"
    / "humanoid_rl_lab_g1_v4j_straight_line_precision_scratch"
    / "2026-09-03_13-44-57_v4j_pilot_seed42_400"
    / "model_399.pt"
)
EXPECTED_CHECKPOINT_SHA256 = "91898D84D0979B73F5CE3DBC7FA3F4FD85AE2600E8CC05403F8EDB6C9DB05203"

OLD_TASK = "Humanoid-RL-Lab-G1-Flat-V4B-Direct-Yaw-v0"
NEW_TASK = "Humanoid-RL-Lab-G1-Flat-V4J-Straight-Line-Precision-v0"

OLD_EXPERIMENT_DIR = "humanoid_rl_lab_g1_v4b_direct_yaw_scratch"
NEW_EXPERIMENT_DIR = "humanoid_rl_lab_g1_v4j_straight_line_precision_scratch"

OLD_RUN_DIR = "2026-09-01_20-08-18_v4b_genuine_scratch_seed42_official1500"
NEW_RUN_DIR = "2026-09-03_13-44-57_v4j_pilot_seed42_400"

OLD_MODEL_LINE = 'CHECKPOINT = RUN_DIR / "model_1499.pt"'
NEW_MODEL_LINE = 'CHECKPOINT = RUN_DIR / "model_399.pt"'

OLD_OUT = "v4b_model1499_play_equivalent_seed42_vx040_20s_canonical_v1"
NEW_OUT = "v4j_model399_play_equivalent_seed42_vx040_20s_pilot_v1"

OLD_LABEL = '"v4a_model1499_play_equivalent_gait_probe"'
NEW_LABEL = '"v4j_model399_play_equivalent_gait_probe"'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected exactly one occurrence of {old!r}, found {count}"
        )
    return source.replace(old, new, 1)


if not SOURCE_PROBE.is_file():
    raise SystemExit(f"ABORT: frozen v4-B canonical probe missing: {SOURCE_PROBE}")

source_sha = sha256(SOURCE_PROBE)
if source_sha != EXPECTED_SOURCE_SHA256:
    raise SystemExit(
        "ABORT: frozen v4-B canonical probe SHA mismatch.\n"
        f"Expected: {EXPECTED_SOURCE_SHA256}\n"
        f"Actual  : {source_sha}\n"
        "Refusing to derive v4-J evaluation from a changed analysis script."
    )

if not NEW_CHECKPOINT.is_file():
    raise SystemExit(f"ABORT: v4-J pilot checkpoint missing: {NEW_CHECKPOINT}")

checkpoint_sha = sha256(NEW_CHECKPOINT)
if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
    raise SystemExit(
        "ABORT: v4-J model399 SHA mismatch.\n"
        f"Expected: {EXPECTED_CHECKPOINT_SHA256}\n"
        f"Actual  : {checkpoint_sha}"
    )

source = SOURCE_PROBE.read_text(encoding="utf-8")

# Change only runtime identity/provenance. The canonical analysis and evaluator instrumentation remain unchanged.
source = replace_once(source, OLD_TASK, NEW_TASK, "task")
source = replace_once(source, OLD_EXPERIMENT_DIR, NEW_EXPERIMENT_DIR, "experiment directory")
source = replace_once(source, OLD_RUN_DIR, NEW_RUN_DIR, "run directory")
source = replace_once(source, OLD_MODEL_LINE, NEW_MODEL_LINE, "checkpoint filename")
source = replace_once(source, OLD_OUT, NEW_OUT, "evaluation output directory")
source = replace_once(source, OLD_LABEL, NEW_LABEL, "evaluator label")

# Presentational labels only.
source = source.replace(
    "G1 v4-B CONTACT-AWARE GAIT PROBE",
    "G1 v4-J MODEL399 CONTACT-AWARE GAIT PROBE",
)
source = source.replace("V4-B MODEL_1499", "V4-J MODEL_399")
source = source.replace(
    "V4B_CONTACT_GAIT_PROBE_CANONICAL_PASS",
    "V4J_MODEL399_CONTACT_GAIT_PROBE_PASS",
)
source = source.replace("v4-B gait probe", "v4-J model399 gait probe")
source = source.replace("v4-B final checkpoint", "v4-J pilot checkpoint")

# Safety: none of the old runtime identities may remain.
for forbidden in (
    OLD_TASK,
    OLD_EXPERIMENT_DIR,
    OLD_RUN_DIR,
    OLD_MODEL_LINE,
    OLD_OUT,
):
    if forbidden in source:
        raise RuntimeError(f"old v4-B runtime identity remains in derived probe: {forbidden}")

# The frozen source already encodes seed42, [0.40,0,0], 3 s warmup, 20 s measurement,
# deterministic inference and PLAY-equivalent evaluator patching. Since the source SHA is frozen
# and only the exact substitutions above are applied, the scientific protocol is inherited unchanged.
required_unchanged_tokens = (
    '"--deterministic"',
    "env_cfg.observations.policy.enable_corruption = False",
    "env_cfg.events.base_external_force_torque = None",
    "env_cfg.events.push_robot = None",
    'print("Command         :", [VX, VY, WZ])',
    'print("Warmup / probe  :", f"{WARMUP_S:.1f} s / {DURATION_S:.1f} s")',
)
missing = [token for token in required_unchanged_tokens if token not in source]
if missing:
    raise RuntimeError(f"canonical protocol/instrumentation token missing: {missing}")

compile(source, "g1_v4j_model399_contact_gait_probe_derived.py", "exec")

print("=" * 120)
print("v4-J MODEL399 CONTACT-GAIT PROBE DERIVATION")
print("=" * 120)
print(f"Frozen source probe       : {SOURCE_PROBE}")
print(f"Frozen source probe SHA   : {source_sha} — PASS")
print(f"Task                      : {NEW_TASK}")
print(f"Checkpoint                : {NEW_CHECKPOINT}")
print(f"Checkpoint SHA            : {checkpoint_sha} — PASS")
print(f"Output directory label    : {NEW_OUT}")
print("Scientific protocol       : inherited unchanged from frozen v4-B canonical probe")
print("Repository source modified: NO")
print()

with tempfile.TemporaryDirectory(prefix="g1_v4j_probe_") as tmp:
    derived = Path(tmp) / "g1_v4j_model399_contact_gait_probe_derived.py"
    derived.write_text(source, encoding="utf-8", newline="\n")

    completed = subprocess.run(
        [sys.executable, str(derived)],
        cwd=str(REPO),
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(
            f"ABORT: derived v4-J contact-gait probe failed with code {completed.returncode}"
        )

print()
print("V4J_MODEL399_CONTACT_GAIT_WRAPPER_PASS")
print("=" * 120)
