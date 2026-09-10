from __future__ import annotations
import hashlib, subprocess, sys, tempfile
from pathlib import Path

REPO = Path(r"C:\humanoid_rl_lab")
SOURCE_PROBE = REPO / "scripts" / "thesis" / "v4" / "g1_v4b_contact_gait_probe.py"
EXPECTED_SOURCE_SHA = "40948E95BA457FCC1AFD2708619054FE734CCEFF515130A997AFFF3958A84848"
RUN_DIR = REPO / "logs" / "rsl_rl" / "humanoid_rl_lab_g1_v4b_direct_yaw_scratch" / "2026-09-04_08-36-41_v4b_genuine_scratch_seed43_official1500"
MODEL = RUN_DIR / "model_1499.pt"
EXPECTED_MODEL_SHA = "34AF5696A402C71B2F08EB4F0A23066E8422E4432D257D8E063B3944CC02C19E"
OLD_RUN = "2026-09-01_20-08-18_v4b_genuine_scratch_seed42_official1500"
NEW_RUN = "2026-09-04_08-36-41_v4b_genuine_scratch_seed43_official1500"
OLD_OUT = "v4b_model1499_play_equivalent_seed42_vx040_20s_canonical_v1"
NEW_OUT = "v4b_trainseed43_model1499_play_equivalent_evalseed42_vx040_20s_canonical_v1"

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()

def replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise RuntimeError(f"{label}: expected 1 occurrence, found {n}: {old!r}")
    return text.replace(old, new, 1)

if not SOURCE_PROBE.is_file():
    raise SystemExit(f"ABORT: frozen v4-B probe missing: {SOURCE_PROBE}")
if sha256(SOURCE_PROBE) != EXPECTED_SOURCE_SHA:
    raise SystemExit("ABORT: frozen v4-B probe SHA mismatch")
if not MODEL.is_file():
    raise SystemExit(f"ABORT: seed43 checkpoint missing: {MODEL}")
if sha256(MODEL) != EXPECTED_MODEL_SHA:
    raise SystemExit("ABORT: seed43 checkpoint SHA mismatch")

output_dir = REPO / "logs" / "evaluations" / "g1" / NEW_OUT
if output_dir.exists():
    raise SystemExit(f"ABORT: output already exists: {output_dir}")

source = SOURCE_PROBE.read_text(encoding="utf-8")
source = replace_once(source, OLD_RUN, NEW_RUN, "run")
source = replace_once(source, OLD_OUT, NEW_OUT, "output")

for token in (
    "Humanoid-RL-Lab-G1-Flat-V4B-Direct-Yaw-v0",
    "SEED = 42",
    "VX = 0.40",
    "VY = 0.0",
    "WZ = 0.0",
    '"--deterministic"',
    "env_cfg.observations.policy.enable_corruption = False",
    "env_cfg.events.base_external_force_torque = None",
    "env_cfg.events.push_robot = None",
):
    if token not in source:
        raise RuntimeError(f"frozen protocol token missing: {token}")

compile(source, "derived_seed43_probe.py", "exec")

print("=" * 116)
print("v4-B TRAIN-SEED43 — CANONICAL CONTACT-AWARE GAIT PROBE")
print("=" * 116)
print(f"Frozen probe SHA    : {EXPECTED_SOURCE_SHA} — PASS")
print(f"Checkpoint SHA      : {EXPECTED_MODEL_SHA} — PASS")
print(f"Run                 : {RUN_DIR}")
print(f"Output              : {output_dir}")
print("Training seed       : 43")
print("Evaluation seed     : 42")
print("Command             : [0.40, 0.0, 0.0]")
print("Protocol            : frozen v4-B Play-equivalent evaluator")
print()

with tempfile.TemporaryDirectory(prefix="g1_v4b_seed43_") as td:
    derived = Path(td) / "derived_seed43_probe.py"
    derived.write_text(source, encoding="utf-8", newline="\n")
    rc = subprocess.run([sys.executable, str(derived)], cwd=str(REPO), check=False)
    if rc.returncode != 0:
        raise SystemExit(f"ABORT: derived probe exit code {rc.returncode}")

for required in ("samples.csv", "gait_probe_analysis.json"):
    if not (output_dir / required).is_file():
        raise SystemExit(f"ABORT: missing output: {output_dir / required}")

print()
print("V4B_TRAINSEED43_MODEL1499_CANONICAL_CONTACT_GAIT_WRAPPER_PASS")
