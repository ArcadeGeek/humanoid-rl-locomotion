from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(r"C:\humanoid_rl_lab")

V2G_CHECKPOINT = ROOT / r"checkpoints\local_g1_yaw_direct_wide_yaw_flat_orientation_v2g\model_600.pt"
V2G_SHA256 = "58468EED03F9EF35A9BC898063A3B14AC36C93A32F65DB42B404BDE93CD694D3"

V4N_CHECKPOINT = (
    ROOT
    / r"logs\rsl_rl\humanoid_rl_lab_g1_v4n_symmetry_augmentation_scratch"
    / r"2026-09-04_13-58-13_v4n_genuine_scratch_seed42_official1500"
    / "model_1499.pt"
)
V4N_SHA256 = "367D318AB4940A3AAE8F0FA568384212762258067DFC4CF9F624FA472BEB9342"

HANDOFF_REPORT = (
    ROOT
    / r"logs\thesis\handoff\paired_000_vs_036_zero_root_vel_start_seeds100_104"
    / "paired_report.txt"
)
BAND_REPORT = (
    ROOT
    / r"logs\thesis\router\low_speed_band_multiseed_offline_audit_seeds100_104"
    / "low_speed_band_report.txt"
)
HOLDOUT_REPORT = (
    ROOT
    / r"logs\thesis\router\untouched_holdout_110_114_offline_audit"
    / "holdout_110_114_report.txt"
)

EXPECTED_MARKERS = {
    HANDOFF_REPORT: "V2G_V4N_PAIRED_000_VS_036_MULTISEED_OFFLINE_AUDIT_PASS",
    BAND_REPORT: "V2G_V4N_LOW_SPEED_BAND_MULTISEED_OFFLINE_AUDIT_PASS",
    HOLDOUT_REPORT: "V2G_V4N_ROUTER_HYSTERESIS_HOLDOUT_110_114_OFFLINE_AUDIT_PASS",
}

ENTER_LOCO_SPEED = 0.20
EXIT_LOCO_SPEED = 0.10
BRAKE_S = 0.36
BRAKE_STEPS = 18

PASS_MARKER = "V2G_V4N_INTEGRATED_ARCHITECTURE_SCIENTIFIC_CLOSURE_PASS"


def fail(msg: str) -> None:
    raise RuntimeError(msg)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def require_file(path: Path) -> None:
    if not path.is_file():
        fail(f"missing required file: {path}")


def require_marker(path: Path, marker: str) -> None:
    require_file(path)
    text = path.read_text(encoding="utf-8")
    if marker not in text:
        fail(f"required PASS marker missing from {path}: {marker}")


def main() -> None:
    # Frozen policy provenance.
    require_file(V2G_CHECKPOINT)
    require_file(V4N_CHECKPOINT)

    actual_v2g = sha256(V2G_CHECKPOINT)
    actual_v4n = sha256(V4N_CHECKPOINT)

    if actual_v2g != V2G_SHA256:
        fail(f"v2-G checkpoint SHA mismatch: {actual_v2g}")
    if actual_v4n != V4N_SHA256:
        fail(f"v4-N checkpoint SHA mismatch: {actual_v4n}")

    # Closed experimental blocks.
    for report, marker in EXPECTED_MARKERS.items():
        require_marker(report, marker)

    out_dir = ROOT / r"logs\thesis\closure"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "integrated_architecture_scientific_closure.txt"

    lines = [
        "=" * 132,
        "v2-G / v4-N INTEGRATED ARCHITECTURE — SCIENTIFIC CLOSURE MANIFEST",
        "=" * 132,
        "",
        "[FROZEN POLICIES]",
        f"v2-G checkpoint            : {V2G_CHECKPOINT}",
        f"v2-G SHA256                : {actual_v2g}",
        f"v4-N checkpoint            : {V4N_CHECKPOINT}",
        f"v4-N SHA256                : {actual_v4n}",
        "",
        "[FROZEN ARCHITECTURE]",
        "STAND controller           : v2-G",
        "LOCOMOTION controller      : v4-N",
        f"EXIT threshold             : {EXIT_LOCO_SPEED:.2f} m/s",
        f"ENTER threshold            : {ENTER_LOCO_SPEED:.2f} m/s",
        f"braking dwell              : {BRAKE_S:.2f} s / {BRAKE_STEPS} steps",
        "BRAKE policy               : v4-N",
        "BRAKE command              : zero",
        "blending                   : ABSENT",
        "",
        "[DEVELOPMENT EVIDENCE]",
        "handoff paired seeds       : 100-104",
        "handoff comparison         : immediate 0.00 s vs frozen dwell 0.36 s",
        "handoff result             : dwell improves all 6 core transient metrics on 5/5 seeds",
        "handoff mechanism          : pre-switch state conditioning, not action smoothing",
        "low-speed calibration      : vx = 0.10 / 0.15 / 0.20 m/s, seeds 100-104",
        "band result                : ordered overlap structure replicated on 5/5 seeds",
        "threshold status           : DEVELOPMENT-CALIBRATED / FROZEN",
        "",
        "[TRUE UNTOUCHED HOLDOUT]",
        "holdout seeds              : 110-114",
        "router transition sequence : exact on 5/5",
        "canceled BRAKE             : 5 steps on 5/5",
        "completed BRAKE            : 18 steps on 5/5",
        "termination                : NO on 5/5",
        "final double support       : 100% on 5/5",
        "post-holdout tuning        : FORBIDDEN / ABSENT",
        "",
        "[SCIENTIFIC STATUS]",
        "v2-G standing benchmark    : CLOSED / POSITIVE",
        "v4-N locomotion            : CLOSED / PROMOTED",
        "v4-N -> v2-G handoff       : CLOSED / POSITIVE",
        "router integration         : CLOSED / FUNCTIONAL",
        "router thresholds          : CLOSED / DEVELOPMENT-CALIBRATED",
        "untouched holdout          : CLOSED / POSITIVE",
        "integrated architecture    : CLOSED / POSITIVE",
        "",
        "[IMPORTANT SCOPE LIMITATION]",
        "Integrated router/handoff development and holdout validation used zero root-reset velocities",
        "as an engineering-only startup conditioning step.",
        "Therefore these results do NOT establish robustness of frozen v2-G to the original randomized",
        "root-velocity reset distribution. The reproducible seed104 randomized-start failure remains a",
        "separate startup-robustness issue and must not be counted as a handoff or router failure.",
        "",
        "[REPORT PROVENANCE]",
        f"handoff report             : {HANDOFF_REPORT}",
        f"band report                : {BAND_REPORT}",
        f"holdout report             : {HOLDOUT_REPORT}",
        "",
        "No additional development or holdout seed is added, removed, or reclassified by this manifest.",
        "No threshold, dwell, checkpoint, policy, reward, or blending parameter is modified.",
        PASS_MARKER,
        "=" * 132,
    ]

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines), flush=True)
    print(f"\nclosure report            : {report_path}", flush=True)


if __name__ == "__main__":
    main()
