#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(r"C:\humanoid_rl_lab")

DEFAULT_SOURCE = (
    REPO
    / r"logs\evaluations\g1\v4n_model1499_play_equivalent_seed42_vx040_20s_canonical_v1"
)
DEFAULT_SAMPLES = DEFAULT_SOURCE / "samples.csv"
DEFAULT_GAIT_JSON = DEFAULT_SOURCE / "gait_probe_analysis.json"
DEFAULT_SUMMARY_JSON = DEFAULT_SOURCE / "summary.json"

DEFAULT_ROOT = REPO / r"logs\thesis\figures_final\03_v4n_final_gait"
DEFAULT_PNG_DIR = DEFAULT_ROOT / "png"
DEFAULT_PDF_DIR = DEFAULT_ROOT / "pdf"
DEFAULT_DATA_DIR = DEFAULT_ROOT / "data"

EXPECTED_SAMPLES_SHA256 = "3B02E8626687C7F00637F3E0FA54F93E451870B3D384831F7A948CD66CDE3F29"
EXPECTED_GAIT_SHA256 = "21157060CFC72B9A75FD5A88B1F4EEF3721A199798253D580EBD1E3EC9C7C4C9"
EXPECTED_SUMMARY_SHA256 = "39EDD8734ACEBDAB032937194AB6523C01D8CB0DB52BB3215C96F415EC724A46"

EXPECTED_CHECKPOINT_SHA256 = "367D318AB4940A3AAE8F0FA568384212762258067DFC4CF9F624FA472BEB9342"
EXPECTED_SEED = 42
EXPECTED_DT = 0.02
EXPECTED_DURATION_S = 20.0
EXPECTED_SAMPLES = 1000
EXPECTED_COMMAND = (0.40, 0.0, 0.0)

NUM_LEFT_STRIDES = 3
MIDPOINT_S = EXPECTED_DURATION_S / 2.0

PNG_NAME = "F03a_v4n_gait_cycle_foot_clearance_contact_v3.png"
PDF_NAME = "F03a_v4n_gait_cycle_foot_clearance_contact_v3.pdf"
DATA_NAME = "F03a_v4n_gait_cycle_foot_clearance_contact_data_v3.csv"
PASS_MARKER = "G1_THESIS_F03A_V4N_GAIT_CYCLE_FOOT_CLEARANCE_CONTACT_V3_PASS"


def fail(msg: str) -> None:
    raise RuntimeError(msg)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def read_samples(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        fail(f"missing samples CSV: {path}")

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) != EXPECTED_SAMPLES:
        fail(f"expected {EXPECTED_SAMPLES} samples, got {len(rows)}")

    required = (
        "time_s",
        "cmd_vx_mps",
        "cmd_vy_mps",
        "cmd_wz_radps",
        "left_foot_z_world_m",
        "right_foot_z_world_m",
        "left_foot_contact",
        "right_foot_contact",
    )
    missing = [c for c in required if c not in reader.fieldnames]
    if missing:
        fail(f"missing required columns: {missing}")

    data = {}
    for col in required:
        values = np.asarray([float(r[col]) for r in rows], dtype=float)
        if not np.all(np.isfinite(values)):
            fail(f"non-finite values in {col}")
        data[col] = values

    return data


def verify_protocol(data: dict[str, np.ndarray], gait: dict) -> None:
    protocol = gait.get("protocol")
    if not isinstance(protocol, dict):
        fail("gait JSON missing protocol")

    if int(protocol.get("seed", -1)) != EXPECTED_SEED:
        fail(f"unexpected seed: {protocol.get('seed')}")

    if str(protocol.get("checkpoint_sha256", "")).upper() != EXPECTED_CHECKPOINT_SHA256:
        fail("checkpoint SHA mismatch inside gait JSON")

    command = tuple(float(x) for x in protocol.get("command", []))
    if len(command) != 3 or any(
        not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12)
        for a, b in zip(command, EXPECTED_COMMAND)
    ):
        fail(f"unexpected gait command: {command}")

    if not math.isclose(
        float(protocol.get("duration_s")),
        EXPECTED_DURATION_S,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        fail("unexpected gait duration")

    if int(protocol.get("samples", -1)) != EXPECTED_SAMPLES:
        fail("unexpected gait sample count")

    dt = float(protocol.get("dt_s"))
    if not math.isclose(dt, EXPECTED_DT, rel_tol=0.0, abs_tol=1e-9):
        fail(f"unexpected gait dt: {dt}")

    # CSV protocol guards.
    time_s = data["time_s"]
    expected_time = np.arange(1, EXPECTED_SAMPLES + 1, dtype=float) * EXPECTED_DT
    if not np.allclose(time_s, expected_time, rtol=0.0, atol=1e-9):
        fail("samples time grid is not 0.02..20.00 s")

    for col, expected in (
        ("cmd_vx_mps", EXPECTED_COMMAND[0]),
        ("cmd_vy_mps", EXPECTED_COMMAND[1]),
        ("cmd_wz_radps", EXPECTED_COMMAND[2]),
    ):
        if not np.allclose(data[col], expected, rtol=0.0, atol=1e-12):
            fail(f"{col} is not constant at expected command")


def touchdown_indices(contact: np.ndarray) -> np.ndarray:
    binary = contact >= 0.5
    return np.flatnonzero(binary[1:] & ~binary[:-1]) + 1


def select_objective_window(data: dict[str, np.ndarray]) -> tuple[int, int, np.ndarray]:
    t = data["time_s"]
    left_td = touchdown_indices(data["left_foot_contact"])
    if len(left_td) < NUM_LEFT_STRIDES + 1:
        fail("not enough left-foot touchdowns")

    candidates = np.flatnonzero(t[left_td] >= MIDPOINT_S)
    if len(candidates) == 0:
        fail("no left touchdown at/after trial midpoint")

    anchor_pos = int(candidates[0])
    end_pos = anchor_pos + NUM_LEFT_STRIDES
    if end_pos >= len(left_td):
        fail("not enough post-midpoint touchdowns for requested strides")

    start_idx = int(left_td[anchor_pos])
    end_idx = int(left_td[end_pos])

    if not math.isclose(
        t[start_idx],
        10.04,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        fail(f"canonical objective window start changed: {t[start_idx]}")
    if not math.isclose(
        t[end_idx],
        11.96,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        fail(f"canonical objective window end changed: {t[end_idx]}")

    selected_touchdowns = left_td[anchor_pos : end_pos + 1]
    return start_idx, end_idx, selected_touchdowns


def contact_baseline(z: np.ndarray, contact: np.ndarray) -> float:
    mask = contact >= 0.5
    if not np.any(mask):
        fail("contact baseline cannot be computed: no contact samples")
    return float(np.median(z[mask]))


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Segoe UI", "DejaVu Sans"],
            "font.size": 15,
            "axes.titlesize": 22,
            "axes.labelsize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "axes.linewidth": 1.25,
            "grid.linewidth": 0.8,
            "grid.alpha": 0.28,
            "legend.fontsize": 14.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate thesis F03a: objective three-stride V4-N foot-clearance/contact window."
        )
    )
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--gait-json", type=Path, default=DEFAULT_GAIT_JSON)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--png-dir", type=Path, default=DEFAULT_PNG_DIR)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    # Provenance guards.
    for label, path, expected_sha in (
        ("samples", args.samples, EXPECTED_SAMPLES_SHA256),
        ("gait", args.gait_json, EXPECTED_GAIT_SHA256),
        ("summary", args.summary_json, EXPECTED_SUMMARY_SHA256),
    ):
        if not path.is_file():
            fail(f"missing {label} source: {path}")
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            fail(
                f"{label} SHA mismatch: expected {expected_sha}, got {actual_sha}"
            )

    data = read_samples(args.samples)
    gait = json.loads(args.gait_json.read_text(encoding="utf-8"))
    _summary = json.loads(args.summary_json.read_text(encoding="utf-8"))

    verify_protocol(data, gait)

    start_idx, end_idx, left_td = select_objective_window(data)

    sl = slice(start_idx, end_idx + 1)
    t_abs = data["time_s"][sl]
    t_rel = t_abs - t_abs[0]

    left_contact = data["left_foot_contact"][sl] >= 0.5
    right_contact = data["right_foot_contact"][sl] >= 0.5

    left_base = contact_baseline(
        data["left_foot_z_world_m"],
        data["left_foot_contact"],
    )
    right_base = contact_baseline(
        data["right_foot_z_world_m"],
        data["right_foot_contact"],
    )

    left_clearance_cm = (
        data["left_foot_z_world_m"][sl] - left_base
    ) * 100.0
    right_clearance_cm = (
        data["right_foot_z_world_m"][sl] - right_base
    ) * 100.0

    if np.min(left_clearance_cm) < -0.05 or np.min(right_clearance_cm) < -0.05:
        fail("unexpected negative foot clearance relative to contact baseline")

    # Canonical gait metrics used only as context, not to select the window.
    gait_metrics = gait.get("gait", {})
    stride_period_s = float(gait_metrics["same_foot_stride_period_median_s"])
    cadence_spm = float(gait_metrics["cadence_touchdowns_spm"])
    swing_apex_m = float(gait_metrics["relative_swing_apex_median_m"])
    single_support_pct = float(gait_metrics["single_support_pct"])

    args.data_dir.mkdir(parents=True, exist_ok=True)
    freeze_path = args.data_dir / DATA_NAME

    with freeze_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "time_relative_s",
                "time_absolute_s",
                "left_clearance_cm",
                "right_clearance_cm",
                "left_contact",
                "right_contact",
            ],
        )
        writer.writeheader()
        for i in range(len(t_rel)):
            writer.writerow(
                {
                    "time_relative_s": t_rel[i],
                    "time_absolute_s": t_abs[i],
                    "left_clearance_cm": left_clearance_cm[i],
                    "right_clearance_cm": right_clearance_cm[i],
                    "left_contact": int(left_contact[i]),
                    "right_contact": int(right_contact[i]),
                }
            )

    configure_style()
    fig, ax = plt.subplots(figsize=(11.8, 7.5))

    left_line, = ax.plot(
        t_rel,
        left_clearance_cm,
        linewidth=2.5,
        label="Left foot",
        zorder=3,
    )
    right_line, = ax.plot(
        t_rel,
        right_clearance_cm,
        linewidth=2.5,
        label="Right foot",
        zorder=3,
    )

    # Contact state is encoded directly on each foot trajectory:
    # thick same-color segments denote samples where that foot is in contact.
    left_contact_curve = np.where(left_contact, left_clearance_cm, np.nan)
    right_contact_curve = np.where(right_contact, right_clearance_cm, np.nan)

    ax.plot(
        t_rel,
        left_contact_curve,
        linewidth=6.2,
        color=left_line.get_color(),
        alpha=0.72,
        solid_capstyle="butt",
        zorder=4,
    )
    ax.plot(
        t_rel,
        right_contact_curve,
        linewidth=6.2,
        color=right_line.get_color(),
        alpha=0.72,
        solid_capstyle="butt",
        zorder=4,
    )

    # Mark the deterministic left-foot stride boundaries.
    td_rel = data["time_s"][left_td] - t_abs[0]
    for x in td_rel:
        ax.axvline(
            x,
            linestyle="--",
            linewidth=1.0,
            color="#B8B8B8",
            alpha=0.65,
            zorder=0,
        )

    ax.text(
        0.985,
        0.972,
        "Thick segments = foot contact",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=13.5,
        color="#202020",
        zorder=7,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#C8C8C8",
            "linewidth": 1.0,
            "alpha": 0.96,
        },
    )

    ax.set_title(
        "V4-N Gait Cycle — Foot Clearance and Contact",
        pad=17,
        fontweight="normal",
    )
    ax.set_xlabel("Time from selected left touchdown [s]", labelpad=11)
    ax.set_ylabel("Foot clearance above contact level [cm]", labelpad=10)

    ax.set_xlim(0.0, float(t_rel[-1]))
    ax.set_ylim(-0.5, 18.0)
    ax.set_xticks(np.arange(0.0, 1.81, 0.30))
    ax.set_yticks(np.arange(0.0, 18.1, 3.0))

    ax.grid(True, linestyle="--", zorder=0)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.25)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.155),
        ncol=2,
        frameon=True,
        framealpha=0.96,
        facecolor="white",
        edgecolor="#C8C8C8",
        borderpad=0.45,
        columnspacing=1.8,
        handlelength=2.6,
    )

    fig.text(
        0.5,
        0.016,
        (
            "$v_x$ command = 0.40 m/s  |  seed 42  |  "
            "3 consecutive left-foot strides  |  10.04–11.96 s"
        ),
        ha="center",
        va="bottom",
        fontsize=13.5,
        color="#555555",
    )

    fig.subplots_adjust(left=0.12, right=0.975, top=0.87, bottom=0.235)

    args.png_dir.mkdir(parents=True, exist_ok=True)
    args.pdf_dir.mkdir(parents=True, exist_ok=True)

    png_path = args.png_dir / PNG_NAME
    pdf_path = args.pdf_dir / PDF_NAME

    fig.savefig(png_path, dpi=180, bbox_inches=None)
    fig.savefig(pdf_path, format="pdf", bbox_inches=None)
    plt.close(fig)

    if not png_path.is_file() or png_path.stat().st_size <= 0:
        fail("PNG output missing or empty")
    if not pdf_path.is_file() or pdf_path.stat().st_size <= 0:
        fail("PDF output missing or empty")

    print("=" * 122)
    print("THESIS FIGURE F03a — GENERATION SUMMARY")
    print("=" * 122)
    print("Status                              : PASS")
    print("Figure                              : V4-N gait cycle — foot clearance and contact (hybrid visual v3)")
    print("Source                              : canonical seed42 vx040 20 s evaluation")
    print(f"Checkpoint SHA256                   : {EXPECTED_CHECKPOINT_SHA256}")
    print("Window selection rule               : first left touchdown >= trial midpoint + 3 left strides")
    print(f"Selected absolute window [s]        : {t_abs[0]:.2f} -> {t_abs[-1]:.2f}")
    print(f"Selected duration [s]               : {t_rel[-1]:.2f}")
    print(f"Selected samples                    : {len(t_rel)}")
    print(
        "Selected left touchdown times [s]  : "
        + ", ".join(f"{x:.2f}" for x in data["time_s"][left_td])
    )
    print(f"Full-run left contact baseline [m]  : {left_base:.9f}")
    print(f"Full-run right contact baseline [m] : {right_base:.9f}")
    print(f"Window left max clearance [cm]      : {np.max(left_clearance_cm):.6f}")
    print(f"Window right max clearance [cm]     : {np.max(right_clearance_cm):.6f}")
    print(f"Median stride period [s]            : {stride_period_s:.6f}")
    print(f"Cadence [steps/min]                 : {cadence_spm:.6f}")
    print(f"Median relative swing apex [cm]     : {100.0 * swing_apex_m:.6f}")
    print(f"Single-support fraction [%]         : {single_support_pct:.6f}")
    print("Selection interpretation            : deterministic temporal rule; no performance cherry-picking")
    print("Visual revision                     : v1 thick-contact encoding + external bottom color legend")
    print(f"Data freeze                         : {freeze_path}")
    print(f"PNG                                 : {png_path}")
    print(f"PDF vector                          : {pdf_path}")
    print("Simulation / policy run             : NO")
    print("Repository/checkpoints              : NOT MODIFIED")
    print(PASS_MARKER)


if __name__ == "__main__":
    main()
