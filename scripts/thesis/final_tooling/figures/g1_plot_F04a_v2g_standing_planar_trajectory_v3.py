#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(r"C:\humanoid_rl_lab")

DEFAULT_SOURCE = (
    REPO
    / r"logs\thesis\standing_probe\frozen_v2g_zero_command_playeq_seed42_20s"
)
DEFAULT_SAMPLES = DEFAULT_SOURCE / "frozen_v2g_zero_command_playeq_seed42_20s_samples.csv"
DEFAULT_META = DEFAULT_SOURCE / "frozen_v2g_zero_command_playeq_seed42_20s_meta.txt"

DEFAULT_ROOT = REPO / r"logs\thesis\figures_final\04_v2g_final_stand"
DEFAULT_PNG_DIR = DEFAULT_ROOT / "png"
DEFAULT_PDF_DIR = DEFAULT_ROOT / "pdf"
DEFAULT_DATA_DIR = DEFAULT_ROOT / "data"

EXPECTED_SAMPLES_SHA256 = "A0CF568473D37190F315516A068BE7594A989EBB879CB88FFEEDD8B965D0FF3E"
EXPECTED_META_SHA256 = "2AB9AC15A7DE34A4F198AF4109BF6BF657A75131F5422C18FE2103A754DAAFA5"
EXPECTED_CHECKPOINT_SHA256 = "58468EED03F9EF35A9BC898063A3B14AC36C93A32F65DB42B404BDE93CD694D3"

EXPECTED_TOTAL_SAMPLES = 1000
VALID_PRE_RESET_SAMPLES = 849
EXPECTED_DT = 0.02
EXPECTED_VALID_DURATION_S = 16.96
EXPECTED_LAST_VALID_TIMESTAMP_S = 16.98
EXPECTED_FIRST_POST_RESET_TIMESTAMP_S = 17.00

EXPECTED_FINAL_DRIFT_M = 0.003230711577176417
EXPECTED_MAX_EXCURSION_M = 0.003231404048306165

V2G_COLOR = "#2F5D7E"

PNG_NAME = "F04a_v2g_standing_planar_trajectory_v3.png"
PDF_NAME = "F04a_v2g_standing_planar_trajectory_v3.pdf"
DATA_NAME = "F04a_v2g_standing_planar_trajectory_data_v3.csv"
PASS_MARKER = "G1_THESIS_F04A_V2G_STANDING_PLANAR_TRAJECTORY_V3_PASS"


def fail(msg: str) -> None:
    raise RuntimeError(msg)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def parse_meta(path: Path) -> dict[str, str]:
    if not path.is_file():
        fail(f"missing metadata file: {path}")

    meta = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        meta[key.strip()] = value.strip()

    required = {
        "checkpoint_sha256",
        "step_dt",
        "record_steps",
        "record_duration_s",
        "cmd_vx_min",
        "cmd_vx_max",
        "reset_count",
        "csv_sha256",
    }
    missing = required - set(meta)
    if missing:
        fail(f"metadata missing fields: {sorted(missing)}")

    if meta["checkpoint_sha256"].upper() != EXPECTED_CHECKPOINT_SHA256:
        fail("checkpoint SHA mismatch in metadata")
    if meta["csv_sha256"].upper() != EXPECTED_SAMPLES_SHA256:
        fail("CSV SHA mismatch in metadata")
    if not math.isclose(float(meta["step_dt"]), EXPECTED_DT, rel_tol=0.0, abs_tol=1e-12):
        fail("metadata step_dt mismatch")
    if int(meta["record_steps"]) != EXPECTED_TOTAL_SAMPLES:
        fail("metadata record_steps mismatch")
    if not math.isclose(float(meta["record_duration_s"]), 20.0, rel_tol=0.0, abs_tol=1e-12):
        fail("metadata record duration mismatch")
    if not math.isclose(float(meta["cmd_vx_min"]), 0.0, rel_tol=0.0, abs_tol=1e-12):
        fail("metadata cmd_vx_min is not zero")
    if not math.isclose(float(meta["cmd_vx_max"]), 0.0, rel_tol=0.0, abs_tol=1e-12):
        fail("metadata cmd_vx_max is not zero")
    if int(meta["reset_count"]) != 1:
        fail("metadata reset_count is not exactly one")

    return meta


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        fail(f"missing samples CSV: {path}")

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) != EXPECTED_TOTAL_SAMPLES:
        fail(f"expected {EXPECTED_TOTAL_SAMPLES} rows, got {len(rows)}")

    required = {
        "sample",
        "sim_time_s",
        "cmd_vx",
        "cmd_vy",
        "cmd_wz",
        "root_x_w",
        "root_y_w",
    }
    missing = required - set(reader.fieldnames or [])
    if missing:
        fail(f"CSV missing fields: {sorted(missing)}")

    return rows


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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate thesis F04a v2: V2-G pre-reset zero-command planar root trajectory."
    )
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--png-dir", type=Path, default=DEFAULT_PNG_DIR)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    if sha256_file(args.samples) != EXPECTED_SAMPLES_SHA256:
        fail("canonical samples SHA mismatch")
    if sha256_file(args.meta) != EXPECTED_META_SHA256:
        fail("canonical metadata SHA mismatch")

    _meta = parse_meta(args.meta)
    rows = read_rows(args.samples)

    # Frozen pre-reset window: samples 1..849 only.
    if int(rows[0]["sample"]) != 1:
        fail("first sample index changed")
    if int(rows[VALID_PRE_RESET_SAMPLES - 1]["sample"]) != VALID_PRE_RESET_SAMPLES:
        fail("last valid sample index changed")
    if int(rows[VALID_PRE_RESET_SAMPLES]["sample"]) != VALID_PRE_RESET_SAMPLES + 1:
        fail("first post-reset sample index changed")

    first_t = float(rows[0]["sim_time_s"])
    last_valid_t = float(rows[VALID_PRE_RESET_SAMPLES - 1]["sim_time_s"])
    first_post_t = float(rows[VALID_PRE_RESET_SAMPLES]["sim_time_s"])

    if not math.isclose(last_valid_t, EXPECTED_LAST_VALID_TIMESTAMP_S, rel_tol=0.0, abs_tol=1e-12):
        fail(f"last valid timestamp changed: {last_valid_t}")
    if not math.isclose(first_post_t, EXPECTED_FIRST_POST_RESET_TIMESTAMP_S, rel_tol=0.0, abs_tol=1e-12):
        fail(f"first post-reset timestamp changed: {first_post_t}")
    if not math.isclose(
        last_valid_t - first_t,
        EXPECTED_VALID_DURATION_S,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        fail("valid elapsed duration changed")

    # Explicitly confirm the reset discontinuity after the valid window.
    pre_x = float(rows[VALID_PRE_RESET_SAMPLES - 1]["root_x_w"])
    pre_y = float(rows[VALID_PRE_RESET_SAMPLES - 1]["root_y_w"])
    post_x = float(rows[VALID_PRE_RESET_SAMPLES]["root_x_w"])
    post_y = float(rows[VALID_PRE_RESET_SAMPLES]["root_y_w"])
    reset_xy_jump = math.hypot(post_x - pre_x, post_y - pre_y)
    if reset_xy_jump <= 0.50:
        fail(f"expected reset discontinuity not found: {reset_xy_jump}")

    valid = rows[:VALID_PRE_RESET_SAMPLES]

    for key in ("cmd_vx", "cmd_vy", "cmd_wz"):
        values = np.asarray([float(r[key]) for r in valid], dtype=float)
        if not np.all(np.isfinite(values)):
            fail(f"non-finite values in {key}")
        if not np.allclose(values, 0.0, rtol=0.0, atol=1e-12):
            fail(f"{key} is not identically zero")

    sim_t = np.asarray([float(r["sim_time_s"]) for r in valid], dtype=float)
    elapsed_t = sim_t - sim_t[0]

    x = np.asarray([float(r["root_x_w"]) for r in valid], dtype=float)
    y = np.asarray([float(r["root_y_w"]) for r in valid], dtype=float)

    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        fail("non-finite root position data")

    dx_mm = (x - x[0]) * 1000.0
    dy_mm = (y - y[0]) * 1000.0
    excursion_m = np.hypot(dx_mm, dy_mm) / 1000.0

    final_drift_m = float(excursion_m[-1])
    max_excursion_m = float(np.max(excursion_m))

    if not math.isclose(final_drift_m, EXPECTED_FINAL_DRIFT_M, rel_tol=0.0, abs_tol=1e-12):
        fail(f"final drift changed: {final_drift_m}")
    if not math.isclose(max_excursion_m, EXPECTED_MAX_EXCURSION_M, rel_tol=0.0, abs_tol=1e-12):
        fail(f"max excursion changed: {max_excursion_m}")

    # Canonical endpoint guards in the plotted relative coordinate frame.
    if not math.isclose(float(dx_mm[-1]), -2.987504005432129, rel_tol=0.0, abs_tol=1e-9):
        fail(f"final dx changed: {dx_mm[-1]}")
    if not math.isclose(float(dy_mm[-1]), -1.2297630310058594, rel_tol=0.0, abs_tol=1e-9):
        fail(f"final dy changed: {dy_mm[-1]}")

    args.data_dir.mkdir(parents=True, exist_ok=True)
    freeze_path = args.data_dir / DATA_NAME

    with freeze_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "elapsed_time_s",
                "delta_x_mm",
                "delta_y_mm",
                "radial_excursion_mm",
            ],
        )
        writer.writeheader()
        for i in range(len(valid)):
            writer.writerow(
                {
                    "elapsed_time_s": elapsed_t[i],
                    "delta_x_mm": dx_mm[i],
                    "delta_y_mm": dy_mm[i],
                    "radial_excursion_mm": 1000.0 * excursion_m[i],
                }
            )

    configure_style()
    fig, ax = plt.subplots(figsize=(11.8, 7.5))

    ax.plot(
        dx_mm,
        dy_mm,
        color=V2G_COLOR,
        linewidth=3.0,
        zorder=3,
    )

    # Start and final points deliberately use distinct shapes.
    ax.scatter(
        [dx_mm[0]],
        [dy_mm[0]],
        s=78,
        facecolor="white",
        edgecolor=V2G_COLOR,
        linewidth=2.0,
        zorder=5,
    )
    ax.scatter(
        [dx_mm[-1]],
        [dy_mm[-1]],
        s=82,
        marker="D",
        color=V2G_COLOR,
        edgecolor="black",
        linewidth=0.8,
        zorder=5,
    )

    ax.annotate(
        "Start",
        xy=(dx_mm[0], dy_mm[0]),
        xytext=(-22, 24),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=13.5,
        color="#202020",
        arrowprops={
            "arrowstyle": "-",
            "color": "#707070",
            "linewidth": 1.0,
        },
        zorder=6,
    )
    ax.annotate(
        "Final",
        xy=(dx_mm[-1], dy_mm[-1]),
        xytext=(20, -22),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=13.5,
        color="#202020",
        arrowprops={
            "arrowstyle": "-",
            "color": "#707070",
            "linewidth": 1.0,
        },
        zorder=6,
    )

    ax.text(
        0.250,
        0.887,
        (
            f"Final drift = {1000.0 * final_drift_m:.2f} mm\n"
            f"Max excursion = {1000.0 * max_excursion_m:.2f} mm"
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=13.5,
        color="#202020",
        zorder=7,
        bbox={
            "boxstyle": "round,pad=0.38",
            "facecolor": "white",
            "edgecolor": "#C8C8C8",
            "linewidth": 1.0,
            "alpha": 0.96,
        },
    )

    ax.axhline(0.0, color="#B8B8B8", linewidth=1.0, linestyle="--", zorder=0)
    ax.axvline(0.0, color="#B8B8B8", linewidth=1.0, linestyle="--", zorder=0)

    ax.set_title(
        "V2-G Standing — Planar Root Trajectory",
        pad=17,
        fontweight="normal",
    )
    ax.set_xlabel("Root displacement Δx  [mm]", labelpad=11)
    ax.set_ylabel("Root displacement Δy  [mm]", labelpad=10)

    ax.set_xlim(-3.35, 0.35)
    ax.set_ylim(-1.60, 0.35)
    ax.set_xticks(np.arange(-3.0, 0.1, 0.5))
    ax.set_yticks(np.arange(-1.5, 0.1, 0.25))

    # Equal axis scale is essential: one millimetre means the same visually
    # in x and y, so the path geometry is not distorted.
    ax.set_aspect("equal", adjustable="box")

    ax.grid(True, linestyle="--", zorder=0)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.25)

    fig.text(
        0.5,
        0.020,
        (
            "zero command  |  seed 42  |  849 pre-reset samples  |  "
            "16.96 s valid standing  |  timeout/reset excluded"
        ),
        ha="center",
        va="bottom",
        fontsize=13.5,
        color="#555555",
    )

    # Increased bottom separation compared with v1.
    fig.subplots_adjust(left=0.12, right=0.975, top=0.87, bottom=0.17)

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

    print("=" * 124)
    print("THESIS FIGURE F04a v2 — GENERATION SUMMARY")
    print("=" * 124)
    print("Status                              : PASS")
    print("Figure                              : V2-G standing planar root trajectory (annotation refinement v3)")
    print("Source                              : frozen zero-command seed42 20 s probe")
    print(f"Checkpoint SHA256                   : {EXPECTED_CHECKPOINT_SHA256}")
    print(f"Full recorded samples               : {len(rows)}")
    print(f"Valid pre-reset samples             : {len(valid)}")
    print(f"Valid elapsed standing duration [s] : {elapsed_t[-1]:.2f}")
    print(f"First post-reset timestamp [s]      : {first_post_t:.2f}")
    print(f"Reset XY discontinuity [m]          : {reset_xy_jump:.9f}")
    print(f"Final delta x [mm]                  : {dx_mm[-1]:.9f}")
    print(f"Final delta y [mm]                  : {dy_mm[-1]:.9f}")
    print(f"Final planar drift [mm]             : {1000.0 * final_drift_m:.9f}")
    print(f"Max planar excursion [mm]           : {1000.0 * max_excursion_m:.9f}")
    print("Axis geometry                       : equal x/y scale")
    print("Visual revision                     : Start label above-left; metric box lower-right left of x=0 reference")
    print("Contamination guard                 : rows 850-1000 excluded from all standing metrics")
    print("Timeout interpretation              : environment timeout/reset; NOT a fall")
    print(f"Data freeze                         : {freeze_path}")
    print(f"PNG                                 : {png_path}")
    print(f"PDF vector                          : {pdf_path}")
    print("Simulation / policy run             : NO")
    print("Repository/checkpoints              : NOT MODIFIED")
    print(PASS_MARKER)


if __name__ == "__main__":
    main()
