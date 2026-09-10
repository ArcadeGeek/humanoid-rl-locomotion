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
DEFAULT_SUMMARY_JSON = DEFAULT_SOURCE / "summary.json"

DEFAULT_ROOT = REPO / r"logs\thesis\figures_final\03_v4n_final_gait"
DEFAULT_PNG_DIR = DEFAULT_ROOT / "png"
DEFAULT_PDF_DIR = DEFAULT_ROOT / "pdf"
DEFAULT_DATA_DIR = DEFAULT_ROOT / "data"

EXPECTED_SAMPLES_SHA256 = "3B02E8626687C7F00637F3E0FA54F93E451870B3D384831F7A948CD66CDE3F29"
EXPECTED_SUMMARY_SHA256 = "39EDD8734ACEBDAB032937194AB6523C01D8CB0DB52BB3215C96F415EC724A46"
EXPECTED_CHECKPOINT_SHA256 = "367D318AB4940A3AAE8F0FA568384212762258067DFC4CF9F624FA472BEB9342"

EXPECTED_SAMPLES = 1000
EXPECTED_DT = 0.02
EXPECTED_DURATION_S = 20.0
EXPECTED_COMMAND_VX = 0.40
MA_WINDOW_S = 0.50
MA_WINDOW_SAMPLES = 25

V4N_COLOR = "#F58518"
REFERENCE_COLOR = "#6BAED6"

PNG_NAME = "F03b_v4n_forward_velocity_tracking_v2.png"
PDF_NAME = "F03b_v4n_forward_velocity_tracking_v2.pdf"
DATA_NAME = "F03b_v4n_forward_velocity_tracking_data_v2.csv"
PASS_MARKER = "G1_THESIS_F03B_V4N_FORWARD_VELOCITY_TRACKING_V2_PASS"


def fail(msg: str) -> None:
    raise RuntimeError(msg)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def read_samples(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        fail(f"missing samples CSV: {path}")

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) != EXPECTED_SAMPLES:
        fail(f"expected {EXPECTED_SAMPLES} samples, got {len(rows)}")

    required = ("time_s", "cmd_vx_mps", "base_vx_mps")
    missing = [c for c in required if c not in reader.fieldnames]
    if missing:
        fail(f"missing columns: {missing}")

    t = np.asarray([float(r["time_s"]) for r in rows], dtype=float)
    cmd = np.asarray([float(r["cmd_vx_mps"]) for r in rows], dtype=float)
    vx = np.asarray([float(r["base_vx_mps"]) for r in rows], dtype=float)

    if not np.all(np.isfinite(t)):
        fail("non-finite time data")
    if not np.all(np.isfinite(cmd)):
        fail("non-finite command data")
    if not np.all(np.isfinite(vx)):
        fail("non-finite velocity data")

    expected_t = np.arange(1, EXPECTED_SAMPLES + 1, dtype=float) * EXPECTED_DT
    if not np.allclose(t, expected_t, rtol=0.0, atol=1e-9):
        fail("unexpected time grid")

    if not np.allclose(cmd, EXPECTED_COMMAND_VX, rtol=0.0, atol=1e-12):
        fail("forward command is not constant 0.40 m/s")

    return t, cmd, vx


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=float)
    if len(values) < window:
        return result

    kernel = np.ones(window, dtype=float) / float(window)
    result[window - 1 :] = np.convolve(values, kernel, mode="valid")
    return result


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
        description="Generate thesis F03b: V4-N forward-velocity tracking over the full canonical 20 s evaluation."
    )
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--png-dir", type=Path, default=DEFAULT_PNG_DIR)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    for label, path, expected_sha in (
        ("samples", args.samples, EXPECTED_SAMPLES_SHA256),
        ("summary", args.summary_json, EXPECTED_SUMMARY_SHA256),
    ):
        if not path.is_file():
            fail(f"missing {label}: {path}")
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            fail(f"{label} SHA mismatch: expected {expected_sha}, got {actual_sha}")

    t, cmd, vx = read_samples(args.samples)
    summary = json.loads(args.summary_json.read_text(encoding="utf-8"))

    checkpoint_sha = str(summary["checkpoint"]["sha256"]).upper()
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        fail(f"checkpoint SHA mismatch in summary: {checkpoint_sha}")

    if int(summary["seed"]) != 42:
        fail(f"unexpected seed: {summary['seed']}")

    # Canonical evaluator schema uses status="complete" for a successfully
    # completed fixed-duration evaluation. Validate completion semantically
    # instead of assuming a generic PASS token.
    if str(summary.get("status", "")).lower() != "complete":
        fail(f"canonical evaluation status is not complete: {summary.get('status')}")

    termination = summary.get("termination")
    if not isinstance(termination, dict):
        fail("canonical summary missing termination block")
    if termination.get("occurred") is not False:
        fail(f"canonical evaluation terminated unexpectedly: {termination}")

    timing = summary.get("timing")
    if not isinstance(timing, dict):
        fail("canonical summary missing timing block")

    if int(timing.get("samples", -1)) != EXPECTED_SAMPLES:
        fail(f"canonical summary sample count mismatch: {timing.get('samples')}")
    if not math.isclose(
        float(timing.get("policy_dt_s")),
        EXPECTED_DT,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        fail(f"canonical summary dt mismatch: {timing.get('policy_dt_s')}")
    if not math.isclose(
        float(timing.get("measured_duration_s")),
        EXPECTED_DURATION_S,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        fail(
            "canonical summary measured duration mismatch: "
            f"{timing.get('measured_duration_s')}"
        )
    if not math.isclose(
        float(timing.get("requested_duration_s")),
        EXPECTED_DURATION_S,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        fail(
            "canonical summary requested duration mismatch: "
            f"{timing.get('requested_duration_s')}"
        )

    command_mean = summary.get("command_mean")
    if not isinstance(command_mean, dict):
        fail("canonical summary missing command_mean block")
    if not math.isclose(
        float(command_mean.get("vx_mps")),
        EXPECTED_COMMAND_VX,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        fail(f"canonical summary vx command mismatch: {command_mean.get('vx_mps')}")
    if not math.isclose(
        float(command_mean.get("vy_mps")),
        0.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        fail(f"canonical summary vy command mismatch: {command_mean.get('vy_mps')}")
    if not math.isclose(
        float(command_mean.get("wz_radps")),
        0.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        fail(f"canonical summary wz command mismatch: {command_mean.get('wz_radps')}")

    reported_mean = float(summary["velocity_tracking"]["base_vx_mean_mps"])
    reported_rmse = float(summary["velocity_tracking"]["base_vx_rmse_mps"])
    reported_std = float(summary["velocity_tracking"]["base_vx_std_mps"])

    computed_mean = float(np.mean(vx))
    computed_rmse = float(np.sqrt(np.mean((vx - EXPECTED_COMMAND_VX) ** 2)))
    computed_std = float(np.std(vx, ddof=0))

    if not math.isclose(computed_mean, reported_mean, rel_tol=0.0, abs_tol=1e-12):
        fail("computed mean vx does not match canonical summary")
    if not math.isclose(computed_rmse, reported_rmse, rel_tol=0.0, abs_tol=1e-12):
        fail("computed RMSE does not match canonical summary")
    if not math.isclose(computed_std, reported_std, rel_tol=0.0, abs_tol=1e-12):
        fail("computed std vx does not match canonical summary")

    ma = moving_average(vx, MA_WINDOW_SAMPLES)

    args.data_dir.mkdir(parents=True, exist_ok=True)
    freeze_path = args.data_dir / DATA_NAME

    with freeze_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "time_s",
                "command_vx_mps",
                "base_vx_raw_mps",
                "base_vx_ma050s_mps",
            ],
        )
        writer.writeheader()

        for i in range(len(t)):
            writer.writerow(
                {
                    "time_s": t[i],
                    "command_vx_mps": cmd[i],
                    "base_vx_raw_mps": vx[i],
                    "base_vx_ma050s_mps": ma[i],
                }
            )

    configure_style()
    fig, ax = plt.subplots(figsize=(11.8, 7.5))

    # Raw signal gives gait-scale variability without dominating the figure.
    ax.plot(
        t,
        vx,
        color=V4N_COLOR,
        linewidth=0.9,
        alpha=0.24,
        label="V4-N measured $v_x$ (raw)",
        zorder=1,
    )

    # Smoothed tracking curve is the main visual signal.
    ax.plot(
        t,
        ma,
        color=V4N_COLOR,
        linewidth=2.8,
        label="V4-N measured $v_x$ (0.5 s MA)",
        zorder=3,
    )

    # Frozen command reference, using the established light-blue reference style.
    ax.axhline(
        EXPECTED_COMMAND_VX,
        color=REFERENCE_COLOR,
        linestyle="--",
        linewidth=2.0,
        alpha=0.95,
        label="Command = 0.40 m/s",
        zorder=2,
    )

    ax.text(
        0.985,
        0.095,
        (
            f"Mean $v_x$ = {computed_mean:.3f} m/s\n"
            f"RMSE = {computed_rmse:.3f} m/s"
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

    ax.set_title(
        "V4-N Forward-Velocity Tracking — Canonical Evaluation",
        pad=17,
        fontweight="normal",
    )
    ax.set_xlabel("Time [s]", labelpad=11)
    ax.set_ylabel("Forward body velocity $v_x$ [m/s]", labelpad=10)

    ax.set_xlim(0.0, EXPECTED_DURATION_S)
    ax.set_ylim(0.32, 0.41)
    ax.set_xticks(np.arange(0.0, 20.1, 2.5))
    ax.set_yticks(np.arange(0.32, 0.401, 0.02))

    ax.grid(True, linestyle="--", zorder=0)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.25)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.155),
        ncol=3,
        frameon=True,
        framealpha=0.96,
        facecolor="white",
        edgecolor="#C8C8C8",
        borderpad=0.45,
        columnspacing=1.5,
        handlelength=2.5,
    )

    fig.text(
        0.5,
        0.016,
        (
            "seed 42  |  20 s canonical evaluation  |  "
            "raw signal + 0.5 s moving average"
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
    print("THESIS FIGURE F03b — GENERATION SUMMARY")
    print("=" * 122)
    print("Status                              : PASS")
    print("Figure                              : V4-N forward-velocity tracking (schema-corrected v2)")
    print("Source                              : full canonical seed42 vx040 20 s evaluation")
    print(f"Checkpoint SHA256                   : {EXPECTED_CHECKPOINT_SHA256}")
    print(f"Samples                             : {len(t)}")
    print(f"dt [s]                              : {EXPECTED_DT:.2f}")
    print(f"Command vx [m/s]                    : {EXPECTED_COMMAND_VX:.2f}")
    print(f"Mean base vx [m/s]                  : {computed_mean:.12f}")
    print(f"RMSE vx [m/s]                       : {computed_rmse:.12f}")
    print(f"Std vx [m/s]                        : {computed_std:.12f}")
    print(f"Raw min / max vx [m/s]              : {np.min(vx):.12f} / {np.max(vx):.12f}")
    print(f"Moving-average window               : {MA_WINDOW_S:.2f} s / {MA_WINDOW_SAMPLES} samples")
    print("Selection interpretation            : full 20 s canonical trial; no temporal cherry-picking")
    print("Completion schema                   : status=complete; termination=false; 1000 samples / 20.0 s")
    print(f"Data freeze                         : {freeze_path}")
    print(f"PNG                                 : {png_path}")
    print(f"PDF vector                          : {pdf_path}")
    print("Simulation / policy run             : NO")
    print("Repository/checkpoints              : NOT MODIFIED")
    print(PASS_MARKER)


if __name__ == "__main__":
    main()
