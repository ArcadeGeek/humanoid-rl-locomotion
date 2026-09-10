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
EXPECTED_COMMAND = (0.40, 0.0, 0.0)

MA_WINDOW_S = 0.50
MA_WINDOW_SAMPLES = 25

V4N_COLOR = "#F58518"

PNG_NAME = "F03c_v4n_body_tilt_stability_v1.png"
PDF_NAME = "F03c_v4n_body_tilt_stability_v1.pdf"
DATA_NAME = "F03c_v4n_body_tilt_stability_data_v1.csv"
PASS_MARKER = "G1_THESIS_F03C_V4N_BODY_TILT_STABILITY_V1_PASS"


def fail(msg: str) -> None:
    raise RuntimeError(msg)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def read_samples(path: Path):
    if not path.is_file():
        fail(f"missing samples CSV: {path}")

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) != EXPECTED_SAMPLES:
        fail(f"expected {EXPECTED_SAMPLES} rows, got {len(rows)}")

    required = (
        "time_s",
        "cmd_vx_mps",
        "cmd_vy_mps",
        "cmd_wz_radps",
        "root_world_up_tilt_rad",
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

    expected_t = np.arange(1, EXPECTED_SAMPLES + 1, dtype=float) * EXPECTED_DT
    if not np.allclose(data["time_s"], expected_t, rtol=0.0, atol=1e-9):
        fail("unexpected 50 Hz time grid")

    for col, expected in (
        ("cmd_vx_mps", EXPECTED_COMMAND[0]),
        ("cmd_vy_mps", EXPECTED_COMMAND[1]),
        ("cmd_wz_radps", EXPECTED_COMMAND[2]),
    ):
        if not np.allclose(data[col], expected, rtol=0.0, atol=1e-12):
            fail(f"{col} does not match expected constant command")

    return data


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
        description="Generate thesis F03c: V4-N body tilt over the full canonical 20 s evaluation."
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

    data = read_samples(args.samples)
    summary = json.loads(args.summary_json.read_text(encoding="utf-8"))

    if str(summary.get("status", "")).lower() != "complete":
        fail(f"canonical status is not complete: {summary.get('status')}")

    if int(summary.get("seed", -1)) != 42:
        fail(f"unexpected seed: {summary.get('seed')}")

    checkpoint_sha = str(summary["checkpoint"]["sha256"]).upper()
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        fail(f"checkpoint SHA mismatch: {checkpoint_sha}")

    termination = summary.get("termination")
    if not isinstance(termination, dict) or termination.get("occurred") is not False:
        fail(f"unexpected termination state: {termination}")

    timing = summary.get("timing")
    if not isinstance(timing, dict):
        fail("missing timing block")

    if int(timing.get("samples", -1)) != EXPECTED_SAMPLES:
        fail("summary sample count mismatch")
    if not math.isclose(
        float(timing.get("policy_dt_s")),
        EXPECTED_DT,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        fail("summary dt mismatch")
    if not math.isclose(
        float(timing.get("measured_duration_s")),
        EXPECTED_DURATION_S,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        fail("summary measured duration mismatch")

    command_mean = summary.get("command_mean")
    if not isinstance(command_mean, dict):
        fail("missing command_mean block")

    summary_command = (
        float(command_mean.get("vx_mps")),
        float(command_mean.get("vy_mps")),
        float(command_mean.get("wz_radps")),
    )
    if any(
        not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12)
        for a, b in zip(summary_command, EXPECTED_COMMAND)
    ):
        fail(f"summary command mismatch: {summary_command}")

    t = data["time_s"]
    tilt_deg = np.degrees(data["root_world_up_tilt_rad"])
    tilt_ma_deg = moving_average(tilt_deg, MA_WINDOW_SAMPLES)

    mean_deg = float(np.mean(tilt_deg))
    std_deg = float(np.std(tilt_deg, ddof=0))
    p95_deg = float(np.percentile(tilt_deg, 95.0))
    max_deg = float(np.max(tilt_deg))
    min_deg = float(np.min(tilt_deg))

    # Frozen numerical guards from the canonical source.
    if not math.isclose(mean_deg, 2.075515856849037, rel_tol=0.0, abs_tol=1e-9):
        fail(f"canonical tilt mean changed: {mean_deg}")
    if not math.isclose(p95_deg, 2.7123677890356754, rel_tol=0.0, abs_tol=1e-9):
        fail(f"canonical tilt p95 changed: {p95_deg}")
    if not math.isclose(max_deg, 2.8091134752497857, rel_tol=0.0, abs_tol=1e-9):
        fail(f"canonical tilt max changed: {max_deg}")

    args.data_dir.mkdir(parents=True, exist_ok=True)
    freeze_path = args.data_dir / DATA_NAME

    with freeze_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "time_s",
                "body_tilt_raw_deg",
                "body_tilt_ma050s_deg",
            ],
        )
        writer.writeheader()
        for i in range(len(t)):
            writer.writerow(
                {
                    "time_s": t[i],
                    "body_tilt_raw_deg": tilt_deg[i],
                    "body_tilt_ma050s_deg": tilt_ma_deg[i],
                }
            )

    configure_style()
    fig, ax = plt.subplots(figsize=(11.8, 7.5))

    ax.plot(
        t,
        tilt_deg,
        color=V4N_COLOR,
        linewidth=0.9,
        alpha=0.24,
        label="V4-N body tilt (raw)",
        zorder=1,
    )

    ax.plot(
        t,
        tilt_ma_deg,
        color=V4N_COLOR,
        linewidth=2.8,
        label="V4-N body tilt (0.5 s MA)",
        zorder=3,
    )

    ax.text(
        0.985,
        0.965,
        "Lower is better",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=13.5,
        color="#555555",
        zorder=7,
    )

    ax.text(
        0.985,
        0.085,
        (
            f"Mean = {mean_deg:.2f}°\n"
            f"95th percentile = {p95_deg:.2f}°\n"
            f"Max = {max_deg:.2f}°"
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
        "V4-N Body Tilt — Canonical Evaluation",
        pad=17,
        fontweight="normal",
    )
    ax.set_xlabel("Time [s]", labelpad=11)
    ax.set_ylabel("Body tilt from vertical [deg]", labelpad=10)

    ax.set_xlim(0.0, EXPECTED_DURATION_S)
    ax.set_ylim(0.0, 3.2)
    ax.set_xticks(np.arange(0.0, 20.1, 2.5))
    ax.set_yticks(np.arange(0.0, 3.1, 0.5))

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
            "20 s canonical evaluation  |  raw signal + 0.5 s moving average"
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
    print("THESIS FIGURE F03c — GENERATION SUMMARY")
    print("=" * 122)
    print("Status                              : PASS")
    print("Figure                              : V4-N body tilt stability")
    print("Source                              : full canonical seed42 vx040 20 s evaluation")
    print(f"Checkpoint SHA256                   : {EXPECTED_CHECKPOINT_SHA256}")
    print(f"Mean tilt [deg]                     : {mean_deg:.12f}")
    print(f"Std tilt [deg]                      : {std_deg:.12f}")
    print(f"P95 tilt [deg]                      : {p95_deg:.12f}")
    print(f"Max tilt [deg]                      : {max_deg:.12f}")
    print(f"Min tilt [deg]                      : {min_deg:.12f}")
    print(f"Moving-average window               : {MA_WINDOW_S:.2f} s / {MA_WINDOW_SAMPLES} samples")
    print("Selection interpretation            : full 20 s canonical trial; no temporal cherry-picking")
    print("Tilt definition                     : root body up-axis angle relative to world up")
    print(f"Data freeze                         : {freeze_path}")
    print(f"PNG                                 : {png_path}")
    print(f"PDF vector                          : {pdf_path}")
    print("Simulation / policy run             : NO")
    print("Repository/checkpoints              : NOT MODIFIED")
    print(PASS_MARKER)


if __name__ == "__main__":
    main()
