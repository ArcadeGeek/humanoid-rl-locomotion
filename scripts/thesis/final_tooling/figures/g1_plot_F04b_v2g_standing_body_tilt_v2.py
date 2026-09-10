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

EXPECTED_MEAN_TILT_DEG = 0.957705962714085
EXPECTED_P95_TILT_DEG = 1.0322748358073957
EXPECTED_MAX_TILT_DEG = 1.119966076482749

MA_WINDOW_S = 0.50
MA_WINDOW_SAMPLES = 25

V2G_COLOR = "#2F5D7E"

PNG_NAME = "F04b_v2g_standing_body_tilt_v2.png"
PDF_NAME = "F04b_v2g_standing_body_tilt_v2.pdf"
DATA_NAME = "F04b_v2g_standing_body_tilt_data_v2.csv"
PASS_MARKER = "G1_THESIS_F04B_V2G_STANDING_BODY_TILT_V2_PASS"


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
        fail("metadata dt mismatch")
    if int(meta["record_steps"]) != EXPECTED_TOTAL_SAMPLES:
        fail("metadata sample count mismatch")
    if int(meta["reset_count"]) != 1:
        fail("metadata reset count mismatch")
    if not math.isclose(float(meta["cmd_vx_min"]), 0.0, rel_tol=0.0, abs_tol=1e-12):
        fail("metadata vx min is not zero")
    if not math.isclose(float(meta["cmd_vx_max"]), 0.0, rel_tol=0.0, abs_tol=1e-12):
        fail("metadata vx max is not zero")

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
        "root_qw",
        "root_qx",
        "root_qy",
        "root_qz",
        "root_x_w",
        "root_y_w",
    }
    missing = required - set(reader.fieldnames or [])
    if missing:
        fail(f"CSV missing fields: {sorted(missing)}")

    return rows


def root_tilt_deg_from_wxyz(
    qw: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    qz: np.ndarray,
) -> np.ndarray:
    norm = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if np.any(~np.isfinite(norm)) or np.any(norm <= 0.0):
        fail("invalid quaternion norm")

    qw = qw / norm
    qx = qx / norm
    qy = qy / norm
    qz = qz / norm

    # For quaternion wxyz, the world-z component of the rotated body-up
    # unit vector is R[2,2] = 1 - 2*(qx^2 + qy^2).
    body_up_dot_world_up = 1.0 - 2.0 * (qx * qx + qy * qy)
    body_up_dot_world_up = np.clip(body_up_dot_world_up, -1.0, 1.0)

    return np.degrees(np.arccos(body_up_dot_world_up))


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
        description="Generate thesis F04b: V2-G pre-reset zero-command standing body tilt."
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

    # Frozen pre-reset window: only samples 1..849 are scientific standing data.
    if int(rows[0]["sample"]) != 1:
        fail("first sample changed")
    if int(rows[VALID_PRE_RESET_SAMPLES - 1]["sample"]) != VALID_PRE_RESET_SAMPLES:
        fail("last valid sample changed")
    if int(rows[VALID_PRE_RESET_SAMPLES]["sample"]) != VALID_PRE_RESET_SAMPLES + 1:
        fail("first post-reset sample changed")

    first_t = float(rows[0]["sim_time_s"])
    last_valid_t = float(rows[VALID_PRE_RESET_SAMPLES - 1]["sim_time_s"])
    first_post_t = float(rows[VALID_PRE_RESET_SAMPLES]["sim_time_s"])

    if not math.isclose(last_valid_t, EXPECTED_LAST_VALID_TIMESTAMP_S, rel_tol=0.0, abs_tol=1e-12):
        fail(f"last valid timestamp changed: {last_valid_t}")
    if not math.isclose(first_post_t, EXPECTED_FIRST_POST_RESET_TIMESTAMP_S, rel_tol=0.0, abs_tol=1e-12):
        fail(f"first post-reset timestamp changed: {first_post_t}")
    if not math.isclose(last_valid_t - first_t, EXPECTED_VALID_DURATION_S, rel_tol=0.0, abs_tol=1e-12):
        fail("valid standing duration changed")

    # Confirm the reset discontinuity exists immediately after the valid window.
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
    t = sim_t - sim_t[0]

    qw = np.asarray([float(r["root_qw"]) for r in valid], dtype=float)
    qx = np.asarray([float(r["root_qx"]) for r in valid], dtype=float)
    qy = np.asarray([float(r["root_qy"]) for r in valid], dtype=float)
    qz = np.asarray([float(r["root_qz"]) for r in valid], dtype=float)

    tilt_deg = root_tilt_deg_from_wxyz(qw, qx, qy, qz)
    tilt_ma_deg = moving_average(tilt_deg, MA_WINDOW_SAMPLES)

    mean_deg = float(np.mean(tilt_deg))
    std_deg = float(np.std(tilt_deg, ddof=0))
    p95_deg = float(np.percentile(tilt_deg, 95.0))
    max_deg = float(np.max(tilt_deg))
    min_deg = float(np.min(tilt_deg))

    if not math.isclose(mean_deg, EXPECTED_MEAN_TILT_DEG, rel_tol=0.0, abs_tol=1e-9):
        fail(f"mean tilt changed: {mean_deg}")
    if not math.isclose(p95_deg, EXPECTED_P95_TILT_DEG, rel_tol=0.0, abs_tol=1e-9):
        fail(f"p95 tilt changed: {p95_deg}")
    if not math.isclose(max_deg, EXPECTED_MAX_TILT_DEG, rel_tol=0.0, abs_tol=1e-9):
        fail(f"max tilt changed: {max_deg}")

    args.data_dir.mkdir(parents=True, exist_ok=True)
    freeze_path = args.data_dir / DATA_NAME

    with freeze_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "elapsed_time_s",
                "recorded_sim_time_s",
                "body_tilt_raw_deg",
                "body_tilt_ma050s_deg",
            ],
        )
        writer.writeheader()
        for i in range(len(valid)):
            writer.writerow(
                {
                    "elapsed_time_s": t[i],
                    "recorded_sim_time_s": sim_t[i],
                    "body_tilt_raw_deg": tilt_deg[i],
                    "body_tilt_ma050s_deg": tilt_ma_deg[i],
                }
            )

    configure_style()
    fig, ax = plt.subplots(figsize=(11.8, 7.5))

    ax.plot(
        t,
        tilt_deg,
        color=V2G_COLOR,
        linewidth=0.9,
        alpha=0.24,
        label="V2-G body tilt (raw)",
        zorder=1,
    )

    ax.plot(
        t,
        tilt_ma_deg,
        color=V2G_COLOR,
        linewidth=2.8,
        label="V2-G body tilt (0.5 s MA)",
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
        0.980,
        0.885,
        (
            f"Mean = {mean_deg:.2f}°\n"
            f"95th percentile = {p95_deg:.2f}°\n"
            f"Max = {max_deg:.2f}°"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
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
        "V2-G Standing — Body Tilt Stability",
        pad=17,
        fontweight="normal",
    )
    ax.set_xlabel("Elapsed standing time [s]", labelpad=11)
    ax.set_ylabel("Body tilt from vertical [deg]", labelpad=10)

    ax.set_xlim(0.0, EXPECTED_VALID_DURATION_S)
    ax.set_ylim(0.90, 1.14)
    ax.set_xticks(np.arange(0.0, 16.1, 2.0))
    ax.set_yticks(np.arange(0.90, 1.141, 0.04))

    ax.grid(True, linestyle="--", zorder=0)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.25)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.165),
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
        0.014,
        (
            "zero command  |  seed 42  |  849 pre-reset samples  |  "
            "16.96 s valid standing  |  timeout/reset excluded"
        ),
        ha="center",
        va="bottom",
        fontsize=13.5,
        color="#555555",
    )

    fig.subplots_adjust(left=0.12, right=0.975, top=0.87, bottom=0.245)

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
    print("THESIS FIGURE F04b — GENERATION SUMMARY")
    print("=" * 124)
    print("Status                              : PASS")
    print("Figure                              : V2-G standing body tilt stability (final visual v2)")
    print("Source                              : frozen zero-command seed42 20 s probe")
    print(f"Checkpoint SHA256                   : {EXPECTED_CHECKPOINT_SHA256}")
    print(f"Valid pre-reset samples             : {len(valid)}")
    print(f"Valid elapsed standing duration [s] : {t[-1]:.2f}")
    print(f"First post-reset timestamp [s]      : {first_post_t:.2f}")
    print(f"Reset XY discontinuity [m]          : {reset_xy_jump:.9f}")
    print(f"Mean tilt [deg]                     : {mean_deg:.12f}")
    print(f"Std tilt [deg]                      : {std_deg:.12f}")
    print(f"P95 tilt [deg]                      : {p95_deg:.12f}")
    print(f"Max tilt [deg]                      : {max_deg:.12f}")
    print(f"Min tilt [deg]                      : {min_deg:.12f}")
    print(f"Moving-average window               : {MA_WINDOW_S:.2f} s / {MA_WINDOW_SAMPLES} samples")
    print("Tilt definition                     : root body-up angle relative to world vertical")
    print("Visual revision                     : metric box moved upper-right below Lower is better")
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
