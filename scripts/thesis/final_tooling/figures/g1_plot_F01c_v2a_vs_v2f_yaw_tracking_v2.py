#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_ROOT = Path(
    r"C:\humanoid_rl_lab\logs\thesis\figures_final"
    r"\01_v2_v3_evolution"
)
DEFAULT_DATA_ROOT = DEFAULT_ROOT / "data" / "yaw_common_eval_v1"
DEFAULT_PNG_DIR = DEFAULT_ROOT / "png"
DEFAULT_PDF_DIR = DEFAULT_ROOT / "pdf"

CASES = ["V2-A", "V2-F"]
SLUGS = {"V2-A": "V2A", "V2-F": "V2F"}

EXPECTED_CHECKPOINT_SHA = {
    "V2-A": "F1340C5E5CB44AE3157CC9C4D4493BBD1BC6316ED64A029FBD514C7FC0F7B413",
    "V2-F": "F650608C94B50F1BD77D64F975B1F0B87033510B74A159CB3A56860DE4271EE7",
}

EXPECTED_COMMANDS = [-0.15, -0.10, -0.05, 0.00, 0.05, 0.10, 0.15]
EXPECTED_SEEDS = (100, 109)
EXPECTED_TRIALS_PER_COMMAND = 10

COLORS = {
    "V2-A": "#4C78A8",
    "V2-F": "#54A24B",
}
IDEAL_COLOR = "#6BAED6"

PNG_NAME = "F01c_v2a_vs_v2f_yaw_tracking_v2.png"
PDF_NAME = "F01c_v2a_vs_v2f_yaw_tracking_v2.pdf"
DATA_NAME = "F01c_v2a_vs_v2f_yaw_tracking_data_v2.csv"
PASS_MARKER = "G1_THESIS_F01C_V2A_V2F_YAW_TRACKING_V2_PASS"


def fail(message: str) -> None:
    raise RuntimeError(message)


def read_summary(path: Path) -> dict:
    if not path.is_file():
        fail(f"missing benchmark summary: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS":
        fail(f"benchmark status is not PASS: {path}")
    return payload


def read_command_csv(path: Path) -> list[dict]:
    if not path.is_file():
        fail(f"missing command aggregate CSV: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def validate_case(case: str, summary: dict, rows: list[dict]) -> None:
    if summary.get("case") != case:
        fail(f"{case}: case mismatch")
    if str(summary.get("checkpoint_sha256", "")).upper() != EXPECTED_CHECKPOINT_SHA[case]:
        fail(f"{case}: checkpoint SHA mismatch")
    if summary.get("yaw_commands_radps") != EXPECTED_COMMANDS:
        fail(f"{case}: yaw command list mismatch")
    if int(summary.get("seed_start", -1)) != EXPECTED_SEEDS[0]:
        fail(f"{case}: seed_start mismatch")
    if int(summary.get("seed_end", -1)) != EXPECTED_SEEDS[1]:
        fail(f"{case}: seed_end mismatch")
    if summary.get("paired_reset_fingerprints_pass") is not True:
        fail(f"{case}: paired reset guard did not pass")
    if int(summary.get("completed_trials", -1)) != 70:
        fail(f"{case}: expected 70/70 completed trials")

    if len(rows) != len(EXPECTED_COMMANDS):
        fail(f"{case}: expected 7 aggregate rows, got {len(rows)}")

    for expected_cmd, row in zip(EXPECTED_COMMANDS, rows):
        cmd = float(row["command_wz_radps"])
        if not math.isclose(cmd, expected_cmd, rel_tol=0.0, abs_tol=1.0e-12):
            fail(f"{case}: command-order mismatch at {expected_cmd}")
        if int(row["completed_trials"]) != EXPECTED_TRIALS_PER_COMMAND:
            fail(f"{case}: incomplete trials at command {expected_cmd}")
        if int(row["total_trials"]) != EXPECTED_TRIALS_PER_COMMAND:
            fail(f"{case}: total trial mismatch at command {expected_cmd}")
        for key in ("tail_mean_wz_radps_mean", "tail_mean_wz_radps_sd"):
            value = float(row[key])
            if not math.isfinite(value):
                fail(f"{case}: non-finite {key} at command {expected_cmd}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate thesis Figure F01c: standardized yaw response, V2-A vs V2-F."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--png-dir", type=Path, default=DEFAULT_PNG_DIR)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    args = parser.parse_args()

    datasets = {}
    frozen_rows = []

    for case in CASES:
        slug = SLUGS[case]
        summary_path = args.data_root / slug / f"{slug}_yaw_benchmark_summary.json"
        csv_path = args.data_root / slug / f"{slug}_yaw_command_aggregate.csv"

        summary = read_summary(summary_path)
        rows = read_command_csv(csv_path)
        validate_case(case, summary, rows)
        datasets[case] = rows

        for row in rows:
            frozen_rows.append(
                {
                    "policy": case,
                    "command_wz_radps": float(row["command_wz_radps"]),
                    "measured_wz_mean_radps": float(row["tail_mean_wz_radps_mean"]),
                    "measured_wz_sd_radps": float(row["tail_mean_wz_radps_sd"]),
                    "completed_trials": int(row["completed_trials"]),
                    "total_trials": int(row["total_trials"]),
                }
            )

    args.png_dir.mkdir(parents=True, exist_ok=True)
    args.pdf_dir.mkdir(parents=True, exist_ok=True)

    data_path = args.data_root.parent.parent / DATA_NAME
    with data_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(frozen_rows[0].keys()))
        writer.writeheader()
        writer.writerows(frozen_rows)

    configure_style()

    fig, ax = plt.subplots(figsize=(11.8, 7.5))

    x = np.array(EXPECTED_COMMANDS, dtype=float)

    # Ideal yaw response.
    ax.plot(
        x,
        x,
        linestyle="--",
        linewidth=2.2,
        color=IDEAL_COLOR,
        label="Ideal response",
        zorder=1,
    )

    marker_map = {"V2-A": "o", "V2-F": "s"}

    for case in CASES:
        rows = datasets[case]
        y = np.array([float(row["tail_mean_wz_radps_mean"]) for row in rows], dtype=float)
        yerr = np.array([float(row["tail_mean_wz_radps_sd"]) for row in rows], dtype=float)

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            label=case,
            color=COLORS[case],
            marker=marker_map[case],
            markersize=8.0,
            linewidth=2.6,
            elinewidth=1.4,
            capsize=4,
            capthick=1.3,
            markeredgecolor=COLORS[case],
            markerfacecolor=COLORS[case],
            zorder=3,
        )

    # Neutral zero references.
    ax.axhline(0.0, linewidth=0.9, color="#B5B5B5", zorder=0)
    ax.axvline(0.0, linewidth=0.9, color="#B5B5B5", zorder=0)

    ax.set_title(
        "Yaw-Rate Tracking — V2-A vs V2-F",
        pad=17,
        fontweight="normal",
    )
    ax.set_xlabel("Commanded yaw rate [rad/s]", labelpad=11)
    ax.set_ylabel("Measured mean yaw rate [rad/s]", labelpad=10)

    ax.set_xlim(-0.18, 0.18)
    ax.set_ylim(-0.18, 0.18)
    ticks = np.arange(-0.15, 0.151, 0.05)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    ax.grid(True, linestyle="--", zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.25)

    ax.legend(
        loc="upper left",
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#C8C8C8",
    )

    fig.text(
        0.5,
        0.045,
        (
            "$v_x$ command = 0.40 m/s  |  seeds 100–109  |  "
            "final 3 s  |  mean ± 1 SD"
        ),
        ha="center",
        va="bottom",
        fontsize=13.5,
        color="#555555",
    )

    fig.subplots_adjust(left=0.12, right=0.975, top=0.87, bottom=0.19)

    png_path = args.png_dir / PNG_NAME
    pdf_path = args.pdf_dir / PDF_NAME

    fig.savefig(png_path, dpi=180, bbox_inches=None)
    fig.savefig(pdf_path, format="pdf", bbox_inches=None)
    plt.close(fig)

    if not png_path.is_file() or png_path.stat().st_size <= 0:
        fail("PNG output missing or empty")
    if not pdf_path.is_file() or pdf_path.stat().st_size <= 0:
        fail("PDF output missing or empty")

    print("=" * 118)
    print("THESIS FIGURE F01c — GENERATION SUMMARY")
    print("=" * 118)
    print("Status                      : PASS")
    print("Figure                      : Yaw-rate tracking — V2-A vs V2-F (final unified style)")
    print("Policies                    : V2-A / V2-F")
    print("Ideal response              : y = x")
    print("Statistic                   : per-command mean across seeds ± 1 SD")
    print(f"Data freeze                 : {data_path}")
    print(f"PNG                         : {png_path}")
    print(f"PDF vector                  : {pdf_path}")
    print("Repository/checkpoints      : NOT MODIFIED")
    print(PASS_MARKER)


if __name__ == "__main__":
    main()
