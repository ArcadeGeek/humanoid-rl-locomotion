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
DEFAULT_DATA_ROOT = DEFAULT_ROOT / "data" / "straight_common_eval_v1"
DEFAULT_PNG_DIR = DEFAULT_ROOT / "png"
DEFAULT_PDF_DIR = DEFAULT_ROOT / "pdf"

VALIDATION_ORDER = ["V2-A", "V2-D", "V2-E", "V2-F", "V2-G", "V3-C"]
PLOT_ORDER = ["V2-A", "V2-E", "V2-F", "V2-G", "V3-C"]
SLUGS = {
    "V2-A": "V2A",
    "V2-D": "V2D",
    "V2-E": "V2E",
    "V2-F": "V2F",
    "V2-G": "V2G",
    "V3-C": "V3C",
}
EXPECTED_CHECKPOINT_SHA = {
    "V2-A": "F1340C5E5CB44AE3157CC9C4D4493BBD1BC6316ED64A029FBD514C7FC0F7B413",
    "V2-D": "E298A2FAA3959421C5AE00CB3908F84EA34C36CB4D6126194921FFD06AC356FD",
    "V2-E": "154AB9548FA829C59764AFA772D6897BD0F29104729131515A8A616F4879645F",
    "V2-F": "F650608C94B50F1BD77D64F975B1F0B87033510B74A159CB3A56860DE4271EE7",
    "V2-G": "58468EED03F9EF35A9BC898063A3B14AC36C93A32F65DB42B404BDE93CD694D3",
    "V3-C": "C9D93ADE8F32D8F52D1583F4596EAB8906377802265B975E66719C5B41F08458",
}

# Frozen thesis palette for the historical v2/v3 block.
# v4-N is globally reserved as orange in later figures.
COLORS = {
    "V2-A": "#4C78A8",  # blue
    "V2-D": "#B44C43",  # muted brick red
    "V2-E": "#8C8C8C",  # neutral gray
    "V2-F": "#54A24B",  # green
    "V2-G": "#2F5D7E",  # dark blue / petrol
    "V3-C": "#7A5195",  # purple
}

EXPECTED_COMMAND = [0.4, 0.0, 0.0]
EXPECTED_SEEDS = (100, 109)
EXPECTED_TRIALS = 10
EXPECTED_COMPLETION = {
    "V2-A": 10,
    "V2-D": 0,
    "V2-E": 10,
    "V2-F": 10,
    "V2-G": 10,
    "V3-C": 10,
}

FIGURE_ID = "F01a"
PNG_NAME = "F01a_v2v3_straight_tracking_error_v4.png"
PDF_NAME = "F01a_v2v3_straight_tracking_error_v4.pdf"
DATA_NAME = "F01a_v2v3_straight_tracking_error_data_v4.csv"
PASS_MARKER = "G1_THESIS_F01A_V2V3_STRAIGHT_TRACKING_ERROR_V4_PASS"


def fail(message: str) -> None:
    raise RuntimeError(message)


def read_summary(path: Path) -> dict:
    if not path.is_file():
        fail(f"missing aggregate summary: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS":
        fail(f"aggregate status is not PASS: {path}")
    return payload


def validate_summary(case: str, payload: dict) -> None:
    if payload.get("case") != case:
        fail(f"{case}: case mismatch in aggregate")
    if str(payload.get("checkpoint_sha256", "")).upper() != EXPECTED_CHECKPOINT_SHA[case]:
        fail(f"{case}: checkpoint SHA mismatch")
    if payload.get("command") != EXPECTED_COMMAND:
        fail(f"{case}: command mismatch: {payload.get('command')}")
    if int(payload.get("seed_start", -1)) != EXPECTED_SEEDS[0]:
        fail(f"{case}: seed_start mismatch")
    if int(payload.get("seed_end", -1)) != EXPECTED_SEEDS[1]:
        fail(f"{case}: seed_end mismatch")
    if int(payload.get("num_seeds", -1)) != EXPECTED_TRIALS:
        fail(f"{case}: num_seeds mismatch")
    if int(payload.get("completed_trials", -1)) != EXPECTED_COMPLETION[case]:
        fail(f"{case}: completed_trials mismatch")
    if int(payload.get("terminated_trials", -1)) != EXPECTED_TRIALS - EXPECTED_COMPLETION[case]:
        fail(f"{case}: terminated_trials mismatch")

    if case == "V2-D":
        if payload.get("tail_mean_abs_vx_tracking_error_mps_mean") is not None:
            fail("V2-D must remain N/A because no trial completed")
        if payload.get("tail_mean_abs_vx_tracking_error_mps_sd") is not None:
            fail("V2-D SD must remain N/A because no trial completed")
    else:
        mean = payload.get("tail_mean_abs_vx_tracking_error_mps_mean")
        sd = payload.get("tail_mean_abs_vx_tracking_error_mps_sd")
        if mean is None or sd is None:
            fail(f"{case}: missing tracking-error aggregate")
        if not math.isfinite(float(mean)) or not math.isfinite(float(sd)):
            fail(f"{case}: non-finite tracking-error aggregate")


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
        description="Generate thesis Figure F01a from frozen Core-1 v2/v3 straight benchmark summaries."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--png-dir", type=Path, default=DEFAULT_PNG_DIR)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    args = parser.parse_args()

    records = []
    payloads = {}

    for case in VALIDATION_ORDER:
        slug = SLUGS[case]
        path = args.data_root / slug / f"{slug}_aggregate_summary.json"
        payload = read_summary(path)
        validate_summary(case, payload)
        payloads[case] = payload

        if case in PLOT_ORDER:
            records.append(
                {
                    "policy": case,
                    "completed_trials": int(payload["completed_trials"]),
                    "total_trials": int(payload["num_seeds"]),
                    "completion_rate_pct": float(payload["completion_rate_pct"]),
                    "tracking_error_mean_mps": payload.get(
                        "tail_mean_abs_vx_tracking_error_mps_mean"
                    ),
                    "tracking_error_sd_mps": payload.get(
                        "tail_mean_abs_vx_tracking_error_mps_sd"
                    ),
                }
            )

    args.png_dir.mkdir(parents=True, exist_ok=True)
    args.pdf_dir.mkdir(parents=True, exist_ok=True)

    # Freeze exactly the data used by the figure.
    data_path = args.data_root.parent.parent / DATA_NAME
    with data_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    configure_style()

    fig, ax = plt.subplots(figsize=(11.8, 7.5))

    x = np.arange(len(PLOT_ORDER), dtype=float)
    width = 0.67

    means = [
        float(payloads[case]["tail_mean_abs_vx_tracking_error_mps_mean"])
        for case in PLOT_ORDER
    ]
    sds = [
        float(payloads[case]["tail_mean_abs_vx_tracking_error_mps_sd"])
        for case in PLOT_ORDER
    ]
    colors = [COLORS[case] for case in PLOT_ORDER]

    bars = ax.bar(
        x,
        means,
        width=width,
        yerr=sds,
        capsize=5,
        color=colors,
        edgecolor="black",
        linewidth=0.75,
        error_kw={
            "elinewidth": 1.5,
            "capthick": 1.5,
        },
        zorder=3,
    )

    # Numeric labels above valid-policy bars.
    for case, bar, mean, sd in zip(PLOT_ORDER, bars, means, sds):
        y = mean + sd + 0.0042
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="normal",
            color="black",
        )

    ax.set_title(
        "Forward-Velocity Tracking Error — Standardized v2/v3 Benchmark",
        pad=17,
        fontweight="normal",
    )
    ax.set_ylabel("Mean absolute tracking error [m/s]", labelpad=10)
    ax.set_xlabel("Policy checkpoint", labelpad=11)

    ax.set_xticks(x)
    ax.set_xticklabels(PLOT_ORDER)

    # Keep a zero baseline because this is an error metric; lower is better.
    ax.set_ylim(0.0, 0.108)
    ax.set_yticks(np.arange(0.0, 0.101, 0.02))
    ax.grid(axis="y", linestyle="--", zorder=0)
    ax.grid(axis="x", visible=False)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.25)

    ax.text(
        0.99,
        0.965,
        "Lower is better",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=13.5,
        color="#555555",
    )

    # Methodological note, matching the visual language of the user's previous thesis figures.
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
    print("THESIS FIGURE F01a — GENERATION SUMMARY")
    print("=" * 118)
    print("Status                      : PASS")
    print("Figure                      : Forward-velocity tracking error (final unified style)")
    print("Policies plotted            : V2-A / V2-E / V2-F / V2-G / V3-C")
    print("V2-D                        : validated as 0/10 completed; omitted because tracking metric is undefined")
    print("Statistic                   : per-seed sustained-tail mean; aggregate mean ± 1 SD")
    print("V2-D treatment              : excluded from this metric figure; retained in benchmark provenance")
    print(f"Data freeze                 : {data_path}")
    print(f"PNG                         : {png_path}")
    print(f"PDF vector                  : {pdf_path}")
    print("Repository/checkpoints      : NOT MODIFIED")
    print(PASS_MARKER)


if __name__ == "__main__":
    main()
