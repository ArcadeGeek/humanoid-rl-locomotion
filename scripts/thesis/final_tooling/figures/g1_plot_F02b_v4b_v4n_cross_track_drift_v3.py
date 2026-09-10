#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(r"C:\humanoid_rl_lab")
DEFAULT_ROOT = REPO / r"logs\thesis\figures_final\02_v4_campaign"
DEFAULT_PNG_DIR = DEFAULT_ROOT / "png"
DEFAULT_PDF_DIR = DEFAULT_ROOT / "pdf"
DEFAULT_DATA_DIR = DEFAULT_ROOT / "data"

INPUTS = {
    "V4-B": {
        "csv": REPO / (
            r"logs\evaluations\g1\v4b_multiseed_direction_dev_100_109_vx040_20s"
            r"\multiseed_direction_summary.csv"
        ),
        "sha256": "DC7E98F05771F26A1A19FFFDF132C5EECE8CA590D8C9286CA175D5A397C43E6C",
        "color": "#54A24B",
    },
    "V4-N": {
        "csv": REPO / (
            r"logs\evaluations\g1\v4n_multiseed_direction_dev_100_109_vx040_20s"
            r"\multiseed_direction_summary.csv"
        ),
        "sha256": "3EDDBD65D86F5D91BC583E421EF56221E40A702562E260B8561C7BD378E73635",
        "color": "#F58518",  # frozen thesis convention: v4-N = orange
    },
}

EXPECTED_SEEDS = list(range(100, 110))
COMMAND_VX = 0.40
DURATION_S = 20.0

PNG_NAME = "F02b_v4b_v4n_cross_track_drift_v3.png"
PDF_NAME = "F02b_v4b_v4n_cross_track_drift_v3.pdf"
DATA_NAME = "F02b_v4b_v4n_cross_track_drift_data_v3.csv"
PASS_MARKER = "G1_THESIS_F02B_V4B_V4N_CROSS_TRACK_DRIFT_V3_PASS"


def fail(msg: str) -> None:
    raise RuntimeError(msg)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def load_case(case: str, path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        fail(f"{case}: missing input CSV: {path}")

    actual_sha = sha256_file(path)
    if actual_sha != INPUTS[case]["sha256"]:
        fail(
            f"{case}: CSV SHA mismatch: expected {INPUTS[case]['sha256']}, got {actual_sha}"
        )

    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != 10:
        fail(f"{case}: expected 10 seed rows, got {len(rows)}")

    required = {
        "seed",
        "samples",
        "mean_base_vx_mps",
        "mean_base_vy_mps",
        "mean_base_wz_radps",
        "cross_track_initial_heading_m",
        "along_track_initial_heading_m",
    }
    missing = required - set(rows[0].keys())
    if missing:
        fail(f"{case}: missing columns: {sorted(missing)}")

    seeds = [int(row["seed"]) for row in rows]
    if seeds != EXPECTED_SEEDS:
        fail(f"{case}: expected seeds {EXPECTED_SEEDS}, got {seeds}")

    for row in rows:
        if int(row["samples"]) != 1000:
            fail(f"{case}: expected 1000 samples for seed {row['seed']}")
        for key in required - {"seed", "samples"}:
            value = float(row[key])
            if not math.isfinite(value):
                fail(f"{case}: non-finite {key} for seed {row['seed']}")

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
        description="Generate thesis F02b: v4-B vs v4-N absolute cross-track drift."
    )
    parser.add_argument("--v4b-csv", type=Path, default=INPUTS["V4-B"]["csv"])
    parser.add_argument("--v4n-csv", type=Path, default=INPUTS["V4-N"]["csv"])
    parser.add_argument("--png-dir", type=Path, default=DEFAULT_PNG_DIR)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    rows_by_case = {
        "V4-B": load_case("V4-B", args.v4b_csv),
        "V4-N": load_case("V4-N", args.v4n_csv),
    }

    values = {}
    signed_values = {}
    vx_errors = {}

    for case in ("V4-B", "V4-N"):
        rows = rows_by_case[case]

        signed = [float(r["cross_track_initial_heading_m"]) for r in rows]
        absolute = [abs(v) for v in signed]
        vx_err = [abs(float(r["mean_base_vx_mps"]) - COMMAND_VX) for r in rows]

        signed_values[case] = signed
        values[case] = absolute
        vx_errors[case] = vx_err

    stats = {}
    for case in ("V4-B", "V4-N"):
        stats[case] = {
            "mean_abs_cross_track_m": statistics.fmean(values[case]),
            "sd_abs_cross_track_m": statistics.stdev(values[case]),
            "mean_signed_cross_track_m": statistics.fmean(signed_values[case]),
            "sd_signed_cross_track_m": statistics.stdev(signed_values[case]),
            "mean_abs_vx_tracking_error_mps": statistics.fmean(vx_errors[case]),
            "sd_abs_vx_tracking_error_mps": statistics.stdev(vx_errors[case]),
        }

    reduction_pct = (
        100.0
        * (
            stats["V4-B"]["mean_abs_cross_track_m"]
            - stats["V4-N"]["mean_abs_cross_track_m"]
        )
        / stats["V4-B"]["mean_abs_cross_track_m"]
    )

    # Data freeze for exact plotted seed-level values.
    args.data_dir.mkdir(parents=True, exist_ok=True)
    freeze_path = args.data_dir / DATA_NAME

    with freeze_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "policy",
            "seed",
            "signed_cross_track_m",
            "absolute_cross_track_m",
            "mean_base_vx_mps",
            "absolute_vx_tracking_error_mps",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for case in ("V4-B", "V4-N"):
            for row in rows_by_case[case]:
                signed = float(row["cross_track_initial_heading_m"])
                vx = float(row["mean_base_vx_mps"])
                writer.writerow(
                    {
                        "policy": case,
                        "seed": int(row["seed"]),
                        "signed_cross_track_m": signed,
                        "absolute_cross_track_m": abs(signed),
                        "mean_base_vx_mps": vx,
                        "absolute_vx_tracking_error_mps": abs(vx - COMMAND_VX),
                    }
                )

    configure_style()
    fig, ax = plt.subplots(figsize=(9.8, 7.2))

    cases = ["V4-B", "V4-N"]
    x = np.arange(len(cases), dtype=float)
    means = np.array([stats[c]["mean_abs_cross_track_m"] for c in cases])
    sds = np.array([stats[c]["sd_abs_cross_track_m"] for c in cases])

    bars = ax.bar(
        x,
        means,
        yerr=sds,
        width=0.46,
        capsize=7,
        edgecolor="black",
        linewidth=1.15,
        error_kw={
            "elinewidth": 1.6,
            "capthick": 1.6,
            "ecolor": "black",
        },
        color=[INPUTS[c]["color"] for c in cases],
        zorder=2,
    )

    # Numeric labels above error bars.
    for i, case in enumerate(cases):
        top = means[i] + sds[i]
        ax.text(
            x[i],
            top + 0.010,
            f"{means[i]:.3f} m",
            ha="center",
            va="bottom",
            fontsize=15,
            color="#202020",
            zorder=6,
        )

    # Explicit comparative annotation in a light legend-style box.
    ax.text(
        0.5,
        0.420,
        f"V4-N: {reduction_pct:.0f}% lower cross-track drift",
        ha="center",
        va="center",
        fontsize=13.5,
        color="#202020",
        zorder=6,
        bbox={
            "boxstyle": "round,pad=0.38",
            "facecolor": "white",
            "edgecolor": "#C8C8C8",
            "linewidth": 1.0,
            "alpha": 0.96,
        },
    )

    ax.set_title("Straight-Line Cross-Track Drift — V4-B vs V4-N", pad=17, fontweight="normal")
    ax.set_ylabel("Absolute cross-track displacement [m]", labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(cases)

    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0.0, 0.44)
    ax.set_yticks(np.arange(0.0, 0.41, 0.10))

    ax.grid(True, axis="y", linestyle="--", zorder=0)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.25)

    fig.text(
        0.5,
        0.022,
        "$v_x$ command = 0.40 m/s  |  seeds 100–109  |  20 s  |  mean ± 1 SD",
        ha="center",
        va="bottom",
        fontsize=13.5,
        color="#555555",
    )

    fig.subplots_adjust(left=0.15, right=0.97, top=0.87, bottom=0.125)

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

    print("=" * 118)
    print("THESIS FIGURE F02b — GENERATION SUMMARY")
    print("=" * 118)
    print("Status                              : PASS")
    print("Figure                              : Straight-line absolute cross-track drift (boxed comparison v3)")
    print("Policies                            : V4-B / V4-N")
    print("Protocol                            : vx=0.40 m/s, 20 s, seeds 100-109")
    print("V4-B |cross-track| mean ± SD [m]    : "
          f"{stats['V4-B']['mean_abs_cross_track_m']:.9f} ± "
          f"{stats['V4-B']['sd_abs_cross_track_m']:.9f}")
    print("V4-N |cross-track| mean ± SD [m]    : "
          f"{stats['V4-N']['mean_abs_cross_track_m']:.9f} ± "
          f"{stats['V4-N']['sd_abs_cross_track_m']:.9f}")
    print(f"Absolute drift reduction V4-N       : {reduction_pct:.6f}%")
    print("V4-B signed cross-track mean [m]     : "
          f"{stats['V4-B']['mean_signed_cross_track_m']:.9f}")
    print("V4-N signed cross-track mean [m]     : "
          f"{stats['V4-N']['mean_signed_cross_track_m']:.9f}")
    print("V4-B forward tracking error [m/s]    : "
          f"{stats['V4-B']['mean_abs_vx_tracking_error_mps']:.9f}")
    print("V4-N forward tracking error [m/s]    : "
          f"{stats['V4-N']['mean_abs_vx_tracking_error_mps']:.9f}")
    print("Interpretation guard                : directional consistency metric; not global superiority")
    print("Visual revision                     : comparison bracket removed; light legend-style annotation box")
    print(f"Data freeze                         : {freeze_path}")
    print(f"PNG                                 : {png_path}")
    print(f"PDF vector                          : {pdf_path}")
    print("Simulation / policy run             : NO")
    print("Repository/checkpoints              : NOT MODIFIED")
    print(PASS_MARKER)


if __name__ == "__main__":
    main()
