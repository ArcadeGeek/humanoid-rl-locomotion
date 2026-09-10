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
    / r"logs\thesis\router\low_speed_band_multiseed_offline_audit_seeds100_104"
)
DEFAULT_AGG = DEFAULT_SOURCE / "low_speed_band_aggregate.csv"
DEFAULT_LONG = DEFAULT_SOURCE / "low_speed_band_long.csv"
DEFAULT_REPORT = DEFAULT_SOURCE / "low_speed_band_report.txt"

DEFAULT_ROOT = REPO / r"logs\thesis\figures_final\05_low_speed_tradeoff"
DEFAULT_PNG_DIR = DEFAULT_ROOT / "png"
DEFAULT_PDF_DIR = DEFAULT_ROOT / "pdf"
DEFAULT_DATA_DIR = DEFAULT_ROOT / "data"

EXPECTED_AGG_SHA256 = "73D4C7750EA62A73C7CBA9D44158FBA70AD16EEABF47431375AA5BD7E8CEEAB4"
EXPECTED_LONG_SHA256 = "5CDF713C29DC5B55970096C8C8110EC223FD601F1ECB9AE5710829DEB9718186"
EXPECTED_REPORT_SHA256 = "81CF8C74CA397BE010E40BBD0D0E9C4D268DA8D1959A15BE6CF30EE944D85B49"

EXPECTED_SEEDS = [100, 101, 102, 103, 104]
EXPECTED_SPEEDS = [0.10, 0.15, 0.20]
EXPECTED_ROWS = 15

V2G_COLOR = "#2F5D7E"
V4N_COLOR = "#F58518"

PNG_NAME = "F05a_v2g_v4n_low_speed_tracking_tradeoff_v1.png"
PDF_NAME = "F05a_v2g_v4n_low_speed_tracking_tradeoff_v1.pdf"
DATA_NAME = "F05a_v2g_v4n_low_speed_tracking_tradeoff_data_v1.csv"
PASS_MARKER = "G1_THESIS_F05A_V2G_V4N_LOW_SPEED_TRACKING_TRADEOFF_V1_PASS"


def fail(msg: str) -> None:
    raise RuntimeError(msg)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        fail(f"missing CSV: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sample_sd(values: np.ndarray) -> float:
    if len(values) < 2:
        fail("sample SD requires at least 2 values")
    return float(np.std(values, ddof=1))


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
        description="Generate thesis F05a: V2-G vs V4-N low-speed forward tracking trade-off."
    )
    parser.add_argument("--aggregate", type=Path, default=DEFAULT_AGG)
    parser.add_argument("--long", type=Path, default=DEFAULT_LONG)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--png-dir", type=Path, default=DEFAULT_PNG_DIR)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    for label, path, expected in (
        ("aggregate", args.aggregate, EXPECTED_AGG_SHA256),
        ("long", args.long, EXPECTED_LONG_SHA256),
        ("report", args.report, EXPECTED_REPORT_SHA256),
    ):
        if not path.is_file():
            fail(f"missing {label}: {path}")
        actual = sha256_file(path)
        if actual != expected:
            fail(f"{label} SHA mismatch: expected {expected}, got {actual}")

    report_text = args.report.read_text(encoding="utf-8")
    required_report_tokens = (
        "seeds                    : 100,101,102,103,104",
        "commands                 : vx=0.10, 0.15, 0.20 m/s",
        "comparison               : paired, same seed, same reset fingerprint",
        "startup conditioning     : zero root reset velocity",
        "router / handoff         : ABSENT / ABSENT",
        "holdout                  : UNTOUCHED",
        "simulator execution      : ABSENT (offline summaries only)",
        "tracking advantage strictly increases 0.10->0.15->0.20 : 5/5 seeds",
        "V2G_V4N_LOW_SPEED_BAND_MULTISEED_OFFLINE_AUDIT_PASS",
    )
    for token in required_report_tokens:
        if token not in report_text:
            fail(f"report protocol token missing: {token}")

    long_rows = read_csv(args.long)
    agg_rows = read_csv(args.aggregate)

    if len(long_rows) != EXPECTED_ROWS:
        fail(f"expected {EXPECTED_ROWS} long rows, got {len(long_rows)}")
    if len(agg_rows) != len(EXPECTED_SPEEDS):
        fail(f"expected 3 aggregate rows, got {len(agg_rows)}")

    required_long = {
        "seed",
        "vx",
        "v2g_tail3_mean_abs_tracking_error_mps",
        "v4n_tail3_mean_abs_tracking_error_mps",
        "tracking_advantage_v4n_mps",
    }
    missing_long = required_long - set(long_rows[0])
    if missing_long:
        fail(f"long CSV missing columns: {sorted(missing_long)}")

    required_agg = {
        "vx",
        "seed_count",
        "v2g_tail3_mean_abs_tracking_error_mps_mean",
        "v2g_tail3_mean_abs_tracking_error_mps_sd",
        "v4n_tail3_mean_abs_tracking_error_mps_mean",
        "v4n_tail3_mean_abs_tracking_error_mps_sd",
        "tracking_advantage_v4n_mps_mean",
        "tracking_advantage_v4n_mps_sd",
        "tracking_advantage_v4n_mps_positive_seeds",
    }
    missing_agg = required_agg - set(agg_rows[0])
    if missing_agg:
        fail(f"aggregate CSV missing columns: {sorted(missing_agg)}")

    observed_pairs = sorted(
        (int(r["seed"]), round(float(r["vx"]), 2))
        for r in long_rows
    )
    expected_pairs = sorted(
        (seed, speed)
        for seed in EXPECTED_SEEDS
        for speed in EXPECTED_SPEEDS
    )
    if observed_pairs != expected_pairs:
        fail("seed × speed design does not match expected complete 5×3 paired grid")

    speeds = np.asarray(EXPECTED_SPEEDS, dtype=float)
    v2g_mean = []
    v2g_sd = []
    v4n_mean = []
    v4n_sd = []
    advantage_mean = []
    advantage_sd = []

    freeze_rows = []

    for speed in EXPECTED_SPEEDS:
        rows = [r for r in long_rows if math.isclose(float(r["vx"]), speed, abs_tol=1e-12)]
        seeds = sorted(int(r["seed"]) for r in rows)
        if seeds != EXPECTED_SEEDS:
            fail(f"speed {speed:.2f}: unexpected seed set {seeds}")

        v2g = np.asarray(
            [float(r["v2g_tail3_mean_abs_tracking_error_mps"]) for r in rows],
            dtype=float,
        )
        v4n = np.asarray(
            [float(r["v4n_tail3_mean_abs_tracking_error_mps"]) for r in rows],
            dtype=float,
        )
        adv = np.asarray(
            [float(r["tracking_advantage_v4n_mps"]) for r in rows],
            dtype=float,
        )

        if not np.all(np.isfinite(v2g)) or not np.all(np.isfinite(v4n)) or not np.all(np.isfinite(adv)):
            fail(f"speed {speed:.2f}: non-finite tracking data")

        # Definition guard: positive advantage means V4-N lower error.
        if not np.allclose(v2g - v4n, adv, rtol=0.0, atol=1e-12):
            fail(f"speed {speed:.2f}: tracking advantage definition mismatch")

        arow = next(r for r in agg_rows if math.isclose(float(r["vx"]), speed, abs_tol=1e-12))

        computed = {
            "v2g_mean": float(np.mean(v2g)),
            "v2g_sd": sample_sd(v2g),
            "v4n_mean": float(np.mean(v4n)),
            "v4n_sd": sample_sd(v4n),
            "adv_mean": float(np.mean(adv)),
            "adv_sd": sample_sd(adv),
        }

        expected = {
            "v2g_mean": float(arow["v2g_tail3_mean_abs_tracking_error_mps_mean"]),
            "v2g_sd": float(arow["v2g_tail3_mean_abs_tracking_error_mps_sd"]),
            "v4n_mean": float(arow["v4n_tail3_mean_abs_tracking_error_mps_mean"]),
            "v4n_sd": float(arow["v4n_tail3_mean_abs_tracking_error_mps_sd"]),
            "adv_mean": float(arow["tracking_advantage_v4n_mps_mean"]),
            "adv_sd": float(arow["tracking_advantage_v4n_mps_sd"]),
        }

        for key in computed:
            if not math.isclose(computed[key], expected[key], rel_tol=0.0, abs_tol=1e-12):
                fail(
                    f"speed {speed:.2f}: long/aggregate mismatch for {key}: "
                    f"{computed[key]} vs {expected[key]}"
                )

        if int(arow["seed_count"]) != 5:
            fail(f"speed {speed:.2f}: aggregate seed count is not 5")
        if int(arow["tracking_advantage_v4n_mps_positive_seeds"]) != 5:
            fail(f"speed {speed:.2f}: V4-N tracking advantage is not positive in all 5 seeds")

        v2g_mean.append(computed["v2g_mean"])
        v2g_sd.append(computed["v2g_sd"])
        v4n_mean.append(computed["v4n_mean"])
        v4n_sd.append(computed["v4n_sd"])
        advantage_mean.append(computed["adv_mean"])
        advantage_sd.append(computed["adv_sd"])

        freeze_rows.append(
            {
                "command_vx_mps": speed,
                "seed_count": 5,
                "v2g_tracking_error_mean_mps": computed["v2g_mean"],
                "v2g_tracking_error_sd_mps": computed["v2g_sd"],
                "v4n_tracking_error_mean_mps": computed["v4n_mean"],
                "v4n_tracking_error_sd_mps": computed["v4n_sd"],
                "v4n_tracking_advantage_mean_mps": computed["adv_mean"],
                "v4n_tracking_advantage_sd_mps": computed["adv_sd"],
            }
        )

    v2g_mean = np.asarray(v2g_mean)
    v2g_sd = np.asarray(v2g_sd)
    v4n_mean = np.asarray(v4n_mean)
    v4n_sd = np.asarray(v4n_sd)
    advantage_mean = np.asarray(advantage_mean)

    if not np.all(np.diff(advantage_mean) > 0.0):
        fail("aggregate V4-N tracking advantage is not strictly increasing with speed")

    # Frozen expected aggregate means as an additional scientific guard.
    expected_v2g = np.asarray([
        0.01721020298401515,
        0.024512495495875673,
        0.03832842207749686,
    ])
    expected_v4n = np.asarray([
        0.015138882373770073,
        0.009569927728176119,
        0.008900183236598972,
    ])
    if not np.allclose(v2g_mean, expected_v2g, rtol=0.0, atol=1e-12):
        fail(f"V2-G aggregate means changed: {v2g_mean}")
    if not np.allclose(v4n_mean, expected_v4n, rtol=0.0, atol=1e-12):
        fail(f"V4-N aggregate means changed: {v4n_mean}")

    args.data_dir.mkdir(parents=True, exist_ok=True)
    freeze_path = args.data_dir / DATA_NAME
    with freeze_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(freeze_rows[0].keys()))
        writer.writeheader()
        writer.writerows(freeze_rows)

    configure_style()
    fig, ax = plt.subplots(figsize=(11.8, 7.5))

    ax.errorbar(
        speeds,
        v2g_mean,
        yerr=v2g_sd,
        color=V2G_COLOR,
        linewidth=2.7,
        marker="o",
        markersize=8.0,
        markerfacecolor=V2G_COLOR,
        markeredgecolor="black",
        markeredgewidth=0.7,
        capsize=5,
        elinewidth=1.7,
        capthick=1.7,
        label="V2-G",
        zorder=4,
    )

    ax.errorbar(
        speeds,
        v4n_mean,
        yerr=v4n_sd,
        color=V4N_COLOR,
        linewidth=2.7,
        marker="s",
        markersize=8.0,
        markerfacecolor=V4N_COLOR,
        markeredgecolor="black",
        markeredgewidth=0.7,
        capsize=5,
        elinewidth=1.7,
        capthick=1.7,
        label="V4-N",
        zorder=4,
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
        0.035,
        0.94,
        "V4-N tracking advantage increases\nat every tested speed (5/5 seeds)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13.2,
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
        "Low-Speed Forward Tracking — V2-G vs V4-N",
        pad=17,
        fontweight="normal",
    )
    ax.set_xlabel("Commanded forward velocity $v_x$ [m/s]", labelpad=11)
    ax.set_ylabel("Mean absolute tracking error [m/s]", labelpad=10)

    ax.set_xlim(0.09, 0.21)
    ax.set_ylim(0.0, 0.043)
    ax.set_xticks(EXPECTED_SPEEDS)
    ax.set_yticks(np.arange(0.0, 0.041, 0.005))

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
        columnspacing=2.0,
        handlelength=2.6,
    )

    fig.text(
        0.5,
        0.014,
        (
            "seeds 100–104  |  paired reset fingerprints  |  final 3 s  |  "
            "mean ± 1 SD  |  development set"
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
    print("THESIS FIGURE F05a — GENERATION SUMMARY")
    print("=" * 124)
    print("Status                              : PASS")
    print("Figure                              : low-speed forward tracking V2-G vs V4-N")
    print("Protocol                            : paired seeds100-104, vx 0.10/0.15/0.20, final 3 s")
    print("Router / handoff                    : ABSENT / ABSENT")
    print("Holdout                             : UNTOUCHED")
    print("Simulation                          : NO (offline audit only)")
    for i, speed in enumerate(EXPECTED_SPEEDS):
        print(
            f"vx={speed:.2f} | "
            f"V2-G err={v2g_mean[i]:.9f}±{v2g_sd[i]:.9f} | "
            f"V4-N err={v4n_mean[i]:.9f}±{v4n_sd[i]:.9f} | "
            f"adv(V4-N)={advantage_mean[i]:.9f}"
        )
    print("Ordered structure                   : V4-N tracking advantage strictly increasing 0.10->0.15->0.20")
    print("Threshold interpretation            : development-calibrated evidence; NOT global optimization")
    print(f"Data freeze                         : {freeze_path}")
    print(f"PNG                                 : {png_path}")
    print(f"PDF vector                          : {pdf_path}")
    print("Repository/checkpoints              : NOT MODIFIED")
    print(PASS_MARKER)


if __name__ == "__main__":
    main()
